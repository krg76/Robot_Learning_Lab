import time
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xarm.wrapper import XArmAPI


# ============================================================
# Interfaces students / instructors can replace
# ============================================================

class ObjectPoseProvider:
    """
    Interface for perception.

    Must return the cube pose in WORLD coordinates, in meters.
    You should replace this with AprilTag / mocap / RGBD tracking.
    """

    def get_object_pose(self) -> Dict[str, np.ndarray]:
        """
        Returns:
            {
                "pos": np.ndarray shape (3,), meters
                "rot": np.ndarray shape (3,), e.g. Euler xyz or any 3-vector convention
            }
        """
        raise NotImplementedError

    def reset_scene(self) -> None:
        """
        Optional hook to reset / resample trackers or wait for stable pose.
        """
        pass
    
class FixedObjectPoseProvider(ObjectPoseProvider):
    def __init__(self, tcp_pos=(0.331, 0.233, 0.172)):
        self.tcp = np.array(tcp_pos, dtype=np.float32)

        # 🔥 convert TCP → object center
        self.offset = np.array([0.0, 0.0, 0.04], dtype=np.float32)

    def get_object_pose(self):
        obj_center = self.tcp - self.offset

        return {
            "pos": obj_center,
            "rot": np.array([1, 0, 0, 0], dtype=np.float32),
        }


# ============================================================
# Config
# ============================================================

@dataclass
class XArmRealEnvConfig:
    ip: str = "192.168.1.233"

    # Fixed wrist pose in degrees.
    # Match this to however your sim tool frame is defined.
    roll: float = 180.0
    pitch: float = 0.0
    yaw: float = 0.0

    # Cartesian control
    control_hz: float = 10.0
    max_delta_xyz: float = 0.05    # meters per env.step action unit — matches sim (1/n_substeps = 1/20 = 0.05)
    default_speed_mm_s: float = 100.0  # slow/gentle: reduces force on cube contact
    default_acc_mm_s2: float = 800.0

    # Workspace limits in WORLD coordinates (meters)
    workspace_low: tuple = (0.1, -0.05, 0.10)
    workspace_high: tuple = (0.65, 0.35, 0.5)

    # Proximity grasp: close gripper when EEF is within this distance of the object.
    # The sim policy's grip action is delayed vs. arm arrival; this override bridges the gap.
    grasp_proximity_m: float = 0.10  # 10 cm — trigger before arm contacts cube

    # Home / hover positions
    # Calibrated for red cube: home_rel = [-0.375, -0.005, 0.099] matching sim exactly
    home_xyz: tuple = (0.25, 0.1797, 0.301)
    pregrasp_height: float = 0.2

    # Gripper
    gripper_open: int = 800
    gripper_closed: int = 120
    gripper_speed: int = 3000

    # Episode / task
    episode_length: int =350
    place_tol: float = 0.02
    lift_z_thresh: float = 0.10    # matches sim _lift_z_thresh

    # Goal in world coordinates (meters).
    # Calibrated for red cube: goal_rel ≈ [-0.15, +0.10, -0.03], matching sim distribution.
    # Place the goal marker at ~(0.475, 0.285) — same x as red cube, 20 cm further in +y.
    default_goal_xyz: tuple = (0.475, 0.2847, 0.172)

    # Settling
    action_wait_s: float = 0.06
    reset_wait_s: float = 1.0


# ============================================================
# Real xArm pick-place env
# ============================================================

class XArmPickPlaceRealEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: XArmRealEnvConfig,
        pose_provider: ObjectPoseProvider,
        goal_sampler: Optional[Callable[[np.random.Generator], np.ndarray]] = None,
    ):
        super().__init__()
        self.cfg = config
        self.pose_provider = pose_provider
        self.goal_sampler = goal_sampler

        self.dt = 1.0 / self.cfg.control_hz
        self.np_random = np.random.default_rng()

        self.arm = XArmAPI(self.cfg.ip)
        self._connect_and_configure()

        self.workspace_low = np.asarray(self.cfg.workspace_low, dtype=np.float32)
        self.workspace_high = np.asarray(self.cfg.workspace_high, dtype=np.float32)
        self.home_xyz = np.asarray(self.cfg.home_xyz, dtype=np.float32)

        self.goal = np.asarray(self.cfg.default_goal_xyz, dtype=np.float32)
        self._init_obj_z = 0.172

        # TABLE CENTER — calibrated for the RED cube so that ALL observations match the sim:
        #   obj_rel = [-0.15, -0.10, -0.07]  (sim: same)
        #   home eef_rel = [-0.375, -0.005, 0.099]  (sim: same)
        #   eef_to_obj = [-0.225, 0.095, 0.169]  (sim: same)
        # Derivation: table_center = red_obj_world - [-0.15, -0.10, -0.07]
        #   red obj_world = [0.475, 0.0847, 0.132]  → table_center = [0.625, 0.1847, 0.202]
        self.table_center = np.array([0.625, 0.1847, 0.202], dtype=np.float32)

        self._step_count = 0
        self._last_eef = None
        self._last_obj = None
        self._last_obj_rot = None
        self._last_time = None
        self._closed_last = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
            dtype=np.float32,
        )

    # --------------------------
    # Robot setup
    # --------------------------

    def _check_code(self, code, msg):
        if code != 0:
            raise RuntimeError(f"{msg} failed with code {code}")

    def _connect_and_configure(self):
        self.arm.connect()
        time.sleep(0.5)

        self.arm.clean_warn()
        self.arm.clean_error()
        time.sleep(0.2)

        self._check_code(self.arm.motion_enable(True), "motion_enable")
        self._check_code(self.arm.set_mode(0), "set_mode")
        self._check_code(self.arm.set_state(0), "set_state")

        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(self.cfg.gripper_speed)
        self.arm.set_gripper_position(self.cfg.gripper_open, wait=True)

    # --------------------------
    # Robot helpers
    # --------------------------

    def _get_eef_position(self):
        code, pose = self.arm.get_position(is_radian=False)
        self._check_code(code, "get_position")
        return np.asarray(pose[:3], dtype=np.float32) / 1000.0

    def _move_eef_xyz(self, xyz, wait=True):
        xyz = np.clip(xyz, self.workspace_low, self.workspace_high)
        xyz_mm = 1000.0 * xyz

        code = self.arm.set_position(
            x=float(xyz_mm[0]),
            y=float(xyz_mm[1]),
            z=float(xyz_mm[2]),
            roll=self.cfg.roll,
            pitch=self.cfg.pitch,
            yaw=self.cfg.yaw,
            speed=self.cfg.default_speed_mm_s,
            mvacc=self.cfg.default_acc_mm_s2,
            is_radian=False,
            wait=wait,
        )
        self._check_code(code, "set_position")

    def _open_gripper(self):
        self.arm.set_gripper_position(self.cfg.gripper_open, wait=True)
        self._closed_last = False

    def _close_gripper(self):
        self.arm.set_gripper_position(self.cfg.gripper_closed, wait=True)
        self._closed_last = True

    # --------------------------
    # Object
    # --------------------------

    def _get_object_state(self):
        pose = self.pose_provider.get_object_pose()

        obj = np.asarray(pose["pos"], dtype=np.float32)

        # ✅ Always valid quaternion
        obj_rot = np.asarray(
            pose.get("rot", np.array([1, 0, 0, 0], dtype=np.float32)),
            dtype=np.float32,
        )

        return obj, obj_rot

    # --------------------------
    # Reward helpers
    # --------------------------

    @property
    def lift_z_target(self):
        return float(self._init_obj_z + self.cfg.lift_z_thresh)

    def is_success(self, obj_table_rel):
        # obj_table_rel is table-centered; self.goal is in world frame — convert to same space.
        obj_world = obj_table_rel + self.table_center
        return np.linalg.norm(obj_world - self.goal) <= self.cfg.place_tol

    # --------------------------
    # OBSERVATION (FIXED)
    # --------------------------

    def _get_obs(self):
        now = time.time()

        # --- world frame ---
        eef_w = self._get_eef_position()
        obj_w, obj_rot = self._get_object_state()
        goal_w = self.goal.copy()

        # 🔥 convert to table-centered frame
        eef = eef_w - self.table_center
        obj = obj_w - self.table_center
        goal = goal_w - self.table_center

        dt = self.dt if self._last_time is None else max(now - self._last_time, 1e-4)

        # velocities — sim stores get_site_xvelp * dt (displacement per step, NOT velocity).
        # Use raw finite-difference displacement to match that convention.
        eef_velp = (
            np.zeros(3, dtype=np.float32)
            if self._last_eef is None
            else (eef - self._last_eef).astype(np.float32)
        )

        obj_velp = (
            np.zeros(3, dtype=np.float32)
            if self._last_obj is None
            else (obj - self._last_obj).astype(np.float32)
        )

        obj_velr = (
            np.zeros(3, dtype=np.float32)
            if self._last_obj_rot is None
            else (obj_rot[:3] - self._last_obj_rot[:3]).astype(np.float32)
        )

        # relations
        eef_to_obj = eef - obj
        obj_to_goal = obj - goal

        # scalars
        scalars = np.array(
            [
                np.linalg.norm(eef_to_obj),
                np.linalg.norm(eef_to_obj[:-1]),
                np.linalg.norm(obj_to_goal),
                np.linalg.norm(obj_to_goal[:-1]),
                self.lift_z_target,
                self.lift_z_target - obj[-1],
            ],
            dtype=np.float32,
        )

        # 🔥 match sim: gripper ≈ 0
        gripper = np.array([0.0], dtype=np.float32)

        obs = np.concatenate(
            [
                eef,
                eef_velp,
                obj,
                obj_rot,
                obj_velp,
                obj_velr,
                goal,
                eef_to_obj,
                obj_to_goal,
                scalars,
                gripper,
            ],
            axis=0,
        ).astype(np.float32)

        # update history
        self._last_eef = eef.copy()
        self._last_obj = obj.copy()
        self._last_obj_rot = obj_rot.copy()
        self._last_time = now

        if self._step_count == 0:
            print(f"[obs debug] eef={eef.round(3)}  obj={obj.round(3)}  goal={goal.round(3)}")
            print(f"[obs debug] lift_z_target={self.lift_z_target:.4f}  lift_remaining={scalars[5]:.4f}")
            print(f"[obs debug] eef_to_obj={eef_to_obj.round(3)}  obj_to_goal={obj_to_goal.round(3)}")

        return obs

    # --------------------------
    # Gym API
    # --------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0
        self.goal = (
            self.goal_sampler(self.np_random)
            if self.goal_sampler
            else self.goal
        )

        self.pose_provider.reset_scene()

        self._open_gripper()
        self._move_eef_xyz(self.home_xyz, wait=True)
        time.sleep(self.cfg.reset_wait_s)

        obj_world, _ = self._get_object_state()
        obj = obj_world - self.table_center
        # Sim stores _init_z in MuJoCo world z (≈0.566 for table-height objects).
        # Real robot world z is ~0.434 m lower (MuJoCo table height 0.636 vs real 0.172).
        # Add the offset so lift_z_target and lift_remaining match the sim's exact values:
        #   lift_z_target = 0.132 + 0.434 + 0.10 = 0.666  (sim: 0.666)
        #   lift_remaining = 0.666 - (-0.07) = 0.736  (sim: 0.736)
        Z_WORLD_OFFSET = 0.4342
        self._init_obj_z = float(obj_world[2]) + Z_WORLD_OFFSET

        self._last_eef = None
        self._last_obj = None
        self._last_obj_rot = None
        self._last_time = None

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        dx, dy, dz, grip = action

        current = self._get_eef_position()
        delta = self.cfg.max_delta_xyz * np.array([dx, dy, dz], dtype=np.float32)
        target = current + delta

        self._move_eef_xyz(target, wait=True)
        time.sleep(self.cfg.action_wait_s)

        obs = self._get_obs()
        eef = obs[0:3]   # table-relative
        obj = obs[6:9]   # table-relative
        dist_to_obj = float(np.linalg.norm(eef - obj))

        # Gripper control: proximity override + policy action.
        # Policy grip action is delayed ~4 steps vs. arm arrival at the object;
        # force-close when within grasp_proximity_m to bridge that gap.
        if dist_to_obj < self.cfg.grasp_proximity_m and not self._closed_last:
            print(f"[grasp] proximity triggered at dist={dist_to_obj:.3f}m")
            self._close_gripper()
        elif grip > 0.0 and not self._closed_last:
            self._close_gripper()
        elif grip < -0.3 and self._closed_last:
            self._open_gripper()

        reward = -dist_to_obj

        self._step_count += 1
        terminated = self.is_success(obj)
        truncated = self._step_count >= self.cfg.episode_length

        return obs, reward, terminated, truncated, {}

    def close(self):
        try:
            self._open_gripper()
            self._move_eef_xyz(self.home_xyz, wait=True)
        except:
            pass
        try:
            self.arm.disconnect()
        except:
            pass