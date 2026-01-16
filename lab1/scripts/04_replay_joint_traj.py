"""
Replay a recorded xArm trajectory file (.traj) using the SDK trajectory playback APIs.

Workflow:
  1) connect
  2) enable motion
  3) set normal mode (Mode 0)
  4) load_trajectory(<traj>)
  5) playback_trajectory()

IMPORTANT:
- Stand clear
- Keep speeds conservative (trajectory playback is controller-defined)
- Be ready to hit E-stop
"""

import argparse
from xarm.wrapper import XArmAPI

from xarm_lab.arm_utils import connect_arm, disconnect_arm, ArmConfig
from xarm_lab.kinematics import ik_from_pose
from xarm_lab.safety import enable_basic_safety, clear_faults

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--traj", required=True, help="Name of the .traj file recorded in teach mode")
    args = ap.parse_args()

    # TODO: initialize XArmAPI
    arm = connect_arm(ArmConfig(ip=args.ip,is_radian=True))

    # TODO: connect

    try:
        # TODO: enable motion
        # TODO: set normal mode (Mode 0)
        # TODO: set state ready
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)

        # TODO: load_trajectory(args.traj)
        arm.load_trajectory(args.traj)
        # TODO: playback_trajectory()
        arm.playback_trajectory()
        print("[OK] Playback command sent.")

    finally:
        # TODO: disconnect
        disconnect_arm(arm)
        #pass


if __name__ == "__main__":
    main()
