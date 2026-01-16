"""
Utilities for connecting to and querying the xArm.

You should NOT modify function signatures.
Fill in TODOs only.
"""

from dataclasses import dataclass
from typing import List
from xarm.wrapper import XArmAPI


@dataclass
class ArmConfig:
    ip: str
    is_radian: bool = True


def connect_arm(cfg: ArmConfig) -> XArmAPI:
    """
    Connect to the robot and put it into a ready-to-move state.

    Steps you likely need:
    - create XArmAPI object
    - connect()
    - clear warnings / errors
    - enable motion
    - set mode / state

    Return:
        XArmAPI instance
    """
    # TODO: initialize arm
    arm = XArmAPI(cfg.ip,cfg.is_radian)

    # TODO: connect to robot
    arm.connect()

    # TODO: clear warnings / errors
    arm.clean_error()

    # TODO: enable motion
    arm.motion_enable(enable=True)

    # TODO: set mode/state if needed
    arm.set_mode(0)
    arm.set_state(state=0)

    return arm


def get_joint_angles(arm: XArmAPI) -> List[float]:
    """
    Return current joint angles.
    """
    # TODO: call SDK API
    return arm.get_servo_angle(is_radian=arm._is_radian)[1]
    #raise NotImplementedError


def get_tcp_pose(arm: XArmAPI) -> List[float]:
    """
    Return TCP pose as [x, y, z, roll, pitch, yaw].
    """
    # TODO: call SDK API
    #raise NotImplementedError
    return arm.get_position()[1]


def disconnect_arm(arm: XArmAPI) -> None:
    """
    Cleanly disconnect from robot.
    """
    # TODO: disconnect safely
    arm.disconnect()