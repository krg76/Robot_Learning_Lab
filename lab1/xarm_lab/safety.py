"""
Robot safety helpers.

These should be called BEFORE any motion commands.
"""

from xarm.wrapper import XArmAPI


def enable_basic_safety(arm: XArmAPI):
    """
    Enable collision checking and other safety features.

    Hint:
    - Look for 'collision' and 'self_collision' in SDK docs.
    """
    # TODO
    #pass
    arm.set_self_collision_detection(True)


def clear_faults(arm: XArmAPI):
    """
    Clear warnings and errors if robot is in a fault state.
    """
    # TODO
    arm.clean_error()
    #pass