"""Load models: what torque a motion demands, and what the motor can give."""

from motion_core.loads.ball_screw import BallScrew
from motion_core.loads.base import LoadModel
from motion_core.loads.motor import EXAMPLE_MOTORS, Motor, get_motor

__all__ = ["EXAMPLE_MOTORS", "BallScrew", "LoadModel", "Motor", "get_motor"]
