"""Serial-robot kinematics, dynamics and models (screw theory, Modern Robotics).

Extracted from the robot web simulator; this is the public surface consumers use.
`dynamics_fast` (JIT) and `urdf_loader` (URDF) stay as submodules -- import them
explicitly, since they carry the `fast` / `robot` extras.
"""

from motion_core.robots.robot_math import (
    SE3,
    SO3,
    Dynamics,
    Kinematics,
    quintic_time_scaling,
)
from motion_core.robots.robot_model import RobotModel
from motion_core.robots.robot_registry import get_robot, list_robots

__all__ = [
    "SE3",
    "SO3",
    "Dynamics",
    "Kinematics",
    "RobotModel",
    "get_robot",
    "list_robots",
    "quintic_time_scaling",
]
