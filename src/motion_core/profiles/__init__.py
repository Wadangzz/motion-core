"""Motion profile primitives shared by both consumers."""

from motion_core.profiles.bspline import PINNED, ClampedBSpline
from motion_core.profiles.double_s import DoubleSProfile, double_s
from motion_core.profiles.path_timing import InfeasiblePath, PathTiming, time_path
from motion_core.profiles.trapezoid import TrapezoidProfile, trapezoid

__all__ = [
    "PINNED",
    "ClampedBSpline",
    "DoubleSProfile",
    "InfeasiblePath",
    "PathTiming",
    "TrapezoidProfile",
    "double_s",
    "time_path",
    "trapezoid",
]
