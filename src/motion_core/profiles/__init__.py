"""Motion profile primitives shared by both consumers."""

from motion_core.profiles.bspline import PINNED, ClampedBSpline
from motion_core.profiles.double_s import DoubleSProfile, double_s
from motion_core.profiles.trapezoid import TrapezoidProfile, trapezoid

__all__ = [
    "PINNED",
    "ClampedBSpline",
    "DoubleSProfile",
    "TrapezoidProfile",
    "double_s",
    "trapezoid",
]
