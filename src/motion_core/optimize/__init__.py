"""Constrained motion optimizers shared by both consumers."""

from motion_core.optimize.peak_torque import PeakTorqueResult, minimize_peak_torque

__all__ = ["PeakTorqueResult", "minimize_peak_torque"]
