"""V0 for RackPinion: conversion factor, hand-checked torque, gear convention."""

import numpy as np

from motion_core.loads import EXAMPLE_MOTORS, RackPinion

MOTOR = EXAMPLE_MOTORS["example_750W"]


def _axis(**kw) -> RackPinion:
    base = {"pitch_mm": 157.08, "gear_ratio": 10.0, "mass_kg": 40,
            "motor": "example_750W", "efficiency": 0.92, "safety_factor": 0.8}
    base.update(kw)
    return RackPinion(**base)


def test_k_includes_reduction():
    # pitch 157.08 mm per pinion rev, 10:1 reducer -> 15.708 mm per motor rev.
    axis = _axis()
    assert np.isclose(axis.k, 0.15708 / (2 * np.pi) / 10.0)
    assert np.isclose(axis.k * 2 * np.pi * 1000.0, 15.708)  # mm per motor rev


def test_torque_matches_hand_calculation():
    axis = _axis()
    j_total = MOTOR.rotor_inertia + 40 * axis.k**2 / 0.92
    tau = axis.torque(np.r_[0.0], np.r_[0.0], np.r_[1000.0])[0]
    assert np.isclose(tau, j_total * 1000.0)
    assert np.isclose(axis.torque(np.r_[0.0], np.r_[50.0], np.r_[0.0])[0], 0.0)


def test_limits_scale_with_safety_factor():
    axis = _axis()
    assert np.isclose(axis.torque_limit(np.r_[0.0])[0], MOTOR.max_torque * 0.8, rtol=1e-3)
    assert axis.rms_limit()[0] == MOTOR.rated_torque * 0.8
