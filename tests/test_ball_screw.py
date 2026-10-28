"""V0: units, conversion-factor round trip, and hand-checked torque values."""

import numpy as np

from motion_core.loads import EXAMPLE_MOTORS, BallScrew

MOTOR = EXAMPLE_MOTORS["example_400W"]


def _axis(**kw) -> BallScrew:
    base = {"lead_mm": 10, "gear_ratio": 1.0, "mass_kg": 15, "motor": "example_400W",
            "efficiency": 0.9, "safety_factor": 0.8}
    base.update(kw)
    return BallScrew(**base)


def test_k_round_trip():
    axis = _axis()
    assert np.isclose(axis.k, 0.010 / (2 * np.pi))
    s = 0.08
    q = s / axis.k
    assert np.isclose(q * axis.k, s)


def test_ndof_and_array_shape():
    axis = _axis()
    one = np.zeros(1)
    assert axis.ndof == 1
    assert axis.torque(one, one, one).shape == (1,)
    assert axis.rms_limit().shape == (1,)


def test_constant_velocity_matches_hand_calculation():
    # Horizontal, no friction terms: cruising needs zero torque.
    axis = _axis()
    assert np.isclose(axis.torque(np.r_[1.0], np.r_[100.0], np.r_[0.0])[0], 0.0)
    # With viscous and Coulomb terms, well above the smoothing width:
    axis = _axis(viscous=1e-4, coulomb=0.05)
    tau = axis.torque(np.r_[1.0], np.r_[100.0], np.r_[0.0])[0]
    assert np.isclose(tau, 1e-4 * 100.0 + 0.05, rtol=1e-6)


def test_acceleration_matches_hand_calculation():
    axis = _axis()
    j_total = MOTOR.rotor_inertia + 15 * axis.k**2 / 0.9
    tau = axis.torque(np.r_[0.0], np.r_[0.0], np.r_[1000.0])[0]
    assert np.isclose(tau, j_total * 1000.0)


def test_vertical_gravity_term():
    axis = _axis(mass_kg=12, orientation="vertical")
    tau = axis.torque(np.r_[0.0], np.r_[0.0], np.r_[0.0])[0]
    assert np.isclose(tau, 12 * 9.80665 * axis.k / 0.9)


def test_torque_limit_follows_speed_curve():
    axis = _axis()
    at_zero = axis.torque_limit(np.r_[0.0])[0]
    at_max = axis.torque_limit(np.r_[MOTOR.max_speed])[0]
    assert np.isclose(at_zero, MOTOR.max_torque * 0.8, rtol=1e-3)
    assert np.isclose(at_max, MOTOR.rated_torque * 0.8, rtol=2e-2)
    assert axis.rms_limit()[0] == MOTOR.rated_torque * 0.8
