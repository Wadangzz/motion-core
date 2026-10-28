"""Peak-torque optimizer on a 1-dof rest-to-rest move: decreases, violates nothing."""

import numpy as np

from motion_core.loads import BallScrew
from motion_core.optimize import minimize_peak_torque
from motion_core.profiles import PINNED, ClampedBSpline

AXIS = BallScrew(lead_mm=10, gear_ratio=1.0, mass_kg=15, motor="example_400W",
                 efficiency=0.9, safety_factor=0.8)
T = 1.0 / 3.0  # move time [s]
STROKE_Q = 0.08 / AXIS.k  # 80 mm as motor angle [rad]
X = np.linspace(0.0, 1.0, 241)


def _motion_factory(basis: ClampedBSpline):
    b, db, ddb = basis.matrices(X)

    def motion(c: np.ndarray):
        return b @ c, (db @ c) / T, (ddb @ c) / T**2

    return motion


def test_peak_decreases_and_limits_hold():
    basis = ClampedBSpline(n_ctrl=14)
    y = STROKE_Q * (X - np.sin(2 * np.pi * X) / (2 * np.pi))  # smooth initial rise
    c0 = basis.fit(X, y)
    free = np.ones(basis.n_ctrl, dtype=bool)
    free[:PINNED] = free[-PINNED:] = False

    motion = _motion_factory(basis)
    tau0 = AXIS.torque(*motion(c0))
    result = minimize_peak_torque(motion, AXIS, c0, free=free)

    assert result.success, result.message
    assert result.peak < np.max(np.abs(tau0))  # objective actually went down
    assert result.history[-1] <= result.history[0] + 1e-9

    # V2: zero violations at the solution.
    q, dq, ddq = motion(result.c)
    tau = AXIS.torque(q, dq, ddq)
    assert np.all(np.abs(tau) <= AXIS.torque_limit(dq) + 1e-9)
    assert np.sqrt(np.mean(tau**2)) <= AXIS.rms_limit()[0] + 1e-9

    # Boundary conditions survived the optimization.
    assert np.isclose(q[0], 0.0, atol=1e-9) and np.isclose(q[-1], STROKE_Q)
    assert np.allclose([dq[0], dq[-1], ddq[0], ddq[-1]], 0.0, atol=1e-6)
