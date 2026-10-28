"""V1: double-S profile against its defining properties, in every branch."""

import numpy as np
import pytest

from motion_core.profiles import double_s

CASES = {
    "full": (1.0, 0.5, 2.0, 20.0),  # cruise + constant-accel phases exist
    "no_cruise": (0.1, 1.0, 2.0, 20.0),  # too short to reach v_max
    "no_const_accel": (1.0, 0.2, 5.0, 10.0),  # v_max reached before a_max
    "jerk_only": (0.01, 1.0, 5.0, 30.0),  # neither v_max nor a_max reached
}


@pytest.mark.parametrize("d,v,a,j", CASES.values(), ids=CASES.keys())
def test_rest_to_rest_within_limits(d, v, a, j):
    p = double_s(d, v, a, j)
    t = np.linspace(-0.05, p.duration + 0.05, 20001)
    q, dq, ddq = p.sample(t)

    # Boundary conditions and the full distance.
    assert np.isclose(q[-1], d, atol=1e-9) and np.isclose(q[0], 0.0, atol=1e-12)
    assert abs(dq[0]) < 1e-12 and abs(dq[-1]) < 1e-12
    assert abs(ddq[0]) < 1e-12 and abs(ddq[-1]) < 1e-12

    # Stated limits hold, and the reported reached-values are honest —
    # evaluated at the exact phase boundaries where the peaks sit.
    assert dq.max() <= v * (1 + 1e-9)
    assert np.abs(ddq).max() <= a * (1 + 1e-9)
    at_peaks = p.sample(np.array([p.t_j, p.t_a]))
    assert np.isclose(at_peaks[2][0], p.a_lim, rtol=1e-9)  # ddq at t_j
    assert np.isclose(at_peaks[1][1], p.v_lim, rtol=1e-9)  # dq at t_a

    # Derivative consistency (numeric jerk also bounded by j_max).
    h = t[1] - t[0]
    assert np.allclose(np.gradient(q, h), dq, atol=1e-3 * max(1.0, dq.max()))
    assert np.allclose(np.gradient(dq, h), ddq, atol=5e-2 * max(1.0, np.abs(ddq).max()))
    jerk = np.gradient(ddq, h)
    assert np.abs(jerk).max() <= j * 1.05  # finite-diff smearing tolerance


def test_cruise_case_hits_both_limits():
    p = double_s(*CASES["full"])
    assert p.t_v > 0 and np.isclose(p.v_lim, 0.5) and np.isclose(p.a_lim, 2.0)


def test_monotone_position():
    for d, v, a, j in CASES.values():
        p = double_s(d, v, a, j)
        q, _, _ = p.sample(np.linspace(0, p.duration, 5001))
        assert np.all(np.diff(q) >= -1e-12)
