"""Trapezoid profile: V1 analytic properties in both branches, plus duration stretch."""

import numpy as np
import pytest

from motion_core.profiles import trapezoid


@pytest.mark.parametrize("d,v,a", [(1.0, 0.5, 2.0), (0.05, 1.0, 2.0)], ids=["cruise", "triangle"])
def test_rest_to_rest_within_limits(d, v, a):
    p = trapezoid(d, v, a)
    t = np.linspace(-0.1, p.duration + 0.1, 10001)
    q, dq, ddq = p.sample(t)
    assert np.isclose(q[-1], d) and np.isclose(q[0], 0.0)
    assert abs(dq[0]) < 1e-12 and abs(dq[-1]) < 1e-12
    assert dq.max() <= v * (1 + 1e-9)
    assert np.abs(ddq).max() <= a * (1 + 1e-9)
    assert np.isclose(dq.max(), p.v_peak, rtol=1e-4)
    h = t[1] - t[0]
    assert np.allclose(np.gradient(q, h), dq, atol=2e-3 * max(1.0, dq.max()))


def test_triangle_when_too_short():
    p = trapezoid(0.05, 1.0, 2.0)
    assert p.t_v == 0.0 and p.v_peak < 1.0
    assert np.isclose(p.v_peak, np.sqrt(0.05 * 2.0))


def test_stretch_hits_exact_time_and_sheds_speed_and_accel():
    fast = trapezoid(1.0, 0.5, 2.0)
    slow = fast.stretched(fast.duration * 1.5)
    assert np.isclose(slow.duration, fast.duration * 1.5)
    assert np.isclose(slow.v_peak, fast.v_peak / 1.5)
    assert np.isclose(slow.a, fast.a / 1.5**2)  # peak torque drops quadratically
    q, _, _ = slow.sample(np.array([slow.duration]))
    assert np.isclose(q[0], 1.0)


def test_compression_is_refused():
    fast = trapezoid(1.0, 0.5, 2.0)
    with pytest.raises(ValueError, match="compress"):
        fast.stretched(fast.duration * 0.9)
