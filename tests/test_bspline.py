"""ClampedBSpline: boundary pinning, derivative consistency, fit accuracy."""

import numpy as np

from motion_core.profiles import PINNED, ClampedBSpline

X = np.linspace(0.0, 1.0, 2001)


def _smooth_rise(x: np.ndarray) -> np.ndarray:
    # Rest-to-rest analytic curve with f'=f''=0 at both ends (cycloidal-like).
    return x - np.sin(2 * np.pi * x) / (2 * np.pi)


def test_pinned_ends_fix_position_velocity_acceleration():
    basis = ClampedBSpline(n_ctrl=12)
    c = np.random.default_rng(0).uniform(-1, 1, 12)
    c[:PINNED] = 0.0
    c[-PINNED:] = 1.0
    f, df, ddf = basis.eval(c, np.array([0.0, 1.0]))
    assert np.allclose(f, [0.0, 1.0], atol=1e-12)
    assert np.allclose(df, 0.0, atol=1e-9)
    assert np.allclose(ddf, 0.0, atol=1e-6)


def test_matrices_agree_with_eval():
    basis = ClampedBSpline(n_ctrl=14)
    c = np.random.default_rng(1).uniform(-1, 1, 14)
    b, db, ddb = basis.matrices(X)
    f, df, ddf = basis.eval(c, X)
    assert np.allclose(b @ c, f)
    assert np.allclose(db @ c, df)
    assert np.allclose(ddb @ c, ddf)


def test_fit_reproduces_smooth_curve_and_its_acceleration():
    # The parameterization must be able to express a baseline before any
    # optimized-vs-baseline comparison means anything (START_HERE step 3).
    basis = ClampedBSpline(n_ctrl=16)
    y = _smooth_rise(X)
    c = basis.fit(X, y)
    f, _, ddf = basis.eval(c, X)
    ddy = 2 * np.pi * np.sin(2 * np.pi * X)  # analytic second derivative
    assert np.max(np.abs(f - y)) < 1e-5
    assert np.max(np.abs(ddf - ddy)) < 0.01 * np.max(np.abs(ddy))
