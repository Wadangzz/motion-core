"""Clamped B-spline motion profile: control points -> (q, dq, ddq) on a grid.

The spline lives on a normalized abscissa x in [0, 1]; derivatives are w.r.t.
x, and the caller applies its own chain rule (cam master angle, time, ...).

Boundary conditions come from the clamped knot vector: with the first ``p``
control points equal, position, velocity and acceleration at x=0 are all pinned
to that value (same at x=1 for the last ``p``), for any degree >= 5 with p=3.
The optimizer therefore fixes three coefficients per end and varies the rest.
"""

import numpy as np
from scipy.interpolate import BSpline

PINNED = 3  # equal control points per end to pin position, velocity, acceleration


class ClampedBSpline:
    """Uniform clamped B-spline basis of a given size on x in [0, 1]."""

    def __init__(self, n_ctrl: int, degree: int = 5):
        if n_ctrl < 2 * PINNED + 1:
            raise ValueError(f"need at least {2 * PINNED + 1} control points, got {n_ctrl}")
        self.n_ctrl = n_ctrl
        self.degree = degree
        n_internal = n_ctrl - degree - 1
        self.knots = np.r_[
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, n_internal + 2)[1:-1],
            np.ones(degree + 1),
        ]

    def matrices(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Design matrices ``(B, dB, ddB)``, each ``(len(x), n_ctrl)``.

        ``f = B @ c`` etc., so an optimization loop pays the basis cost once.
        """
        basis = BSpline(self.knots, np.eye(self.n_ctrl), self.degree)
        return basis(x), basis.derivative(1)(x), basis.derivative(2)(x)

    def eval(self, c: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Profile value and first two derivatives w.r.t. x at ``x``."""
        spl = BSpline(self.knots, np.asarray(c, dtype=float), self.degree)
        return spl(x), spl.derivative(1)(x), spl.derivative(2)(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least-squares control points for samples ``(x, y)``.

        The three boundary coefficients per end are pinned to ``y[0]`` / ``y[-1]``
        so the fit keeps rest-to-rest boundary conditions exactly.
        """
        b, _, _ = self.matrices(np.asarray(x, dtype=float))
        c = np.empty(self.n_ctrl)
        c[:PINNED] = y[0]
        c[-PINNED:] = y[-1]
        free = slice(PINNED, self.n_ctrl - PINNED)
        resid = np.asarray(y, dtype=float) - b[:, :PINNED] @ c[:PINNED] - b[:, -PINNED:] @ c[-PINNED:]
        c[free], *_ = np.linalg.lstsq(b[:, free], resid, rcond=None)
        return c
