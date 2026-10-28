"""Peak-torque minimization over a parameterized motion, epigraph form + SLSQP.

``max |tau|`` is non-smooth, so it never appears in the objective: an auxiliary
variable ``t`` bounds the torque from both sides (epigraph transform) and the
solver minimizes ``t``. All torque constraints are normalized by the initial
peak so the problem is well scaled regardless of the mechanism's units.

The motion callback owns the geometry: it maps parameters ``c`` to
``(q, dq, ddq)`` on a fixed evaluation grid in physical units ([rad], [rad/s],
[rad/s^2]). The grid being fixed is what holds cycle time constant. The RMS
constraint treats the grid as uniform in time.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

Motion = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class PeakTorqueResult:
    c: np.ndarray  # optimized motion parameters
    peak: float  # max |tau| over the grid [N·m]
    rms: float  # cycle RMS torque [N·m]
    success: bool
    message: str
    history: list[float]  # objective (normalized peak) per iteration


def minimize_peak_torque(
    motion: Motion,
    load,
    c0: np.ndarray,
    free: np.ndarray | None = None,
    maxiter: int = 200,
) -> PeakTorqueResult:
    """Minimize peak torque of ``motion(c)`` under the load's limits (layer L6).

    ``free`` is a boolean mask over ``c0``: only masked-true entries are
    decision variables, the rest stay at their ``c0`` value (boundary
    conditions). V2 constraints enforced: torque-speed limit and cycle RMS.
    """
    c0 = np.asarray(c0, dtype=float)
    if free is None:
        free = np.ones(c0.shape, dtype=bool)

    def assemble(x: np.ndarray) -> np.ndarray:
        c = c0.copy()
        c[free] = x[:-1]
        return c

    def torque_of(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, dq, ddq = motion(assemble(x))
        return np.asarray(load.torque(q, dq, ddq)), np.asarray(dq)

    tau0, _ = torque_of(np.r_[c0[free], 0.0])
    tau_ref = np.max(np.abs(tau0))
    if tau_ref == 0.0:
        raise ValueError("initial motion demands zero torque; nothing to minimize")
    rms_lim = float(np.asarray(load.rms_limit()).max())

    # One memoized evaluation per x: SLSQP calls objective and constraints
    # separately, and numeric differentiation multiplies the call count.
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def evaluated(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        if key not in cache:
            cache.clear()  # keep only the current point and its FD neighbors
            cache[key] = torque_of(x)
        return cache[key]

    def epigraph(x: np.ndarray) -> np.ndarray:
        tau, _ = evaluated(x)
        t = x[-1] * tau_ref
        return np.r_[t - tau, t + tau] / tau_ref

    def speed_limit(x: np.ndarray) -> np.ndarray:
        tau, dq = evaluated(x)
        lim = np.asarray(load.torque_limit(dq))
        return np.r_[lim - tau, lim + tau] / tau_ref

    def rms_margin(x: np.ndarray) -> float:
        tau, _ = evaluated(x)
        return (rms_lim - np.sqrt(np.mean(tau**2))) / tau_ref

    history: list[float] = []
    res = minimize(
        lambda x: x[-1],
        x0=np.r_[c0[free], 1.0],
        jac=lambda x: np.r_[np.zeros(x.size - 1), 1.0],
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": epigraph},
            {"type": "ineq", "fun": speed_limit},
            {"type": "ineq", "fun": rms_margin},
        ],
        options={"maxiter": maxiter, "ftol": 1e-8},
        callback=lambda x: history.append(float(x[-1])),
    )

    c = assemble(res.x)
    tau, _ = torque_of(res.x)
    return PeakTorqueResult(
        c=c,
        peak=float(np.max(np.abs(tau))),
        rms=float(np.sqrt(np.mean(tau**2))),
        success=bool(res.success),
        message=str(res.message),
        history=history,
    )
