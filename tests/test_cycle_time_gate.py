"""Stage-1 gate #2: double-S vs quintic at the same torque limit — >= 15% faster.

The quintic (the simulator's existing generator) answers "given T, move
smoothly"; the double-S answers "given the limits, move fast". Both are held to
the identical torque-speed bound of the same BallScrew axis, the quintic by
bisecting for its minimal feasible T, the double-S by back-calculating the
acceleration budget from the torque limit (design doc §5.2). Fixed before the
work started: if the double-S is not >= 15% faster, that is a failure to report,
not a number to adjust.
"""

import numpy as np

from motion_core.loads import BallScrew
from motion_core.profiles import double_s
from motion_core.robots import quintic_time_scaling

AXIS = BallScrew(lead_mm=10, gear_ratio=1.0, mass_kg=15, motor="example_400W",
                 efficiency=0.9, safety_factor=0.8)
D = 0.08 / AXIS.k  # 80 mm as motor angle [rad]
N = 2001


def _within_limits(q, dq, ddq, v_max) -> bool:
    """Torque-speed bound AND the speed budget — a cruising horizontal axis
    demands no torque, so without the speed check the quintic would simply
    overspeed the motor to go fast."""
    tau = AXIS.torque(q, dq, ddq)
    speed_ok = np.abs(dq).max() <= v_max * (1 + 1e-12)  # cruise sits exactly on the bound
    return bool(np.all(np.abs(tau) <= AXIS.torque_limit(dq)) and speed_ok)


def _budgets() -> tuple[float, float]:
    """(v_max, a_max) from the motor, conventions of design doc §5.2 step 1.

    Speed budget: rated speed, where the full peak torque is still available.
    Acceleration: usable torque at that speed minus the static demand, over the
    effective inertia (both read off the LoadModel, no constants duplicated).
    """
    v_max = AXIS._motor.rated_speed
    j_eff = AXIS.torque(np.r_[0.0], np.r_[0.0], np.r_[1.0])[0]
    tau_static = AXIS.torque(np.r_[0.0], np.r_[v_max], np.r_[0.0])[0]
    a_max = (AXIS.torque_limit(np.r_[v_max])[0] - abs(tau_static)) / j_eff
    return v_max, a_max


def _min_feasible_quintic_t(v_max, lo: float = 1e-3, hi: float = 10.0) -> float:
    for _ in range(60):
        mid = (lo + hi) / 2
        t = np.linspace(0.0, mid, N)
        s, ds, dds = quintic_time_scaling(t, mid)
        if _within_limits(D * s, D * ds, D * dds, v_max):
            hi = mid
        else:
            lo = mid
    return hi


def test_double_s_beats_quintic_by_15pct_at_same_torque_limit():
    v_max, a_max = _budgets()
    j_max = a_max / 0.005  # 5 ms jerk phase — preset, stated with the result

    ds_profile = double_s(D, v_max, a_max, j_max)
    t = np.linspace(0.0, ds_profile.duration, N)
    assert _within_limits(*ds_profile.sample(t), v_max)  # V2 on the fast profile

    t_quintic = _min_feasible_quintic_t(v_max)
    reduction = 1.0 - ds_profile.duration / t_quintic
    assert reduction >= 0.15, (
        f"double-S {ds_profile.duration:.3f}s vs quintic {t_quintic:.3f}s: "
        f"only {reduction:.1%}"
    )
