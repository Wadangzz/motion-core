"""Judge a closed-loop run against the V3 gate (design doc §9).

The loop itself does not live here. Friction, torque saturation, the discrete
ZOH control rate and the controllers are the simulator's, and duplicating them
would be a second implementation to keep honest. This module takes the time
series a run produced and says whether it passed, which is the part that has to
mean the same thing for every caller.

The gate is three conditions, all of which must hold:

* no saturated sample,
* per-joint peak following error within ``FOLLOW_MARGIN * peak torque / kp``,
* a settled end: residual speed within ``REST_SPEED`` and position error within
  ``kd * REST_SPEED / kp``.

Every bound is derived rather than picked, because §9.1 asks for numbers fixed
before the measurement and only a formula can be. Under a PD law with exact
gravity compensation the dynamic torque is whatever the feedback term produces,
so the quasi-steady error is ``e ~ tau_dyn / kp``; the margin covers the
Coriolis, friction and ZOH terms that estimate omits. The bound scales with the
peak torque the run actually commanded, not with the actuator limit: a
trajectory timed by a velocity limit stays far below the limit, and dividing by
the limit there would permit an error the dynamics cannot explain.

The settle bound is the loop's own stopping rule read back. A run ends when its
rest event fires, which is a velocity threshold, not true equilibrium: the
damping term still holds ``e ~ kd * v_rest / kp`` at that instant. Judging
against a tighter number would fail a feasible trajectory for the integrator's
choice of when to stop.

Computed torque cancels ``tau_dyn`` by feedforward, so both expressions stay
upper bounds there, with room to spare.
"""

import numpy as np

FOLLOW_MARGIN = 1.5     # Coriolis, friction and ZOH, which e ~ tau/kp omits
REST_SPEED = 2e-2       # rad/s, the simulator's own rest threshold (VEL_TOL)


def follow_bound(tau, kp):
    """Per-joint peak following error the commanded torque can explain, in rad.

    :param tau: commanded joint torques of the run, ``(n, dof)``
    :param kp: proportional gain of the run
    """
    peak = np.abs(np.asarray(tau, dtype=float)).max(axis=0)
    return FOLLOW_MARGIN * peak / float(kp)


def settle_bound(kp, kd):
    """Position error still held by the damping term at the rest threshold, rad."""
    return float(kd) * REST_SPEED / float(kp)


def saturated_samples(tau, tau_max):
    """Per-joint count of samples whose commanded torque exceeds the limit.

    ``tau`` must be the *commanded* torque, before the actuator clip. A clipped
    series cannot answer this question: a saturated sample and one that merely
    touches the limit are then the same number.
    """
    tau = np.asarray(tau, dtype=float)
    limit = np.asarray(tau_max, dtype=float)
    return np.count_nonzero(np.abs(tau) > limit, axis=0)


def gate(t, q, q_ref, tau, tau_max, kp, kd):
    """Judge one closed-loop run. Returns the metrics and the verdict.

    :param t: sample times, ``(n,)``, ascending
    :param q: measured joint positions, ``(n, dof)``
    :param q_ref: reference positions at the same samples, ``(n, dof)``
    :param tau: commanded joint torques at the same samples, ``(n, dof)``
    :param tau_max: actuator torque limit, scalar or ``(dof,)``, N*m
    :param kp: proportional gain of the run
    :param kd: derivative gain of the run
    :return: dict of ``saturated`` (per joint), ``peak_error`` (per joint),
        ``follow_bound`` (per joint), ``settle_error``, ``settle_speed``,
        ``settle_bound`` and ``passed``
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    error = np.abs(np.asarray(q_ref, dtype=float) - q)
    # Residual speed by difference: the simulator reports no velocity series in
    # torque mode, and asking a caller for one it does not have buys nothing.
    speed = np.abs(np.gradient(q, t, axis=0))

    saturated = saturated_samples(tau, tau_max)
    peak_error = error.max(axis=0)
    follow = follow_bound(tau, kp)
    settle = settle_bound(kp, kd)
    settle_error = float(error[-1].max())
    settle_speed = float(speed[-1].max())

    return {
        "saturated": saturated,
        "peak_error": peak_error,
        "follow_bound": follow,
        "settle_error": settle_error,
        "settle_speed": settle_speed,
        "settle_bound": settle,
        "passed": bool(
            not saturated.any()
            and np.all(peak_error <= follow)
            and settle_error <= settle
            and settle_speed <= REST_SPEED
        ),
    }
