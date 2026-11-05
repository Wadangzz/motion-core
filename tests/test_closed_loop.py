"""V3: the gate that judges a closed-loop run (design doc §9).

The loop runs in the simulator, so these tests feed the judgment synthetic
series whose verdict is known by construction. What is worth locking is the
boundary behaviour: which side of each bound passes, that the bounds follow the
derivation rather than a stored number, and that a torque series sitting exactly
on the actuator limit is not reported as saturated.
"""

import numpy as np
import pytest

from motion_core.verify.closed_loop import (
    FOLLOW_MARGIN,
    REST_SPEED,
    follow_bound,
    gate,
    saturated_samples,
    settle_bound,
)

DOF = 2
KP, KD = 100.0, 20.0
TAU_MAX = 50.0
TAU_PEAK = 0.5 * TAU_MAX
HOLD = 0.3      # the simulator keeps integrating past the motion, to rest
N = 66


def _run(error_scale=0.0, end_error=0.0, end_speed=0.0):
    """A rest-to-rest run: error swells mid-motion and dies at the end."""
    t = np.linspace(0.0, 1.0 + HOLD, N)
    # The run has to carry the hold after the motion, as the simulator's does:
    # it integrates until rest and cuts there. Judged at the last sample of the
    # motion itself, every real trajectory is still moving.
    phase = np.minimum(t, 1.0)
    shape = np.sin(np.pi * phase)[:, None] * np.ones(DOF)
    s = 0.5 * (1.0 - np.cos(np.pi * phase))
    q_ref = np.column_stack([s, 2.0 * s])
    q = q_ref - error_scale * shape
    q[t >= 1.0] -= end_error        # the whole hold, so the offset adds no speed
    if end_speed:                       # a last sample that is still moving
        q[-1] += end_speed * (t[-1] - t[-2])
    return t, q, q_ref, TAU_PEAK * shape


def test_clean_run_passes():
    report = gate(*_run(), TAU_MAX, KP, KD)
    assert report["passed"]
    assert not report["saturated"].any()


def test_bounds_are_the_derived_expressions():
    """The torque the run commanded sets the following bound, not the limit.

    A trajectory timed by a velocity limit peaks well under the actuator limit,
    and dividing by the limit there would allow an error the dynamics cannot
    account for.
    """
    tau = _run()[3]
    assert follow_bound(tau, KP) == pytest.approx(FOLLOW_MARGIN * TAU_PEAK / KP)
    assert settle_bound(KP, KD) == pytest.approx(KD * REST_SPEED / KP)


@pytest.mark.parametrize("factor, passes", [(0.99, True), (1.01, False)],
                         ids=["inside_bound", "over_bound"])
def test_peak_following_error_decides_at_the_bound(factor, passes):
    bound = FOLLOW_MARGIN * TAU_PEAK / KP
    report = gate(*_run(error_scale=factor * bound), TAU_MAX, KP, KD)
    assert report["passed"] is passes
    assert report["peak_error"].max() == pytest.approx(factor * bound)


def test_saturation_fails_the_gate_but_touching_the_limit_does_not():
    """A clipped series reads as exactly at the limit; that is not a violation.

    The caller has to pass the commanded torque for saturation to be visible at
    all, which is why the limit itself is the passing side.
    """
    t, q, q_ref, tau = _run()
    tau[10, 1] = TAU_MAX
    assert gate(t, q, q_ref, tau, TAU_MAX, KP, KD)["passed"]
    tau[10, 1] = TAU_MAX * 1.001
    report = gate(t, q, q_ref, tau, TAU_MAX, KP, KD)
    assert not report["passed"]
    assert list(report["saturated"]) == [0, 1]


def test_per_joint_limits_are_honoured():
    tau = np.full((N, DOF), 0.6 * TAU_MAX)
    assert list(saturated_samples(tau, [TAU_MAX, TAU_MAX / 2])) == [0, N]


@pytest.mark.parametrize("kwargs, passes",
                         [({"end_error": 0.99 * KD * REST_SPEED / KP}, True),
                          ({"end_error": 1.01 * KD * REST_SPEED / KP}, False),
                          ({"end_speed": 1.01 * REST_SPEED}, False)],
                         ids=["settled", "offset_left", "still_moving"])
def test_the_settled_end_decides_at_the_rest_threshold(kwargs, passes):
    assert gate(*_run(**kwargs), TAU_MAX, KP, KD)["passed"] is passes
