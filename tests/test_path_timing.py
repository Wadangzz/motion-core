"""time_path: torque-budget timing of a fixed joint-space path (design doc §5.2, §6.1).

Success criteria fixed before writing the implementation:
- 1 dof, constant inertia: reduces to the hand calculation of
  ``test_cycle_time_gate`` (same budgets, same double-S duration).
- V2 on the output: the sampled trajectory stays inside the load's
  torque-speed bound and the stated joint speed limit, zero violations.
- Roadmap stage-2 gate: on a multi-joint UR5 move, the derived T beats the
  minimal feasible quintic (bisected under the identical limits) by >= 15%.
- Infeasible path (gravity alone saturates a joint) raises ``InfeasiblePath``
  naming the joint, instead of returning a timing.
"""

import numpy as np
import pytest

from motion_core.loads import ArticulatedRobot, BallScrew
from motion_core.profiles import InfeasiblePath, double_s, time_path
from motion_core.robots import quintic_time_scaling, robot_registry

N = 2001

# Synthetic-but-plausible joint ratings (N·m): stated data, not UR5e spec.
UR5_PEAK = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
UR5_SPEED = np.pi  # rad/s, all joints


def _ur5():
    model = robot_registry.get_robot("ur5")
    load = ArticulatedRobot(model, max_torque=UR5_PEAK, rated_torque=UR5_PEAK / 2)
    q0 = model.ready.astype(float)
    q1 = q0 + np.array([1.2, -0.5, 0.7, -0.9, 0.6, 1.5])
    return load, q0, q1


def _violations(load, timing_sample, dq_max):
    q, dq, ddq = timing_sample
    tau = load.torque(q, dq, ddq)
    over_tau = np.abs(tau) > load.torque_limit(dq) * (1 + 1e-9)
    over_dq = np.abs(dq) > np.asarray(dq_max) * (1 + 1e-9)
    return int(over_tau.sum() + over_dq.sum())


def test_1dof_reduces_to_cycle_time_gate_hand_calc():
    axis = BallScrew(lead_mm=10, gear_ratio=1.0, mass_kg=15, motor="example_400W",
                     efficiency=0.9, safety_factor=0.8)
    d = 0.08 / axis.k  # 80 mm as motor angle [rad]
    v_max = axis._motor.rated_speed

    timing = time_path(axis, np.r_[0.0], np.r_[d], dq_max=v_max)

    # Same budgets as test_cycle_time_gate._budgets, by hand:
    j_eff = axis.torque(np.r_[0.0], np.r_[0.0], np.r_[1.0])[0]
    a_max = axis.torque_limit(np.r_[v_max])[0] / j_eff  # horizontal: no static demand
    ref = double_s(d, v_max, a_max, a_max / 0.005)
    assert timing.duration == pytest.approx(ref.duration, rel=1e-9)
    assert timing.binding == "joint_speed"  # 80 mm at rated speed cruises


def test_ur5_within_limits_and_beats_quintic_by_15pct():
    load, q0, q1 = _ur5()
    timing = time_path(load, q0, q1, dq_max=UR5_SPEED)
    t = np.linspace(0.0, timing.duration, N)
    assert _violations(load, timing.sample(t), UR5_SPEED) == 0  # V2

    # Minimal feasible quintic over the same path under the identical limits.
    dq_path = q1 - q0
    lo, hi = 1e-3, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        s, ds, dds = quintic_time_scaling(np.linspace(0.0, mid, N), mid)
        sample = (q0 + s[:, None] * dq_path, ds[:, None] * dq_path,
                  dds[:, None] * dq_path)
        if _violations(load, sample, UR5_SPEED) == 0:
            hi = mid
        else:
            lo = mid

    reduction = 1.0 - timing.duration / hi
    assert reduction >= 0.15, (
        f"time_path {timing.duration:.3f}s vs quintic {hi:.3f}s: only {reduction:.1%}"
    )


def test_zero_move_has_zero_duration():
    load, q0, _ = _ur5()
    timing = time_path(load, q0, q0, dq_max=UR5_SPEED)
    assert timing.duration == 0.0
    q, dq, ddq = timing.sample(np.linspace(0.0, 1.0, 5))
    assert np.allclose(q, q0) and not dq.any() and not ddq.any()


def test_gravity_saturated_joint_raises_infeasible():
    model = robot_registry.get_robot("ur5")
    load = ArticulatedRobot(model, max_torque=1.0, rated_torque=0.5)  # ~gravity level
    q0 = model.ready.astype(float)
    with pytest.raises(InfeasiblePath) as e:
        time_path(load, q0, q0 + 0.5, dq_max=UR5_SPEED)
    assert e.value.joint >= 0 and 0.0 <= e.value.s <= 1.0
