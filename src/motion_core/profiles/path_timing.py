"""Time the traversal of a fixed path from the torque limit (design doc §5.2).

The path is an input (design doc §1, non-goal): a straight line in ``q`` between
two rest states, ``q(s) = q0 + s * dq_path`` for ``s`` in [0, 1]. What is
computed here is the *timing*: the ``s(t)`` that respects the load's torque
budget, so the duration ``T`` comes out of the physics instead of being chosen
by hand. That substitution is the point of the whole project (design doc §2.2);
``quintic_time_scaling`` is the "given T, move smoothly" baseline it replaces.

**This is not time-optimal.** The budget is a single conservative pair
``(s_dot_max, s_ddot_max)`` valid over the whole path: every term is taken at its
worst case over the grid and the terms are summed in absolute value, so the
bound holds wherever the axis happens to be, at the cost of leaving torque on
the table where the path is cheap. A genuine path-parameterized optimum varies
the bound with ``s`` and belongs to the TOPP family (Bobrow,
Pfeiffer-Johanni, TOPP-RA), kept as the baseline to compare against (design
doc §6.2), not reimplemented here. For one dof with constant inertia the two
coincide, which is why this reduces to ``test_cycle_time_gate``'s hand
calculation.

Only the ``LoadModel`` protocol is used: the inertia, static and velocity terms
are read back by probing ``torque()``, which is exact for models linear in
``ddq`` (all of them: ``tau = M(q) ddq + ...``). So the same function times a
ball-screw cam axis and a six-axis arm.
"""

from dataclasses import dataclass

import numpy as np

from motion_core.profiles.double_s import DoubleSProfile, double_s

JERK_TIME = 0.005  # [s] default jerk-phase duration; j_max = a_max / JERK_TIME


class InfeasiblePath(Exception):
    """No timing exists: holding the path already exhausts the torque limit."""

    def __init__(self, message: str, joint: int, s: float):
        super().__init__(message)
        self.joint = joint
        self.s = s


@dataclass(frozen=True)
class PathTiming:
    """Timing of one rest-to-rest traversal, plus which limit set it."""

    q0: np.ndarray
    dq_path: np.ndarray  # q1 - q0
    profile: "DoubleSProfile | None"  # None for a zero-length move
    s_dot_max: float
    s_ddot_max: float
    binding: str  # "peak_torque" | "joint_speed" | "none"
    binding_joint: int  # joint index that set s_ddot_max (-1 if none)
    binding_s: float  # path parameter where it bound (-1.0 if none)

    @property
    def duration(self) -> float:
        return 0.0 if self.profile is None else self.profile.duration

    def sample(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, dq, ddq) at times ``t``, each shaped ``(len(t), ndof)``.

        Outside [0, T] the axis rests at an endpoint, as in ``DoubleSProfile``.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.profile is None:
            z = np.zeros((t.size, self.dq_path.size))
            return self.q0 + z, z, z.copy()
        s, ds, dds = self.profile.sample(t)
        s = np.where(t > self.profile.duration, 1.0, np.where(t < 0, 0.0, s))
        return (
            self.q0 + s[:, None] * self.dq_path,
            ds[:, None] * self.dq_path,
            dds[:, None] * self.dq_path,
        )


def _terms(load, q0, dq_path, s_dot, grid):
    """Decompose the torque demand along the path by probing the LoadModel.

    Returns ``(s, inertia, static, velocity)``; the arrays are ``(grid, ndof)``:
    ``inertia`` is ``M(q) dq_path`` (the coefficient of ``s_ddot``), ``static``
    the demand at rest (gravity, Coulomb offset), ``velocity`` the extra demand
    from moving at ``s_dot`` (centrifugal/Coriolis, viscous) at zero ``s_ddot``.
    """
    s = np.linspace(0.0, 1.0, grid)
    q = q0 + s[:, None] * dq_path
    zero = np.zeros_like(q)
    static = np.asarray(load.torque(q, zero, zero), dtype=float)
    inertia = np.asarray(load.torque(q, zero, np.tile(dq_path, (grid, 1))), dtype=float) - static
    velocity = np.asarray(load.torque(q, s_dot * np.tile(dq_path, (grid, 1)), zero), dtype=float)
    return s, inertia, static, velocity - static


def time_path(load, q0, q1, dq_max, *, jerk_time: float = JERK_TIME,
              grid: int = 51) -> PathTiming:
    """Time the straight path ``q0 -> q1`` within ``load``'s torque budget.

    ``dq_max`` [rad/s] is the per-joint speed limit (scalar or length-ndof); the
    ``LoadModel`` protocol carries a torque-speed curve but no speed rating, so
    the caller states it. ``grid`` is the number of path samples the budget is
    taken over; the worst case over this grid is what binds, so a coarse grid
    is optimistic; 51 is the default used by the tests.
    """
    q0 = np.asarray(q0, dtype=float)
    dq_path = np.asarray(q1, dtype=float) - q0
    n = dq_path.size
    dq_max = np.broadcast_to(np.asarray(dq_max, dtype=float), (n,))
    if np.any(dq_max <= 0):
        raise ValueError("dq_max must be positive")

    moving = np.abs(dq_path) > 0
    if not moving.any():
        return PathTiming(q0, dq_path, None, 0.0, 0.0, "none", -1, -1.0)

    # Speed budget: the first joint to hit its rating caps the path speed.
    s_dot_max = float(np.min(dq_max[moving] / np.abs(dq_path[moving])))

    # Acceleration budget: usable torque minus the demand that s_ddot cannot
    # avoid, over the inertia seen along the path. Absolute values throughout:
    # the terms are not assumed to help each other anywhere on the path.
    s, inertia, static, velocity = _terms(load, q0, dq_path, s_dot_max, grid)
    limit = np.asarray(load.torque_limit(s_dot_max * np.tile(dq_path, (grid, 1))), dtype=float)
    headroom = limit - np.abs(static) - np.abs(velocity)
    with np.errstate(divide="ignore", invalid="ignore"):
        budget = np.where(np.abs(inertia) > 0, headroom / np.abs(inertia), np.inf)
    flat = int(np.argmin(budget))
    i_s, i_j = divmod(flat, n)
    s_ddot_max = float(budget[i_s, i_j])
    if s_ddot_max <= 0:
        raise InfeasiblePath(
            f"joint {i_j} has no torque left for acceleration at s={s[i_s]:.3f}: "
            f"limit {limit[i_s, i_j]:.3f} N·m vs static {abs(static[i_s, i_j]):.3f} + "
            f"velocity {abs(velocity[i_s, i_j]):.3f} N·m",
            joint=i_j, s=float(s[i_s]),
        )

    profile = double_s(1.0, s_dot_max, s_ddot_max, s_ddot_max / jerk_time)
    binding = "joint_speed" if profile.t_v > 0 else "peak_torque"
    return PathTiming(q0, dq_path, profile, s_dot_max, s_ddot_max,
                      binding, i_j, float(s[i_s]))
