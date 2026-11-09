"""ArticulatedRobot: LoadModel over the existing RNE dynamics, layers L2 + L3 + L6.

No equations are derived here — ``Dynamics.inverse_dynamics`` (parity-tested
against numba, cross-checked against MuJoCo) does the physics. This class only
adapts it to the ``LoadModel`` protocol and adds motor limits, smoothed joint
friction and a contact wrench at the tip, exactly as ``BallScrew`` does for the
linear axis.

Coordinates follow ``docs/theory/conventions.md``: a robot works directly in
``q`` [rad], there is no linear-to-motor factor ``k``. Torque limits come from
explicit per-joint motor ratings passed by the caller, not from the URDF —
URDF effort limits are position-limit metadata, not a torque-speed model.
"""

from dataclasses import dataclass, field

import numpy as np

from motion_core.robots import Dynamics, RobotModel


@dataclass(frozen=True)
class ArticulatedRobot:
    """Serial arm (or a 1-dof mechanism modeled as one). ``ndof == model.n``.

    ``tip_wrench`` is what the end effector pushes on its environment with:
    ``[moment; force]`` in the end-effector frame, Modern Robotics convention. A
    gripper squeezing an object at N newtons is a force entry of N. Its
    joint-torque contribution is ``model.jacobian_body(q).T @ tip_wrench``, which
    is a test, and it stays constant in that frame, which is what a grip force
    is. Leave it at 0 for free-space motion.

    A payload bolted to the flange is **not** this. Its weight points down in the
    world, so in the end-effector frame it swings as the arm moves, and a
    constant field cannot say that. Put the mass in the model instead:
    ``urdf_loader`` merges the inertia of links attached by fixed joints, which is
    where a gripper's own weight belongs.

    It is a field and not an argument to ``torque`` because the ``LoadModel``
    contract is ``(q, dq, ddq)`` and nothing else, the same reason a cam axis
    converts its master angle internally.
    """

    model: RobotModel
    max_torque: "float | np.ndarray"  # N·m per joint, peak (speed-independent for now)
    rated_torque: "float | np.ndarray"  # N·m per joint, continuous — the RMS bound
    safety_factor: float = 0.8
    viscous: float = 0.0  # [N·m·s/rad] per joint
    coulomb: float = 0.0  # [N·m] per joint
    smoothing_eps: float = 1.0  # [rad/s] width of the tanh Coulomb smoothing
    tip_wrench: "float | np.ndarray" = 0.0  # [N·m, N] see the class docstring
    _max: np.ndarray = field(init=False, repr=False)
    _rated: np.ndarray = field(init=False, repr=False)
    _tip: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        n = self.model.n
        object.__setattr__(self, "_max", np.broadcast_to(self.max_torque, (n,)).astype(float))
        object.__setattr__(self, "_rated", np.broadcast_to(self.rated_torque, (n,)).astype(float))
        object.__setattr__(self, "_tip", np.broadcast_to(self.tip_wrench, (6,)).astype(float))

    @property
    def ndof(self) -> int:
        return self.model.n

    def torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Required joint torque [N·m] via RNE, plus smoothed friction.

        Accepts a single configuration (length ``ndof``), a trajectory
        ``(N, ndof)``, or — when ``ndof == 1`` — a flat length-N sample array;
        the output matches the input shape.
        """
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        ddq = np.asarray(ddq, dtype=float)
        flat_1dof = q.ndim == 1 and self.ndof == 1
        single = q.ndim == 1 and not flat_1dof
        pts = np.atleast_2d(q.reshape(-1, 1) if flat_1dof else q)
        vel = np.atleast_2d(dq.reshape(-1, 1) if flat_1dof else dq)
        acc = np.atleast_2d(ddq.reshape(-1, 1) if flat_1dof else ddq)

        tau = np.empty_like(pts)
        for i in range(pts.shape[0]):
            tau[i] = Dynamics.inverse_dynamics(
                pts[i], vel[i], acc[i], self.model.gravity, self._tip, *self.model.dyn_args
            )
        tau += self.viscous * vel + self.coulomb * np.tanh(vel / self.smoothing_eps)
        if flat_1dof:
            return tau[:, 0].reshape(q.shape)
        return tau[0] if single else tau

    def torque_limit(self, dq: np.ndarray) -> np.ndarray:
        """Usable peak torque [N·m], shaped like ``dq`` (constant over speed)."""
        dq = np.asarray(dq, dtype=float)
        if dq.ndim == 1 and self.ndof == 1:
            return np.full(dq.shape, self._max[0] * self.safety_factor)
        return np.broadcast_to(self._max * self.safety_factor, dq.shape).copy()

    def rms_limit(self) -> np.ndarray:
        return self._rated * self.safety_factor
