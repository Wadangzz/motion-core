"""BallScrew: single-axis LoadModel, layers L0 + L1 + L3 + L6.

All quantities are in motor-shaft coordinates (``docs/theory/conventions.md``):
``q`` is motor angle [rad], the linear-to-motor factor is
``k = lead / (2*pi) * gear_ratio`` [m/rad], and

    tau = (J_m + m*k^2/eta) * ddq  +  B * dq  +  tau_f * tanh(dq/eps)  +  m*g*k/eta

where the gravity term exists only for ``orientation="vertical"`` with positive
``q`` moving the mass up. Efficiency ``eta`` divides the load-side terms as a
constant (L1 first-order model; back-driving is not treated separately).
Coulomb friction is smoothed with ``tanh(dq / eps)`` because the optimizer
differentiates ``torque`` numerically; the width ``eps`` [rad/s] is a
constructor argument, default 1.0 — about 10 rpm, well under any working speed.
Stribeck effects are out of scope.
"""

from dataclasses import dataclass, field

import numpy as np

from motion_core.loads.motor import Motor, get_motor

G = 9.80665  # m/s^2


@dataclass(frozen=True)
class BallScrew:
    """Ball-screw axis driven through an optional gearbox. ``ndof == 1``."""

    lead_mm: float
    gear_ratio: float
    mass_kg: float
    motor: "Motor | str"
    efficiency: float = 0.9
    safety_factor: float = 0.8
    orientation: str = "horizontal"  # "horizontal" | "vertical"
    viscous: float = 0.0  # B [N·m·s/rad], motor shaft
    coulomb: float = 0.0  # tau_f [N·m], motor shaft
    smoothing_eps: float = 1.0  # [rad/s] width of the tanh Coulomb smoothing
    _motor: Motor = field(init=False, repr=False)

    def __post_init__(self):
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(f"unknown orientation: {self.orientation!r}")
        object.__setattr__(self, "_motor", get_motor(self.motor))

    @property
    def ndof(self) -> int:
        return 1

    @property
    def k(self) -> float:
        """Linear-to-motor conversion [m/rad]: ``lead / (2*pi) * gear_ratio``."""
        return self.lead_mm / 1000.0 / (2 * np.pi) * self.gear_ratio

    def torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Required motor torque [N·m]. Broadcasts elementwise (1-dof model)."""
        dq = np.asarray(dq, dtype=float)
        ddq = np.asarray(ddq, dtype=float)
        j_total = self._motor.rotor_inertia + self.mass_kg * self.k**2 / self.efficiency
        tau = j_total * ddq + self.viscous * dq
        tau = tau + self.coulomb * np.tanh(dq / self.smoothing_eps)
        if self.orientation == "vertical":
            tau = tau + self.mass_kg * G * self.k / self.efficiency
        return tau

    def torque_limit(self, dq: np.ndarray) -> np.ndarray:
        """Usable peak torque [N·m]: torque-speed curve times the safety factor."""
        return self._motor.torque_available(dq) * self.safety_factor

    def rms_limit(self) -> np.ndarray:
        """Continuous rating [N·m] times the safety factor, for the cycle RMS check."""
        return np.atleast_1d(self._motor.rated_torque * self.safety_factor)
