"""RackPinion: single-axis LoadModel, layers L0 + L1 + L3 + L6.

Same physics as ``BallScrew`` — a mass driven at ``k`` [m/rad] — with the
rack-and-pinion parameter vocabulary. Note the gear convention difference:

    BallScrew:  k = lead  / (2*pi) * gear_ratio   (screw revs per motor rev)
    RackPinion: k = pitch / (2*pi) / gear_ratio   (gear_ratio = reduction,
                                                   motor revs per pinion rev)

``pitch_mm`` is rack travel per pinion revolution. Coulomb smoothing and the
constant-efficiency L1 model follow ``ball_screw.py``; pinion and reducer
inertia are neglected (synthetic benchmark mechanisms state their own numbers).
"""

from dataclasses import dataclass, field

import numpy as np

from motion_core.loads.motor import Motor, get_motor

G = 9.80665  # m/s^2


@dataclass(frozen=True)
class RackPinion:
    """Rack-and-pinion axis behind a reducer. ``ndof == 1``."""

    pitch_mm: float  # rack travel per pinion revolution
    gear_ratio: float  # reduction: motor revs per pinion rev (10.0 = 10:1)
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
        """Linear-to-motor conversion [m/rad]: ``pitch / (2*pi) / gear_ratio``."""
        return self.pitch_mm / 1000.0 / (2 * np.pi) / self.gear_ratio

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
