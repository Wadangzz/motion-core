"""Motor data for layer L6: torque-speed curve and thermal (RMS) rating.

Every motor here is a synthetic example written for this project — no vendor
catalog data. The torque-speed curve is a first-order shape: constant maximum
torque up to rated speed, then a linear derate down to rated torque at maximum
speed. Real curves come from the datasheet; this one exists so the optimizer
sees a speed-dependent bound at all.
"""

from dataclasses import dataclass

import numpy as np

RPM = 2 * np.pi / 60.0  # rad/s per rpm


@dataclass(frozen=True)
class Motor:
    """Servo motor limits, motor-shaft coordinates."""

    name: str
    rated_torque: float  # N·m, continuous — the RMS thermal bound
    max_torque: float  # N·m, short-time peak at low speed
    rated_speed: float  # rad/s, end of the constant-peak-torque region
    max_speed: float  # rad/s, peak torque has derated to rated torque here
    rotor_inertia: float  # kg·m^2

    def torque_available(self, dq: np.ndarray) -> np.ndarray:
        """Peak torque [N·m] available at speed ``dq`` [rad/s], smooth in ``dq``.

        Smoothness matters: the optimizer differentiates this numerically. The
        corner at rated speed is softened with a softplus of width 1% of the
        torque derate span. Past ``max_speed`` the line keeps falling, so an
        overspeed shows up as an unsatisfiable torque bound rather than a
        separate constraint.
        """
        speed = np.abs(np.asarray(dq, dtype=float))
        slope = (self.max_torque - self.rated_torque) / (self.max_speed - self.rated_speed)
        derated = self.max_torque - slope * (speed - self.rated_speed)
        # Smooth min(max_torque, derated).
        width = 0.01 * (self.max_torque - self.rated_torque)
        return self.max_torque - width * np.logaddexp(0.0, (self.max_torque - derated) / width)


EXAMPLE_MOTORS = {
    # Plausible numbers for a 400 W low-inertia servo. Synthetic, not a catalog entry.
    "example_400W": Motor(
        name="example_400W",
        rated_torque=1.3,
        max_torque=3.9,
        rated_speed=3000 * RPM,
        max_speed=6000 * RPM,
        rotor_inertia=0.28e-4,
    ),
}


def get_motor(motor: "Motor | str") -> Motor:
    """Accept either a ``Motor`` or the name of a packaged example motor."""
    if isinstance(motor, Motor):
        return motor
    return EXAMPLE_MOTORS[motor]
