"""LoadModel protocol shared by every optimizer and the verifier.

All quantities are **arrays**, not scalars: a single industrial axis returns
length-1 arrays and a 6-axis robot returns length-6, so the optimizers and the
verifier are independent of the number of degrees of freedom.

Coordinate and unit conventions live in ``docs/theory/conventions.md`` and every
implementation follows them. The generalized coordinate ``q`` is motor angle
[rad] for a single axis and the joint vector [rad] for a robot. A cam master
angle is converted to ``q`` by the implementation, never by the caller.
"""

from typing import Protocol

import numpy as np


class LoadModel(Protocol):
    """Maps motion to required torque, and reports what the motor can deliver."""

    @property
    def ndof(self) -> int:
        """Number of degrees of freedom. Every array below has this length."""
        ...

    def torque(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Required torque [N·m] at the given position, velocity and acceleration.

        Must be smooth in all three arguments: the optimizer differentiates it
        numerically. Discontinuous terms (Coulomb friction ``sgn``) are smoothed
        by the implementation, which documents the smoothing width.
        """
        ...

    def torque_limit(self, dq: np.ndarray) -> np.ndarray:
        """Usable torque at the given speed [N·m].

        Torque-speed curve times the safety factor, so the optimizer sees one
        speed-dependent bound (layer L6).
        """
        ...

    def rms_limit(self) -> np.ndarray:
        """Continuous rated torque [N·m] for the cycle RMS thermal check.

        On high-duty cam motion this constraint usually binds before the peak.
        """
        ...
