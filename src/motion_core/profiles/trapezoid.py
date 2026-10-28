"""Trapezoidal velocity profile — what a positioning module actually generates.

Rest-to-rest, symmetric accel/decel. ``trapezoid`` is time-optimal under
(v_max, a_max); ``stretched`` scales a profile to a longer imposed total time,
lowering speed and acceleration together — the synchronized-arrival primitive
(design doc §5.3), which leaves torque margin on the stretched axis.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrapezoidProfile:
    d: float  # displacement, > 0
    v_peak: float  # speed actually reached (= command speed of the module)
    a: float  # accel = decel magnitude
    t_a: float  # ramp duration
    t_v: float  # cruise duration (0 for a triangular move)

    @property
    def duration(self) -> float:
        return 2 * self.t_a + self.t_v

    def stretched(self, duration: float) -> "TrapezoidProfile":
        """The same move taking exactly ``duration`` (>= the current one).

        Uniform time scaling: speed drops by ``lam`` and acceleration by
        ``lam**2``, so a synchronized axis also sheds peak torque. Refuses to
        compress — that would need more torque, not less.
        """
        lam = duration / self.duration
        if lam < 1 - 1e-12:
            raise ValueError(f"cannot compress {self.duration:.4f}s into {duration:.4f}s")
        return TrapezoidProfile(d=self.d, v_peak=self.v_peak / lam, a=self.a / lam**2,
                                t_a=self.t_a * lam, t_v=self.t_v * lam)

    def sample(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, dq, ddq) at times ``t``; rests at the endpoints outside [0, T]."""
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, 0.0, self.duration)
        q_ramp = self.a * self.t_a**2 / 2  # distance covered by one ramp
        t1, t2 = self.t_a, self.t_a + self.t_v

        q = np.where(
            tc < t1,
            self.a * tc**2 / 2,
            np.where(
                tc < t2,
                q_ramp + self.v_peak * (tc - t1),
                self.d - self.a * (self.duration - tc) ** 2 / 2,
            ),
        )
        dq = np.where(tc < t1, self.a * tc,
                      np.where(tc < t2, self.v_peak, self.a * (self.duration - tc)))
        ddq = np.where(tc < t1, self.a, np.where(tc < t2, 0.0, -self.a))
        outside = (t < 0) | (t > self.duration)
        return q, np.where(outside, 0.0, dq), np.where(outside, 0.0, ddq)


def trapezoid(d: float, v_max: float, a_max: float) -> TrapezoidProfile:
    """Time-optimal trapezoid over ``d`` (> 0); triangular when too short for v_max."""
    if min(d, v_max, a_max) <= 0:
        raise ValueError("d, v_max, a_max must all be positive")
    v_peak = min(v_max, float(np.sqrt(d * a_max)))
    t_a = v_peak / a_max
    t_v = d / v_peak - t_a
    return TrapezoidProfile(d=d, v_peak=v_peak, a=a_max, t_a=t_a, t_v=t_v)


