"""Seven-segment double-S (jerk-limited) rest-to-rest profile.

Time-optimal under symmetric velocity / acceleration / jerk bounds, following
the standard construction (Biagiotti & Melchiorri ch. 3.4). Degenerate moves —
too short to reach ``v_max`` or ``a_max`` — fall into the usual branches.

This is the "fastest within limits" family; the quintic polynomial is the
"smooth for a given T" family. The difference between the two, with T derived
from the torque limit instead of chosen by hand, is the cycle-time claim of the
whole project (design doc §2.2, §8).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoubleSProfile:
    """Sampled as position/velocity/acceleration vs time [s]; units follow ``d``."""

    d: float  # displacement, > 0
    v_lim: float  # velocity actually reached
    a_lim: float  # acceleration actually reached
    j_max: float
    t_j: float  # jerk phase duration
    t_a: float  # total acceleration ramp duration (= deceleration, symmetric)
    t_v: float  # constant-velocity duration

    @property
    def duration(self) -> float:
        return 2 * self.t_a + self.t_v

    def sample(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, dq, ddq) at times ``t``; clamped to the rest states outside [0, T]."""
        t = np.asarray(t, dtype=float)
        j = self.j_max
        durations = np.array([
            self.t_j, self.t_a - 2 * self.t_j, self.t_j, self.t_v,
            self.t_j, self.t_a - 2 * self.t_j, self.t_j,
        ])
        jerks = j * np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        starts = np.r_[0.0, np.cumsum(durations)]

        # Phase-start states by stepping the piecewise-constant jerk.
        q0, v0, a0 = [0.0], [0.0], [0.0]
        for dur, jk in zip(durations, jerks):
            q0.append(q0[-1] + v0[-1] * dur + a0[-1] * dur**2 / 2 + jk * dur**3 / 6)
            v0.append(v0[-1] + a0[-1] * dur + jk * dur**2 / 2)
            a0.append(a0[-1] + jk * dur)

        tc = np.clip(t, 0.0, starts[-1])
        idx = np.clip(np.searchsorted(starts, tc, side="right") - 1, 0, 6)
        dt = tc - starts[idx]
        jk = jerks[idx]
        q0 = np.asarray(q0)
        v0 = np.asarray(v0)
        a0 = np.asarray(a0)
        q = q0[idx] + v0[idx] * dt + a0[idx] * dt**2 / 2 + jk * dt**3 / 6
        dq = v0[idx] + a0[idx] * dt + jk * dt**2 / 2
        ddq = a0[idx] + jk * dt
        # Outside [0, T] the axis rests at an endpoint.
        dq = np.where((t < 0) | (t > starts[-1]), 0.0, dq)
        ddq = np.where((t < 0) | (t > starts[-1]), 0.0, ddq)
        return q, dq, ddq


def double_s(d: float, v_max: float, a_max: float, j_max: float) -> DoubleSProfile:
    """Time-optimal rest-to-rest double-S over distance ``d`` (> 0)."""
    if min(d, v_max, a_max, j_max) <= 0:
        raise ValueError("d, v_max, a_max, j_max must all be positive")

    # Ramp shape if v_max is the binding limit.
    if v_max * j_max >= a_max**2:  # a_max is reached
        t_j = a_max / j_max
        t_a = t_j + v_max / a_max
    else:  # v_max reached before a_max
        t_j = np.sqrt(v_max / j_max)
        t_a = 2 * t_j
    t_v = d / v_max - t_a

    if t_v >= 0:
        v_lim = v_max
    else:
        # No cruise phase. First assume a_max is still reached:
        t_v = 0.0
        t_j = a_max / j_max
        t_a = (t_j + np.sqrt(t_j**2 + 4 * d / a_max)) / 2
        if t_a < 2 * t_j:
            # Not even a_max is reached: pure jerk-limited move.
            t_j = (d / (2 * j_max)) ** (1 / 3)
            t_a = 2 * t_j
        v_lim = j_max * t_j * (t_a - t_j)

    a_lim = j_max * t_j
    return DoubleSProfile(d=d, v_lim=v_lim, a_lim=a_lim, j_max=j_max,
                          t_j=t_j, t_a=t_a, t_v=t_v)
