"""V3.5: the RNE against an independent engine (design doc §6.4, §9).

``test_dynamics_parity`` pits numpy against numba, which is the same equations
twice over: a wrong equation passes both. This file is the gate that can catch
one, because MuJoCo solves the problem with its own code.

Both sides get the same ``(Mlist, Glist, Slist)``, so a difference here is an
implementation difference and nothing else. The design gate is 1%; the measured
difference is around 1e-15 relative, so the assertion is tightened to 1e-9 to
leave a real regression nowhere to hide. Passing says the equations are right,
not that the model matches a real machine (that is V5, on a bench).

This file does not depend on numba, unlike the numpy/JIT parity gates, so it
must not live beside them under their module-level skip.
"""

import numpy as np
import pytest

from motion_core.robots import Dynamics
from motion_core.robots import ur5_model as ur5
from motion_core.robots.robot_math import quintic_time_scaling

MODEL = (ur5.MLIST, ur5.GLIST, ur5.SLIST)
GATE = 1e-9             # design doc asks 1%; measured margin is ~1e-15
NO_TIP = np.zeros(6)


def _random_states(n=50, seed=1):
    """Broad coverage, including velocities that make the Coriolis terms bite."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield (rng.standard_normal(ur5.N), rng.standard_normal(ur5.N),
               rng.standard_normal(ur5.N))

def _trajectory(n=60, duration=1.5):
    """A generated rest-to-rest trajectory, which is what the engine is for."""
    start = ur5.READY
    delta = np.array([0.8, -0.6, 0.9, -0.4, 0.5, 1.2])
    for t in np.linspace(0.0, duration, n):
        s, ds, dds = quintic_time_scaling(t, duration)
        yield start + s * delta, ds * delta, dds * delta


def _worst_relative(states):
    from motion_core.verify.mujoco_cross import inverse_dynamics

    worst, peak = 0.0, 0.0
    for theta, dtheta, ddtheta in states:
        ours = Dynamics.inverse_dynamics(theta, dtheta, ddtheta, ur5.GRAVITY,
                                         NO_TIP, *MODEL)
        theirs = inverse_dynamics(*MODEL, ur5.GRAVITY, theta, dtheta, ddtheta)
        worst = max(worst, float(np.max(np.abs(ours - theirs))))
        peak = max(peak, float(np.max(np.abs(ours))))
    # Relative to the peak torque of the run, not per sample: near a zero
    # crossing a per-sample ratio blows up while the error stays negligible.
    return worst / peak


@pytest.mark.parametrize("states", [_random_states, _trajectory],
                         ids=["random_states", "generated_trajectory"])
def test_mujoco_inverse_dynamics_parity(states):
    """Same states through both engines; joint torque difference within gate."""
    pytest.importorskip("mujoco")
    assert _worst_relative(states()) <= GATE


def test_mjcf_matches_the_model_it_was_built_from():
    """The generated MJCF carries our inertias, not a stand-in robot's.

    The whole gate rests on both engines seeing one model, so check the bridge
    rather than trusting it.
    """
    mujoco = pytest.importorskip("mujoco")
    from motion_core.verify.mujoco_cross import mjcf_from_mr

    model = mujoco.MjModel.from_xml_string(mjcf_from_mr(*MODEL, ur5.GRAVITY))
    assert model.nq == ur5.N
    assert np.allclose(model.body_mass[1:], [g[3, 3] for g in ur5.GLIST])
    assert np.allclose(model.opt.gravity, ur5.GRAVITY)
    # Armature and damping would add torque that the RNE knows nothing about.
    assert np.all(model.dof_armature == 0.0)
    assert np.all(model.dof_damping == 0.0)
