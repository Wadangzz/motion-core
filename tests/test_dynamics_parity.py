"""Parity gates for the dynamics implementations.

Two of these come across with the extracted code; the third is new and is the
one that can actually catch a wrong equation.

- numpy vs JIT: same equations, two implementations. Catches porting bugs.
- vs MuJoCo (V3.5): an independent implementation. Catches wrong equations.
  Optional dependency, skipped when absent. A match here says the implementation
  is right, not that the model matches a real machine -- that is the bench test.
"""

import numpy as np
import pytest

from motion_core.robots import Dynamics
from motion_core.robots import dynamics_fast as df
from motion_core.robots import ur5_model as ur5

MODEL = (ur5.MLIST, ur5.GLIST, ur5.SLIST)
STACK = df.stack_model(*MODEL)
TOL = 1e-10                          # bit-identical expected; float slack only

pytestmark = pytest.mark.skipif(
    not df.HAVE_NUMBA, reason="numba absent: the JIT path falls back to the numpy one"
)


def _states(n=200, seed=1):
    """Reproducible random (theta, dtheta, ddtheta, tau, Ftip) states."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield (rng.standard_normal(ur5.N), rng.standard_normal(ur5.N),
               rng.standard_normal(ur5.N), rng.standard_normal(ur5.N),
               rng.standard_normal(6))


def test_inverse_dynamics_parity():
    md = 0.0
    for th, dth, ddth, _, Ftip in _states():
        a = Dynamics.inverse_dynamics(th, dth, ddth, ur5.GRAVITY, Ftip, *MODEL)
        b = df.inverse_dynamics(th, dth, ddth, ur5.GRAVITY, Ftip, *STACK)
        md = max(md, float(np.max(np.abs(a - b))))
    assert md < TOL, f"inverse_dynamics max|diff| = {md}"


def test_mass_matrix_parity():
    md = 0.0
    for th, *_ in _states():
        a = Dynamics.mass_matrix(th, *MODEL)
        b = df.mass_matrix(th, *STACK)
        md = max(md, float(np.max(np.abs(a - b))))
    assert md < TOL, f"mass_matrix max|diff| = {md}"


def test_gravity_forces_parity():
    md = 0.0
    for th, *_ in _states():
        a = Dynamics.gravity_forces(th, ur5.GRAVITY, *MODEL)
        b = df.gravity_forces(th, ur5.GRAVITY, *STACK)
        md = max(md, float(np.max(np.abs(a - b))))
    assert md < TOL, f"gravity_forces max|diff| = {md}"


def test_coriolis_forces_parity():
    md = 0.0
    for th, dth, *_ in _states():
        a = Dynamics.coriolis_forces(th, dth, *MODEL)
        b = df.coriolis_forces(th, dth, *STACK)
        md = max(md, float(np.max(np.abs(a - b))))
    assert md < TOL, f"coriolis_forces max|diff| = {md}"


def test_forward_dynamics_parity():
    md = 0.0
    for th, dth, _, tau, Ftip in _states():
        a = Dynamics.forward_dynamics(th, dth, tau, ur5.GRAVITY, Ftip, *MODEL)
        b = df.forward_dynamics(th, dth, tau, ur5.GRAVITY, Ftip, *STACK)
        md = max(md, float(np.max(np.abs(a - b))))
    assert md < TOL, f"forward_dynamics max|diff| = {md}"


@pytest.mark.parametrize("th", [ur5.READY, np.zeros(ur5.N)])
def test_named_poses_parity(th):
    """Also identical at the operating poses: home (singular) and ready."""
    th = np.asarray(th, float)
    a = Dynamics.gravity_forces(th, ur5.GRAVITY, *MODEL)
    b = df.gravity_forces(th, ur5.GRAVITY, *STACK)
    assert np.max(np.abs(a - b)) < TOL


@pytest.mark.skip(reason="not implemented: V3.5 cross-check")
def test_mujoco_inverse_dynamics_parity():
    """Same trajectory through both engines; joint torque difference <= 1%."""
    pytest.importorskip("mujoco")
    raise NotImplementedError
