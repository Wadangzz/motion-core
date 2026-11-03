"""Parity gates for the dynamics implementations.

numpy vs JIT: the same equations written twice, so this catches porting bugs
and nothing else. A wrong equation passes both sides.

The gate that can catch a wrong equation is V3.5, against MuJoCo, and it lives
in test_mujoco_cross.py.
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


# V3.5 (vs MuJoCo) lives in test_mujoco_cross.py: it must not sit behind this
# module's numba skip, having nothing to do with the JIT path.
