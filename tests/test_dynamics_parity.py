"""Parity gates for the dynamics implementations.

Two of these come across with the extracted code; the third is new and is the
one that can actually catch a wrong equation.

- numpy vs JIT: same equations, two implementations. Catches porting bugs.
- vs MuJoCo (V3.5): an independent implementation. Catches wrong equations.
  Optional dependency, skipped when absent. A match here says the implementation
  is right, not that the model matches a real machine -- that is the bench test.
"""

import pytest


@pytest.mark.skip(reason="not implemented: dynamics not extracted yet")
def test_numpy_jit_parity():
    raise NotImplementedError


@pytest.mark.skip(reason="not implemented: V3.5 cross-check")
def test_mujoco_inverse_dynamics_parity():
    """Same trajectory through both engines; joint torque difference <= 1%."""
    pytest.importorskip("mujoco")
    raise NotImplementedError
