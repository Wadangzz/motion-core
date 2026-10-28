"""ArticulatedRobot vs the analytic 1-dof pendulum — the RNE wrapper adds no physics.

The model is built by hand (no URDF, no optional deps): a uniform rod of mass m
and length L, revolute about +y at the origin, COM at r = L/2 along +x at home.
Rotating about a fixed axis the inertia is constant, J = I_yy + m r^2, and the
potential is V = -m g r sin(theta), so

    tau = J * ddtheta - m*g*r*cos(theta)
"""

import numpy as np

from motion_core.loads import ArticulatedRobot
from motion_core.robots import RobotModel

M, L, R = 8.0, 0.5, 0.25
IYY = M * L**2 / 12
GRAV = 9.81


def _translation(x: float) -> np.ndarray:
    t = np.eye(4)
    t[0, 3] = x
    return t


def _pendulum() -> ArticulatedRobot:
    g_mat = np.zeros((6, 6))
    g_mat[:3, :3] = np.diag([0.002, IYY, IYY])
    g_mat[3:, 3:] = M * np.eye(3)
    model = RobotModel(
        Mlist=[_translation(R), _translation(L - R)],
        Glist=[g_mat],
        Slist=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).T,
        M_home=_translation(L),
        gravity=(0.0, 0.0, -GRAV),
    )
    return ArticulatedRobot(model, max_torque=80.0, rated_torque=25.0, safety_factor=0.8)


def _analytic(theta, ddtheta):
    return (IYY + M * R**2) * ddtheta - M * GRAV * R * np.cos(theta)


def test_rne_matches_analytic_pendulum():
    load = _pendulum()
    theta = np.linspace(-1.2, 1.2, 25)
    dtheta = np.linspace(-3.0, 3.0, 25)
    ddtheta = np.linspace(-30.0, 30.0, 25)
    tau = load.torque(theta, dtheta, ddtheta)  # flat 1-dof sample arrays
    assert np.allclose(tau, _analytic(theta, ddtheta), atol=1e-9)


def test_shapes_single_point_and_trajectory():
    load = _pendulum()
    assert load.ndof == 1
    assert load.torque(np.zeros(1), np.zeros(1), np.zeros(1)).shape == (1,)
    assert load.torque(np.zeros((7, 1)), np.zeros((7, 1)), np.zeros((7, 1))).shape == (7, 1)
    assert load.rms_limit().shape == (1,)
    assert load.torque_limit(np.zeros(9)).shape == (9,)
    assert np.allclose(load.torque_limit(np.zeros(9)), 80.0 * 0.8)


def test_friction_terms_add_on_top():
    plain = _pendulum()
    model = plain.model
    rubbed = ArticulatedRobot(model, max_torque=80.0, rated_torque=25.0,
                              viscous=0.02, coulomb=0.5)
    dq = np.r_[5.0]
    extra = rubbed.torque(np.zeros(1), dq, np.zeros(1)) - plain.torque(np.zeros(1), dq, np.zeros(1))
    assert np.isclose(extra[0], 0.02 * 5.0 + 0.5 * np.tanh(5.0), rtol=1e-6)
