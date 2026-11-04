"""Cross-check the RNE implementation against MuJoCo (design doc V3.5).

``test_dynamics_parity`` compares numpy against numba. Those are the same
equations written twice, so a wrong equation passes both. MuJoCo is an
independent codebase, so agreeing with it is evidence about the equations
themselves rather than about the port.

The comparison holds the *model* fixed and varies only the implementation: the
MJCF is generated from the very ``(Mlist, Glist, Slist)`` handed to the RNE, so
a torque difference is an implementation difference. Loading a menagerie robot
instead would compare two different sets of inertias and say nothing about the
algorithm, which is why the design doc words the gate as "same parameters".

A match here means the equations are implemented right. It does not mean the
model matches a real machine; that is the bench measurement (V5).

Frames follow Modern Robotics: link frame {i} sits at the centre of mass of
link i, which is what lets ``Glist[i]`` be block diagonal and lets the MJCF
body frame double as the inertial frame.
"""

import numpy as np

from motion_core.robots.robot_math import SE3, SO3


def _quat_wxyz(rot):
    """SO(3) -> MuJoCo quaternion order (w, x, y, z)."""
    x, y, z, w = SO3.to_quat(rot)
    return w, x, y, z


def _fmt(values):
    return " ".join(f"{v:.17g}" for v in values)


def mjcf_from_mr(mlist, glist, slist, gravity):
    """Build an MJCF serial chain from Modern Robotics parameters.

    :param mlist: ``M_{i-1,i}`` home transforms, ``(n+1, 4, 4)`` (the last entry
        is the tool frame and carries no inertia, so it becomes no body)
    :param glist: spatial inertia of each link in its own frame, ``(n, 6, 6)``
    :param slist: joint screw axes in the space frame, ``(6, n)`` by column
    :param gravity: gravity vector in the space frame, ``(3,)``
    :return: MJCF XML text
    """
    mlist = np.asarray(mlist, dtype=float)
    glist = np.asarray(glist, dtype=float)
    slist = np.asarray(slist, dtype=float)
    n = glist.shape[0]

    head = [
        '<mujoco model="mr_chain">',
        '  <compiler angle="radian"/>',
        f'  <option gravity="{_fmt(gravity)}"/>',
        "  <worldbody>",
    ]
    body, close = [], []
    t_0i = np.eye(4)
    for i in range(n):
        t_0i = t_0i @ mlist[i]
        # Screw axis in link frame {i}. The body frame is {i}, and MJCF wants
        # the joint axis and a point on it in that frame.
        axis = SE3.Adjoint(np.linalg.inv(t_0i)) @ slist[:, i]
        omega, vel = axis[:3], axis[3:]
        point = np.cross(omega, vel)  # zero pitch: v = -omega x q
        inertia = glist[i][:3, :3]
        mass = glist[i][3, 3]
        # MJCF fullinertia order: ixx iyy izz ixy ixz iyz
        full = (inertia[0, 0], inertia[1, 1], inertia[2, 2],
                inertia[0, 1], inertia[0, 2], inertia[1, 2])
        pad = "  " * (i + 2)
        pos = _fmt(mlist[i][:3, 3])
        quat = _fmt(_quat_wxyz(mlist[i][:3, :3]))
        open_body = f'{pad}<body name="link{i + 1}" pos="{pos}" quat="{quat}">'
        mass_tag = f'{pad}  <inertial pos="0 0 0" mass="{mass:.17g}"'
        joint_tag = f'{pad}  <joint name="j{i + 1}" type="hinge"'
        body += [
            open_body,
            f'{mass_tag} fullinertia="{_fmt(full)}"/>',
            f'{joint_tag} axis="{_fmt(omega)}" pos="{_fmt(point)}"/>',
        ]
        close.insert(0, f"{pad}</body>")
    return "\n".join(head + body + close + ["  </worldbody>", "</mujoco>"])


def inverse_dynamics(mlist, glist, slist, gravity, theta, dtheta, ddtheta):
    """Joint torques for one state, computed by MuJoCo.

    Counterpart of ``Dynamics.inverse_dynamics`` with no tip wrench. Armature,
    damping and friction stay at the MJCF defaults of zero, so both sides solve
    the same rigid-body problem.
    """
    import mujoco  # optional dependency: verification only, never the solver

    model = mujoco.MjModel.from_xml_string(
        mjcf_from_mr(mlist, glist, slist, gravity)
    )
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(theta, dtype=float)
    data.qvel[:] = np.asarray(dtheta, dtype=float)
    data.qacc[:] = np.asarray(ddtheta, dtype=float)
    mujoco.mj_inverse(model, data)
    return np.array(data.qfrc_inverse, dtype=float)
