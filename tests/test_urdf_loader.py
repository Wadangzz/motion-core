"""urdf_loader: 고정관절로 붙은 링크(그리퍼·툴)의 관성 합치기.

로더는 테스트가 없었다. 레지스트리의 URDF 로봇은 첫 사용 때 네트워크로 URDF 를
받고, 테스트는 하드코딩 UR5(`ur5_model`)만 썼다. 여기는 합성 URDF 하나를 손계산과
대조한다: 네트워크도 `robot_descriptions` 도 필요 없다.

기구는 1축이다. 회전관절 하나(z축, 원점) + 링크1(질량 1, COM x=1, 점질량) + 고정관절로
x=2 에 붙은 툴(점질량). 합친 강체의 값은 손으로 떨어진다:

    질량  1 + m_tool
    COM   (1*1 + 2*m_tool) / (1 + m_tool)
    관성  평행축 정리. m_tool=1 이면 COM x=1.5, 두 점질량이 각각 d=0.5 →
          diag(0, 2*1*0.25, 2*1*0.25) = diag(0, 0.5, 0.5)
"""

import numpy as np
import pytest

pytest.importorskip("yourdfpy")

from motion_core.robots.urdf_loader import load_mr_model

_URDF = """<robot name="fixed_tool_rig">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="1 0 0"/><mass value="1"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <link name="tool">
    <inertial>
      <origin xyz="0 0 0"/><mass value="{tool_mass}"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="link1"/><child link="tool"/><origin xyz="2 0 0"/>
  </joint>
</robot>
"""


def _rig(tmp_path, tool_mass):
    p = tmp_path / "rig.urdf"
    p.write_text(_URDF.format(tool_mass=tool_mass))
    return load_mr_model(str(p), ee_link="tool")


def test_fixed_joint_link_mass_is_merged(tmp_path):
    """툴 질량이 직전 액추에이터 링크에 들어가고, COM 과 관성이 같이 옮겨진다."""
    m = _rig(tmp_path, tool_mass=1.0)

    assert m["n"] == 1                                   # 고정관절은 자유도가 아니다
    G = m["Glist"][0]
    assert G[3, 3] == pytest.approx(2.0)                 # 1 + 1
    np.testing.assert_allclose(G[:3, :3], np.diag([0.0, 0.5, 0.5]), atol=1e-12)
    # Mlist[0] = base → 링크1 COM. 합친 COM 은 x=1.5
    np.testing.assert_allclose(m["Mlist"][0][:3, 3], [1.5, 0, 0], atol=1e-12)
    # Mlist[1] = 링크1 COM → EE(tool, x=2)
    np.testing.assert_allclose(m["Mlist"][1][:3, 3], [0.5, 0, 0], atol=1e-12)


def test_massless_fixed_link_leaves_the_link_untouched(tmp_path):
    """질량 0 인 툴(플랜지 등)은 아무것도 바꾸지 않는다. 합치기의 하위호환 계약이다."""
    m = _rig(tmp_path, tool_mass=0.0)

    G = m["Glist"][0]
    assert G[3, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(G[:3, :3], np.zeros((3, 3)), atol=1e-12)
    np.testing.assert_allclose(m["Mlist"][0][:3, 3], [1.0, 0, 0], atol=1e-12)


def test_tool_mass_does_not_move_the_kinematics(tmp_path):
    """질량은 동역학만 바꾼다. 스크류축과 EE 프레임은 그대로다."""
    light = _rig(tmp_path, tool_mass=0.0)
    heavy = _rig(tmp_path, tool_mass=5.0)

    np.testing.assert_array_equal(light["Slist"], heavy["Slist"])
    np.testing.assert_array_equal(light["M_home"], heavy["M_home"])
    np.testing.assert_allclose(heavy["M_home"][:3, 3], [2, 0, 0], atol=1e-12)


# ---------------------------------------------------------------------------
# ee_link 미지정 경로: 체인을 문서 순서가 아니라 트리에서 유도한다.
#
# 문서 순서 가정은 조용히 틀린다. SO-ARM101 URDF 는 관절을 역순(6,5,4,3,2,1)으로
# 적어서 그 가정으로 만들면 모델이 스크램블되고, 형제 관절(그리퍼 손가락 2개)은
# 직렬 사슬로 이어붙는다(레지스트리의 omx 가 그랬다). 둘 다 예외가 아니라 잘못된
# 숫자로 나오므로, 유도 불가는 에러여야 한다.

_REVERSED = """<robot name="reversed_doc_order">
  <link name="base"/>
  <link name="mid"><inertial><origin xyz="0 0 0"/><mass value="1"/>
    <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <link name="tip"><inertial><origin xyz="0 0 0"/><mass value="1"/>
    <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <joint name="joint2" type="revolute">
    <parent link="mid"/><child link="tip"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="mid"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""

_SIBLINGS = """<robot name="two_fingers">
  <link name="base"/>
  <link name="palm"><inertial><origin xyz="0 0 0"/><mass value="1"/>
    <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
  <link name="left"/>
  <link name="right"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="palm"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="finger_left" type="prismatic">
    <parent link="palm"/><child link="left"/>
    <origin xyz="1 0 0"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="0.02" effort="1" velocity="1"/>
  </joint>
  <joint name="finger_right" type="prismatic">
    <parent link="palm"/><child link="right"/>
    <origin xyz="1 0 0"/><axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.02" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _write(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_chain_order_comes_from_the_tree_not_the_document(tmp_path):
    """관절을 역순으로 적은 URDF 도 base→tip 순서로 나온다."""
    m = load_mr_model(_write(tmp_path, _REVERSED, "rev.urdf"))

    assert m["joint_names"] == ["joint1", "joint2"]
    # joint1 은 원점, joint2 는 x=1 축. 문서 순서로 읽으면 뒤집혀서 -Slist 열이 바뀐다.
    np.testing.assert_allclose(m["Slist"][:, 0], [0, 0, 1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(m["Slist"][:, 1], [0, 0, 1, 0, -1, 0], atol=1e-12)


def test_branching_actuators_raise_instead_of_serializing(tmp_path):
    """형제 관절(손가락 2개)은 직렬이 아니다. 조용히 이어붙이지 말고 거부한다."""
    with pytest.raises(ValueError, match="finger_right"):
        load_mr_model(_write(tmp_path, _SIBLINGS, "fingers.urdf"))
