"""URDF → Modern Robotics 동역학 파라미터 (Mlist, Glist, Slist, M_home).

yourdfpy 로 파싱, 직렬 체인(serial arm) 가정. 회전·직동 관절을 한 스크류축으로 통일.
UR5 하드코딩값(ur5_model)과 비교 검증 완료: Slist/Mlist diff < 1e-11, Glist 정확 일치.

고정관절로 붙은 링크(그리퍼·툴·플랜지)의 질량은 직전 액추에이터 링크에 합친다.
안 합치면 조용히 빠진다: Panda 핸드 0.73kg 을 빼면 DROID 실측 토크 대비 2축 오차가
7.6% → 18.1% 로 2.4배가 된다(2026-11-09 실측).

반환 dict 는 robot_math.Dynamics / dynamics_fast 가 그대로 받는 (Mlist, Glist, Slist) 형식.
"""
import numpy as np

_ACT = ("revolute", "continuous", "prismatic")


def load_mr_model(urdf_path, ee_link=None):
    """URDF 경로 → MR 모델 dict.

    ee_link 지정 시 base→ee_link **경로 위의 관절만** 사용(그리퍼 등 분기 제외).
    미지정 시 트리에서 유도한다(_serial_ee_link): 액추에이터가 한 사슬이면 그 마지막
    링크가 EE 고, 분기하면 ValueError. 문서 순서는 쓰지 않는다.
    """
    from yourdfpy import URDF  # robot extra — import 시점에 요구하지 않는다

    r = URDF.load(urdf_path)
    links = {lk.name: lk for lk in r.robot.links}
    if ee_link is None:
        ee_link = _serial_ee_link(r)
    path = _path_joints(r, ee_link)                  # fixed 포함 경로 순서

    Slist, names, limits = [], [], []
    parts, R_link = [], []           # parts[k]=k번째 액추에이터 링크에 붙은 강체들(world)
    T = np.eye(4)                    # home 에서 현재 링크 프레임의 world 변환
    for j in path:
        T = T @ np.asarray(j.origin, float)
        if j.type not in _ACT:
            # fixed 는 변환을 누적하고, 그 자식 링크의 관성은 직전 액추에이터 링크에
            # 합친다. 액추에이터가 아직 없으면 고정 베이스라 동역학에 기여하지 않는다.
            if parts:
                parts[-1].append(_world_body(links[j.child].inertial, T))
            continue
        ax = np.asarray(j.axis, float)
        ax = ax / np.linalg.norm(ax)
        w = T[:3, :3] @ ax           # 축 방향(world)
        q = T[:3, 3]                 # 축 위 한 점(world)
        if j.type == "prismatic":
            Slist.append(np.concatenate([np.zeros(3), w]))
        else:
            Slist.append(np.concatenate([w, -np.cross(w, q)]))
        parts.append([_world_body(links[j.child].inertial, T)])
        R_link.append(T[:3, :3].copy())
        names.append(j.name)
        lim = getattr(j, "limit", None)
        if lim is not None and j.type != "continuous":
            limits.append((float(lim.lower), float(lim.upper)))
        else:
            limits.append((-np.pi, np.pi))

    n = len(names)
    M_home = T.copy()                # EE = ee_link (또는 마지막 액추에이터 링크) 프레임

    # 링크마다 합쳐 (COM 프레임, 공간관성) 으로. 프레임 방향은 링크 프레임을 쓴다
    # (URDF 관성 origin 의 회전이 단위면 합치기 전 값과 비트 동일).
    WC = [np.eye(4)]                 # WC[0]=base 프레임, WC[k]=k번째 액추에이터 COM 프레임
    Glist = []
    for k in range(n):
        mass, com, inertia = _merge_bodies(parts[k])
        C = np.eye(4)
        C[:3, :3], C[:3, 3] = R_link[k], com
        WC.append(C)
        G = np.zeros((6, 6))
        G[:3, :3] = R_link[k].T @ inertia @ R_link[k]     # COM 관성텐서(링크 프레임)
        G[3:, 3:] = mass * np.eye(3)
        Glist.append(G)

    Mlist = [WC[1]]                  # M01 = WC1 (base→링크1 COM)
    for i in range(1, n):
        Mlist.append(np.linalg.inv(WC[i]) @ WC[i + 1])
    Mlist.append(np.linalg.inv(WC[n]) @ M_home)          # 링크n COM → EE

    return {
        "name": r.robot.name,
        "n": n,
        "Mlist": Mlist,
        "Glist": Glist,
        "Slist": np.array(Slist).T,  # 6×n (열 = 관절)
        "M_home": M_home,
        "joint_names": names,
        "joint_limits": limits,
    }


def _world_body(ine, T):
    """링크의 <inertial> → home 의 world 프레임 기준 (질량, COM 위치, 관성텐서).

    ``ine`` 가 없는 링크(관성 미기재)는 질량 0 으로 본다.
    """
    if ine is None:
        return 0.0, T[:3, 3].copy(), np.zeros((3, 3))
    C = T @ np.asarray(ine.origin, float)            # COM 프레임(world)
    R = C[:3, :3]
    return float(ine.mass), C[:3, 3], R @ np.asarray(ine.inertia, float) @ R.T


def _merge_bodies(bodies):
    """강체 여러 개를 하나로 (평행축 정리). 전부 같은 world 방향 기준."""
    mass = sum(b[0] for b in bodies)
    if mass == 0.0:                                  # 질량 미기재 링크뿐 → COM 미정의
        return 0.0, bodies[0][1], sum(b[2] for b in bodies)
    com = sum(m * c for m, c, _ in bodies) / mass
    inertia = np.zeros((3, 3))
    for m, c, I in bodies:
        d = c - com
        inertia += I + m * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def _path_joints(r, ee_link):
    """base(root) → ee_link 경로의 관절 리스트(fixed 포함, 순서대로)."""
    by_child = {j.child: j for j in r.robot.joints}
    seq = []
    cur = ee_link
    while cur in by_child:            # child→parent 로 거슬러 올라감
        seq.append(by_child[cur])
        cur = by_child[cur].parent
    return list(reversed(seq))


def _serial_ee_link(r):
    """액추에이터가 한 사슬인지 확인하고 마지막 액추에이터 링크를 돌려준다.

    문서 순서(= 체인 순서) 가정이 조용히 틀리는 자리가 둘이다. 관절을 역순으로 적은
    URDF(SO-ARM101)는 모델이 스크램블되고, 형제 관절(그리퍼 손가락 2개)은 직렬
    사슬로 이어붙는다. 둘 다 예외가 아니라 잘못된 숫자로 나오므로, 가장 깊은
    액추에이터까지의 경로가 액추에이터 전부를 담지 못하면 거부한다.
    """
    act = [j for j in r.robot.joints if j.type in _ACT]
    if not act:
        raise ValueError("액추에이터 관절이 없다")
    by_child = {j.child: j for j in r.robot.joints}

    def depth(link):
        d = 0
        while link in by_child:
            link, d = by_child[link].parent, d + 1
        return d

    ee = max((j.child for j in act), key=depth)
    on_path = {j.name for j in _path_joints(r, ee)}
    off = [j.name for j in act if j.name not in on_path]
    if off:
        raise ValueError(
            f"액추에이터가 직렬이 아니다: {ee} 경로 밖의 관절 {off}. "
            "ee_link 로 체인을 지정하라")
    return ee
