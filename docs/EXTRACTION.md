# robot-web-simulator 에서 motion-core 추출

이 문서는 **절차서**다. 왜 이렇게 하는지는
`D:\design_docs\motion-platform-design.md` §2, 현재 상태는
`D:\design_docs\motionopt-scaffold\START_HERE.md`.

> 🔴 **사용자 승인 전에 실행하지 않는다.** 기존 레포(`robot_web_simulator`)를 수정하는
> 작업이다. START_HERE §2 의 A/B/C 결정이 먼저다. 아래는 **A(진짜 분리)** 기준이다.

---

## 0. 원본 사실 (2026-10-26 실측)

- 원본: `C:\Users\JLT_JY_PC\Documents\GitHub\robot_web_simulator\backend\app\core\`
- 마지막 커밋: 2026-06-25 (`chore(docs): 백엔드 레거시 문서 3종 정리`)
- **코어 전체에 FastAPI 의존이 0** — 그래서 추출이 사실상 이동 작업이다
- 코어 모듈을 참조하는 상위 코드는 **7곳뿐**이다 — `sim.py` · `interactive.py` ·
  `controllers/kinematics.py` · `controllers/simulate.py` · `routers/robots.py` ·
  `routers/interactive_ws.py` · `tests/test_dynamics_parity.py`
- 작업 트리 상태(2026-10-27 확인): 브랜치 `develop`, `backend/app/sim.py` 1개 수정됨.
  **추출이 같은 파일의 import 를 재작성하므로 착수 전에 정리해야 한다**

---

## 1. 사전 확인

```bash
cd /c/Users/JLT_JY_PC/Documents/GitHub/robot_web_simulator
git status --porcelain          # 비어 있어야 한다. 아니면 사용자에게 확인
git branch --show-current
git switch -c feature/extract-motion-core
```

`motion-core` 는 이미 `C:\Users\JLT_JY_PC\Documents\GitHub\motion-core` 에서 git 이
초기화되어 있고 스캐폴드 커밋이 남아 있다. 추출은 그 위에 이어서 커밋한다.

---

## 2. 파일별 처리

| 원본 (`backend/app/core/`) | 크기 | 외부 의존 | 대상 (`src/motion_core/`) |
| --- | --- | --- | --- |
| `robot_math.py` | 17KB | numpy, scipy | `robots/robot_math.py` |
| `dynamics_fast.py` | 5KB | numpy, numba | `robots/dynamics_fast.py` |
| `robot_model.py` | 2.8KB | `app.core.robot_math` | `robots/robot_model.py` |
| `robot_registry.py` | 2.6KB | `app.core.robot_model`, importlib | `robots/robot_registry.py` |
| `ur5_model.py` | 3.5KB | numpy | `robots/ur5_model.py` |
| `urdf_loader.py` | 3.5KB | yourdfpy | `robots/urdf_loader.py` |
| `paths.py` | 0.6KB | os | **앱에 남긴다** — 앱 경로 규약 |

**파일명은 바꾸지 않는다.** 이동과 리네임을 섞으면 깨진 지점을 못 찾는다.
`robot_math.py` 를 `spatial/kinematics/dynamics` 로 쪼개는 것은 이동이 끝나고
테스트가 녹색이 된 뒤 별도 커밋으로 한다.

### import 재작성

원본에 있는 내부 import 는 두 곳뿐이다.

```
robot_model.py     : from app.core.robot_math  import SE3, Kinematics
                   → from motion_core.robots.robot_math import SE3, Kinematics

robot_registry.py  : from app.core.robot_model import RobotModel
                   → from motion_core.robots.robot_model import RobotModel
```

`robots/__init__.py` 에 공개 API 를 모은다 — 외부 소비자는 이것만 본다.

```python
from motion_core.robots.robot_math import SO3, SE3, Kinematics, Dynamics
from motion_core.robots.robot_model import RobotModel
from motion_core.robots.robot_registry import get_robot        # 실제 이름 확인 후
```

`quintic_time_scaling` 은 `robot_math.py` 안에 있다. 궤적 생성기이므로 최종 위치는
`motion_core/profiles/` 가 맞다 — 다만 **이동 단계에서는 건드리지 않고**, 프로파일
구현을 시작할 때 함께 옮긴다(교체 대상이므로 그때 비교 기준으로 남긴다).

### 선택 의존 처리

`motion-core/tests/test_boundaries.py` 가 **모듈 레벨 `import numba` / `import yourdfpy`
를 실패로 잡는다.** 둘은 함수 안 가드 import 로 바꾸거나, 해당 모듈을 지연 로드한다.

- `dynamics_fast.py` — numba 가 없으면 numpy 구현으로 폴백. `fast` extra
- `urdf_loader.py` — yourdfpy 는 함수 안에서 import. `robot` extra

이 테스트가 바로 통과하지 않는 것이 정상이다. **테스트를 고치지 말고 코드를 고친다.**

---

## 3. parity 테스트 이식

원본: `backend/tests/test_dynamics_parity.py` (numpy ↔ numba 비트일치)

→ `motion-core/tests/test_dynamics_parity.py` 의 `test_numpy_jit_parity`
(현재 `@pytest.mark.skip`)를 실물로 교체한다. 같은 파일의
`test_mujoco_inverse_dynamics_parity` 는 **V3.5** 자리이며 3단계까지 skip 유지.

---

## 4. 시뮬레이터를 소비자로 전환

```bash
cd /c/Users/JLT_JY_PC/Documents/GitHub/robot_web_simulator/backend
# 원본 삭제
git rm app/core/robot_math.py app/core/dynamics_fast.py app/core/robot_model.py \
       app/core/robot_registry.py app/core/ur5_model.py app/core/urdf_loader.py
# 의존 추가 (개발 중에는 editable)
uv add --editable ../../motion-core    # 또는 pip install -e
```

남은 `app/` 전체에서 참조를 재작성한다.

```bash
grep -rn "app\.core\.\(robot_math\|dynamics_fast\|robot_model\|robot_registry\|ur5_model\|urdf_loader\)" \
  app --include=*.py
```

→ `motion_core.robots.X` 로 바꾼다. `app.core.paths` 는 그대로 둔다.

**게이트: 시뮬레이터의 기존 테스트가 전부 통과해야 1단계가 끝난 것이다.**
통과 못 하면 되돌리지 말고 원인을 고친다 — 어차피 지금이 가장 싼 시점이다.

---

## 5. 완료 조건

```bash
cd /c/Users/JLT_JY_PC/Documents/GitHub/motion-core
pip install -e ".[dev]"
pytest -q                      # parity 실물 테스트 포함해 통과
ruff check .

cd /c/Users/JLT_JY_PC/Documents/GitHub/robot_web_simulator/backend
uv run pytest -q               # 기존 테스트 전부 통과 ← 1단계 게이트
```

- 두 레포가 **같은 작업 단위**이므로 커밋을 짝지어 남긴다
- `motion-core` 가 `app` 을 import 하지 않는지는 경계 테스트가 자동으로 본다
- 끝나면 START_HERE.md §4(현재 상태)와 §5(다음 작업)를 갱신한다
