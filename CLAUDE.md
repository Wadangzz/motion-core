# CLAUDE.md

## 먼저 읽을 것

**현재 상태·다음 작업의 정본은 `..\motionopt\START_HERE.md` 다.**
작업을 시작하기 전에 읽는다. 설계 의도(수식·검증 V0~V5·로드맵)는
`D:\design_docs\motion-platform-design.md`, 끝난 추출의 이력은 `docs/EXTRACTION.md`.

**식을 건드리기 전에 `docs/theory/conventions.md` 를 읽는다.** 좌표·단위 규약이
거기 한 번만 정의돼 있고, 이 분야 부호 오류의 1순위가 프레임 혼동이다.

소비자(`../motionopt`, `../robot_web_simulator`)와 같이 바뀌는 작업이 많다 →
세션을 `C:\Users\JLT_JY_PC\Documents\GitHub\` 에서 시작하면 세 레포가 범위에 들어온다.

## 이 레포가 무엇인가

공유 엔진. 두 소비자(로봇 웹 시뮬레이터 · 산업용 `motionopt`)가 **editable 의존**으로
이걸 쓴다. 그래서 두 가지가 테스트로 강제된다 (`tests/test_boundaries.py`):

1. **선택 의존을 모듈 레벨에서 import 하지 않는다** — numba · mujoco · yourdfpy ·
   casadi · matplotlib · yaml. `numpy + scipy` 만으로 import 돼야 한다.
   numba 는 `dynamics_fast.py` 의 `try/except` + `HAVE_NUMBA` 폴백(없으면 같은 식을
   순수 NumPy 로), yourdfpy 는 함수 안 import 다
2. **소비자를 import 하지 않는다** — `motionopt` · `app` · `robot_web_simulator`

걸리면 **테스트를 고치지 말고 코드를 고친다.**

## 현재 코드 상태

- `robots/` ✅ 시뮬레이터에서 추출 완료 — robot_math · dynamics_fast · robot_model ·
  robot_registry · ur5_model · urdf_loader. numpy↔numba parity 테스트가 붙어 있다
- `loads/base.py` — `LoadModel` Protocol (모든 반환이 **배열**: 1축은 길이 1, 6축 로봇은
  길이 6. 최적화기가 자유도와 무관해지는 이유)
- `profiles/ optimize/ verify/` ⬜ 빈 패키지 — **본론이 여기다**

## 명령

```bash
uv pip install -e ".[dev]"            # CI 와 같은 경로
uv run --no-sync python -m pytest -q  # 10 passed, 1 skipped(mujoco=V3.5)
uv run --no-sync ruff check .         # 깨끗해야 한다 (line-length 100)
```

numba 없는 설치를 확인할 때:
`PYTHONPATH=src uv run --with pytest --with numpy --with scipy -m pytest tests -q`
→ 3 passed, 8 skipped.

## 잊으면 다시 헤매는 것

- **`max |tau|` 를 목적함수에 직접 넣지 않는다** — 비평활. epigraph 변환(보조변수 `t`)
  필수. SLSQP 가 안 돌면 솔버가 아니라 정식화를 먼저 의심한다
- **쿨롱 마찰 `sgn` 은 평활화한다** — 최적화기가 수치 미분한다. 평활 폭을 docstring 에 적는다
- **MuJoCo 는 검증 전용**(V3.5, ≤1%). gradient 가 없어 최적화 루프에 넣지 않는다.
  numpy↔numba parity 는 같은 수식의 두 구현이라 수식이 틀리면 둘 다 틀린다 — 그 맹점을
  MuJoCo 가 덮는다. 단 시뮬 둘이 맞아도 실기 증거는 아니다(그건 V5)
- **선행 연구를 베이스라인으로 명시** — 경로 고정 토크제한 시간최적은 기풀린 문제
  (Bobrow, Pfeiffer-Johanni, TOPP-RA). 새 알고리즘이라 주장하지 않는다
- 벤치마크는 전부 합성 기구다. 고객사 데이터·`.gx3` 지식·사내 도구 코드는 들어오지
  않는다 (공개 예정 레포)
