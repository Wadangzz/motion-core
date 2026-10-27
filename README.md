# motion-core

**Rigid-body dynamics and torque-limited motion optimization — one engine, two
very different machines.**

`motion-core` answers two questions about any mechanism: *what torque does this
motion demand*, and *what is the fastest motion the motor can actually deliver*.
It models the load — inertia, friction, gravity, position-dependent terms — and
solves for the profile that minimizes peak torque, thermal (RMS) torque or cycle
time under the real torque-speed limit of the drive.

The same interface and the same solver cover a single ball-screw axis and a
six-axis serial robot. That is the claim this package exists to make: industrial
motion design and robot trajectory generation are the same constrained
optimization problem, and only the load model and the output format differ.

No training data. Physics and constraints are enough.

> **Status: early development.** The dynamics layer has been extracted from an
> existing robot simulator and is in place; the optimization layer is new and the
> API below is still the target design.

---

## Who uses it

```
                        motion-core
            (dynamics · LoadModel · profiles · verify)
                     ╱                      ╲
    robot-web-simulator                   motionopt
    3D playback, closed-loop              cam profiles, positioning
    controllers, URDF robots              tables, PLC export
```

Two consumers, installed as separate projects, both depending on this package.
The split is the design: a web simulator should not have to carry a PLC exporter,
and a PLC tool should not have to carry three.js. What they genuinely share lives
here, and nothing else does.

`motion-core` never imports a consumer, and importing it never requires an
optional dependency — both are
[tests](tests/test_boundaries.py), not promises.

---

## What it does

| Area | Contents | Status |
| --- | --- | --- |
| Spatial math | SO(3)/SE(3) Lie groups, exp/log, screw axes (Modern Robotics conventions) | ✅ |
| Kinematics | PoE forward kinematics, body Jacobian, damped-least-squares IK, manipulability | ✅ |
| Dynamics | Recursive Newton-Euler, mass matrix, Coriolis, gravity, forward dynamics | ✅ |
| JIT path | Same equations accelerated, checked for parity against numpy | ✅ |
| Load models | `LoadModel` interface · `BallScrew` (1-DOF) · `ArticulatedRobot` (n-DOF) | 🚧 |
| Profiles | 7-segment double-S (jerk-limited time-optimal), trapezoid, polynomial, B-spline | 🚧 |
| Optimization | Epigraph formulation, SLSQP, infeasibility diagnosis | 🚧 |
| Verification | Constraint checks, cross-engine comparison against MuJoCo | 🚧 |

The first four are in `motion_core.robots`, extracted from the simulator that
used to own them, with the numpy-vs-JIT parity tests that came with them. 🚧 is
in progress — not written yet.

---

## The load model interface

Every optimizer and the verifier only ever call three methods:

```python
class LoadModel(Protocol):
    @property
    def ndof(self) -> int: ...
    def torque(self, q, dq, ddq) -> np.ndarray: ...   # what the motion demands
    def torque_limit(self, dq) -> np.ndarray: ...     # what the motor can give
    def rms_limit(self) -> np.ndarray: ...            # thermal ceiling
```

Everything is an **array**, so a single axis (length 1) and a six-axis robot
(length 6) go through identical code. A cam master angle is converted to the
generalized coordinate `q` inside the implementation — never by the caller.
Conventions are fixed once in
[`docs/theory/conventions.md`](docs/theory/conventions.md); read it before
touching a formula, because most sign errors in motion code are frame errors.

Two implementations:

| | `BallScrew` | `ArticulatedRobot` |
| --- | --- | --- |
| L0 rigid inertia, Coulomb friction | ✅ | ✅ |
| L1 transmission efficiency, viscous friction | ✅ | ✅ |
| L2 position-dependent inertia `M(q)`, Coriolis | — | ✅ |
| L3 position-dependent gravity `g(q)` | ✅ | ✅ |
| L6 torque-speed curve, RMS thermal limit | ✅ | ✅ |

Two-inertia compliance (L4), process force (L5) and a learned residual
correction (L7) are **deliberately not implemented**. Writing wrappers before a
single case is validated is how an interface gets frozen in the wrong shape; the
layers are named in the design notes so it is clear what gets switched on later,
and that is all.

Coulomb friction is smoothed near zero speed: the optimizer differentiates the
torque function numerically, and `sgn` is not differentiable.

---

## Formulation

Minimizing `max |tau|` directly is non-smooth, which is why a gradient-based
solver stalls on it. The peak and the jerk bound become variables, so every
constraint the solver sees is smooth:

```
minimize    w1 * t  +  w2 * tau_rms(c)  +  w3 * u
over        c (control points), t, u

subject to  |tau(c; i)|  <= t                 at every grid point i
            |jerk(c; i)| <= u
            |tau(c; i)|  <= torque_limit(speed_i)
            tau_rms(c)   <= rms_limit
            continuity in position, velocity, acceleration at segment bounds
```

Terms are normalized before weighting — torque in N·m and jerk differ by orders
of magnitude, and unnormalized weights are meaningless. SciPy SLSQP first,
CasADi + IPOPT when the problem outgrows it. If SLSQP struggles, suspect the
formulation before the solver.

When no feasible profile exists, the answer is a **report** — which constraint
binds, where, and what cycle time or motor torque would be needed — not a
marginal profile that trips an overload alarm on the floor.

### On prior art

Torque-limited time-optimal traversal of a *given path* is a solved problem:
Bobrow and Pfeiffer–Johanni, and more recently TOPP-RA. This package claims no
new algorithm there and uses those methods as the baseline. What is being
attempted is narrower and more practical: one load-model interface and one
formulation that covers a 1-DOF industrial mechanism and a multi-DOF robot, with
the verification to show it holds in both.

---

## Verification

| | Check | Passes when |
| --- | --- | --- |
| V0 | Unit and frame round-trip against `conventions.md` | error at numerical noise |
| V1 | Standard profiles against their closed-form solutions | matches |
| V2 | Generated profile back through the load model | zero violations |
| V3 | Closed-loop tracking with a real controller (in the simulator) | no saturation |
| V3.5 | Inverse dynamics cross-checked against MuJoCo, same trajectory | torque difference ≤ 1% |
| V5 | Measured peak torque vs prediction, on a bench | error ≤ 15% |

**V3.5 exists because V2 and V3 grade the model with itself.** The numpy-vs-JIT
parity test compares two implementations of the same equations — if the equations
are wrong, both agree and both are wrong. MuJoCo is an independent
implementation, so the comparison catches implementation error. It does not catch
*modeling* error: two simulators agreeing is not evidence about a real machine,
which is what V5 is for.

MuJoCo is a test-only extra (`pip install motion-core[validation]`) and is never
used inside the optimization loop — it provides no gradients, and pulling in a
JAX stack to get them would cost more than it returns.

---

## Install

```bash
pip install motion-core                 # numpy + scipy only
pip install "motion-core[fast]"         # JIT dynamics
pip install "motion-core[robot]"        # URDF loading
pip install "motion-core[validation]"   # MuJoCo cross-check (tests)
pip install -e ".[dev]"                 # everything, for development
```

---

## Layout

```
src/motion_core/
├── robots/        # SO(3)/SE(3), kinematics, RNE dynamics, JIT path, URDF models
├── loads/         # LoadModel protocol, BallScrew, ArticulatedRobot
├── profiles/      # double-S, trapezoid, polynomial, B-spline primitives
├── optimize/      # epigraph formulation, solver adapters, infeasibility report
└── verify/        # constraint checks, cross-engine comparison

benchmarks/cases/  # synthetic machines (YAML)
benchmarks/run.py  # reproduces the results table
docs/theory/       # conventions, derivations, formulation notes
tests/             # includes test_boundaries.py -- the dependency rule, as a test
```

---

## Results

Benchmarks run on **synthetic machines** in
[`benchmarks/cases/`](benchmarks/cases) — written for this project, no customer
or proprietary data anywhere in this repository.

| Case | What | Baseline | Duration | Peak torque |
| --- | --- | --- | --- | --- |
| `robot_ur5` | UR5 joint-space move | 5th-order polynomial, hand-picked T | TBD | TBD |

The bar, set before the work started so a weak result cannot be reported as a win:

| Metric | Condition | Target |
| --- | --- | --- |
| Peak torque reduction | 3 synthetic machines, same cycle time, vs modified sine | **≥ 10%** |
| Cycle time reduction | same torque limit, time-optimal vs quintic | **≥ 15%** |
| Torque prediction error | measured vs model, on a bench | ≤ 15% |

If peak torque reduction comes in under 10%, the approach does not pay for
itself and this says so.

Reproduce with `python benchmarks/run.py`.

---

## License

MIT
