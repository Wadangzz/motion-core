"""The reason this package can be a dependency, as a test.

``motion-core`` is imported by a web simulator, by an industrial motion-data
generator, and by a demonstration-dataset auditor. It must install with
numpy + scipy alone, and it must not know anything about any of them. These
tests fail if that erodes.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).parent.parent / "src" / "motion_core"

# Allowed in the package, but only behind a guarded import inside a function,
# so that importing motion_core never requires them.
OPTIONAL = {"numba", "mujoco", "yourdfpy", "robot_descriptions", "casadi",
            "matplotlib", "yaml"}

# Consumers. The engine never imports its users.
CONSUMERS = ("motionopt", "robot_web_simulator", "app", "feasibility_audit")


def _module_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _all_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_optional_deps_stay_optional():
    offenders = {}
    for path in PKG.rglob("*.py"):
        hits = _module_level_imports(path) & OPTIONAL
        if hits:
            offenders[str(path)] = sorted(hits)
    assert not offenders, f"move these to a guarded in-function import: {offenders}"


def test_engine_does_not_import_its_consumers():
    offenders = {}
    for path in PKG.rglob("*.py"):
        hits = {m for m in _all_imports(path) if m.split(".")[0] in CONSUMERS}
        if hits:
            offenders[str(path)] = sorted(hits)
    assert not offenders, f"motion-core must not depend on a consumer: {offenders}"
