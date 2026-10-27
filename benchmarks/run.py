"""Reproduce the README results table.

The case list is whatever sits in ``cases/*.yaml`` -- there is no second list to
keep in sync. Not implemented yet: this prints the table skeleton so the harness
and the README agree on shape before any numbers exist.
"""

import pathlib
import sys

CASES = sorted((pathlib.Path(__file__).parent / "cases").glob("*.yaml"))


def main() -> int:
    if not CASES:
        print("no cases found", file=sys.stderr)
        return 1
    print(f"{'case':<20} {'baseline':<24} {'duration':>10} {'peak torque':>12}")
    for path in CASES:
        print(f"{path.stem:<20} {'(from yaml)':<24} {'TBD':>10} {'TBD':>12}")
    print("\nnot implemented: optimizers are not wired up yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
