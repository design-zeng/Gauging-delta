# tests/legacy/

This directory contains `perception.py`, the original reference implementation
from the IEEE TPAMI 2025 paper on the Gauging-delta algorithm.

---

## Purpose

`perception.py` is the parity baseline. Every correctness test in
`tests/test_parity.py` runs both `Perception` (legacy) and `GaugingDelta`
(refactored) on the same input and asserts that the outputs are identical
(ARI = 1.000 on all six benchmark datasets).

## Status

- **Read-only.** Do not modify this file under any circumstances.
- It is not part of the `gauging_delta` package and is not installed as a
  dependency. It is only imported by the test suite via a `sys.path` insertion
  in `tests/conftest.py` and `tests/test_parity.py`.
- The `__init__.py` in this directory is empty and exists only to satisfy
  Python's package discovery.

## Why it is kept here

Keeping the reference implementation adjacent to the tests makes the parity
contract self-contained: anyone who clones the repository can run
`uv run pytest tests/test_parity.py -v` and verify bit-exact agreement
without any external dependency.
