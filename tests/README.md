# tests/

Test suite for Gauging-delta. 77 tests total: 53 full-mode, 22 lite-mode, and 2
shared edge-case tests.

---

## Running Tests

```bash
# Full suite
make test

# Full suite, verbose output
uv run pytest tests/ -v

# Skip slow seed-parity tests (10 random-blob seeds)
uv run pytest -m "not slow" tests/ -v

# Single file
uv run pytest tests/test_parity.py -v
uv run pytest tests/test_unit.py -v
uv run pytest tests/test_lite.py -v

# Single test
uv run pytest tests/test_parity.py::test_dataset_parity -v

# Single parametrized case
uv run pytest "tests/test_parity.py::test_dataset_parity[compound]" -v
```

---

## Test Files

### `conftest.py`

Session-scoped shared fixtures. Loads all six benchmark datasets once and runs
`perception.py` once per dataset per session so individual tests do not pay the
re-run cost.

Key fixtures:

| Fixture | Scope | Returns |
|---|---|---|
| `datasets` | session | `{name: (X, y_true)}` for all 6 datasets |
| `legacy_labels` | session | `{name: np.ndarray}` — raw cluster IDs from `Perception.fit` |

The `legacy` subdirectory is inserted into `sys.path` so `from perception import Perception`
resolves correctly.

---

### `test_parity.py`

Verifies that `GaugingDelta` reproduces `perception.py` exactly. Three categories:

**1. Dataset parity** (12 tests — 6 datasets × ARI + cluster-count checks)

`GaugingDelta` must achieve ARI=1.000 vs legacy on every benchmark dataset.

```bash
uv run pytest tests/test_parity.py -k "dataset_parity or cluster_count" -v
```

**2. Seed parity** (10 tests, marked `slow`)

ARI=1.000 on 10 random 3-blob datasets with fixed seeds. Skipped by default in
`make test` via `-m "not slow"`.

```bash
uv run pytest tests/test_parity.py -m slow -v
```

**3. Intermediate merge-sequence parity** (2 tests — flame + 3_blobs)

The full ordered merge sequence `(lead, child, distance)` must match legacy
step-by-step on `flame` and `3_blobs`. This catches subtle ordering or
tie-breaking regressions that would not show up in end-state ARI alone.

---

### `test_unit.py`

Unit tests for individual components. Does not depend on `legacy_labels`.

| Class | Tests |
|---|---|
| `TestComputeAngle` | Right angle, zero vector, straight angle, rounding parity, batch vs scalar |
| `TestProximity` | Lead selection (larger cluster wins), equal-size tie-break, singleton fallback |
| `TestConfig` | Default values, field override, frozen enforcement, all constant groups |
| `TestSwappableComponents` | Custom proximity Protocol, config passthrough |
| `TestSklearnAPI` | `fit` returns self, `labels_` dtype, `n_clusters_` attribute |

---

### `test_lite.py`

Tests for `mode="lite"` (centroid linkage, no continuity gate). 22 tests across
five classes:

| Class | Tests |
|---|---|
| `TestLiteAPI` | fit/fit_predict return shapes, preserve_labels, n_clusters_target |
| `TestLiteValidation` | Invalid mode raises ValueError, mode attribute stored, "full" is default |
| `TestLiteMemory` | No `_near_ref`, no `_pd_*` arrays, no `_row_*` tracking, has `_lite_*` tree |
| `TestLiteQuality` | Well-separated blobs ARI > 0.4, all 5 benchmark datasets run without error |
| `TestLiteEdgeCases` | 2 points, 3 collinear, duplicate points, single point |
| `TestLiteDeterminism` | Repeated calls on the same data produce identical labels |

---

## Markers

| Marker | Used on | Behavior |
|---|---|---|
| `slow` | `test_seed_parity` (10 seeds) | Skipped by `make test` and `uv run pytest -m "not slow"` |

---

## `legacy/` Subdirectory

Contains `perception.py` — the original IEEE TPAMI 2025 reference implementation.
See `legacy/README.md` for details.
