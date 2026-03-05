# Gauging-delta Optimization Log

Branch: `refactor/time-space-complexity-optimization`

---

## Config-Extraction Changes (WIP — pre-optimization prerequisite)

### What changed

7 magic numbers previously hardcoded inside `continuity.py` and `threshold.py` were
extracted into `GaugingDeltaConfig` (frozen dataclass in `src/gauging_delta/config.py`).

**3 compactness fields** (added to `GaugingDeltaConfig`):

| Field | Default | Source |
|-------|---------|--------|
| `compact_fallback` | `0.5` | `continuity.py` — returned when cluster has ≤ `compact_min_size` points |
| `compact_min_size` | `2` | `continuity.py` — minimum points to compute sigma/mean compactness |
| `compact_history_window` | `5` | `continuity.py` — sliding window for recent merge distances |

**3 transition-smoothness fields** (added to `GaugingDeltaConfig`):

| Field | Default | Source |
|-------|---------|--------|
| `transition_external_zero_fallback` | `2.0` | `continuity.py` — returned when either external count is 0 |
| `transition_lopsided_override` | `1.5` | `continuity.py` — returned when distribution is asymmetric |
| `transition_lopsided_ratio` | `0.1` | `continuity.py` — ratio threshold for detecting lopsided distributions |

**1 vision-scale field** (added to `GaugingDeltaConfig`):

| Field | Default | Source |
|-------|---------|--------|
| `vision_scale_dist_ratio_fallback` | `1.0` | `threshold.py` — dist_ratio when avg contextual distance is 0 |

### Why

- Enables config-driven tuning without code changes
- Sets up hyperparameter sweeps for performance optimization
- All algorithm constants now live in one auditable location (`config.py`)

### Parity status

ARI = 1.000 on all 6 benchmark datasets (flame, 3_blobs, pathbased, 3-spiral, jain, compound).

---

## Cycle Template

```
## Cycle N — <short title>

**Date:** YYYY-MM-DD
**Files changed:** list of files

### Change description

<What was changed and why>

### Test results

- `test_parity.py`: PASS / FAIL (ARI = ?)
- `test_unit.py`: PASS / FAIL

### Stress test (50 configs, seed=42, workers=1)

| Generator | Mean new time (s) | ARI mean |
|-----------|------------------|----------|
| collinear_clusters | | |
| duplicate_points | | |
| extreme_scale | | |
| high_dimensional | | |
| near_zero_separation | | |
| singleton_merges | | |
| spirals | | |
| unequal_mass | | |
| well_separated_blobs | | |
| **OVERALL** | | |

Peak memory: ? MB

### Holistic benchmark

| Dataset | ARI | Time (s) |
|---------|-----|----------|
| flame | | |
| 3_blobs | | |
| pathbased | | |
| 3-spiral | | |
| jain | | |
| compound | | |

### Delta vs previous cycle

- Time: ?% faster / slower
- Memory: ?% less / more
- ARI: unchanged / changed (describe)
```

---

## Cycle 0 — Baseline (config-extraction only, no algorithmic changes)

**Date:** 2026-03-04
**Files changed:** `config.py`, `continuity.py`, `threshold.py`

### Change description

Config-extraction prerequisite only. No algorithmic changes. Establishes the
timing/parity baseline against which future optimization cycles are measured.

### Test results

- `test_parity.py`: PASS (ARI = 1.000 on all 6 datasets)
- `test_unit.py`: PASS

### Stress test (50 configs, seed=42, workers=1)

| Generator | Mean new time (s) | ARI mean |
|-----------|-------------------|----------|
| collinear_clusters | 0.1780 | 1.000000 |
| duplicate_points | 0.8825 | 1.000000 |
| extreme_scale | 0.3285 | 1.000000 |
| high_dimensional | 0.1794 | 1.000000 |
| near_zero_separation | 0.3318 | 1.000000 |
| singleton_merges | 0.2499 | 1.000000 |
| spirals | 0.2954 | 1.000000 |
| unequal_mass | 0.1893 | 1.000000 |
| well_separated_blobs | 0.1553 | 1.000000 |
| **OVERALL** | **0.2689** | **1.000000** |

Total wall time (50 configs): 13.45s · Max single config: 2.08s

### Holistic benchmark

| Dataset | N | ARI | New time (s) | New peak mem (MB) | Speedup vs legacy |
|---------|---|-----|-------------|-------------------|-------------------|
| flame | 240 | 1.000 | 4.64 | 61.3 | 1.52x |
| 3_blobs | 300 | 1.000 | 6.26 | 96.2 | 1.52x |
| pathbased | 300 | 1.000 | 6.53 | 96.3 | 1.30x |
| 3-spiral | 312 | 1.000 | 6.16 | 104.0 | 0.87x |
| jain | 373 | 1.000 | 9.56 | 147.7 | 1.19x |
| compound | 399 | 1.000 | 11.53 | 168.2 | 1.38x |

Synthetic N=500 new time: 16.93s (legacy: 21.93s, speedup 1.30x, peak 265.9 MB)

### Delta vs previous cycle

Baseline — no prior cycle.
