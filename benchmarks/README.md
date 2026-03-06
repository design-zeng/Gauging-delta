# benchmarks/

Benchmarking, profiling, and parity verification tools for Gauging-delta.

---

## Development Cycle

### `cycle_bench.py`

The primary tool for the per-cycle optimization workflow. Three subcommands:

| Subcommand | Purpose |
|---|---|
| `profile` | cProfile + tracemalloc to identify hotspots and peak memory |
| `bench` | Compare refactored code against the saved cycle baseline |
| `verify` | Targeted gate-parity check on only the gates you changed |

**Gate-to-module mapping:**

| Gate | Module | What it tests |
|---|---|---|
| G2 | `proximity.py` | rho, lead_id, child_id |
| G3 | `threshold.py` | T_i, T_j, xi_s, vision_scale, adp_prox |
| G4 | derived | rejection decision |
| G5 | `continuity.py` | smoothness score |
| G6 | derived | final accept/reject |

```bash
# Profile hotspots at default size (N=5000)
uv run python benchmarks/cycle_bench.py profile

# Profile at custom sizes
uv run python benchmarks/cycle_bench.py profile --sizes 100 500 1000

# Benchmark against saved baseline
uv run python benchmarks/cycle_bench.py bench

# Save the current run as the new baseline
uv run python benchmarks/cycle_bench.py bench --save

# Verify only gates 2 and 3
uv run python benchmarks/cycle_bench.py verify --gates 2,3

# Verify gate 2 on a specific dataset
uv run python benchmarks/cycle_bench.py verify --gates 2 --dataset flame
```

---

## Quality Verification

### `gate_parity.py`

Runs the legacy (`perception.py`) and refactored implementations side-by-side,
comparing intermediate gate outputs at every merge attempt. When a gate diverges,
**borrow mode** injects the legacy output into the refactored pipeline to isolate
root causes from cascading failures.

```bash
# Verify all 6 benchmark datasets
uv run python benchmarks/gate_parity.py

# Single dataset
uv run python benchmarks/gate_parity.py --dataset flame

# Borrow legacy output at gate 2 to isolate downstream divergence
uv run python benchmarks/gate_parity.py --borrow 2

# Reproduce a specific stress-test configuration
uv run python benchmarks/gate_parity.py --stress-config 109

# Stop at the first divergence (faster feedback)
uv run python benchmarks/gate_parity.py --quick
```

### `benchmark_parity.py`

Multi-level parity verification with per-stage timing and memory profiling.
Uses binary search to pinpoint the exact merge decision where a divergence first
occurs.

| Level | Granularity |
|---|---|
| `holistic` | End-to-end ARI + total time + peak memory |
| `stage` | Per-stage timing/memory + cluster-state comparison |
| `component` | Per-merge-decision component parity (rho, T_i, T_j, xi_s, smoothness) |
| `decision` | Full dump of a single identified merge decision |

```bash
uv run python benchmarks/benchmark_parity.py

uv run python benchmarks/benchmark_parity.py --level stage --sizes 100 500

uv run python benchmarks/benchmark_parity.py --level component --sizes 500

# Dump full detail for merge decision 58
uv run python benchmarks/benchmark_parity.py --level decision --sizes 500 --decision 58
```

### `stress_test.py`

Numerical stress test targeting specific instability hazards: division by zero,
nan/inf propagation, and floating-point precision loss. Generates synthetic
configurations and verifies that legacy and refactored implementations agree
on every one.

```bash
uv run python benchmarks/stress_test.py

# Custom run: 200 configs, fixed seed
uv run python benchmarks/stress_test.py --n-configs 200 --seed 42

# Deep mode: more extreme numerical configurations
uv run python benchmarks/stress_test.py --deep --n-configs 50

# Reproduce a specific failing config by index
uv run python benchmarks/stress_test.py --reproduce 17
```

Or via Make:

```bash
make bench-stress
```

---

## Performance

### `benchmark_all.py`

Publication-quality runtime and memory scaling comparison across three variants:

1. **Original** — `perception.py` (Perception class)
2. **Max-Parity** — `GaugingDelta` (ARI=1.000 vs original)
3. **Max-Fast** — same as Max-Parity (fast variant was removed)

Fits `O(N^α)` curves to measured data and saves plots + raw JSON results.

```bash
uv run python benchmarks/benchmark_all.py

uv run python benchmarks/benchmark_all.py --sizes 100 500 1000 --repeats 2

# Skip the slow original implementation
uv run python benchmarks/benchmark_all.py --skip-original --workers 8
```

### `test_performance.py`

Pytest-based performance tests. Measures execution time and peak memory for
dataset sizes up to 10,000 points and compares both implementations.

```bash
make bench-test
# or
uv run pytest benchmarks/test_performance.py -v
```

### `run_full_benchmark.py`

Sequential overnight runner designed for 60 GB RAM systems. Serialises progress
to disk after every task so interrupted runs can be resumed. Runs in three phases:

| Phase | Variants | Size range |
|---|---|---|
| 1 | All three | moderate |
| 2 | Max-Parity + Max-Fast only | larger |
| 3 | Max-Fast only | largest |

```bash
uv run python benchmarks/run_full_benchmark.py

# Single phase
uv run python benchmarks/run_full_benchmark.py --phase 1

# Resume an interrupted run
uv run python benchmarks/run_full_benchmark.py --resume

# Dry-run: print tasks without executing
uv run python benchmarks/run_full_benchmark.py --dry-run
```

---

## Shared Helper

### `_plotting.py`

Activates `scienceplots` style (`science`, `no-latex`, `grid`) when available;
falls back to matplotlib default. Imported by `benchmark_parity.py`,
`stress_test.py`, and `benchmark_all.py` to ensure consistent plot appearance.

---

## `results/` Directory

Saved outputs written by the benchmark scripts:

| File | Written by | Contents |
|---|---|---|
| `cycle_baseline.json` | `cycle_bench.py bench --save` | Full-mode timing baseline for next cycle comparison |
| `cycle_baseline_lite.json` | `cycle_bench.py bench --save` | Lite-mode timing baseline |
| `cycle_profile.json` | `cycle_bench.py profile` | cProfile + tracemalloc snapshot (full mode) |
| `cycle_profile_lite.json` | `cycle_bench.py profile` | cProfile + tracemalloc snapshot (lite mode) |
| `benchmark_parity.json` | `benchmark_parity.py` | Per-level parity results |
| `divergences.json` | `gate_parity.py` | Gate-level divergence records |
| `stress_test.json` | `stress_test.py` | Stress-test summary + pass/fail counts |
| `stress_divergences.json` | `stress_test.py` | Failing stress configurations |
| `benchmark_progress.json` | `run_full_benchmark.py` | Resume checkpoint for overnight runs |
| `scaling_analysis.png` | `benchmark_all.py` / `cycle_bench.py` | Runtime/memory scaling plots |
| `memray.bin` / `memray_flamegraph.html` | memray profiling | Memory flamegraph |
