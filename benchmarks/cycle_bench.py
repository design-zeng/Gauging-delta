"""
Cycle benchmarking, profiling, and targeted verification tool.

Three subcommands for the per-cycle optimization workflow:

  profile  — cProfile + tracemalloc to identify hotspots and memory usage
  bench    — Benchmark refactored code against previous cycle's saved baseline
  verify   — Targeted gate-parity check (only test changed gates)

Usage:
    uv run python benchmarks/cycle_bench.py profile
    uv run python benchmarks/cycle_bench.py profile --sizes 100 500 1000
    uv run python benchmarks/cycle_bench.py bench
    uv run python benchmarks/cycle_bench.py bench --save
    uv run python benchmarks/cycle_bench.py verify --gates 2,3
    uv run python benchmarks/cycle_bench.py verify --gates 2 --dataset flame

Gate-to-module mapping:
    G2: proximity   → proximity.py
    G3: threshold   → threshold.py
    G4: rejection   → derived from G2+G3
    G5: continuity  → continuity.py
    G6: accept      → derived from G4+G5
    Merge exec      → algorithm.py (_do_merge) — affects next iteration's G2 inputs
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import pstats
import sys
import time
import tracemalloc
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
BASELINE_PATH = RESULTS_DIR / "cycle_baseline.json"
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
BENCHMARK_DATASETS = {
    "flame": DATA_DIR / "flame.txt",
    "3_blobs": DATA_DIR / "3_blobs.txt",
    "pathbased": DATA_DIR / "pathbased.txt",
    "3-spiral": DATA_DIR / "3-spiral.txt",
    "jain": DATA_DIR / "jain.txt",
    "compound": DATA_DIR / "compound.txt",
}


def _load_dataset(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",")[:, :2]


def _make_blobs(n: int, dim: int = 2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20, 20, size=(5, dim))
    parts = []
    per = n // 5
    for i, c in enumerate(centers):
        count = per + (1 if i < n - per * 5 else 0)
        parts.append(rng.normal(loc=c, scale=1.0, size=(count, dim)))
    return np.vstack(parts)


# Default synthetic configurations: (label, N, D)
SYNTH_SCALING = [
    ("s100", 100, 2),
    ("s250", 250, 2),
    ("s500", 500, 2),
    ("s750", 750, 2),
    ("s1000", 1000, 2),
]
SYNTH_DIMENSIONAL = [
    ("d10", 500, 10),
    ("d50", 500, 50),
]
DEFAULT_SEEDS = [42, 123, 777]


def _print_time_estimate(label: str, seconds: float) -> None:
    if seconds < 60:
        print(f"{label}: {seconds:.1f} seconds")
    elif seconds < 3600:
        print(f"{label}: {seconds / 60:.1f} minutes")
    elif seconds < 86400:
        print(f"{label}: {seconds / 3600:.1f} hours")
    else:
        print(f"{label}: {seconds / 86400:.1f} days")


# ---------------------------------------------------------------------------
# Complexity models for extrapolation
# ---------------------------------------------------------------------------

def _f_n2(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n**2 + b


def _f_n2logn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n**2 * np.log(n) + b


def _f_n3(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n**3 + b


def _f_nlogn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * n * np.log(n) + b


MODELS = {
    "O(N log N)": _f_nlogn,
    "O(N^2)": _f_n2,
    "O(N^2 log N)": _f_n2logn,
    "O(N^3)": _f_n3,
}


def _fit_best_model(sizes: np.ndarray, times: np.ndarray) -> tuple[str, float, float]:
    """Fit complexity models, return (best_name, R^2, extrapolated_70k_seconds)."""
    best_name, best_r2, best_popt = "unknown", -1.0, None
    for name, func in MODELS.items():
        try:
            popt, _ = curve_fit(func, sizes, times, p0=[1e-8, 0], maxfev=20000)
            predicted = func(sizes, *popt)
            ss_res = np.sum((times - predicted) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 > best_r2:
                best_name, best_r2, best_popt = name, r2, popt
        except RuntimeError:
            pass

    extrapolated = 0.0
    if best_popt is not None:
        extrapolated = float(MODELS[best_name](np.array([70000.0]), *best_popt)[0])
    return best_name, best_r2, extrapolated


# ---------------------------------------------------------------------------
# profile subcommand
# ---------------------------------------------------------------------------

def _run_timed(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run GaugingDelta, return (labels, time_s, peak_bytes)."""
    from gauging_delta import GaugingDelta

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model = GaugingDelta(preserve_labels=True)
    model.fit(X.copy())
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return np.asarray(model.labels_, dtype=int), elapsed, float(peak)


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile runtime and memory, identify hotspots, extrapolate to N=70K."""
    from gauging_delta import GaugingDelta

    sizes = args.sizes
    dataset = args.dataset or "compound"

    print("=" * 60)
    print("Cycle Profiler")
    print("=" * 60)

    # --- 1. cProfile on the target dataset ---
    path = BENCHMARK_DATASETS.get(dataset)
    if path and path.exists():
        X = _load_dataset(path)
    else:
        print(f"Dataset '{dataset}' not found, using synthetic N=500")
        X = _make_blobs(500)

    print(f"\ncProfile on {dataset} (N={len(X)}):")
    print("-" * 60)

    pr = cProfile.Profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pr.enable()
        model = GaugingDelta(preserve_labels=True)
        model.fit(X.copy())
        pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    # --- 2. Per-function breakdown (top bottlenecks) ---
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2)
    ps2.sort_stats("tottime")
    ps2.print_stats(10)
    print("Top functions by total time:")
    print(s2.getvalue())

    # --- 3. Memory snapshot ---
    print("Memory profile:")
    print("-" * 60)
    gc.collect()
    tracemalloc.start()
    model2 = GaugingDelta(preserve_labels=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model2.fit(X.copy())
    snapshot = tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  Peak memory: {peak / 1e6:.1f} MB")
    top_stats = snapshot.statistics("lineno")
    print(f"  Top 10 allocations:")
    for stat in top_stats[:10]:
        print(f"    {stat}")

    # --- 4. Quick scaling check (single seed, for complexity estimate) ---
    print(f"\nScaling check (single seed, sizes={sizes}):")
    print("-" * 60)
    timing_data: list[tuple[int, float, float]] = []

    for n in sizes:
        X_synth = _make_blobs(n, seed=42)
        _, elapsed, mem = _run_timed(X_synth)
        timing_data.append((n, elapsed, mem))
        print(f"  N={n:>6d}  time={elapsed:>8.3f}s  mem={mem / 1e6:>8.1f} MB")

    if len(timing_data) >= 3:
        ns = np.array([t[0] for t in timing_data], dtype=float)
        ts = np.array([t[1] for t in timing_data])
        best_model, r2, ext_70k = _fit_best_model(ns, ts)

        print(f"\n  Best-fit complexity: {best_model} (R^2={r2:.4f})")
        if ext_70k > 0:
            _print_time_estimate("  N=70K extrapolation", ext_70k)

    # --- 5. Save profile results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profile_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "n_points": len(X),
        "peak_memory_bytes": peak,
        "scaling": [
            {"n": t[0], "time_s": t[1], "peak_bytes": t[2]} for t in timing_data
        ],
    }
    if len(timing_data) >= 3:
        profile_data["best_fit_model"] = best_model
        profile_data["best_fit_r2"] = r2
        profile_data["extrapolated_70k_s"] = ext_70k

    out_path = RESULTS_DIR / "cycle_profile.json"
    with open(out_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    print(f"\n  Saved profile to {out_path}")


# ---------------------------------------------------------------------------
# bench subcommand
# ---------------------------------------------------------------------------

def _run_multi_seed(
    n: int, dim: int, seeds: list[int],
) -> tuple[float, float, float, float]:
    """Run multiple seeds, return (median_time, std_time, median_mem, std_mem)."""
    times, mems = [], []
    for seed in seeds:
        X = _make_blobs(n, dim=dim, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, elapsed, mem = _run_timed(X)
        times.append(elapsed)
        mems.append(mem)
    return (
        float(np.median(times)),
        float(np.std(times)),
        float(np.median(mems)),
        float(np.std(mems)),
    )


def cmd_bench(args: argparse.Namespace) -> None:
    """Benchmark against previous cycle's baseline."""
    seeds = args.seeds

    print("=" * 60)
    print(f"Cycle Benchmark (seeds={seeds})")
    print("=" * 60)

    # Load baseline if exists
    baseline = None
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
        print(f"  Baseline loaded: cycle {baseline.get('cycle', '?')} "
              f"({baseline.get('timestamp', '?')[:19]})")
    else:
        print("  No baseline found — first run (timings only)")

    results: dict[str, dict] = {"datasets": {}, "synthetic": {}}

    # --- Fixed benchmark datasets (single run, deterministic) ---
    print(f"\n{'Dataset':<15s} {'N':>5s} {'D':>3s} {'Time (s)':>10s} {'Mem (MB)':>10s}", end="")
    if baseline:
        print(f" {'Prev (s)':>10s} {'Speedup':>10s}", end="")
    print()
    print("-" * (48 + (22 if baseline else 0)))

    for name, path in BENCHMARK_DATASETS.items():
        if not path.exists():
            continue
        X = _load_dataset(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, elapsed, mem = _run_timed(X)

        results["datasets"][name] = {
            "n": len(X), "time_s": elapsed, "peak_bytes": mem,
        }

        line = f"{name:<15s} {len(X):>5d} {X.shape[1]:>3d} {elapsed:>10.3f} {mem / 1e6:>10.1f}"
        if baseline and name in baseline.get("datasets", {}):
            prev = baseline["datasets"][name]["time_s"]
            speedup = prev / elapsed if elapsed > 0 else float("inf")
            line += f" {prev:>10.3f} {speedup:>9.2f}x"
        print(line)

    # --- Synthetic configs: scaling + dimensional, multi-seed ---
    synth_configs = SYNTH_SCALING + SYNTH_DIMENSIONAL
    print(f"\n{'Config':<15s} {'N':>5s} {'D':>3s} {'Time (s)':>10s} {'± Std':>8s} "
          f"{'Mem (MB)':>10s}", end="")
    if baseline:
        print(f" {'Prev (s)':>10s} {'Speedup':>10s}", end="")
    print()
    print("-" * (55 + (22 if baseline else 0)))

    for label, n, dim in synth_configs:
        med_t, std_t, med_m, std_m = _run_multi_seed(n, dim, seeds)

        results["synthetic"][label] = {
            "n": n, "dim": dim, "seeds": seeds,
            "time_s": med_t, "time_std": std_t,
            "peak_bytes": med_m, "mem_std": std_m,
        }

        line = (f"{label:<15s} {n:>5d} {dim:>3d} {med_t:>10.3f} {std_t:>8.3f} "
                f"{med_m / 1e6:>10.1f}")
        if baseline and label in baseline.get("synthetic", {}):
            prev = baseline["synthetic"][label]["time_s"]
            speedup = prev / med_t if med_t > 0 else float("inf")
            line += f" {prev:>10.3f} {speedup:>9.2f}x"
        print(line)

    # --- Extrapolation (scaling configs only, D=2) ---
    scale_timings = [
        (r["n"], r["time_s"])
        for label, r in results["synthetic"].items()
        if label.startswith("s")
    ]
    if len(scale_timings) >= 3:
        ns = np.array([t[0] for t in scale_timings], dtype=float)
        ts = np.array([t[1] for t in scale_timings])
        best_model, r2, ext_70k = _fit_best_model(ns, ts)
        print(f"\n  Complexity fit (D=2): {best_model} (R^2={r2:.4f})")
        _print_time_estimate("  N=70K estimate (D=2)", ext_70k)

    # --- Memory extrapolation ---
    scale_mems = [
        (r["n"], r["peak_bytes"])
        for label, r in results["synthetic"].items()
        if label.startswith("s")
    ]
    if len(scale_mems) >= 3:
        ns = np.array([t[0] for t in scale_mems], dtype=float)
        ms = np.array([t[1] for t in scale_mems])
        coeffs = np.polyfit(ns**2, ms, 1)
        est_70k = coeffs[0] * 70000**2 + coeffs[1]
        print(f"  N=70K memory estimate (D=2): {est_70k / 1e9:.0f} GB")

    # --- Save as new baseline if --save ---
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cycle_num = (baseline.get("cycle", 0) + 1) if baseline else 0
        save_data = {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "seeds": seeds,
            "datasets": results["datasets"],
            "synthetic": results["synthetic"],
        }
        with open(BASELINE_PATH, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Saved as cycle {cycle_num} baseline to {BASELINE_PATH}")
    elif not args.save and baseline:
        print("\n  (use --save to update baseline)")


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    """Targeted gate parity verification."""
    from benchmarks.gate_parity import (
        BENCHMARK_DATASETS as GP_DATASETS,
        _load_dataset as gp_load,
        _print_report,
        run_gate_parity,
    )

    gate_filter = None
    if args.gates:
        gate_filter = [int(g.strip()) for g in args.gates.split(",")]

    gate_desc = ", ".join(f"G{g}" for g in gate_filter) if gate_filter else "all"
    print("=" * 60)
    print(f"Targeted Gate Parity (gates: {gate_desc})")
    print("=" * 60)

    sources: list[tuple[str, np.ndarray]] = []
    if args.dataset:
        path = GP_DATASETS.get(args.dataset)
        if path and path.exists():
            sources.append((args.dataset, gp_load(path)))
        else:
            print(f"Dataset '{args.dataset}' not found")
            sys.exit(1)
    else:
        for name, path in GP_DATASETS.items():
            if path.exists():
                sources.append((name, gp_load(path)))

    all_pass = True
    for name, X in sources:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report = run_gate_parity(
                X, name, quick=args.quick, borrow_gate=args.borrow, gate_filter=gate_filter
            )
        _print_report(report)
        if report.all_divergences:
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"PASS — gates [{gate_desc}] match on all datasets")
    else:
        print("DIVERGENCES FOUND")
    sys.exit(0 if all_pass else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cycle benchmarking and profiling tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # profile
    p_prof = sub.add_parser("profile", help="Profile hotspots and memory")
    p_prof.add_argument("--dataset", type=str, default="compound", help="Dataset to profile")
    p_prof.add_argument(
        "--sizes", nargs="+", type=int, default=[100, 250, 500, 750, 1000],
        help="Synthetic sizes for scaling measurement",
    )

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark against previous cycle")
    p_bench.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds for multi-seed benchmark (default: 42 123 777)",
    )
    p_bench.add_argument("--save", action="store_true", help="Save results as new baseline")

    # verify
    p_ver = sub.add_parser("verify", help="Targeted gate parity check")
    p_ver.add_argument(
        "--gates", type=str, default=None,
        help="Comma-separated gate numbers (e.g. '2,3'). Default: all gates.",
    )
    p_ver.add_argument("--dataset", type=str, help="Single dataset name")
    p_ver.add_argument("--borrow", type=int, default=None, help="Gate to borrow from legacy")
    p_ver.add_argument("--quick", action="store_true", help="Stop at first divergence")

    args = parser.parse_args()

    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
