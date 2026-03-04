"""
Unified benchmark: Runtime, Space, and Parity for three Gauging-δ versions.

Compares:
  1. Original   – perception.py (Perception class)
  2. Max-Parity – GaugingDelta  (100% ARI vs original)
  3. Max-Fast   – GaugingDelta (same as max-parity; Fast variant was removed)

Generates publication-quality plots with SciencePlots and saves raw
results to JSON for reproducibility.

Usage:
    python benchmarks/benchmark_all.py
    python benchmarks/benchmark_all.py --sizes 100 500 1000 --repeats 2
    python benchmarks/benchmark_all.py --skip-original --workers 8
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import sys
import time
import tracemalloc
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import adjusted_rand_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
ASSETS_DIR = ROOT / "assets"

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_SIZES = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
N_BLOBS = 5
DIMS = 2
SEED = 42


# ---------------------------------------------------------------------------
# Complexity model functions for curve fitting
# ---------------------------------------------------------------------------

def f_nlogn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """O(N log N) model."""
    return a * n * np.log(n) + b


def f_n2(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """O(N²) model."""
    return a * n ** 2 + b


def f_n2logn(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """O(N² log N) model."""
    return a * n ** 2 * np.log(n) + b


def f_n3(n: np.ndarray, a: float, b: float) -> np.ndarray:
    """O(N³) model."""
    return a * n ** 3 + b


MODELS = {
    "O(N log N)": f_nlogn,
    "O(N²)": f_n2,
    "O(N² log N)": f_n2logn,
    "O(N³)": f_n3,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_blobs(n_points: int, seed: int = SEED) -> np.ndarray:
    """Generate deterministic well-separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    per_blob = n_points // N_BLOBS
    remainder = n_points - per_blob * N_BLOBS
    centers = rng.uniform(-20, 20, size=(N_BLOBS, DIMS))
    parts = []
    for i, c in enumerate(centers):
        count = per_blob + (1 if i < remainder else 0)
        parts.append(rng.normal(loc=c, scale=1.0, size=(count, DIMS)))
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Runner functions (each in its own process for clean memory measurement)
# ---------------------------------------------------------------------------

def _run_original(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run legacy Perception and return (labels, time_s, peak_bytes)."""
    sys.path.insert(0, str(ROOT))
    from perception import Perception

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    p = Perception()
    with redirect_stdout(io.StringIO()):
        labels, _ = p.fit(X.copy())

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return np.asarray(labels, dtype=int), elapsed, float(peak)


def _run_max_parity(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run GaugingDelta (max-parity) and return (labels, time_s, peak_bytes)."""
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


def _run_max_fast(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run GaugingDelta (same as max_parity — GaugingDeltaFast was removed)."""
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


# Wrapper for multiprocessing (must be top-level picklable)
def _worker(args: tuple) -> dict:
    """Run a single benchmark task. Returns result dict."""
    version, n, repeat_idx, seed = args
    X = make_blobs(n, seed=seed)

    try:
        if version == "original":
            labels, elapsed, peak = _run_original(X)
        elif version == "max_parity":
            labels, elapsed, peak = _run_max_parity(X)
        elif version == "max_fast":
            labels, elapsed, peak = _run_max_fast(X)
        else:
            raise ValueError(f"Unknown version: {version}")

        return {
            "version": version,
            "n": n,
            "repeat": repeat_idx,
            "time_s": elapsed,
            "peak_bytes": peak,
            "labels": labels.tolist(),
            "error": None,
        }
    except Exception as e:
        return {
            "version": version,
            "n": n,
            "repeat": repeat_idx,
            "time_s": None,
            "peak_bytes": None,
            "labels": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmarks(
    sizes: list[int],
    repeats: int | None,
    workers: int,
    skip_original: bool,
    cached_results: dict | None = None,
    save_path: Path | None = None,
) -> dict:
    """Run all benchmarks and return structured results dict.

    If *save_path* is given, results are serialised to JSON after **every**
    individual task so that overnight runs never lose progress.
    """
    versions = ["original", "max_parity", "max_fast"]
    if skip_original:
        versions = ["max_parity", "max_fast"]

    # Build task list
    tasks = []
    for n in sizes:
        n_repeats = repeats if repeats else (3 if n <= 1000 else 1)
        for version in versions:
            # Check cache
            if cached_results and _has_cached(cached_results, version, n, n_repeats):
                continue
            for r in range(n_repeats):
                tasks.append((version, n, r, SEED + r))

    total = len(tasks)
    print(f"\n{'=' * 70}")
    print(f"Gauging-δ Unified Benchmark")
    print(f"Sizes: {sizes}")
    print(f"Versions: {versions}")
    print(f"Total tasks: {total}")
    print(f"Workers: {workers}")
    print(f"{'=' * 70}\n")

    # Run tasks
    raw_results = []
    if cached_results:
        raw_results = cached_results.get("raw", [])

    if total > 0:
        completed = 0
        t_start = time.perf_counter()

        if workers > 1:
            with Pool(processes=workers) as pool:
                for result in pool.imap_unordered(_worker, tasks):
                    raw_results.append(result)
                    completed += 1
                    _print_progress(completed, total, t_start, result)
                    if save_path:
                        _incremental_save(raw_results, sizes, versions, save_path)
        else:
            for task in tasks:
                result = _worker(task)
                raw_results.append(result)
                completed += 1
                _print_progress(completed, total, t_start, result)
                if save_path:
                    _incremental_save(raw_results, sizes, versions, save_path)

    # Aggregate results
    return _aggregate(raw_results, sizes, versions)


def _has_cached(cached: dict, version: str, n: int, n_repeats: int) -> bool:
    """Check if results are already cached for this version/size."""
    raw = cached.get("raw", [])
    count = sum(1 for r in raw if r["version"] == version and r["n"] == n and r["error"] is None)
    return count >= n_repeats


def _incremental_save(raw: list[dict], sizes: list[int], versions: list[str], path: Path) -> None:
    """Persist current results to JSON after each task (crash-safe)."""
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "sizes": sizes,
        "raw": [{k: v for k, v in r.items() if k != "labels"} for r in raw],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(save_data, f, indent=2)
    tmp.replace(path)  # atomic on POSIX; best-effort on Windows


def _print_progress(completed: int, total: int, t_start: float, result: dict) -> None:
    elapsed = time.perf_counter() - t_start
    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
    status = "OK" if result["error"] is None else f"ERR: {result['error'][:40]}"
    time_str = f"{result['time_s']:.1f}s" if result["time_s"] else "N/A"
    mem_str = f"{result['peak_bytes'] / 1e6:.0f}MB" if result["peak_bytes"] else "N/A"
    print(
        f"  [{completed:>4d}/{total}] {result['version']:>12s} N={result['n']:>6d} "
        f"r={result['repeat']}  {time_str:>10s}  {mem_str:>8s}  "
        f"ETA={eta / 60:.1f}min  {status}"
    )


def _aggregate(raw: list[dict], sizes: list[int], versions: list[str]) -> dict:
    """Aggregate raw results into summary statistics."""
    summary = {}

    for version in versions:
        v_data = {
            "sizes": [],
            "times_mean": [],
            "times_std": [],
            "mem_mean": [],
            "mem_std": [],
        }
        for n in sizes:
            runs = [r for r in raw if r["version"] == version and r["n"] == n and r["error"] is None]
            if not runs:
                continue
            times = [r["time_s"] for r in runs]
            mems = [r["peak_bytes"] for r in runs]
            v_data["sizes"].append(n)
            v_data["times_mean"].append(float(np.mean(times)))
            v_data["times_std"].append(float(np.std(times)))
            v_data["mem_mean"].append(float(np.mean(mems)))
            v_data["mem_std"].append(float(np.std(mems)))

        summary[version] = v_data

    # Compute parity (ARI vs original)
    parity = {}
    for version in versions:
        if version == "original":
            continue
        ari_data = {"sizes": [], "ari_mean": [], "ari_std": []}
        for n in sizes:
            orig_runs = [r for r in raw if r["version"] == "original" and r["n"] == n and r["error"] is None and r["labels"]]
            ver_runs = [r for r in raw if r["version"] == version and r["n"] == n and r["error"] is None and r["labels"]]
            if not orig_runs or not ver_runs:
                continue
            aris = []
            for o, v in zip(orig_runs, ver_runs):
                ari = adjusted_rand_score(o["labels"], v["labels"])
                aris.append(ari)
            ari_data["sizes"].append(n)
            ari_data["ari_mean"].append(float(np.mean(aris)))
            ari_data["ari_std"].append(float(np.std(aris)))
        parity[version] = ari_data

    return {"raw": raw, "summary": summary, "parity": parity}


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fit_best_model(
    sizes: np.ndarray,
    values: np.ndarray,
) -> dict[str, tuple[np.ndarray, float]]:
    """Fit complexity models and return {name: (params, R²)}."""
    results = {}
    for name, func in MODELS.items():
        try:
            popt, _ = curve_fit(func, sizes, values, p0=[1e-8, 0], maxfev=20000)
            predicted = func(sizes, *popt)
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            results[name] = (popt, r2)
        except RuntimeError:
            results[name] = (np.array([0.0, 0.0]), -1.0)
    return results


# ---------------------------------------------------------------------------
# Plotting with SciencePlots
# ---------------------------------------------------------------------------

VERSION_LABELS = {
    "original": "Original (Perception)",
    "max_parity": "Max-Parity (GaugingDelta)",
    "max_fast": "Max-Fast (GaugingDelta)",
}
VERSION_COLORS = {
    "original": "#d62728",
    "max_parity": "#1f77b4",
    "max_fast": "#2ca02c",
}
VERSION_MARKERS = {
    "original": "s",
    "max_parity": "o",
    "max_fast": "^",
}

MODEL_COLORS = {
    "O(N log N)": "#17becf",
    "O(N²)": "#ff7f0e",
    "O(N² log N)": "#9467bd",
    "O(N³)": "#e377c2",
}


def _use_science_style():
    """Try to use SciencePlots, fall back to default if unavailable."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex", "grid"])
    except (ImportError, OSError):
        plt.style.use("default")
        print("  [WARN] scienceplots not installed; using default style.")


def plot_runtime(summary: dict, out_dir: Path) -> None:
    """Runtime comparison plot with curve fits."""
    _use_science_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for version, data in summary.items():
        if not data["sizes"]:
            continue
        sizes = np.array(data["sizes"], dtype=float)
        times = np.array(data["times_mean"])
        stds = np.array(data["times_std"])

        ax.errorbar(
            sizes, times, yerr=stds,
            fmt=VERSION_MARKERS.get(version, "o"),
            color=VERSION_COLORS.get(version, "gray"),
            capsize=3, markersize=4, linewidth=1,
            label=VERSION_LABELS.get(version, version),
        )

        # Fit best model and draw curve
        if len(sizes) >= 3:
            fits = fit_best_model(sizes, times)
            best_name = max(fits, key=lambda k: fits[k][1])
            best_popt, best_r2 = fits[best_name]
            if best_r2 > 0.5:
                dense_n = np.linspace(sizes[0], sizes[-1], 300)
                y_fit = MODELS[best_name](dense_n, *best_popt)
                ax.plot(
                    dense_n, y_fit, "--",
                    color=VERSION_COLORS.get(version, "gray"),
                    alpha=0.5, linewidth=0.8,
                )

    ax.set_xlabel("Number of Points (N)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Comparison")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_runtime.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark_runtime.png/pdf")


def plot_memory(summary: dict, out_dir: Path) -> None:
    """Memory comparison plot with curve fits."""
    _use_science_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for version, data in summary.items():
        if not data["sizes"]:
            continue
        sizes = np.array(data["sizes"], dtype=float)
        mems = np.array(data["mem_mean"]) / 1e6  # MB
        stds = np.array(data["mem_std"]) / 1e6

        ax.errorbar(
            sizes, mems, yerr=stds,
            fmt=VERSION_MARKERS.get(version, "o"),
            color=VERSION_COLORS.get(version, "gray"),
            capsize=3, markersize=4, linewidth=1,
            label=VERSION_LABELS.get(version, version),
        )

        # Fit best model
        if len(sizes) >= 3:
            fits = fit_best_model(sizes, mems)
            best_name = max(fits, key=lambda k: fits[k][1])
            best_popt, best_r2 = fits[best_name]
            if best_r2 > 0.5:
                dense_n = np.linspace(sizes[0], sizes[-1], 300)
                y_fit = MODELS[best_name](dense_n, *best_popt)
                ax.plot(
                    dense_n, y_fit, "--",
                    color=VERSION_COLORS.get(version, "gray"),
                    alpha=0.5, linewidth=0.8,
                )

    ax.set_xlabel("Number of Points (N)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory Comparison")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xscale("log")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_memory.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark_memory.png/pdf")


def plot_parity(parity: dict, out_dir: Path) -> None:
    """Parity (ARI) bar chart per version per dataset size."""
    _use_science_style()
    fig, ax = plt.subplots(figsize=(5, 3))

    bar_width = 0.35
    offset = 0
    for version, data in parity.items():
        if not data["sizes"]:
            continue
        sizes = np.array(data["sizes"])
        aris = np.array(data["ari_mean"])
        stds = np.array(data["ari_std"])

        x = np.arange(len(sizes)) + offset * bar_width
        ax.bar(
            x, aris, bar_width, yerr=stds,
            label=VERSION_LABELS.get(version, version),
            color=VERSION_COLORS.get(version, "gray"),
            alpha=0.85, capsize=2,
        )
        offset += 1

    if parity:
        # Use sizes from first available version
        first_data = next(iter(parity.values()))
        sizes = first_data["sizes"]
        ax.set_xticks(np.arange(len(sizes)) + bar_width * (len(parity) - 1) / 2)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=6, rotation=45)

    ax.set_xlabel("Number of Points (N)")
    ax.set_ylabel("ARI vs Original")
    ax.set_title("Parity (Adjusted Rand Index)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_parity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark_parity.png/pdf")


def plot_combined(summary: dict, parity: dict, out_dir: Path) -> None:
    """Combined 2×2 figure for README."""
    _use_science_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # --- Runtime (top-left) ---
    ax = axes[0, 0]
    for version, data in summary.items():
        if not data["sizes"]:
            continue
        sizes = np.array(data["sizes"], dtype=float)
        times = np.array(data["times_mean"])
        ax.plot(
            sizes, times,
            marker=VERSION_MARKERS.get(version, "o"),
            color=VERSION_COLORS.get(version, "gray"),
            markersize=3, linewidth=1,
            label=VERSION_LABELS.get(version, version),
        )
    ax.set_xlabel("N")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Memory (top-right) ---
    ax = axes[0, 1]
    for version, data in summary.items():
        if not data["sizes"]:
            continue
        sizes = np.array(data["sizes"], dtype=float)
        mems = np.array(data["mem_mean"]) / 1e6
        ax.plot(
            sizes, mems,
            marker=VERSION_MARKERS.get(version, "o"),
            color=VERSION_COLORS.get(version, "gray"),
            markersize=3, linewidth=1,
            label=VERSION_LABELS.get(version, version),
        )
    ax.set_xlabel("N")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Memory")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(fontsize=5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Runtime curve fit (bottom-left) ---
    ax = axes[1, 0]
    for version, data in summary.items():
        if not data["sizes"] or len(data["sizes"]) < 3:
            continue
        sizes = np.array(data["sizes"], dtype=float)
        times = np.array(data["times_mean"])

        fits = fit_best_model(sizes, times)
        best_name = max(fits, key=lambda k: fits[k][1])
        best_popt, best_r2 = fits[best_name]

        ax.scatter(
            sizes, times, s=12,
            marker=VERSION_MARKERS.get(version, "o"),
            color=VERSION_COLORS.get(version, "gray"),
            zorder=5,
        )

        if best_r2 > 0.5:
            dense_n = np.linspace(sizes[0], sizes[-1], 300)
            y_fit = MODELS[best_name](dense_n, *best_popt)
            ax.plot(
                dense_n, y_fit, "-",
                color=VERSION_COLORS.get(version, "gray"),
                linewidth=1, alpha=0.7,
                label=f"{VERSION_LABELS.get(version, version)[:10]}… {best_name} R²={best_r2:.3f}",
            )

    ax.set_xlabel("N")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Curve Fit")
    ax.legend(fontsize=4.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Parity (bottom-right) ---
    ax = axes[1, 1]
    if parity:
        bar_width = 0.35
        offset = 0
        first_sizes = None
        for version, data in parity.items():
            if not data["sizes"]:
                continue
            if first_sizes is None:
                first_sizes = data["sizes"]
            sizes_arr = np.arange(len(data["sizes"]))
            aris = np.array(data["ari_mean"])
            ax.bar(
                sizes_arr + offset * bar_width, aris, bar_width,
                label=VERSION_LABELS.get(version, version),
                color=VERSION_COLORS.get(version, "gray"),
                alpha=0.85,
            )
            offset += 1
        if first_sizes:
            ax.set_xticks(np.arange(len(first_sizes)) + bar_width * (offset - 1) / 2)
            ax.set_xticklabels([str(s) for s in first_sizes], fontsize=5, rotation=45)
    ax.set_xlabel("N")
    ax.set_ylabel("ARI")
    ax.set_title("Parity vs Original")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Gauging-delta Benchmark", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"benchmark_combined.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved benchmark_combined.png/pdf")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results: dict) -> None:
    """Print a summary table to console."""
    summary = results["summary"]
    parity = results["parity"]

    print(f"\n{'=' * 90}")
    print(f"{'Version':>20s}  {'N':>6s}  {'Time (s)':>10s}  {'Mem (MB)':>10s}  {'ARI':>8s}")
    print(f"{'-' * 90}")

    for version in ["original", "max_parity", "max_fast"]:
        if version not in summary:
            continue
        data = summary[version]
        ari_data = parity.get(version, {})
        ari_sizes = ari_data.get("sizes", [])
        ari_vals = ari_data.get("ari_mean", [])

        for i, n in enumerate(data["sizes"]):
            t = data["times_mean"][i]
            m = data["mem_mean"][i] / 1e6
            ari_idx = ari_sizes.index(n) if n in ari_sizes else -1
            ari_str = f"{ari_vals[ari_idx]:.4f}" if ari_idx >= 0 else "baseline"
            print(f"{VERSION_LABELS.get(version, version):>20s}  {n:>6d}  {t:>10.2f}  {m:>10.1f}  {ari_str:>8s}")

    print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Gauging-δ Unified Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help=f"Dataset sizes to benchmark (default: {DEFAULT_SIZES})",
    )
    parser.add_argument(
        "--skip-original", action="store_true",
        help="Skip running original Perception (use cached results if available)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Override number of repeats per size (default: 3 for N<=1000, 1 otherwise)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory for results (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1; set >1 for multiprocessing)",
    )
    args = parser.parse_args()

    sizes = args.sizes or DEFAULT_SIZES
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Load cached results if available
    json_path = out_dir / "benchmark_results.json"
    cached = None
    if json_path.exists() and args.skip_original:
        print(f"  Loading cached results from {json_path}")
        with open(json_path) as f:
            cached = json.load(f)

    # Run benchmarks
    results = run_benchmarks(
        sizes=sizes,
        repeats=args.repeats,
        workers=args.workers,
        skip_original=args.skip_original,
        cached_results=cached,
        save_path=json_path,
    )

    # Save raw results
    out_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "sizes": sizes,
        "raw": [
            {k: v for k, v in r.items() if k != "labels"}
            for r in results["raw"]
        ],
        "summary": results["summary"],
        "parity": results["parity"],
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # Print summary
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_runtime(results["summary"], ASSETS_DIR)
    plot_memory(results["summary"], ASSETS_DIR)
    if results["parity"]:
        plot_parity(results["parity"], ASSETS_DIR)
    plot_combined(results["summary"], results["parity"], ASSETS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
