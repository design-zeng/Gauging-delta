"""
Full benchmark runner for 60 GB RAM systems.

Runs **sequentially** (one task at a time) to minimise peak memory.
Progress is serialised to disk after every individual task so that an
overnight run can be safely resumed without losing completed work.
All warnings are suppressed for clean console output.

Phases:
  1 – All three variants at moderate sizes.
  2 – Skip original; max_parity + max_fast at larger sizes.
  3 – max_fast only at the largest sizes.

Memory budget (approximate, 2-D data, float64):
  max_parity : ~140 bytes / N²  →  safe limit ≈ 18 000 points @ 60 GB
  max_fast   :  O(N·k) sparse   →  comfortably handles 50 000+ points

Usage:
    uv run python benchmarks/run_full_benchmark.py
    uv run python benchmarks/run_full_benchmark.py --phase 1
    uv run python benchmarks/run_full_benchmark.py --resume
    uv run python benchmarks/run_full_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# ── Suppress all warnings globally ────────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmarks" / "results"
PROGRESS_FILE = RESULTS_DIR / "benchmark_progress.json"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES: dict[int, dict] = {
    1: {
        "label": "All variants, moderate sizes",
        "sizes": [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000],
        "repeats": 3,
        "versions": ["original", "max_parity", "max_fast"],
    },
    2: {
        "label": "max_parity + max_fast, larger sizes",
        "sizes": [12500, 15000, 17500],
        "repeats": 1,
        "versions": ["max_parity", "max_fast"],
    },
    3: {
        "label": "max_fast only, largest sizes",
        "sizes": [20000, 25000],
        "repeats": 1,
        "versions": ["max_fast"],
    },
}

SEED = 42

# ---------------------------------------------------------------------------
# Task runners (in-process, sequential, one at a time)
# ---------------------------------------------------------------------------

def _make_blobs(n_points: int, seed: int = SEED) -> "np.ndarray":
    import numpy as np
    rng = np.random.default_rng(seed)
    n_blobs, dims = 5, 2
    per_blob = n_points // n_blobs
    remainder = n_points - per_blob * n_blobs
    centers = rng.uniform(-20, 20, size=(n_blobs, dims))
    parts = []
    for i, c in enumerate(centers):
        count = per_blob + (1 if i < remainder else 0)
        parts.append(rng.normal(loc=c, scale=1.0, size=(count, dims)))
    return np.vstack(parts)


def _run_single(version: str, X: "np.ndarray") -> tuple["np.ndarray", float, float]:
    """Run one benchmark task and return (labels, time_s, peak_bytes)."""
    import gc
    import io
    import time
    import tracemalloc
    from contextlib import redirect_stdout

    import numpy as np

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    if version == "original":
        from perception import Perception
        p = Perception()
        with redirect_stdout(io.StringIO()):
            labels, _ = p.fit(X.copy())
        labels = np.asarray(labels, dtype=int)
    elif version == "max_parity":
        from gauging_delta import GaugingDelta
        model = GaugingDelta(preserve_labels=True)
        model.fit(X.copy())
        labels = np.asarray(model.labels_, dtype=int)
    elif version == "max_fast":
        from gauging_delta import GaugingDeltaFast
        model = GaugingDeltaFast(preserve_labels=True)
        model.fit(X.copy())
        labels = np.asarray(model.labels_, dtype=int)
    else:
        raise ValueError(f"Unknown version: {version}")

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return labels, elapsed, float(peak)


# ---------------------------------------------------------------------------
# Progress management
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    """Load serialised progress (completed tasks + raw results)."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "raw": []}


def _save_progress(progress: dict) -> None:
    """Atomically persist progress to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(PROGRESS_FILE)


def _task_key(version: str, n: int, repeat: int) -> str:
    return f"{version}|{n}|{repeat}"


def _is_completed(progress: dict, version: str, n: int, repeat: int) -> bool:
    return _task_key(version, n, repeat) in progress["completed"]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(phases_to_run: list[int], *, dry_run: bool = False, resume: bool = False) -> None:
    """Execute benchmark tasks one at a time, saving after each."""
    import numpy as np

    progress = _load_progress() if resume else {"completed": [], "raw": []}

    # Build full sequential task list
    tasks: list[tuple[int, str, int, int]] = []  # (phase, version, n, repeat)
    for phase_num in phases_to_run:
        cfg = PHASES[phase_num]
        for n in cfg["sizes"]:
            for version in cfg["versions"]:
                for r in range(cfg["repeats"]):
                    tasks.append((phase_num, version, n, r))

    # Filter out already completed tasks
    remaining = [
        t for t in tasks
        if not _is_completed(progress, t[1], t[2], t[3])
    ]

    total_all = len(tasks)
    total_remaining = len(remaining)
    already_done = total_all - total_remaining

    print(f"\n{'=' * 70}")
    print(f"  Gauging-delta  Full Benchmark Suite (sequential)")
    print(f"  Phases       : {phases_to_run}")
    print(f"  Total tasks  : {total_all}")
    print(f"  Already done : {already_done}")
    print(f"  Remaining    : {total_remaining}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"{'=' * 70}\n")

    if dry_run:
        for phase_num, version, n, r in remaining:
            print(f"  [DRY] Phase {phase_num}  {version:>12s}  N={n:>6d}  r={r}")
        print(f"\n  {total_remaining} tasks would be executed.\n")
        return

    if total_remaining == 0:
        print("  All tasks already completed. Use without --resume to re-run.\n")
        return

    t_start = time.perf_counter()
    completed_count = 0

    for phase_num, version, n, r in remaining:
        # Generate data
        X = _make_blobs(n, seed=SEED + r)

        # Run task
        try:
            labels, elapsed, peak = _run_single(version, X)
            result = {
                "version": version,
                "n": n,
                "repeat": r,
                "phase": phase_num,
                "time_s": elapsed,
                "peak_bytes": peak,
                "n_clusters": int(len(set(labels.tolist()))),
                "error": None,
            }
        except Exception as e:
            result = {
                "version": version,
                "n": n,
                "repeat": r,
                "phase": phase_num,
                "time_s": None,
                "peak_bytes": None,
                "n_clusters": None,
                "error": str(e),
            }

        # Record result
        progress["raw"].append(result)
        progress["completed"].append(_task_key(version, n, r))
        progress["last_updated"] = datetime.now().isoformat()

        # Save immediately
        _save_progress(progress)

        # Print progress
        completed_count += 1
        wall = time.perf_counter() - t_start
        eta = (wall / completed_count) * (total_remaining - completed_count) if completed_count > 0 else 0
        status = "OK" if result["error"] is None else f"ERR: {result['error'][:40]}"
        time_str = f"{result['time_s']:.1f}s" if result["time_s"] else "N/A"
        mem_str = f"{result['peak_bytes'] / 1e6:.0f}MB" if result["peak_bytes"] else "N/A"
        print(
            f"  [{already_done + completed_count:>4d}/{total_all}] "
            f"P{phase_num} {version:>12s} N={n:>6d} r={r}  "
            f"{time_str:>10s}  {mem_str:>8s}  "
            f"ETA={eta / 60:.1f}min  {status}"
        )

        # Free memory aggressively
        del X
        gc.collect()

    print(f"\n{'=' * 70}")
    print(f"  All {total_all} tasks complete.")
    print(f"  Total wall time: {(time.perf_counter() - t_start) / 60:.1f} min")
    print(f"  Results saved to: {PROGRESS_FILE}")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full benchmark runner (60 GB RAM, sequential, crash-safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Run only this phase (default: all phases sequentially)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print task list without executing",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved progress (skip completed tasks)",
    )
    args = parser.parse_args()

    phases_to_run = [args.phase] if args.phase else sorted(PHASES.keys())
    run_all(phases_to_run, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
