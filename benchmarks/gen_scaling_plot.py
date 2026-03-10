"""Generate assets/scaling_{runtime,memory,timesplit}_{light,dark}.png for the README.

Runs GaugingDelta (full and lite modes) across diverse randomly-parameterized
datasets. 30 examples per (N, mode) cell using 7 weighted generators.

Timing is split into "distance init" and "clustering" phases via monkey-patch
on ``_init_distances``, so precomputed-matrix users can see what they skip.

Dataset generators (Hypothesis-inspired, deterministically seeded):
  bench_blobs     — Well-separated Gaussians, randomized k/separation/std (weight 2)
  bench_noisy     — Overlapping Gaussians, high std (weight 2)
  bench_uniform   — Uniform random [0,20]², no structure (weight 2)
  bench_spirals   — Interleaved spiral arms (weight 1)
  bench_unequal   — Highly imbalanced cluster sizes 90/10% (weight 1)
  bench_singleton — Dense clusters + scattered singletons (weight 1)
  bench_collinear — Clusters along a line (weight 1)

Usage:
    uv run python benchmarks/gen_scaling_plot.py [--theme light|dark|both]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
import tracemalloc
import types
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _plotting import apply_theme
from scipy.optimize import curve_fit


matplotlib.use("Agg")

# Minimal Tufte-inspired style — no grid, no top/right spines, sans-serif.
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": "#333333",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
RESULTS_DIR = ROOT / "benchmarks" / "results"
sys.path.insert(0, str(ROOT))

FULL_SIZES = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000]
LITE_SIZES = [
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    7500,
    10000,
    15000,
    20000,
    30000,
    50000,
]

N_EXAMPLES = 30

# ---------------------------------------------------------------------------
# Deterministic benchmark generators (all produce (target_n, 2) arrays)
#
# Each generator accepts (target_n, rng) where rng is a seeded
# random.Random instance used to pick randomized parameters, and an
# np.random.RandomState derived from it for data generation.
# ---------------------------------------------------------------------------

# Weighted generator table: name → (function, weight)
# Weight determines how many times the generator appears in the selection pool.
_GENERATORS = {}


def _register(weight):
    def decorator(fn):
        _GENERATORS[fn.__name__] = (fn, weight)
        return fn

    return decorator


@_register(weight=2)
def bench_blobs(target_n, rng):
    """Well-separated Gaussians, randomized k/separation/std."""
    k = rng.randint(3, 8)
    separation = rng.uniform(8.0, 25.0)
    std = rng.uniform(0.3, 2.0)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    per = target_n // k
    parts = []
    for i in range(k):
        count = per + (1 if i < target_n - per * k else 0)
        center = nprng.uniform(-separation, separation, size=2)
        parts.append(nprng.normal(center, std, (count, 2)))
    return np.vstack(parts)


@_register(weight=2)
def bench_noisy(target_n, rng):
    """Overlapping Gaussians, high std."""
    k = rng.randint(3, 7)
    std = rng.uniform(3.0, 6.0)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    per = target_n // k
    parts = []
    for i in range(k):
        count = per + (1 if i < target_n - per * k else 0)
        center = nprng.uniform(-10, 10, size=2)
        parts.append(nprng.normal(center, std, (count, 2)))
    return np.vstack(parts)


@_register(weight=2)
def bench_uniform(target_n, rng):
    """Uniform random in [0, 20]² — no cluster structure."""
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    return nprng.uniform(0, 20, size=(target_n, 2))


@_register(weight=1)
def bench_spirals(target_n, rng):
    """Interleaved spiral arms."""
    n_arms = rng.randint(2, 4)
    noise = rng.uniform(0.05, 0.3)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    per = target_n // n_arms
    parts = []
    for arm in range(n_arms):
        count = per + (1 if arm < target_n - per * n_arms else 0)
        theta = np.linspace(0, 4 * np.pi, count) + arm * 2 * np.pi / n_arms
        r = np.linspace(0.5, 5, count)
        x = r * np.cos(theta) + nprng.normal(0, noise, count)
        y = r * np.sin(theta) + nprng.normal(0, noise, count)
        parts.append(np.column_stack([x, y]))
    return np.vstack(parts)


@_register(weight=1)
def bench_unequal(target_n, rng):
    """Highly imbalanced cluster sizes (90/10%)."""
    large_frac = rng.uniform(0.85, 0.95)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    n_large = int(target_n * large_frac)
    n_small = max(target_n - n_large, 2)
    large = nprng.normal(loc=[0, 0], scale=1.0, size=(n_large, 2))
    small = nprng.normal(loc=[8, 8], scale=0.3, size=(n_small, 2))
    return np.vstack([large, small])


@_register(weight=1)
def bench_singleton(target_n, rng):
    """Dense clusters + scattered singletons."""
    n_clusters = rng.randint(2, 4)
    scatter_frac = rng.uniform(0.3, 0.6)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    n_clustered = int(target_n * (1 - scatter_frac))
    n_scattered = target_n - n_clustered
    parts = []
    per = n_clustered // n_clusters
    for i in range(n_clusters):
        count = per + (1 if i < n_clustered - per * n_clusters else 0)
        center = nprng.uniform(-5, 5, size=2)
        parts.append(nprng.normal(center, 0.5, (count, 2)))
    parts.append(nprng.uniform(-20, 20, size=(n_scattered, 2)))
    return np.vstack(parts)


@_register(weight=1)
def bench_collinear(target_n, rng):
    """Clusters along a line."""
    k = rng.randint(3, 6)
    spacing = rng.uniform(3.0, 8.0)
    nprng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    per = target_n // k
    parts = []
    for i in range(k):
        count = per + (1 if i < target_n - per * k else 0)
        center = np.array([i * spacing, 0.0])
        parts.append(nprng.normal(center, 0.3, (count, 2)))
    return np.vstack(parts)


# Build the weighted selection pool once at import time.
_GENERATOR_POOL = []
for _name, (_fn, _w) in _GENERATORS.items():
    _GENERATOR_POOL.extend([(_name, _fn)] * _w)


def generate_dataset(target_n, size_idx, ex_idx):
    """Generate one dataset with reproducible seeding.

    A ``random.Random`` seeded from (size_idx, ex_idx) picks the generator
    (from the weighted pool) and all its randomized parameters.
    """
    rng = random.Random(size_idx * 1000 + ex_idx)
    name, fn = rng.choice(_GENERATOR_POOL)
    return fn(target_n, rng), name


# ---------------------------------------------------------------------------
# Runner with timing split
# ---------------------------------------------------------------------------


def run_once_split(X, mode):
    """Return (t_total, t_dist, t_cluster, peak_bytes) for one run.

    Monkey-patches ``_init_distances`` to measure the distance-init phase
    separately from the clustering phase.
    """
    from gauging_delta import GaugingDelta

    gd = GaugingDelta(mode=mode)
    orig = gd._init_distances.__func__

    def timed_init(self, _orig=orig):
        t0 = time.perf_counter()
        _orig(self)
        self._t_init = time.perf_counter() - t0

    gd._init_distances = types.MethodType(timed_init, gd)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd.fit(X.copy())
    t_total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_dist = getattr(gd, "_t_init", 0.0)
    t_cluster = t_total - t_dist
    return t_total, t_dist, t_cluster, float(peak)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_split(sizes, mode, n_examples=N_EXAMPLES):
    """Measure timing split + memory across generated datasets.

    Returns list of dicts, one per size.
    """
    results = []
    for si, n in enumerate(sizes):
        totals, dists, clusters, mems = [], [], [], []
        for ei in range(n_examples):
            X, _name = generate_dataset(n, si, ei)
            t_total, t_dist, t_cluster, peak = run_once_split(X, mode)
            totals.append(t_total)
            dists.append(t_dist)
            clusters.append(t_cluster)
            mems.append(peak)
            del X
            gc.collect()
        results.append(
            {
                "n": n,
                "med_total": float(np.median(totals)),
                "std_total": float(np.std(totals)),
                "med_dist": float(np.median(dists)),
                "std_dist": float(np.std(dists)),
                "med_cluster": float(np.median(clusters)),
                "std_cluster": float(np.std(clusters)),
                "med_mem": float(np.median(mems)),
                "std_mem": float(np.std(mems)),
            }
        )
        r = results[-1]
        print(
            f"  N={n:>6,}  median={r['med_total']:>7.2f}s ±{r['std_total']:.2f}"
            f"  (dist={r['med_dist']:.2f}s, cluster={r['med_cluster']:.2f}s)"
            f"  mem={r['med_mem'] / 1e6:>8.1f} MB  ({n_examples} runs)"
        )
    return results


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------


def power_law(n, a, b):
    return a * n**b


def fit_power(sizes, values):
    """Fit a*N^b; return (a, b, R²)."""
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    try:
        popt, _ = curve_fit(
            power_law,
            sizes,
            values,
            p0=[1e-6, 1.5],
            bounds=([0, 0.3], [np.inf, 4.0]),
            maxfev=10000,
        )
        pred = power_law(sizes, *popt)
        ss_res = float(np.sum((values - pred) ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return float(popt[0]), float(popt[1]), r2
    except Exception:
        return 1e-10, 2.0, 0.0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_time(s):
    if s < 60:
        return f"{s:.0f} s"
    return f"{s / 60:.1f} min"


def fmt_mem(b):
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    return f"{b / 1e6:.0f} MB"


# ---------------------------------------------------------------------------
# Plotting (3 separate figures — Catppuccin Mocha/Latte themed)
# ---------------------------------------------------------------------------


def _save_fig(fig, name, theme):
    """Save a figure to assets/ with theme suffix and close it."""
    ASSETS_DIR.mkdir(exist_ok=True)
    stem, ext = name.rsplit(".", 1)
    out = ASSETS_DIR / f"{stem}_{theme}.{ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot \u2192 {out}")


def _add_time_refs(ax, tc):
    """Add faint reference lines at 1 s and 1 min."""
    for val, label in [(1, "1 s"), (60, "1 min")]:
        ax.axhline(val, color=tc.ref_line, linewidth=0.6, zorder=0)
        ax.text(
            ax.get_xlim()[0] * 1.3, val * 1.15, label,
            fontsize=7, color=tc.ref_text, va="bottom",
        )


def _label_at_end(ax, x, y, text, color, offset=(1.08, 1.0)):
    """Place a direct label at the end of a line."""
    ax.text(
        x * offset[0], y * offset[1], text,
        fontsize=9, color=color, va="center", fontstyle="italic",
    )


def plot_results(full_res, lite_res, fits, theme="light"):
    """Generate 3 separate scaling plots for one theme and save to assets/."""
    tc = apply_theme(theme)

    full_ns = np.array([r["n"] for r in full_res], dtype=float)
    lite_ns = np.array([r["n"] for r in lite_res], dtype=float)

    full_med_t = np.array([r["med_total"] for r in full_res])
    full_std_t = np.array([r["std_total"] for r in full_res])
    lite_med_t = np.array([r["med_total"] for r in lite_res])
    lite_std_t = np.array([r["std_total"] for r in lite_res])

    full_med_dist = np.array([r["med_dist"] for r in full_res])
    full_std_dist = np.array([r["std_dist"] for r in full_res])
    full_med_clust = np.array([r["med_cluster"] for r in full_res])
    full_std_clust = np.array([r["std_cluster"] for r in full_res])

    full_med_m = np.array([r["med_mem"] for r in full_res])
    lite_med_m = np.array([r["med_mem"] for r in lite_res])

    # Per-mode extrapolation ranges — each starts from its own last measured point
    extrap_end = 100_000
    full_extrap = np.concatenate([
        [full_ns[-1]],
        np.logspace(np.log10(full_ns[-1]) + 0.01, np.log10(extrap_end), 50),
    ])
    lite_extrap = np.concatenate([
        [lite_ns[-1]],
        np.logspace(np.log10(lite_ns[-1]) + 0.01, np.log10(extrap_end), 50),
    ])

    n_gen = len(_GENERATORS)
    footnote = f"median over {N_EXAMPLES} runs \u00d7 {n_gen} generators per N"
    DASH = (0, (6, 3))

    # ================================================================
    # Figure 1: Runtime Scaling
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    fa_t, fb_t, _fr2_t = fits["full_time"]
    la_t, lb_t, _lr2_t = fits["lite_time"]

    # Measured data with error bars
    ax1.errorbar(
        full_ns, full_med_t, yerr=full_std_t,
        fmt="o-", color=tc.full, capsize=3, capthick=1, markersize=5,
        linewidth=2, label="Full mode",
    )
    ax1.errorbar(
        lite_ns, lite_med_t, yerr=lite_std_t,
        fmt="s-", color=tc.lite, capsize=3, capthick=1, markersize=5,
        linewidth=2, label="Lite mode",
    )

    # Power-law extrapolation dashes (per-mode, no gap)
    ax1.loglog(
        full_extrap, power_law(full_extrap, fa_t, fb_t),
        linestyle=DASH, color=tc.full, alpha=0.6, linewidth=1.5,
    )
    ax1.loglog(
        lite_extrap, power_law(lite_extrap, la_t, lb_t),
        linestyle=DASH, color=tc.lite, alpha=0.6, linewidth=1.5,
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (points)")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title("Runtime Scaling")
    ax1.legend(loc="upper left")
    _add_time_refs(ax1, tc)

    # Direct fit labels near extrapolation ends
    _label_at_end(
        ax1, full_extrap[-1], power_law(full_extrap[-1], fa_t, fb_t),
        f"$O(N^{{{fb_t:.2f}}})$", tc.full,
    )
    _label_at_end(
        ax1, lite_extrap[-1], power_law(lite_extrap[-1], la_t, lb_t),
        f"$O(N^{{{lb_t:.2f}}})$", tc.lite,
    )

    fig1.text(0.5, -0.02, footnote, ha="center", fontsize=8, color=tc.footnote)
    fig1.tight_layout()
    _save_fig(fig1, "scaling_runtime.png", theme)

    # ================================================================
    # Figure 2: Full Mode Time Split
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    fa_d, fb_d, _fr2_d = fits["full_dist"]
    fa_c, fb_c, _fr2_c = fits["full_cluster"]

    # Measured data with error bars
    ax2.errorbar(
        full_ns, full_med_dist, yerr=full_std_dist,
        fmt="o-", color=tc.dist, capsize=3, capthick=1, markersize=5,
        linewidth=2, label="Distance init",
    )
    ax2.errorbar(
        full_ns, full_med_clust, yerr=full_std_clust,
        fmt="s-", color=tc.lite, capsize=3, capthick=1, markersize=5,
        linewidth=2, label="Clustering",
    )

    # Power-law extrapolation dashes — shows crossover at ~N=42K
    ax2.loglog(
        full_extrap, power_law(full_extrap, fa_d, fb_d),
        linestyle=DASH, color=tc.dist, alpha=0.6, linewidth=1.5,
    )
    ax2.loglog(
        full_extrap, power_law(full_extrap, fa_c, fb_c),
        linestyle=DASH, color=tc.lite, alpha=0.6, linewidth=1.5,
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N (points)")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title("Full Mode: Time Split")
    ax2.legend(loc="upper left")
    _add_time_refs(ax2, tc)

    # Direct fit labels at extrapolation ends
    _label_at_end(
        ax2, full_extrap[-1], power_law(full_extrap[-1], fa_d, fb_d),
        f"$O(N^{{{fb_d:.2f}}})$", tc.dist,
    )
    _label_at_end(
        ax2, full_extrap[-1], power_law(full_extrap[-1], fa_c, fb_c),
        f"$O(N^{{{fb_c:.2f}}})$", tc.lite,
    )

    fig2.text(0.5, -0.02, footnote, ha="center", fontsize=8, color=tc.footnote)
    fig2.tight_layout()
    _save_fig(fig2, "scaling_timesplit.png", theme)

    # ================================================================
    # Figure 3: Memory Scaling
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    fa_m, fb_m, _fr2_m = fits["full_mem"]
    la_m, lb_m, _lr2_m = fits["lite_mem"]

    full_med_m_mb = full_med_m / 1e6
    lite_med_m_mb = lite_med_m / 1e6

    # Measured data (memory is near-deterministic, no error bars needed)
    ax3.plot(
        full_ns, full_med_m_mb, "o-", color=tc.full, markersize=5,
        linewidth=2, label="Full mode",
    )
    ax3.plot(
        lite_ns, lite_med_m_mb, "s-", color=tc.lite, markersize=5,
        linewidth=2, label="Lite mode",
    )

    # Power-law extrapolation dashes (per-mode, no gap)
    ax3.loglog(
        full_extrap, power_law(full_extrap, fa_m, fb_m) / 1e6,
        linestyle=DASH, color=tc.full, alpha=0.6, linewidth=1.5,
    )
    ax3.loglog(
        lite_extrap, power_law(lite_extrap, la_m, lb_m) / 1e6,
        linestyle=DASH, color=tc.lite, alpha=0.6, linewidth=1.5,
    )

    # 64 GB ceiling
    ax3.axhline(
        64_000, color=tc.ceil, linewidth=1.2, linestyle=":", alpha=0.85,
    )
    ax3.text(
        full_ns[0] * 1.3, 64_000 * 1.15, "64 GB",
        fontsize=9, color=tc.ceil, va="bottom",
    )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("N (points)")
    ax3.set_ylabel("Peak Memory (MB)")
    ax3.set_title("Memory Scaling")
    ax3.legend(loc="upper left")

    # Reference lines at key memory values
    for val, label in [(1000, "1 GB"), (10_000, "10 GB")]:
        ax3.axhline(val, color=tc.ref_line, linewidth=0.6, zorder=0)
        ax3.text(
            full_ns[0] * 1.3, val * 1.15, label,
            fontsize=7, color=tc.ref_text, va="bottom",
        )

    # Direct fit labels near extrapolation ends
    _label_at_end(
        ax3, full_extrap[-1], power_law(full_extrap[-1], fa_m, fb_m) / 1e6,
        f"$O(N^{{{fb_m:.2f}}})$", tc.full,
    )
    _label_at_end(
        ax3, lite_extrap[-1], power_law(lite_extrap[-1], la_m, lb_m) / 1e6,
        f"$O(N^{{{lb_m:.2f}}})$", tc.lite,
    )

    fig3.text(0.5, -0.02, footnote, ha="center", fontsize=8, color=tc.footnote)
    fig3.tight_layout()
    _save_fig(fig3, "scaling_memory.png", theme)


# ---------------------------------------------------------------------------
# Table + JSON output
# ---------------------------------------------------------------------------


def print_table(full_res, lite_res, fits):
    """Print README-ready markdown table with dist-init sub-column."""
    print(f"\n## README scaling table (measured rows, median over {N_EXAMPLES} runs):\n")
    header = (
        f"| {'N':>8} | {'Full total':>12} | {'(dist init)':>12} "
        f"| {'Full memory':>12} | {'Lite total':>12} | {'Lite memory':>12} |"
    )
    sep = f"|{'-' * 10}:|{'-' * 14}:|{'-' * 14}:|{'-' * 14}:|{'-' * 14}:|{'-' * 14}:|"
    print(header)
    print(sep)

    lite_lookup = {r["n"]: r for r in lite_res}
    for r in full_res:
        n = r["n"]
        lr = lite_lookup.get(n)
        lt = fmt_time(lr["med_total"]) if lr else "\u2014"
        lm = fmt_mem(lr["med_mem"]) if lr else "\u2014"
        print(
            f"| {n:>8,} "
            f"| {fmt_time(r['med_total']):>12} "
            f"| {fmt_time(r['med_dist']):>12} "
            f"| {fmt_mem(r['med_mem']):>12} "
            f"| {lt:>12} "
            f"| {lm:>12} |"
        )

    # Lite-only sizes
    for r in lite_res:
        if r["n"] not in {fr["n"] for fr in full_res}:
            print(
                f"| {r['n']:>8,} "
                f"| {'\u2014':>12} "
                f"| {'\u2014':>12} "
                f"| {'\u2014':>12} "
                f"| {fmt_time(r['med_total']):>12} "
                f"| {fmt_mem(r['med_mem']):>12} |"
            )

    # Projected
    fa_t, fb_t, _ = fits["full_time"]
    la_t, lb_t, _ = fits["lite_time"]
    fa_m, fb_m, _ = fits["full_mem"]
    la_m, lb_m, _ = fits["lite_mem"]
    fa_d, fb_d, _ = fits["full_dist"]

    projected = [20_000, 35_000, 100_000, 500_000, 1_000_000]
    print("\n## Projected rows (power-law extrapolation):\n")
    print(header)
    print(sep)
    for n in projected:
        nn = float(n)
        ft = f"~{fmt_time(float(power_law(np.array([nn]), fa_t, fb_t)[0]))}"
        fd = f"~{fmt_time(float(power_law(np.array([nn]), fa_d, fb_d)[0]))}"
        fm_b = float(power_law(np.array([nn]), fa_m, fb_m)[0])
        fm = ("**" + fmt_mem(fm_b) + "**") if fm_b >= 64e9 else fmt_mem(fm_b)
        lt = f"~{fmt_time(float(power_law(np.array([nn]), la_t, lb_t)[0]))}"
        lm = fmt_mem(float(power_law(np.array([nn]), la_m, lb_m)[0]))
        print(f"| {n:>8,} | {ft:>12} | {fd:>12} | {fm:>12} | {lt:>12} | {lm:>12} |")


def save_json(full_res, lite_res, fits):
    """Save results and fits to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "scaling_results.json"
    gen_info = [f"{name}({w})" for name, (_, w) in _GENERATORS.items()]
    data = {
        "meta": {
            "n_examples": N_EXAMPLES,
            "full_sizes": FULL_SIZES,
            "lite_sizes": LITE_SIZES,
            "generators": gen_info,
        },
        "full": full_res,
        "lite": lite_res,
        "fits": {k: {"a": v[0], "b": v[1], "r2": v[2]} for k, v in fits.items()},
    }
    out.write_text(json.dumps(data, indent=2))
    print(f"Saved JSON \u2192 {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_cached():
    """Load results + fits from cached scaling_results.json if available."""
    path = RESULTS_DIR / "scaling_results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    full_res = data["full"]
    lite_res = data["lite"]
    fits = {k: (v["a"], v["b"], v["r2"]) for k, v in data["fits"].items()}
    return full_res, lite_res, fits


def main():
    parser = argparse.ArgumentParser(description="Generate scaling plots for the README")
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "both"],
        default="both",
        help="Which Catppuccin theme(s) to generate (default: both)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking; regenerate plots from cached scaling_results.json",
    )
    args = parser.parse_args()
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]

    if args.plot_only:
        cached = load_cached()
        if cached is None:
            print("No cached results found — run without --plot-only first.")
            sys.exit(1)
        full_res, lite_res, fits = cached
        print(f"Loaded cached results from {RESULTS_DIR / 'scaling_results.json'}")
    else:
        n_generators = len(_GENERATORS)
        print("=" * 75)
        print(f"Scaling benchmark  ({n_generators} generators \u00d7 {N_EXAMPLES} examples/cell)")
        print(f"Full sizes: {FULL_SIZES}")
        print(f"Lite sizes: {LITE_SIZES}")
        print("=" * 75)

        # ── Full mode ──────────────────────────────────────────────────────
        print("\nFull mode:")
        full_res = measure_split(FULL_SIZES, "full")

        # ── Lite mode ──────────────────────────────────────────────────────
        print("\nLite mode:")
        lite_res = measure_split(LITE_SIZES, "lite")

        # ── Fit power laws ─────────────────────────────────────────────────
        full_ns = [r["n"] for r in full_res]
        lite_ns = [r["n"] for r in lite_res]

        fits = {
            "full_time": fit_power(full_ns, [r["med_total"] for r in full_res]),
            "full_dist": fit_power(full_ns, [r["med_dist"] for r in full_res]),
            "full_cluster": fit_power(full_ns, [r["med_cluster"] for r in full_res]),
            "full_mem": fit_power(full_ns, [r["med_mem"] for r in full_res]),
            "lite_time": fit_power(lite_ns, [r["med_total"] for r in lite_res]),
            "lite_mem": fit_power(lite_ns, [r["med_mem"] for r in lite_res]),
        }

        print(f"\nFull:  time O(N^{fits['full_time'][1]:.2f}) R\u00b2={fits['full_time'][2]:.4f}")
        print(f"  dist-init O(N^{fits['full_dist'][1]:.2f}) R\u00b2={fits['full_dist'][2]:.4f}")
        print(
            f"  clustering O(N^{fits['full_cluster'][1]:.2f})"
            f" R\u00b2={fits['full_cluster'][2]:.4f}"
        )
        print(f"  mem O(N^{fits['full_mem'][1]:.2f}) R\u00b2={fits['full_mem'][2]:.4f}")
        print(f"Lite:  time O(N^{fits['lite_time'][1]:.2f}) R\u00b2={fits['lite_time'][2]:.4f}")
        print(f"  mem O(N^{fits['lite_mem'][1]:.2f}) R\u00b2={fits['lite_mem'][2]:.4f}")

        print_table(full_res, lite_res, fits)
        save_json(full_res, lite_res, fits)

    # ── Plot + output ──────────────────────────────────────────────────────
    for theme in themes:
        print(f"\nGenerating {theme} theme plots...")
        plot_results(full_res, lite_res, fits, theme=theme)


if __name__ == "__main__":
    main()
