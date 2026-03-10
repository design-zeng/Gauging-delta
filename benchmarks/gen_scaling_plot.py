"""Generate assets/scaling_benchmark.png for the README.

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
    uv run python benchmarks/gen_scaling_plot.py
"""

from __future__ import annotations

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
import scienceplots  # noqa: F401 — required for scientific style
from scipy.optimize import curve_fit


matplotlib.use("Agg")
plt.style.use(["science", "no-latex", "grid"])

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
# Plotting (1×3 panels)
# ---------------------------------------------------------------------------


def plot_results(full_res, lite_res, fits):
    """Generate 1×3 panel plot and save to assets/."""
    # Colorblind-safe palette (Okabe-Ito)
    COLOR_FULL = "#0072B2"
    COLOR_LITE = "#009E73"
    COLOR_CEIL = "#D55E00"

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

    extrap_max = max(full_ns[-1], lite_ns[-1])
    extrap_ns = np.logspace(np.log10(extrap_max), np.log10(100_000), 60)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # === Panel 1: Runtime Scaling ===
    fa_t, fb_t, fr2_t = fits["full_time"]
    la_t, lb_t, lr2_t = fits["lite_time"]

    # Measured data with ±1σ shaded bands
    ax1.plot(full_ns, full_med_t, "o-", color=COLOR_FULL, markersize=5, linewidth=1.3)
    ax1.fill_between(
        full_ns,
        np.maximum(full_med_t - full_std_t, 1e-6),
        full_med_t + full_std_t,
        alpha=0.15,
        color=COLOR_FULL,
        label=r"Full mode $\tilde{x} \pm \sigma$",
    )
    ax1.plot(lite_ns, lite_med_t, "s-", color=COLOR_LITE, markersize=5, linewidth=1.3)
    ax1.fill_between(
        lite_ns,
        np.maximum(lite_med_t - lite_std_t, 1e-6),
        lite_med_t + lite_std_t,
        alpha=0.15,
        color=COLOR_LITE,
        label=r"Lite mode $\tilde{x} \pm \sigma$",
    )

    # Power-law extrapolations
    ax1.loglog(
        extrap_ns,
        power_law(extrap_ns, fa_t, fb_t),
        "--",
        color=COLOR_FULL,
        alpha=0.5,
        linewidth=1,
        label=f"Full fit $O(N^{{{fb_t:.2f}}})$  $R^2\\!={fr2_t:.3f}$",
    )
    ax1.loglog(
        extrap_ns,
        power_law(extrap_ns, la_t, lb_t),
        "--",
        color=COLOR_LITE,
        alpha=0.5,
        linewidth=1,
        label=f"Lite fit $O(N^{{{lb_t:.2f}}})$  $R^2\\!={lr2_t:.3f}$",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("$N$")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime Scaling", fontsize=12)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # === Panel 2: Full Mode Time Split (stacked area + uncertainty) ===
    _fa_d, fb_d, fr2_d = fits["full_dist"]
    _fa_c, fb_c, fr2_c = fits["full_cluster"]

    full_total_top = full_med_dist + full_med_clust

    # Stacked area fills
    ax2.fill_between(
        full_ns,
        0,
        full_med_dist,
        alpha=0.25,
        color=COLOR_FULL,
    )
    ax2.fill_between(
        full_ns,
        full_med_dist,
        full_total_top,
        alpha=0.25,
        color=COLOR_LITE,
    )

    # Median lines with markers
    ax2.plot(
        full_ns,
        full_med_dist,
        "o-",
        color=COLOR_FULL,
        markersize=5,
        linewidth=1.3,
        label=f"Distance init $O(N^{{{fb_d:.2f}}})$  $R^2\\!={fr2_d:.3f}$",
    )
    ax2.plot(
        full_ns,
        full_total_top,
        "s-",
        color=COLOR_LITE,
        markersize=5,
        linewidth=1.3,
        label=f"+ Clustering $O(N^{{{fb_c:.2f}}})$  $R^2\\!={fr2_c:.3f}$",
    )

    # ±1σ uncertainty bands on each boundary
    ax2.fill_between(
        full_ns,
        np.maximum(full_med_dist - full_std_dist, 1e-6),
        full_med_dist + full_std_dist,
        alpha=0.12,
        color=COLOR_FULL,
    )
    # Combined uncertainty on the top boundary: σ_total = sqrt(σ_dist² + σ_clust²)
    std_top = np.sqrt(full_std_dist**2 + full_std_clust**2)
    ax2.fill_between(
        full_ns,
        np.maximum(full_total_top - std_top, 1e-6),
        full_total_top + std_top,
        alpha=0.12,
        color=COLOR_LITE,
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("$N$")
    ax2.set_ylabel("Runtime (s)")
    ax2.set_title("Full Mode: Time Split", fontsize=12)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Annotation: what precomputed users skip
    mid = len(full_ns) // 2
    ax2.annotate(
        "precomputed users\nskip this region",
        xy=(full_ns[mid], full_med_dist[mid]),
        xytext=(full_ns[max(mid - 2, 0)], full_med_dist[mid] * 0.15),
        fontsize=8,
        color=COLOR_FULL,
        fontstyle="italic",
        arrowprops={"arrowstyle": "->", "color": COLOR_FULL, "lw": 0.8},
    )

    # === Panel 3: Memory Scaling ===
    fa_m, fb_m, fr2_m = fits["full_mem"]
    la_m, lb_m, lr2_m = fits["lite_mem"]

    full_med_m_mb = full_med_m / 1e6
    lite_med_m_mb = lite_med_m / 1e6

    # Memory is deterministic (std ≈ 0), so plot without bands
    ax3.plot(
        full_ns,
        full_med_m_mb,
        "o-",
        color=COLOR_FULL,
        markersize=5,
        linewidth=1.3,
        label=r"Full mode",
    )
    ax3.plot(
        lite_ns,
        lite_med_m_mb,
        "s-",
        color=COLOR_LITE,
        markersize=5,
        linewidth=1.3,
        label=r"Lite mode",
    )

    # Power-law extrapolations
    ax3.loglog(
        extrap_ns,
        power_law(extrap_ns, fa_m, fb_m) / 1e6,
        "--",
        color=COLOR_FULL,
        alpha=0.5,
        linewidth=1,
        label=f"Full fit $O(N^{{{fb_m:.2f}}})$  $R^2\\!={fr2_m:.3f}$",
    )
    ax3.loglog(
        extrap_ns,
        power_law(extrap_ns, la_m, lb_m) / 1e6,
        "--",
        color=COLOR_LITE,
        alpha=0.5,
        linewidth=1,
        label=f"Lite fit $O(N^{{{lb_m:.2f}}})$  $R^2\\!={lr2_m:.3f}$",
    )
    ax3.axhline(
        64_000,
        color=COLOR_CEIL,
        linewidth=1.0,
        linestyle=":",
        alpha=0.85,
        label="64 GB",
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("$N$")
    ax3.set_ylabel("Peak Memory (MB)")
    ax3.set_title("Memory Scaling", fontsize=12)
    ax3.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # === Figure-level ===
    n_generators = len(_GENERATORS)
    fig.suptitle(
        f"Gauging-$\\delta$ Scaling"
        f"  ({n_generators} generators"
        f" $\\times$ {N_EXAMPLES} examples/cell"
        f"$\\;\\cdot\\;$ shaded = $\\pm 1\\sigma$)",
        fontsize=12,
    )
    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    out = ASSETS_DIR / "scaling_benchmark.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot \u2192 {out}")


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


def main():
    n_generators = len(_GENERATORS)
    print("=" * 75)
    print(f"Scaling benchmark  ({n_generators} generators \u00d7 {N_EXAMPLES} examples/cell)")
    print(f"Full sizes: {FULL_SIZES}")
    print(f"Lite sizes: {LITE_SIZES}")
    print("=" * 75)

    # ── Full mode ──────────────────────────────────────────────────────────
    print("\nFull mode:")
    full_res = measure_split(FULL_SIZES, "full")

    # ── Lite mode ──────────────────────────────────────────────────────────
    print("\nLite mode:")
    lite_res = measure_split(LITE_SIZES, "lite")

    # ── Fit power laws ─────────────────────────────────────────────────────
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
    print(f"  clustering O(N^{fits['full_cluster'][1]:.2f}) R\u00b2={fits['full_cluster'][2]:.4f}")
    print(f"  mem O(N^{fits['full_mem'][1]:.2f}) R\u00b2={fits['full_mem'][2]:.4f}")
    print(f"Lite:  time O(N^{fits['lite_time'][1]:.2f}) R\u00b2={fits['lite_time'][2]:.4f}")
    print(f"  mem O(N^{fits['lite_mem'][1]:.2f}) R\u00b2={fits['lite_mem'][2]:.4f}")

    # ── Plot + output ──────────────────────────────────────────────────────
    plot_results(full_res, lite_res, fits)
    print_table(full_res, lite_res, fits)
    save_json(full_res, lite_res, fits)


if __name__ == "__main__":
    main()
