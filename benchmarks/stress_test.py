"""
Numerical stress test for parity between legacy and new implementations.

Generators target specific numerical instability hazards (division by zero,
nan/inf propagation, precision loss) and verify that both implementations
agree on every dataset.

Usage:
    uv run python benchmarks/stress_test.py
    uv run python benchmarks/stress_test.py --n-configs 200 --seed 42
    uv run python benchmarks/stress_test.py --deep --n-configs 50
    uv run python benchmarks/stress_test.py --reproduce 17
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
ASSETS_DIR = ROOT / "assets"
sys.path.insert(0, str(ROOT))

from benchmarks._plotting import use_science_style  # noqa: E402


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------


@dataclass
class TestConfig:
    """Fully reproducible test configuration."""

    config_id: int
    generator: str
    n_samples: int
    n_clusters: int
    n_dims: int
    seed: int
    params: dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single stress test."""

    config: TestConfig
    ari: float
    legacy_time_s: float
    new_time_s: float
    passed: bool
    actual_n: int = 0
    n_divergences: int = 0
    first_divergence: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Data generators — each targets a specific numerical hazard
# ---------------------------------------------------------------------------


def gen_near_zero_separation(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Clusters with near-zero separation.

    Numerical hazard: proximity denominator near zero -> rho -> inf/nan.
    """
    separation = cfg.params.get("separation", 1e-6)
    parts = []
    for i in range(cfg.n_clusters):
        center = np.zeros(cfg.n_dims)
        center[0] = i * separation
        pts = rng.normal(loc=center, scale=separation * 0.1, size=(cfg.n_samples // cfg.n_clusters, cfg.n_dims))
        parts.append(pts)
    return np.vstack(parts)


def gen_duplicate_points(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Dataset with exact duplicate points.

    Numerical hazard: zero distances -> division by zero in proximity/linkage.
    """
    dup_ratio = cfg.params.get("dup_ratio", 0.3)
    n_unique = int(cfg.n_samples * (1 - dup_ratio))
    n_dup = cfg.n_samples - n_unique

    parts = []
    for _i in range(cfg.n_clusters):
        center = rng.uniform(-10, 10, size=cfg.n_dims)
        n_per = n_unique // cfg.n_clusters
        pts = rng.normal(loc=center, scale=1.0, size=(n_per, cfg.n_dims))
        parts.append(pts)

    unique = np.vstack(parts)
    # Duplicate random points
    dup_idx = rng.choice(len(unique), size=n_dup, replace=True)
    return np.vstack([unique, unique[dup_idx]])


def gen_extreme_scale(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Data at extreme scales (very small or very large).

    Numerical hazard: float precision loss in angle rounding.
    """
    scale = cfg.params.get("scale", 1e-8)
    parts = []
    for i in range(cfg.n_clusters):
        center = np.full(cfg.n_dims, i * 5.0) * scale
        n_per = cfg.n_samples // cfg.n_clusters
        pts = rng.normal(loc=center, scale=scale * 0.5, size=(n_per, cfg.n_dims))
        parts.append(pts)
    return np.vstack(parts)


def gen_collinear_clusters(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Clusters along a line.

    Numerical hazard: degenerate angles -> norm products near zero.
    """
    parts = []
    for i in range(cfg.n_clusters):
        center = np.zeros(cfg.n_dims)
        center[0] = i * 5.0
        n_per = cfg.n_samples // cfg.n_clusters
        pts = rng.normal(loc=center, scale=0.3, size=(n_per, cfg.n_dims))
        parts.append(pts)
    return np.vstack(parts)


def gen_singleton_merges(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Mostly singletons with a few clusters.

    Numerical hazard: empty merge_history -> fallback paths in proximity, threshold.
    """
    n_clustered = cfg.n_samples // 2
    n_scattered = cfg.n_samples - n_clustered
    parts = []
    for _i in range(cfg.n_clusters):
        center = rng.uniform(-5, 5, size=cfg.n_dims)
        n_per = n_clustered // cfg.n_clusters
        pts = rng.normal(loc=center, scale=0.5, size=(n_per, cfg.n_dims))
        parts.append(pts)
    # Scattered singletons
    scattered = rng.uniform(-20, 20, size=(n_scattered, cfg.n_dims))
    parts.append(scattered)
    return np.vstack(parts)


def gen_unequal_mass(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Highly unequal cluster sizes (95%/5% split).

    Numerical hazard: mass_smoothness near zero -> division guard in continuity.
    """
    large_ratio = cfg.params.get("large_ratio", 0.95)
    n_large = int(cfg.n_samples * large_ratio)
    n_small = cfg.n_samples - n_large

    center_large = np.zeros(cfg.n_dims)
    center_small = np.full(cfg.n_dims, 8.0)

    large = rng.normal(loc=center_large, scale=1.0, size=(n_large, cfg.n_dims))
    small = rng.normal(loc=center_small, scale=0.3, size=(max(n_small, 2), cfg.n_dims))
    return np.vstack([large, small])


def gen_high_dimensional(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Higher-dimensional data (3D/5D/10D).

    Numerical hazard: angle computation in higher dims.
    """
    parts = []
    for _i in range(cfg.n_clusters):
        center = rng.uniform(-5, 5, size=cfg.n_dims)
        n_per = cfg.n_samples // cfg.n_clusters
        pts = rng.normal(loc=center, scale=1.0, size=(n_per, cfg.n_dims))
        parts.append(pts)
    return np.vstack(parts)


def gen_well_separated_blobs(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Control group: trivially separable well-separated blobs."""
    parts = []
    for _i in range(cfg.n_clusters):
        center = rng.uniform(-20, 20, size=cfg.n_dims)
        n_per = cfg.n_samples // cfg.n_clusters
        pts = rng.normal(loc=center, scale=0.5, size=(n_per, cfg.n_dims))
        parts.append(pts)
    return np.vstack(parts)


def gen_spirals(rng: np.random.Generator, cfg: TestConfig) -> np.ndarray:
    """Spiral clusters (non-convex).

    The shape that broke during rewrite — important regression test.
    """
    n_per = cfg.n_samples // cfg.n_clusters
    parts = []
    for k in range(cfg.n_clusters):
        theta = np.linspace(0, 4 * np.pi, n_per) + k * 2 * np.pi / cfg.n_clusters
        r = np.linspace(0.5, 5, n_per)
        x = r * np.cos(theta) + rng.normal(0, 0.15, n_per)
        y = r * np.sin(theta) + rng.normal(0, 0.15, n_per)
        if cfg.n_dims == 2:
            pts = np.column_stack([x, y])
        else:
            extra = rng.normal(0, 0.1, (n_per, cfg.n_dims - 2))
            pts = np.column_stack([x, y, extra])
        parts.append(pts)
    return np.vstack(parts)


GENERATORS = {
    "near_zero_separation": (gen_near_zero_separation, 0.15),
    "duplicate_points": (gen_duplicate_points, 0.10),
    "extreme_scale": (gen_extreme_scale, 0.10),
    "collinear_clusters": (gen_collinear_clusters, 0.10),
    "singleton_merges": (gen_singleton_merges, 0.10),
    "unequal_mass": (gen_unequal_mass, 0.10),
    "high_dimensional": (gen_high_dimensional, 0.10),
    "well_separated_blobs": (gen_well_separated_blobs, 0.15),
    "spirals": (gen_spirals, 0.10),
}


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


def generate_configs(n_configs: int, base_seed: int) -> list[TestConfig]:
    """Generate a diverse set of test configurations."""
    rng = np.random.default_rng(base_seed)
    configs: list[TestConfig] = []

    # Build weighted generator list
    gen_names = []
    gen_weights = []
    for name, (_, weight) in GENERATORS.items():
        gen_names.append(name)
        gen_weights.append(weight)

    # Normalize weights
    total = sum(gen_weights)
    gen_weights = [w / total for w in gen_weights]

    for i in range(n_configs):
        gen_name = rng.choice(gen_names, p=gen_weights)
        seed = int(rng.integers(0, 2**31))

        # Vary parameters
        n_samples = int(rng.choice([30, 50, 75, 100, 150, 200]))
        n_clusters = int(rng.choice([2, 3, 4, 5]))
        n_dims = 2  # Default

        params: dict = {}

        if gen_name == "near_zero_separation":
            params["separation"] = float(rng.choice([1e-4, 1e-6, 1e-8]))
        elif gen_name == "duplicate_points":
            params["dup_ratio"] = float(rng.choice([0.1, 0.3, 0.5]))
        elif gen_name == "extreme_scale":
            params["scale"] = float(rng.choice([1e-8, 1e-6, 1e6, 1e8]))
        elif gen_name == "unequal_mass":
            params["large_ratio"] = float(rng.choice([0.9, 0.95, 0.99]))
        elif gen_name == "high_dimensional":
            n_dims = int(rng.choice([3, 5, 10]))
        elif gen_name == "spirals":
            n_clusters = int(rng.choice([2, 3]))

        configs.append(
            TestConfig(
                config_id=i,
                generator=gen_name,
                n_samples=n_samples,
                n_clusters=n_clusters,
                n_dims=n_dims,
                seed=seed,
                params=params,
            )
        )

    return configs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_legacy(X: np.ndarray) -> np.ndarray:
    """Run legacy Perception, return labels."""
    from perception import Perception

    p = Perception(k=None)
    with redirect_stdout(io.StringIO()):
        labels_raw, _ = p.fit(X.copy())
    return np.asarray(labels_raw, dtype=int)


def _run_new(X: np.ndarray) -> np.ndarray:
    """Run GaugingDelta, return labels."""
    from gauging_delta import GaugingDelta

    model = GaugingDelta(preserve_labels=True)
    model.fit(X.copy())
    return np.asarray(model.labels_, dtype=int)


def run_single_config(cfg: TestConfig, deep: bool = False) -> TestResult:
    """Run both implementations on a single config and compare."""
    rng = np.random.default_rng(cfg.seed)
    gen_func = GENERATORS[cfg.generator][0]

    try:
        X = gen_func(rng, cfg)
    except Exception as e:
        return TestResult(
            config=cfg, ari=0.0, legacy_time_s=0.0, new_time_s=0.0,
            passed=False, error=f"Generator error: {e}",
        )

    actual_n = len(X)

    # Run legacy
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t0 = time.perf_counter()
            l_labels = _run_legacy(X)
            l_time = time.perf_counter() - t0
    except Exception as e:
        return TestResult(
            config=cfg, ari=0.0, legacy_time_s=0.0, new_time_s=0.0,
            passed=False, actual_n=actual_n, error=f"Legacy error: {e}",
        )

    # Run new
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t0 = time.perf_counter()
            n_labels = _run_new(X)
            n_time = time.perf_counter() - t0
    except Exception as e:
        return TestResult(
            config=cfg, ari=0.0, legacy_time_s=l_time, new_time_s=0.0,
            passed=False, actual_n=actual_n, error=f"New error: {e}",
        )

    ari = adjusted_rand_score(l_labels, n_labels)
    passed = ari >= 1.0 - 1e-9

    result = TestResult(
        config=cfg, ari=ari, legacy_time_s=l_time, new_time_s=n_time,
        passed=passed, actual_n=actual_n,
    )

    # If failed or deep mode, run component-level comparison
    if not passed or deep:
        try:
            import warnings

            from benchmarks.benchmark_parity import (
                _compare_components,
                _run_legacy_component,
                _run_new_component,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, l_logs = _run_legacy_component(X)
                _, n_logs = _run_new_component(X)

            divs = _compare_components(l_logs, n_logs, f"stress_{cfg.config_id}")
            result.n_divergences = len(divs)
            if divs:
                result.first_divergence = asdict(divs[0])
        except Exception as e:
            result.first_divergence = {"error": str(e)}

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[TestResult]) -> None:
    """Print statistical summary to console."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error is None)
    errors = sum(1 for r in results if r.error is not None)

    aris = [r.ari for r in results if r.error is None]

    print(f"\n{'=' * 70}")
    print("Stress Test Summary")
    print(f"{'=' * 70}")
    print(f"  Total configs:  {total}")
    print(f"  Passed (ARI=1): {passed} ({100 * passed / total:.1f}%)")
    print(f"  Failed:         {failed}")
    print(f"  Errors:         {errors}")

    if aris:
        print("\n  ARI distribution:")
        print(f"    min:    {min(aris):.6f}")
        print(f"    mean:   {np.mean(aris):.6f}")
        print(f"    median: {np.median(aris):.6f}")
        print(f"    max:    {max(aris):.6f}")

    # Breakdown by generator
    print(f"\n  {'Generator':<25s} {'Count':>6s} {'Pass':>6s} {'Fail':>6s} {'Err':>5s} {'ARI mean':>10s}")
    print(f"  {'-' * 58}")
    gen_groups: dict[str, list[TestResult]] = {}
    for r in results:
        gen_groups.setdefault(r.config.generator, []).append(r)
    for gen_name in sorted(gen_groups.keys()):
        grp = gen_groups[gen_name]
        n_pass = sum(1 for r in grp if r.passed)
        n_fail = sum(1 for r in grp if not r.passed and r.error is None)
        n_err = sum(1 for r in grp if r.error is not None)
        grp_aris = [r.ari for r in grp if r.error is None]
        mean_ari = float(np.mean(grp_aris)) if grp_aris else 0.0
        print(f"  {gen_name:<25s} {len(grp):>6d} {n_pass:>6d} {n_fail:>6d} {n_err:>5d} {mean_ari:>10.6f}")

    # List failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("\n  Failures:")
        for r in failures[:10]:
            if r.error:
                print(f"    #{r.config.config_id} [{r.config.generator}] ERROR: {r.error[:60]}")
            else:
                print(f"    #{r.config.config_id} [{r.config.generator}] ARI={r.ari:.6f} "
                      f"N={r.config.n_samples} seed={r.config.seed}")
                if r.first_divergence:
                    fd = r.first_divergence
                    print(f"      First divergence: {fd.get('divergent_field', '?')} "
                          f"legacy={fd.get('legacy_value', '?')} new={fd.get('new_value', '?')}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_stress_summary(results: list[TestResult], out_dir: Path) -> None:
    """ARI distribution + timing by generator."""
    use_science_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ARI distribution
    ax = axes[0]
    aris = [r.ari for r in results if r.error is None]
    if aris:
        ax.hist(aris, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.5)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="Perfect parity")
        ax.set_xlabel("ARI")
        ax.set_ylabel("Count")
        ax.set_title("ARI Distribution")
        ax.legend(fontsize=7)

    # Timing by generator
    ax = axes[1]
    gen_groups: dict[str, list[float]] = {}
    for r in results:
        if r.error is None:
            gen_groups.setdefault(r.config.generator, []).append(r.new_time_s)

    if gen_groups:
        names = sorted(gen_groups.keys())
        positions = range(len(names))
        data = [gen_groups[n] for n in names]
        bp = ax.boxplot(data, positions=list(positions), vert=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([n[:12] for n in names], fontsize=5, rotation=45, ha="right")
        ax.set_ylabel("New Time (s)")
        ax.set_title("Timing by Generator")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"stress_test_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved stress_test_summary.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Gauging-delta Numerical Stress Test for Parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-configs", type=int, default=100,
        help="Number of random test configurations (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for config generation (default: 42)",
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Run component-level comparison on ALL configs (not just failures)",
    )
    parser.add_argument(
        "--reproduce", type=int, default=None,
        help="Re-run a specific config ID from a previous run",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1, max: 4)",
    )
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 4))

    print(f"\n{'=' * 70}")
    print("Gauging-delta Numerical Stress Test")
    print(f"Configs: {args.n_configs}  Seed: {args.seed}  Deep: {args.deep}  Workers: {args.workers}")
    print(f"{'=' * 70}\n")

    # Generate configs
    configs = generate_configs(args.n_configs, args.seed)

    # Reproduce mode
    if args.reproduce is not None:
        cfg = next((c for c in configs if c.config_id == args.reproduce), None)
        if cfg is None:
            print(f"Config #{args.reproduce} not found in generated configs")
            return
        configs = [cfg]
        print(f"Reproducing config #{args.reproduce}: {cfg.generator} "
              f"N={cfg.n_samples} seed={cfg.seed}")

    # Run all configs
    results: list[TestResult] = []
    t_start = time.perf_counter()

    if args.workers > 1:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_single_config, cfg, args.deep): cfg
                for cfg in configs
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                results.append(result)

                status = "PASS" if result.passed else ("ERR" if result.error else "FAIL")
                ari_str = f"ARI={result.ari:.4f}" if result.error is None else result.error[:30]
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / (i + 1)) * (len(configs) - i - 1)
                print(
                    f"  [{i + 1:>4d}/{len(configs)}] {result.config.generator:<25s} "
                    f"N={result.actual_n:>4d} {status:>4s}  {ari_str}  "
                    f"ETA={eta:.0f}s"
                )
        # Re-sort by config_id for deterministic output
        results.sort(key=lambda r: r.config.config_id)
    else:
        # Sequential execution
        for i, cfg in enumerate(configs):
            result = run_single_config(cfg, deep=args.deep)
            results.append(result)

            status = "PASS" if result.passed else ("ERR" if result.error else "FAIL")
            ari_str = f"ARI={result.ari:.4f}" if result.error is None else result.error[:30]
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / (i + 1)) * (len(configs) - i - 1)
            print(
                f"  [{i + 1:>4d}/{len(configs)}] {cfg.generator:<25s} "
                f"N={result.actual_n:>4d} {status:>4s}  {ari_str}  "
                f"ETA={eta:.0f}s"
            )

    # Print summary
    print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "stress_test.json"
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "n_configs": args.n_configs,
        "seed": args.seed,
        "deep": args.deep,
        "results": [
            {
                "config": asdict(r.config),
                "actual_n": r.actual_n,
                "ari": r.ari,
                "legacy_time_s": r.legacy_time_s,
                "new_time_s": r.new_time_s,
                "passed": r.passed,
                "n_divergences": r.n_divergences,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved results to {json_path}")

    # Save divergences
    failures = [r for r in results if not r.passed]
    if failures:
        div_path = RESULTS_DIR / "stress_divergences.json"
        div_data = []
        for r in failures:
            entry = {
                "config": asdict(r.config),
                "ari": r.ari,
                "error": r.error,
                "first_divergence": r.first_divergence,
            }
            div_data.append(entry)
        with open(div_path, "w") as f:
            json.dump(div_data, f, indent=2)
        print(f"  Saved {len(failures)} failure(s) to {div_path}")

    # Generate plots
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_stress_summary(results, ASSETS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
