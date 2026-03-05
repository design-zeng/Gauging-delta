"""
Gate-by-gate parity verification with borrow-and-continue.

Runs legacy and refactored implementations side-by-side, comparing intermediate
gate outputs at each merge attempt.  When a gate diverges, the borrow mode
injects the legacy gate's output into the refactored pipeline to isolate root
causes from cascading failures.

Gates per merge attempt:
  G2: proximity   → rho, lead_id, child_id
  G3: threshold   → T_i, T_j, xi_s, vision_scale, adp_prox
  G4: rejection   → rejected (bool)
  G5: continuity  → smoothness
  G6: accept      → merged (bool)

Usage:
    uv run python benchmarks/gate_parity.py                          # all 6 datasets
    uv run python benchmarks/gate_parity.py --dataset flame          # single dataset
    uv run python benchmarks/gate_parity.py --borrow 2               # borrow at gate 2
    uv run python benchmarks/gate_parity.py --stress-config 109      # reproduce stress config
    uv run python benchmarks/gate_parity.py --quick                  # stop at first divergence
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GateValues:
    """Gate outputs for one merge attempt, one implementation."""

    gate: int
    values: dict[str, Any]


@dataclass
class GateDivergence:
    """A specific divergence between legacy and new at one gate."""

    attempt: int
    gate: int
    gate_name: str
    field: str
    legacy_value: Any
    new_value: Any
    delta: float | None = None


@dataclass
class AttemptRecord:
    """Gate values from both sides for one merge attempt."""

    idx: int
    c1: int
    c2: int
    legacy_gates: dict[int, dict[str, Any]] = field(default_factory=dict)
    new_gates: dict[int, dict[str, Any]] = field(default_factory=dict)
    divergences: list[GateDivergence] = field(default_factory=list)


@dataclass
class GateParityReport:
    """Full report for a dataset."""

    dataset: str
    n_points: int
    n_attempts: int
    first_divergence: GateDivergence | None
    all_divergences: list[GateDivergence]
    borrow_gate: int | None
    borrow_downstream_divergences: list[GateDivergence]


# ---------------------------------------------------------------------------
# Gate names and comparison tolerances
# ---------------------------------------------------------------------------

GATE_NAMES = {
    2: "proximity",
    3: "threshold",
    4: "rejection",
    5: "continuity",
    6: "accept",
}

GATE_TOLERANCES: dict[str, float] = {
    "rho": 1e-10,
    "T_i": 1e-10,
    "T_j": 1e-10,
    "xi_s": 1e-10,
    "adp_prox": 1e-10,
    "smoothness": 1e-10,
    "vision_scale": 1e-10,
}


def _values_match(field_name: str, legacy_val: Any, new_val: Any) -> bool:
    """Compare two gate values with appropriate tolerance."""
    if isinstance(legacy_val, bool) or isinstance(new_val, bool):
        return legacy_val == new_val
    if isinstance(legacy_val, (int, np.integer)) and isinstance(new_val, (int, np.integer)):
        return int(legacy_val) == int(new_val)
    if isinstance(legacy_val, float) and isinstance(new_val, float):
        # Handle nan/inf
        if math.isnan(legacy_val) and math.isnan(new_val):
            return True
        if math.isinf(legacy_val) and math.isinf(new_val):
            return (legacy_val > 0) == (new_val > 0)
        tol = GATE_TOLERANCES.get(field_name, 1e-10)
        return abs(legacy_val - new_val) <= tol
    return legacy_val == new_val


# ---------------------------------------------------------------------------
# Legacy gate extraction via monkey-patching
# ---------------------------------------------------------------------------


def _run_legacy_with_gates(X: np.ndarray) -> tuple[np.ndarray, list[AttemptRecord]]:
    """Run legacy Perception, capturing per-gate values at each merge attempt."""
    from perception import Perception

    attempts: list[AttemptRecord] = []
    step = [0]

    # Captured intermediates
    cap_prox: list[tuple] = []
    cap_thresh: list[tuple] = []
    cap_smoothness: list[float] = []

    _orig_prox = Perception.compute_proximity
    _orig_thresh = Perception.compute_adaptive_threshold
    _orig_cont = Perception.compute_continuation
    _orig_vg = Perception.vision_generic

    def _patched_prox(self, c1, c2):
        result = _orig_prox(self, c1, c2)
        cap_prox.clear()
        cap_prox.append(result)  # (rho, distance, lead_id, child_id)
        return result

    def _patched_thresh(self, k, lead_id, child_id, d_ij, rho):
        result = _orig_thresh(self, k, lead_id, child_id, d_ij, rho)
        cap_thresh.clear()
        cap_thresh.append(result)  # (T_i, T_j, xi_s, adp_prox)
        return result

    def _patched_cont(self, c1, c2, threshold_cont, mean_dist, prox_threshold):
        result = _orig_cont(self, c1, c2, threshold_cont, mean_dist, prox_threshold)
        cap_smoothness.clear()
        cap_smoothness.append(result)
        return result

    def _patched_vg(self, c1, c2):
        d_ij = self.clusters_dist[frozenset((c1, c2))]["distance_info"]["near_dist"]["distance"]

        result = _orig_vg(self, c1, c2)
        is_merge = result[0]

        rec = AttemptRecord(idx=step[0], c1=c1, c2=c2)

        # G2: proximity
        if cap_prox:
            rho, dist, lead_id, child_id = cap_prox[0]
            rec.legacy_gates[2] = {
                "rho": float(rho),
                "d_ij": float(dist),
                "lead_id": int(lead_id),
                "child_id": int(child_id),
            }
        else:
            rec.legacy_gates[2] = {"rho": float("nan"), "d_ij": float(d_ij), "lead_id": c1, "child_id": c2}

        # G3: threshold
        if cap_thresh:
            T_i, T_j, xi_s, adp_prox = cap_thresh[0]
            rec.legacy_gates[3] = {
                "T_i": float(T_i),
                "T_j": float(T_j),
                "xi_s": float(xi_s),
                "adp_prox": float(adp_prox),
            }
        else:
            rec.legacy_gates[3] = {"T_i": 0.0, "T_j": 0.0, "xi_s": 1.0, "adp_prox": 0.0}

        # G4: rejection
        rho_val = rec.legacy_gates[2]["rho"]
        T_i_val = rec.legacy_gates[3]["T_i"]
        T_j_val = rec.legacy_gates[3]["T_j"]
        rejected = rho_val > T_i_val or rho_val > T_j_val
        rec.legacy_gates[4] = {"rejected": rejected}

        # G5: continuity (only if not rejected)
        if not rejected and cap_smoothness:
            rec.legacy_gates[5] = {"smoothness": float(cap_smoothness[0])}
        elif not rejected:
            rec.legacy_gates[5] = {"smoothness": float("nan")}

        # G6: accept
        rec.legacy_gates[6] = {"merged": is_merge}

        attempts.append(rec)
        step[0] += 1
        cap_prox.clear()
        cap_thresh.clear()
        cap_smoothness.clear()
        return result

    Perception.compute_proximity = _patched_prox
    Perception.compute_adaptive_threshold = _patched_thresh
    Perception.compute_continuation = _patched_cont
    Perception.vision_generic = _patched_vg
    try:
        p = Perception(k=None)
        with redirect_stdout(io.StringIO()):
            labels_raw, _ = p.fit(X.copy())
    finally:
        Perception.compute_proximity = _orig_prox
        Perception.compute_adaptive_threshold = _orig_thresh
        Perception.compute_continuation = _orig_cont
        Perception.vision_generic = _orig_vg

    return np.asarray(labels_raw, dtype=int), attempts


# ---------------------------------------------------------------------------
# Refactored gate extraction via monkey-patching
# ---------------------------------------------------------------------------


def _run_new_with_gates(X: np.ndarray) -> tuple[np.ndarray, list[AttemptRecord]]:
    """Run GaugingDelta, capturing per-gate values at each merge attempt."""
    import gauging_delta.algorithm as _alg_mod
    from gauging_delta import GaugingDelta
    from gauging_delta.continuity import DefaultContinuity
    from gauging_delta.proximity import DefaultProximity

    attempts: list[AttemptRecord] = []
    step = [0]

    cap_prox: list = []
    cap_thresh: list = []
    cap_smooth: list = []

    _orig_prox = DefaultProximity.compute
    _orig_thresh = _alg_mod.compute_adaptive_threshold
    _orig_cont = DefaultContinuity.compute
    _orig_tm = GaugingDelta._try_merge

    def _patched_prox(self, C_i, C_j, d_ij, fallback):
        result = _orig_prox(self, C_i, C_j, d_ij, fallback)
        cap_prox.clear()
        cap_prox.append(result)
        return result

    def _patched_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, clusters_dist, cfg):
        result = _orig_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, clusters_dist, cfg)
        cap_thresh.clear()
        cap_thresh.append(result)
        return result

    def _patched_cont(self, lead, child, threshold, d_ij_norm, adp_prox, X_data, point_dists):
        result = _orig_cont(self, lead, child, threshold, d_ij_norm, adp_prox, X_data, point_dists)
        cap_smooth.clear()
        cap_smooth.append(result)
        return result

    def _patched_tm(self, c1, c2):
        d_ij = float(
            self._clusters_dist[frozenset((c1, c2))]["distance_info"]["near_dist"]["distance"]
        )

        result = _orig_tm(self, c1, c2)

        rec = AttemptRecord(idx=step[0], c1=c1, c2=c2)

        # G2: proximity
        if cap_prox:
            p = cap_prox[0]
            rec.new_gates[2] = {
                "rho": float(p.rho),
                "d_ij": float(d_ij),
                "lead_id": int(p.lead_id),
                "child_id": int(p.child_id),
            }
        else:
            rec.new_gates[2] = {"rho": float("nan"), "d_ij": d_ij, "lead_id": c1, "child_id": c2}

        # G3: threshold
        if cap_thresh:
            t = cap_thresh[0]
            rec.new_gates[3] = {
                "T_i": float(t.T_i),
                "T_j": float(t.T_j),
                "xi_s": float(t.xi_s),
                "adp_prox": float(t.adp_prox),
            }
        else:
            rec.new_gates[3] = {"T_i": 0.0, "T_j": 0.0, "xi_s": 1.0, "adp_prox": 0.0}

        # G4: rejection
        rho_val = rec.new_gates[2]["rho"]
        T_i_val = rec.new_gates[3]["T_i"]
        T_j_val = rec.new_gates[3]["T_j"]
        rejected = rho_val > T_i_val or rho_val > T_j_val
        rec.new_gates[4] = {"rejected": rejected}

        # G5: continuity
        if not rejected and cap_smooth:
            rec.new_gates[5] = {"smoothness": float(cap_smooth[0])}
        elif not rejected:
            rec.new_gates[5] = {"smoothness": float("nan")}

        # G6: accept
        rec.new_gates[6] = {"merged": result is not None}

        attempts.append(rec)
        step[0] += 1
        cap_prox.clear()
        cap_thresh.clear()
        cap_smooth.clear()
        return result

    DefaultProximity.compute = _patched_prox
    _alg_mod.compute_adaptive_threshold = _patched_thresh
    DefaultContinuity.compute = _patched_cont
    GaugingDelta._try_merge = _patched_tm
    try:
        model = GaugingDelta(preserve_labels=True)
        model.fit(X.copy())
    finally:
        DefaultProximity.compute = _orig_prox
        _alg_mod.compute_adaptive_threshold = _orig_thresh
        DefaultContinuity.compute = _orig_cont
        GaugingDelta._try_merge = _orig_tm

    return np.asarray(model.labels_, dtype=int), attempts


# ---------------------------------------------------------------------------
# Gate comparison
# ---------------------------------------------------------------------------


def _compare_attempts(
    legacy_attempts: list[AttemptRecord],
    new_attempts: list[AttemptRecord],
    dataset: str,
    *,
    quick: bool = False,
    gate_filter: list[int] | None = None,
) -> list[GateDivergence]:
    """Compare gate outputs attempt-by-attempt. Returns all divergences.

    If *gate_filter* is given, only those gates are compared (e.g. [2, 3]).
    """
    divergences: list[GateDivergence] = []
    gates_to_check = gate_filter if gate_filter else [2, 3, 4, 5, 6]

    n = min(len(legacy_attempts), len(new_attempts))
    if len(legacy_attempts) != len(new_attempts):
        divergences.append(
            GateDivergence(
                attempt=-1,
                gate=-1,
                gate_name="attempt_count",
                field="count",
                legacy_value=len(legacy_attempts),
                new_value=len(new_attempts),
                delta=abs(len(legacy_attempts) - len(new_attempts)),
            )
        )

    for i in range(n):
        la = legacy_attempts[i]
        na = new_attempts[i]

        # Check pair alignment first (always checked — prerequisite for gate comparison)
        if la.c1 != na.c1 or la.c2 != na.c2:
            divergences.append(
                GateDivergence(
                    attempt=i,
                    gate=1,
                    gate_name="pair_select",
                    field="cluster_pair",
                    legacy_value=f"({la.c1}, {la.c2})",
                    new_value=f"({na.c1}, {na.c2})",
                )
            )
            break  # Pairs diverged — cascade from prior merge

        # Compare only requested gates
        for gate_num in gates_to_check:
            lg = la.legacy_gates.get(gate_num, {})
            ng = na.new_gates.get(gate_num, {})

            if not lg or not ng:
                continue

            for fld in lg:
                if fld not in ng:
                    continue
                if not _values_match(fld, lg[fld], ng[fld]):
                    delta = None
                    if isinstance(lg[fld], (int, float)) and isinstance(ng[fld], (int, float)):
                        try:
                            delta = abs(float(lg[fld]) - float(ng[fld]))
                        except (ValueError, OverflowError):
                            pass
                    divergences.append(
                        GateDivergence(
                            attempt=i,
                            gate=gate_num,
                            gate_name=GATE_NAMES.get(gate_num, f"gate{gate_num}"),
                            field=fld,
                            legacy_value=lg[fld],
                            new_value=ng[fld],
                            delta=delta,
                        )
                    )

        if quick and divergences:
            break

    return divergences


# ---------------------------------------------------------------------------
# Borrow-and-continue
# ---------------------------------------------------------------------------


def _run_borrow_and_continue(
    X: np.ndarray,
    legacy_attempts: list[AttemptRecord],
    borrow_gate: int,
    first_div: GateDivergence,
) -> list[GateDivergence]:
    """Re-run refactored code, injecting legacy output at borrow_gate.

    Tests whether downstream gates match when the divergent gate is corrected.
    Returns divergences in gates AFTER borrow_gate.
    """
    import gauging_delta.algorithm as _alg_mod
    from gauging_delta import GaugingDelta
    from gauging_delta.continuity import DefaultContinuity
    from gauging_delta.proximity import DefaultProximity

    attempt_idx = first_div.attempt
    downstream_divs: list[GateDivergence] = []

    # Build lookup of legacy gate values by attempt index
    legacy_by_idx = {a.idx: a for a in legacy_attempts}

    cap_prox: list = []
    cap_thresh: list = []
    cap_smooth: list = []
    step = [0]

    _orig_prox = DefaultProximity.compute
    _orig_thresh = _alg_mod.compute_adaptive_threshold
    _orig_cont = DefaultContinuity.compute
    _orig_tm = GaugingDelta._try_merge

    def _patched_prox(self, C_i, C_j, d_ij, fallback):
        result = _orig_prox(self, C_i, C_j, d_ij, fallback)
        cap_prox.clear()
        cap_prox.append(result)

        # Borrow at gate 2: replace proximity result with legacy values
        if borrow_gate == 2 and step[0] in legacy_by_idx:
            la = legacy_by_idx[step[0]]
            lg2 = la.legacy_gates.get(2, {})
            if lg2:
                # Create a patched result with legacy values
                from gauging_delta.proximity import ProximityResult

                result = ProximityResult(
                    rho=lg2["rho"],
                    distance=lg2["d_ij"],
                    lead_id=lg2["lead_id"],
                    child_id=lg2["child_id"],
                )
                cap_prox[-1] = result
        return result

    def _patched_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, clusters_dist, cfg):
        result = _orig_thresh(lead, child, d_ij, rho, all_clusters, dist_matrix, clusters_dist, cfg)
        cap_thresh.clear()
        cap_thresh.append(result)

        # Borrow at gate 3: replace threshold result with legacy values
        if borrow_gate == 3 and step[0] in legacy_by_idx:
            la = legacy_by_idx[step[0]]
            lg3 = la.legacy_gates.get(3, {})
            if lg3:
                from gauging_delta.threshold import ThresholdResult

                result = ThresholdResult(
                    T_i=lg3["T_i"],
                    T_j=lg3["T_j"],
                    xi_s=lg3["xi_s"],
                    adp_prox=lg3["adp_prox"],
                )
                cap_thresh[-1] = result
        return result

    def _patched_cont(self, lead, child, threshold, d_ij_norm, adp_prox, X_data, point_dists):
        result = _orig_cont(self, lead, child, threshold, d_ij_norm, adp_prox, X_data, point_dists)
        cap_smooth.clear()
        cap_smooth.append(result)

        # Borrow at gate 5: replace smoothness with legacy value
        if borrow_gate == 5 and step[0] in legacy_by_idx:
            la = legacy_by_idx[step[0]]
            lg5 = la.legacy_gates.get(5, {})
            if lg5 and "smoothness" in lg5:
                result = lg5["smoothness"]
                cap_smooth[-1] = result
        return result

    def _patched_tm(self, c1, c2):
        d_ij = float(
            self._clusters_dist[frozenset((c1, c2))]["distance_info"]["near_dist"]["distance"]
        )
        result = _orig_tm(self, c1, c2)

        # Capture and compare downstream gates for this attempt
        if step[0] in legacy_by_idx:
            la = legacy_by_idx[step[0]]
            for gate_num in [2, 3, 4, 5, 6]:
                if gate_num <= borrow_gate:
                    continue  # Skip gates at or before borrow point
                lg = la.legacy_gates.get(gate_num, {})
                ng: dict[str, Any] = {}
                if gate_num == 2 and cap_prox:
                    p = cap_prox[0]
                    ng = {"rho": float(p.rho), "d_ij": d_ij, "lead_id": int(p.lead_id), "child_id": int(p.child_id)}
                elif gate_num == 3 and cap_thresh:
                    t = cap_thresh[0]
                    ng = {"T_i": float(t.T_i), "T_j": float(t.T_j), "xi_s": float(t.xi_s), "adp_prox": float(t.adp_prox)}
                elif gate_num == 4:
                    rho_v = float(cap_prox[0].rho) if cap_prox else float("nan")
                    ti = float(cap_thresh[0].T_i) if cap_thresh else 0.0
                    tj = float(cap_thresh[0].T_j) if cap_thresh else 0.0
                    ng = {"rejected": rho_v > ti or rho_v > tj}
                elif gate_num == 5 and cap_smooth:
                    ng = {"smoothness": float(cap_smooth[0])}
                elif gate_num == 6:
                    ng = {"merged": result is not None}

                if not lg or not ng:
                    continue
                for fld in lg:
                    if fld in ng and not _values_match(fld, lg[fld], ng[fld]):
                        delta = None
                        if isinstance(lg[fld], (int, float)) and isinstance(ng[fld], (int, float)):
                            try:
                                delta = abs(float(lg[fld]) - float(ng[fld]))
                            except (ValueError, OverflowError):
                                pass
                        downstream_divs.append(
                            GateDivergence(
                                attempt=step[0],
                                gate=gate_num,
                                gate_name=GATE_NAMES.get(gate_num, f"gate{gate_num}"),
                                field=fld,
                                legacy_value=lg[fld],
                                new_value=ng[fld],
                                delta=delta,
                            )
                        )

        step[0] += 1
        cap_prox.clear()
        cap_thresh.clear()
        cap_smooth.clear()
        return result

    DefaultProximity.compute = _patched_prox
    _alg_mod.compute_adaptive_threshold = _patched_thresh
    DefaultContinuity.compute = _patched_cont
    GaugingDelta._try_merge = _patched_tm
    try:
        model = GaugingDelta(preserve_labels=True)
        model.fit(X.copy())
    finally:
        DefaultProximity.compute = _orig_prox
        _alg_mod.compute_adaptive_threshold = _orig_thresh
        DefaultContinuity.compute = _orig_cont
        GaugingDelta._try_merge = _orig_tm

    return downstream_divs


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_gate_parity(
    X: np.ndarray,
    dataset_name: str,
    *,
    quick: bool = False,
    borrow_gate: int | None = None,
    gate_filter: list[int] | None = None,
) -> GateParityReport:
    """Run full gate-parity comparison on a dataset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        legacy_labels, legacy_attempts = _run_legacy_with_gates(X)
        new_labels, new_attempts = _run_new_with_gates(X)

    divergences = _compare_attempts(
        legacy_attempts, new_attempts, dataset_name, quick=quick, gate_filter=gate_filter
    )

    first_div = divergences[0] if divergences else None

    # Borrow-and-continue if requested and there's a divergence
    borrow_divs: list[GateDivergence] = []
    actual_borrow = borrow_gate
    if first_div and borrow_gate is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            borrow_divs = _run_borrow_and_continue(X, legacy_attempts, borrow_gate, first_div)
    elif first_div and borrow_gate is None:
        # Auto-borrow at the first divergent gate
        actual_borrow = first_div.gate
        if actual_borrow in (2, 3, 5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                borrow_divs = _run_borrow_and_continue(
                    X, legacy_attempts, actual_borrow, first_div
                )

    return GateParityReport(
        dataset=dataset_name,
        n_points=len(X),
        n_attempts=min(len(legacy_attempts), len(new_attempts)),
        first_divergence=first_div,
        all_divergences=divergences,
        borrow_gate=actual_borrow,
        borrow_downstream_divergences=borrow_divs,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def _load_stress_config(config_id: int) -> np.ndarray:
    """Reproduce a stress test config by ID."""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from stress_test import GENERATORS, generate_configs

    configs = generate_configs(200, base_seed=42)
    for cfg in configs:
        if cfg.config_id == config_id:
            gen_fn, _weight = GENERATORS[cfg.generator]
            rng = np.random.default_rng(cfg.seed)
            return gen_fn(rng, cfg)
    raise ValueError(f"Config ID {config_id} not found in 200 configs (seed=42)")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(report: GateParityReport) -> None:
    """Print human-readable gate parity report."""
    header = f"Gate Parity — {report.dataset} (N={report.n_points}, {report.n_attempts} attempts)"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")

    if not report.all_divergences:
        print("  PASS — all gates match")
        return

    print(f"  {len(report.all_divergences)} divergence(s) found\n")

    if report.first_divergence:
        d = report.first_divergence
        print(f"  First divergence:")
        print(f"    Attempt #{d.attempt}, Gate {d.gate} ({d.gate_name})")
        print(f"    Field:   {d.field}")
        print(f"    Legacy:  {d.legacy_value}")
        print(f"    New:     {d.new_value}")
        if d.delta is not None:
            print(f"    Delta:   {d.delta}")

    # Show up to 5 divergences
    if len(report.all_divergences) > 1:
        print(f"\n  All divergences (first 10):")
        for d in report.all_divergences[:10]:
            print(f"    #{d.attempt} G{d.gate}({d.gate_name}).{d.field}: "
                  f"legacy={d.legacy_value} new={d.new_value}")
        if len(report.all_divergences) > 10:
            print(f"    ... and {len(report.all_divergences) - 10} more")

    if report.borrow_gate is not None:
        print(f"\n  Borrow-and-continue at Gate {report.borrow_gate}:")
        if not report.borrow_downstream_divergences:
            print(f"    Downstream gates PASS — root cause is Gate {report.borrow_gate} only")
        else:
            print(f"    {len(report.borrow_downstream_divergences)} downstream divergence(s):")
            for d in report.borrow_downstream_divergences[:5]:
                print(f"      #{d.attempt} G{d.gate}({d.gate_name}).{d.field}: "
                      f"legacy={d.legacy_value} new={d.new_value}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate-by-gate parity verification")
    parser.add_argument("--dataset", type=str, help="Single dataset name (e.g. flame)")
    parser.add_argument("--stress-config", type=int, help="Stress test config ID to reproduce")
    parser.add_argument("--borrow", type=int, help="Gate number to borrow from legacy (2, 3, or 5)")
    parser.add_argument("--quick", action="store_true", help="Stop at first divergence")
    parser.add_argument(
        "--gates",
        type=str,
        default=None,
        help="Comma-separated gate numbers to compare (e.g. '2,3'). Default: all gates.",
    )
    args = parser.parse_args()

    print("Gate-by-Gate Parity Verification")
    print("=" * 40)

    sources: list[tuple[str, np.ndarray]] = []

    if args.stress_config is not None:
        X = _load_stress_config(args.stress_config)
        sources.append((f"stress_{args.stress_config}", X))
    elif args.dataset:
        path = BENCHMARK_DATASETS.get(args.dataset)
        if path and path.exists():
            sources.append((args.dataset, _load_dataset(path)))
        else:
            print(f"Dataset '{args.dataset}' not found")
            sys.exit(1)
    else:
        for name, path in BENCHMARK_DATASETS.items():
            if path.exists():
                sources.append((name, _load_dataset(path)))

    gate_filter = None
    if args.gates:
        gate_filter = [int(g.strip()) for g in args.gates.split(",")]

    all_pass = True
    for name, X in sources:
        report = run_gate_parity(
            X, name, quick=args.quick, borrow_gate=args.borrow, gate_filter=gate_filter
        )
        _print_report(report)
        if report.all_divergences:
            all_pass = False

    print(f"\n{'=' * 40}")
    if all_pass:
        print("ALL DATASETS PASS — perfect gate parity")
    else:
        print("DIVERGENCES FOUND — see details above")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
