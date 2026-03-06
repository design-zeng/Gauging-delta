# src/gauging_delta/

Python package implementing the Gauging-delta non-parametric hierarchical
clustering algorithm (IEEE TPAMI 2025). Provides an sklearn-compatible API with
swappable proximity, continuity, and linkage components.

---

## Module Layout

| Module | Purpose | Paper reference |
|---|---|---|
| `algorithm.py` | Main orchestrator: `fit`, `fit_predict`, `labels_`, `n_clusters_` | Algorithm 1 |
| `config.py` | `GaugingDeltaConfig` — all algorithm constants in one frozen dataclass | Throughout |
| `cluster.py` | `Cluster` dataclass: point indices, center, merge history, stats | — |
| `_types.py` | Protocol definitions: `ProximityMetric`, `ContinuityMetric`, `LinkageMetric`, `ProximityResult` | — |
| `proximity.py` | Default proximity metric ρ = d_ij / μ_historical | Eq. 3 |
| `threshold.py` | Adaptive threshold T = β × T_stat × ξ_s (fixed composition glue, not swappable) | Eq. 4 |
| `continuity.py` | Multi-scale density/angle transition analysis | Section II.C |
| `linkage.py` | Single-linkage cluster distance with legacy tie-breaking | — |
| `angles.py` | Parity-safe angle computation (`compute_angle`, `compute_angle_batch`) | — |
| `__init__.py` | Public re-exports: `GaugingDelta`, `GaugingDeltaConfig`, default components | — |

---

## Algorithm Flow

```
fit(X)
  │
  ├─ _init_distances_full / _init_distances_lite
  │    Build dense distance matrix + per-point sorted distance arrays (full)
  │    or KD-tree over initial cluster centroids (lite)
  │
  ├─ Extract sorted pairs via argsort
  │
  └─ Merge loop  (until no more pairs)
       │
       ├─ Extract closest pair (C_i, C_j)
       │
       ├─ G2: Compute ρ = d_ij / μ_historical  [proximity.py]
       │
       ├─ G3: Compute adaptive threshold T = β × T_stat × ξ_s  [threshold.py]
       │
       ├─ G4: Reject if ρ > T  (no merge)
       │
       ├─ G5: Compute continuity / smoothness  [continuity.py]
       │
       ├─ G6: Reject if smoothness < threshold_continuity  (no merge)
       │
       └─ _do_merge_full / _do_merge_lite
            Merge C_j into C_i, update distances, update incremental min cache

  └─ Build final label assignments
```

---

## Swappable Component Architecture

Three components implement Protocols defined in `_types.py` and can be replaced
with custom implementations:

```python
from gauging_delta import GaugingDelta, GaugingDeltaConfig

gd = GaugingDelta(
    proximity=CustomProximity(),    # ProximityMetric Protocol
    continuity=CustomContinuity(),  # ContinuityMetric Protocol
    config=GaugingDeltaConfig(threshold_continuity=0.2),
)
gd.fit(X)
```

| Protocol | Method signature | Default implementation |
|---|---|---|
| `ProximityMetric` | `compute(C_i, C_j, d_ij, fallback) -> ProximityResult` | `DefaultProximity` |
| `ContinuityMetric` | `compute(C_i, C_j, X, config) -> float` | `DefaultContinuity` |
| `LinkageMetric` | `compute(C_i, C_j, dist_matrix) -> float` | `DefaultLinkage` (full mode only) |

`threshold.py` is **not** swappable — it implements fixed composition logic that
must stay aligned with the paper's Eq. 4 for parity guarantees.

---

## Full vs Lite Mode

`GaugingDelta(mode="full")` (default) and `GaugingDelta(mode="lite")` differ in
linkage and continuity:

| Aspect | `full` | `lite` |
|---|---|---|
| Linkage | Single-linkage via `_near_ref` (N×N point-ref array) | Centroid distance via brute-force on `_lite_centers` |
| Continuity | Multi-scale angle/density analysis (Section II.C) | Skipped |
| Persistent arrays | `_dist_matrix`, `_near_ref`, `_pd_indices`, `_pd_dists`, `_row_mins`, `_row_argmins` | `_lite_centers` (K×D), `_lite_ids`, `_lite_id_to_pos` |
| Memory complexity | O(N²) | O(N·D) |
| Quality | ARI = 1.000 vs legacy on all 6 datasets | Lower — tends to under-merge |
| Suitable for | Exact reproduction + general use | Large-scale exploratory clustering (N up to millions) |

Each of the three core methods dispatches to mode-specific implementations:

```python
_init_distances  →  _init_distances_full  /  _init_distances_lite
_try_merge       →  _try_merge_full       /  _try_merge_lite
_do_merge        →  _do_merge_full        /  _do_merge_lite
```

---

## Critical Parity Details

The following must be preserved exactly to maintain ARI = 1.000 vs `perception.py`:

1. **Angle rounding** — `round(norm_prod, 4)` is used only for the zero check;
   the actual division uses the unrounded product.
2. **i-increment** — the sorted-pair index advances only when the pair distance
   is `inf` or the last-merged-cluster heuristic does not fire, never on merge
   failure.
3. **Mass ratio division** — no guard on `mass[i] / mass[i-1]`; numpy `inf`
   propagation is intentional and triggers the early break.
4. **Proximity division** — numpy semantics for `nan`/`inf` on zero denominator,
   not Python `ZeroDivisionError`.
5. **Merge history order** — `lead.past_dists.append(d_ij)` before
   `lead.past_dists.extend(child.past_dists)`.
