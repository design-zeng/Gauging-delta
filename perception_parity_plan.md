# Perception Parity Plan

## Goal
Ensure the refactored GaugingDelta implementation matches the legacy Perception output
with **exact label match** and **merge-by-merge decision parity** on small datasets.

---

## 1. Data Flow Mapping (Legacy -> Refactor)

### 1) Initialization
- Legacy: Perception.fit builds DIST_MATRIX, FAIL_MATRIX, and initial_clusters for each
  point, then calls initiate_dists to populate distance structures and MIN_BTN_CLUSTER_DIST.
- Refactor: GaugingDelta.fit initializes NeighborGraph, computes fallback distance from
  pairwise point distances, and creates Cluster objects.

### 2) Distance Matrix / Pair Distances
- Legacy: initiate_dists computes all pairwise near distances (closest points) and
  fills DIST_MATRIX; MIN_BTN_CLUSTER_DIST is 1st percentile of those distances.
- Refactor: NeighborGraph stores sparse edges for performance, but _get_sorted_cluster_pairs
  reconstructs a dense distance matrix when needed; fallback distance stored as _fallback_dist.

### 3) Pair Selection & Ordering
- Legacy: get_indices_of_k_smallest uses argpartition on the full distance matrix with
  k = X.shape[0], then iterates only those k pairs each round.
- Refactor: _get_sorted_cluster_pairs computes a dense matrix for all active clusters
  and returns all pairs sorted by distance; duplicate pairs are de-duplicated.

### 4) Mergeability (Proximity + Threshold + Continuity)
- Legacy: vision_generic
  - compute_proximity -> rho, lead/child
  - compute_adaptive_threshold -> threshold1/2, shape_diff, adp_prox (vision_scale)
  - compute_continuation(lead, child, T_continuity/shape_diff, distance/proximity, adp_prox)
- Refactor: _compute_mergeability
  - compute_proximity_rho -> rho, lead/child
  - compute_beta_ij + compute_xi_s -> beta_ij, xi_s
  - compute_adaptive_threshold_T -> T_i, T_j
  - compute_continuity(lead, child, T_continuity/shape_diff, d_ij, T_adaptive)

### 5) Merge Decision
- Legacy: merge if rho <= T1 and rho <= T2, then if continuity > T_continuity/shape_diff.
- Refactor: same logic using rho, T_i/T_j, and continuity > T_continuity/shape_diff.

### 6) Cluster Update
- Legacy: merge_2_cluster updates traces, data, center, mean/std, past_std, past_dists,
  then update_clusters recomputes distances in DIST_MATRIX and updates MIN_BTN_CLUSTER_DIST.
- Refactor: _merge_cluster_data updates point_indices, center, mu/sigma, sigma_history,
  merge_history/merging_dists, then NeighborGraph merge + _update_min_cluster_dist.

### 7) Output Labels
- Legacy: labels are the surviving cluster IDs (original point indices).
- Refactor: _build_labels re-labels clusters to contiguous integers based on sorted IDs.

---

## 2. Parity Risks / Differences to Watch

1) **Pair ordering scope**: legacy uses only the smallest N pairs each round (k = N),
   while refactor sorts all valid pairs and de-duplicates. This can change merge order
   or stop conditions.

2) **Label identity**: refactor re-labels clusters to contiguous IDs. This will not
   match legacy labels exactly even if clustering is identical.

3) **Vision scale / beta_ij**:
   - Legacy uses nearest neighbors from both clusters and their intersection.
   - Refactor uses only C_i neighbors; distances to C_j are approximated.

4) **Force distance**: legacy compute_force_btn_clusters uses mix_dist (average of near
   and center distances). Refactor uses near distance for compute_F_ij.

5) **Shape similarity (xi_s)**: refactor uses recent-vs-historical sigma ratios per
   cluster, while legacy shape_diff compares cross-cluster std ratios.

6) **Merge traces & density history**: legacy stores traces (p1, p2) and past_densities
   on merge; refactor currently does not add new merge edges or density_history entries.

---

## 3. Test Suite Audit & Minimal Parity Suite

### Existing tests
- test_parity.py: unit-level sanity checks (angles/proximity/threshold), plus basic
  clustering validity without legacy comparison.
- test_pipeline_parity.py: step-based parity checks but most assertions are “positive
  or in range,” not strict equality; no full merge sequence parity.
- test_exact_parity.py: property-based tests with ARI=1.0 (not exact label match),
  plus several benchmark datasets (still large for legacy).

### Gaps
- No **exact label equality** checks (ARI allows label permutations).
- No **merge-by-merge parity** tests (merge sequence + per-merge stats).
- Beta/xi_s tests are sanity-only (not against legacy).

### Proposed minimal parity suite (small datasets)
1) **Deterministic micro datasets** (<= 20 points):
   - 6-point “two blobs” (simple_data)
   - 2-point close/far
   - triangle / square / short line
2) **Merge sequence parity**:
   - Record merge list (lead_id, child_id, d_ij, rho, T_i/T_j, continuity).
   - Compare sequences exactly between legacy and refactor.
3) **Exact label parity**:
   - Use np.array_equal on label arrays after ensuring refactor preserves legacy IDs
     (or add a “preserve_labels” mode).
4) **Performance control**:
   - Skip Hypothesis for parity debugging or cap to <= 3 examples, <= 20 points.

---

## 4. Debug Playbook (Step-by-Step)

1) **Choose dataset** (<= 20 points), fix random seed.
2) **Initialization checkpoint**:
   - Compare MIN_BTN_CLUSTER_DIST / fallback distance.
   - Compare initial pair distances for a few known point pairs.
3) **Pair ordering checkpoint (Round 1)**:
   - Log first k pairs from legacy (k = N) and refactor.
4) **Mergeability checkpoint**:
   - For each candidate pair, log rho, T_i/T_j, beta_ij/xi_s, continuity.
   - Compare at first divergence (same pair, different decision).
5) **Merge update checkpoint**:
   - After each merge: cluster sizes, center, mu/sigma, merge_history length.
   - Compare MIN_BTN_CLUSTER_DIST update.
6) **Stop at first divergence** and inspect which sub-metric first differs.

Suggested instrumentation (low impact):
- Monkeypatch legacy.merge_2_cluster and refactor._merge_cluster_data to log merges.
- Add optional debug callbacks to GaugingDelta.fit for per-merge stats.

---

## 5. Next Actions
1) Decide on label strategy (preserve legacy IDs vs contiguous re-labeling).
2) Implement merge logging utilities for both paths.
3) Add small-dataset merge sequence parity tests.
4) Tighten beta_ij/xi_s parity checks once behavior is aligned.
