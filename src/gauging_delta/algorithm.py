"""Gauging-delta hierarchical clustering algorithm.

sklearn-compatible API with swappable proximity, continuity, and linkage metrics.
Port of ``Perception.fit`` (perception.py L48-146) and supporting methods.
"""

from __future__ import annotations

import numpy as np

from gauging_delta._types import ContinuityMetric, LinkageMetric, ProximityMetric
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig
from gauging_delta.continuity import DefaultContinuity
from gauging_delta.linkage import DefaultLinkage
from gauging_delta.proximity import DefaultProximity
from gauging_delta.threshold import compute_adaptive_threshold


class GaugingDelta:
    """Gauging-delta clustering with sklearn-compatible API.

    Parameters
    ----------
    n_clusters : int or None
        Target number of clusters.  ``None`` = let the algorithm decide.
    config : GaugingDeltaConfig or None
        All magic numbers.  Defaults are the paper's values.
    proximity, continuity, linkage : Protocol-compatible objects or None
        Swappable algorithm components.
    preserve_labels : bool
        If *True*, ``labels_`` keeps the internal cluster IDs instead
        of renumbering to ``0 … n_clusters_-1``.
    """

    def __init__(
        self,
        *,
        n_clusters: int | None = None,
        config: GaugingDeltaConfig | None = None,
        proximity: ProximityMetric | None = None,
        continuity: ContinuityMetric | None = None,
        linkage: LinkageMetric | None = None,
        preserve_labels: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.config = config or GaugingDeltaConfig()
        self.proximity = proximity or DefaultProximity()
        self.continuity = continuity or DefaultContinuity(self.config)
        self.linkage = linkage or DefaultLinkage()
        self.preserve_labels = preserve_labels

    # ----- sklearn-compatible interface ------------------------------------

    def fit(self, X: np.ndarray) -> GaugingDelta:
        """Run Gauging-delta on data matrix *X* (n_samples, n_features)."""
        self._X = np.asarray(X, dtype=float)
        n = len(self._X)

        # --- Initialise singleton clusters (perception.py L56-69) ---
        self._clusters: dict[int, Cluster] = {
            i: Cluster(
                label=i,
                point_indices=[i],
                center=self._X[i].copy(),
            )
            for i in range(n)
        }

        # --- Pairwise distances (perception.py L72, L176-220) ---
        self._dist_matrix = np.full((n, n), np.inf)
        self._near_ref = np.empty((n, n), dtype=int)  # _near_ref[i,j] = point in i nearest to j
        self._init_distances()

        # --- Main merge loop (perception.py L75-127) ---
        early_stop = False
        while len(self._clusters) > 1:
            pre_length = len(self._clusters)

            k_nearest = self._get_sorted_pairs()
            last_merged: int | None = None
            i = 0

            while i < k_nearest.shape[1]:
                c1, c2 = int(k_nearest[0, i]), int(k_nearest[1, i])

                # Early stop on target cluster count
                if self.n_clusters is not None and len(self._clusters) == self.n_clusters:
                    early_stop = True
                    break

                # Skip dead pairs (perception.py L97)
                if np.isinf(self._dist_matrix[c1, c2]):
                    i += 1
                    continue

                # last_merged heuristic (perception.py L104-113)
                if last_merged is not None:
                    _c = self._get_nearest_cluster(last_merged)
                    if (
                        _c is not None
                        and self._dist_matrix[last_merged, _c] < self._dist_matrix[c1, c2]
                    ):
                        c1, c2 = last_merged, _c
                    else:
                        i += 1
                    last_merged = None
                else:
                    i += 1

                # Try mergeability pipeline (perception.py L259-342)
                # NOTE: legacy does NOT increment i on merge failure.
                # The else:i+=1 at perception.py L122-123 pairs with
                # if ~np.isinf (L97), not with if is_complete (L116).
                merged_id = self._try_merge(c1, c2)
                if merged_id is not None:
                    last_merged = merged_id

            if pre_length == len(self._clusters) or early_stop:
                break

        # Post-processing: force to n_clusters if needed (perception.py L128-139)
        if self.n_clusters is not None and self.n_clusters < len(self._clusters):
            self._post_processing()

        # --- Build output labels ---
        self._build_labels()
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X).labels_

    # ----- Distance initialisation -----------------------------------------

    def _init_distances(self) -> None:
        """Port of ``initiate_dists`` (perception.py L176-220).

        For singleton clusters, near_dist = center_dist = |X[i] - X[j]|.
        We compute all pairwise distances in one vectorized call via
        scipy.spatial.distance.cdist, then build sorted neighbor arrays.
        """
        from scipy.spatial.distance import cdist

        n = len(self._X)

        # Vectorized pairwise distances — O(N²·D) in one C call
        pairwise = cdist(self._X, self._X)  # (N, N)
        self._dist_matrix = pairwise.copy()
        np.fill_diagonal(self._dist_matrix, np.inf)

        # For singletons: nearest point in cluster i to cluster j = point i itself
        # _near_ref[i, j] = i for all j (vectorized init)
        idx = np.arange(n)
        self._near_ref[:] = idx[:, np.newaxis]

        # Build sorted neighbor arrays from pairwise matrix (replaces _point_dists dict)
        # _pd_indices[i] = neighbor indices sorted by distance from point i
        # _pd_dists[i] = corresponding sorted distances
        # kind="stable" matches legacy Python list.sort tie-breaking (lower index first)
        # Mask self-distances to inf so self never appears in sorted neighbors
        # (simple [:, 1:] fails for duplicate points where pairwise[i,j]=0=pairwise[i,i])
        pw_no_self = pairwise.copy()
        np.fill_diagonal(pw_no_self, np.inf)
        full_order = np.argsort(pw_no_self, axis=1, kind="stable")[:, : n - 1]
        self._pd_indices = full_order
        self._pd_dists = np.take_along_axis(pairwise, full_order, axis=1)

        # MIN_BTN_CLUSTER_DIST (perception.py L213)
        # Track matrix minimum incrementally: position + value
        self._cached_dist_min = float(self._dist_matrix.min())
        flat_idx = int(np.argmin(self._dist_matrix))
        self._min_pos = (flat_idx // n, flat_idx % n)
        upper = pairwise[np.triu_indices(n, k=1)]
        upper.sort()
        self._fallback_dist = float(upper[int(len(upper) * self.config.min_dist_percentile)])

    # ----- Sorted pairs ----------------------------------------------------

    def _get_sorted_pairs(self) -> np.ndarray:
        """Port of ``get_indices_of_k_smallest`` (perception.py L80, L247-255).

        Returns (2, k) array of (row, col) pairs sorted by distance.
        """
        k = len(self._X)
        mat = self._dist_matrix
        flat = mat.ravel()
        k = min(k, flat.size)
        idx = np.argpartition(flat, k)[:k]
        ind = np.array(np.unravel_index(idx, mat.shape))
        values = mat[tuple(ind)]
        order = np.argsort(values)
        return ind[:, order]

    def _get_nearest_cluster(self, cid: int) -> int | None:
        """Nearest active cluster to *cid* (perception.py L105)."""
        row = self._dist_matrix[cid, :]
        k = 1
        k = min(k, row.size)
        idx = np.argpartition(row, k)[:k]
        idx = idx[np.argsort(row[idx])]
        nearest = int(idx[0])
        if np.isinf(self._dist_matrix[cid, nearest]):
            return None
        return nearest

    # ----- Mergeability pipeline -------------------------------------------

    def _try_merge(self, c1: int, c2: int) -> int | None:
        """Run proximity → threshold → continuity pipeline.

        Returns lead cluster ID on success, None on failure.
        Port of ``vision_generic`` (perception.py L309-342).
        """
        C_i = self._clusters[c1]
        C_j = self._clusters[c2]
        d_ij: float = float(self._dist_matrix[c1, c2])

        # Step 1: Proximity (perception.py L310)
        prox = self.proximity.compute(C_i, C_j, d_ij, self._fallback_dist)
        rho = prox.rho
        lead_id, child_id = prox.lead_id, prox.child_id
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # Step 2: Adaptive threshold (perception.py L312-313)
        thr = compute_adaptive_threshold(
            lead,
            child,
            d_ij,
            rho,
            self._clusters,
            self._dist_matrix,
            self.config,
        )

        # Gate: proximity > threshold → reject (perception.py L318)
        if rho > thr.T_i or rho > thr.T_j:
            return None

        # Step 3: Continuity (perception.py L323-325)
        # Set reference points for continuity
        lead.ref_point = int(self._near_ref[lead_id, child_id])
        child.ref_point = int(self._near_ref[child_id, lead_id])

        cont_threshold = self.config.threshold_continuity / thr.xi_s
        d_ij_norm = d_ij / rho if rho != 0 else d_ij
        smoothness = self.continuity.compute(
            lead,
            child,
            cont_threshold,
            d_ij_norm,
            thr.adp_prox,
            self._X,
            (self._pd_indices, self._pd_dists),
        )

        # Gate: smoothness > threshold → accept merge (perception.py L328)
        if smoothness > cont_threshold:
            # Record history BEFORE merge (perception.py L332-336)
            lead.density_history.append(smoothness)
            lead.merge_history.append(d_ij)
            lead.merging_dists.append(d_ij)

            self._do_merge(lead_id, child_id)
            return lead_id

        return None

    # ----- Merge execution -------------------------------------------------

    def _do_merge(self, lead_id: int, child_id: int) -> None:
        """Combine *child* into *lead* and update distances.

        Port of ``merge_2_cluster`` (perception.py L293-307) +
        ``update_clusters`` (perception.py L277-291).
        """
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # --- Merge data (perception.py L298-306) ---
        p1 = int(self._near_ref[lead_id, child_id])
        p2 = int(self._near_ref[child_id, lead_id])
        lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)

        # Extend history from child (perception.py L306)
        lead.merge_history.extend(child.merge_history)

        # --- Update clusters (perception.py L277-291) ---

        # Snapshot child's distances and refs BEFORE deletion
        # Lance-Williams: near_dist(A∪B, C) = min(near_dist(A,C), near_dist(B,C))
        child_near_row = self._dist_matrix[child_id, :].copy()
        child_ref_self = self._near_ref[child_id, :].copy()
        child_ref_other = self._near_ref[:, child_id].copy()

        del self._clusters[child_id]
        self._dist_matrix[:, child_id] = np.inf
        self._dist_matrix[child_id, :] = np.inf

        active = np.argwhere(~np.isinf(self._dist_matrix[lead_id, :])).reshape(-1)

        # Vectorized Lance-Williams update: only update where child was closer
        # Strict < so lead wins on exact tie (matches legacy argmin first-occurrence)
        child_dists = child_near_row[active]
        lead_dists = self._dist_matrix[lead_id, active]
        mask = child_dists < lead_dists
        winning = active[mask]

        self._dist_matrix[lead_id, winning] = child_near_row[winning]
        self._dist_matrix[winning, lead_id] = child_near_row[winning]
        self._near_ref[lead_id, winning] = child_ref_self[winning]
        self._near_ref[winning, lead_id] = child_ref_other[winning]

        # Update MIN_BTN_CLUSTER_DIST (perception.py L290)
        # Incremental: track position of matrix minimum.  Only the lead row
        # changed (Lance-Williams: values decrease or stay) and child row → inf.
        min_i, min_j = self._min_pos
        if min_i == child_id or min_j == child_id:
            # Cached position gone (child merged away).  Full scan needed
            # because the global min could be anywhere in the unchanged matrix.
            self._cached_dist_min = float(self._dist_matrix.min())
            if not np.isinf(self._cached_dist_min):
                flat_idx = int(np.argmin(self._dist_matrix))
                n = self._dist_matrix.shape[0]
                self._min_pos = (flat_idx // n, flat_idx % n)
        else:
            # Cached position still valid.  Lead row may have decreased.
            lead_row_min = float(np.min(self._dist_matrix[lead_id, :]))
            if lead_row_min < self._cached_dist_min:
                self._cached_dist_min = lead_row_min
                self._min_pos = (lead_id, int(np.argmin(self._dist_matrix[lead_id, :])))
        if not np.isinf(self._cached_dist_min):
            self._fallback_dist = max(self._cached_dist_min, self._fallback_dist)

    # ----- Post-processing -------------------------------------------------

    def _post_processing(self) -> None:
        """Force merge down to n_clusters (perception.py L161-173)."""
        if self.n_clusters is None:
            return
        ranking = sorted(self._clusters.items(), key=lambda x: len(x[1]))
        top_k = ranking[-self.n_clusters :]
        remaining = ranking[: len(ranking) - self.n_clusters]

        top_k_ids = {c[0] for c in top_k}
        merging_pairs: list[list[int]] = []

        for cid, _ in remaining:
            dists = [(tid, self._dist_matrix[tid, cid]) for tid in top_k_ids]
            nearest = min(dists, key=lambda x: x[1])
            merging_pairs.append([nearest[0], cid])

        for lead_id, child_id in merging_pairs:
            self._force_merge(lead_id, child_id)

    def _force_merge(self, lead_id: int, child_id: int) -> None:
        """Unconditional merge for post-processing (perception.py L172)."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        if not np.isinf(self._dist_matrix[lead_id, child_id]):
            p1 = int(self._near_ref[lead_id, child_id])
            p2 = int(self._near_ref[child_id, lead_id])
            lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = np.linalg.norm(self._X[lead.point_indices] - lead.center, axis=1)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)
        lead.merge_history.extend(child.merge_history)

        del self._clusters[child_id]
        self._dist_matrix[:, child_id] = np.inf
        self._dist_matrix[child_id, :] = np.inf

    # ----- Label construction ----------------------------------------------

    def _build_labels(self) -> None:
        """Assign final labels to each point."""
        n = len(self._X)
        labels = np.full(n, -1, dtype=int)

        if self.preserve_labels:
            for cid, cluster in self._clusters.items():
                for p in cluster.point_indices:
                    labels[p] = cid
        else:
            for new_label, (_, cluster) in enumerate(sorted(self._clusters.items())):
                for p in cluster.point_indices:
                    labels[p] = new_label

        self.labels_ = labels
        self.n_clusters_ = len(self._clusters)
