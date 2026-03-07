"""Gauging-delta hierarchical clustering algorithm.

sklearn-compatible API with swappable proximity, continuity, and linkage metrics.
Port of ``Perception.fit`` (perception.py L48-146) and supporting methods.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from gauging_delta._types import ContinuityMetric, LinkageMetric, ProximityMetric
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig
from gauging_delta.continuity import DefaultContinuity
from gauging_delta.linkage import DefaultLinkage
from gauging_delta.proximity import DefaultProximity
from gauging_delta.threshold import compute_adaptive_threshold


class GaugingDelta(ClusterMixin, BaseEstimator):
    """Gauging-delta clustering with sklearn-compatible API.

    Parameters
    ----------
    n_clusters : int or None
        Target number of clusters.  ``None`` = let the algorithm decide.
    mode : {'full', 'lite'}, default='full'
        Algorithm variant.

        - ``'full'``: Single-linkage distances with angle-based continuity
          analysis.  Highest quality; O(N²) memory for 4 dense arrays.
        - ``'lite'``: Centroid-linkage distances, no continuity gate.
          O(N·D) memory; suitable for large datasets.
    metric : str or callable, default='euclidean'
        Distance metric.  Accepts any metric supported by
        :func:`scipy.spatial.distance.cdist` (e.g. ``'euclidean'``,
        ``'cosine'``, ``'cityblock'``), a callable ``f(u, v) -> float``,
        or ``'precomputed'``.

        When ``'precomputed'``, *X* passed to :meth:`fit` must be a
        distance matrix — square *(N, N)*, condensed 1-D (from
        :func:`~scipy.spatial.distance.pdist`), or a DataFrame.
        Requires ``mode='full'``; continuity is automatically disabled.

        Non-euclidean metrics must satisfy the triangle inequality
        (validated at ``fit()`` time).
    config : GaugingDeltaConfig or None
        Algorithm constants.
    proximity : ProximityMetric or None
        Custom proximity metric.  ``None`` uses the default.
    continuity : ContinuityMetric, False, or None
        Continuity gate.  ``None`` uses the mode default (on for full,
        off for lite).  ``False`` explicitly disables the gate.  A
        :class:`ContinuityMetric` instance provides a custom implementation.
    linkage : LinkageMetric or None
        Custom linkage metric.  Full mode uses single-linkage by default;
        lite mode uses centroid-linkage.
    preserve_labels : bool
        If *True*, ``labels_`` keeps the internal cluster IDs instead
        of renumbering to ``0 … n_clusters_-1``.
    """

    def __init__(
        self,
        *,
        n_clusters: int | None = None,
        mode: str = "full",
        metric: str | callable = "euclidean",
        config: GaugingDeltaConfig | None = None,
        proximity: ProximityMetric | None = None,
        continuity: ContinuityMetric | None | bool = None,
        linkage: LinkageMetric | None = None,
        preserve_labels: bool = False,
    ) -> None:
        self.n_clusters = n_clusters
        self.mode = mode
        self.metric = metric
        self.config = config
        self.proximity = proximity
        self.continuity = continuity
        self.linkage = linkage
        self.preserve_labels = preserve_labels

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags

    # ----- Distance helpers --------------------------------------------------

    def _cdist(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Pairwise distances between rows of *X* and *Y*."""
        from scipy.spatial.distance import cdist

        return cdist(X, Y, metric=self.metric)

    def _row_dists(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distances from each row of *X* to single point *y*. Returns (len(X),)."""
        if self.metric == "euclidean" or self._precomputed:
            return np.linalg.norm(X - y, axis=1)
        from scipy.spatial.distance import cdist

        return cdist(X, y.reshape(1, -1), metric=self.metric).ravel()

    def _point_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two single points."""
        if self.metric == "euclidean" or self._precomputed:
            diff = a - b
            return float(math.sqrt(float(diff @ diff)))
        from scipy.spatial.distance import cdist

        return float(cdist(a.reshape(1, -1), b.reshape(1, -1), metric=self.metric)[0, 0])

    # ----- Metric validation ------------------------------------------------

    def _validate_metric(self) -> None:
        """Spot-check triangle inequality on a sample of point triples."""
        n = len(self._X)
        rng = np.random.RandomState(0)
        sample = rng.choice(n, size=min(n, 9), replace=False)
        for i in range(0, len(sample) - 2, 3):
            a, b, c = int(sample[i]), int(sample[i + 1]), int(sample[i + 2])
            d_ab = self._point_dist(self._X[a], self._X[b])
            d_bc = self._point_dist(self._X[b], self._X[c])
            d_ac = self._point_dist(self._X[a], self._X[c])
            # Check all three sides as the "long" side
            for d_long, d_sum in [
                (d_ab, d_ac + d_bc),
                (d_ac, d_ab + d_bc),
                (d_bc, d_ab + d_ac),
            ]:
                if d_long > d_sum + 1e-10:
                    raise ValueError(
                        f"metric violates triangle inequality on points "
                        f"x[{a}], x[{b}], x[{c}]. "
                        f"Gauging-delta requires a valid distance metric."
                    )

    # ----- Precomputed distance matrix validation ---------------------------

    @staticmethod
    def _validate_precomputed(X: np.ndarray) -> np.ndarray:
        """Validate and normalise a precomputed distance matrix.

        Accepts:
        - Square (N, N) symmetric matrix with zero diagonal.
        - Condensed 1-D vector of length N*(N-1)/2 (scipy ``pdist`` output).

        Returns the square (N, N) matrix.
        """
        if X.ndim == 1:
            from scipy.spatial.distance import squareform

            try:
                X = squareform(X)
            except ValueError:
                raise ValueError(
                    "1-D input with metric='precomputed' must be a condensed "
                    "distance vector of length N*(N-1)/2 (from scipy.spatial."
                    "distance.pdist).  Pass a square (N, N) matrix instead."
                ) from None
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(
                f"metric='precomputed' requires a square distance matrix or "
                f"condensed 1-D vector, got shape {X.shape}"
            )
        if np.any(X < -1e-10):
            raise ValueError("Distance matrix contains negative values.")
        if np.any(np.abs(np.diag(X)) > 1e-10):
            raise ValueError("Distance matrix diagonal must be zero.")
        if not np.allclose(X, X.T, atol=1e-10):
            raise ValueError("Distance matrix must be symmetric.")
        return X

    # ----- sklearn-compatible interface ------------------------------------

    def fit(self, X, y=None):
        """Run Gauging-delta on data matrix *X*.

        Parameters
        ----------
        X : array-like
            Feature matrix of shape *(n_samples, n_features)*, or a
            precomputed distance matrix when ``metric='precomputed'``.
            Distance matrices may be square *(n, n)*, condensed 1-D
            (length *n*(n-1)/2*, as from :func:`scipy.spatial.distance.pdist`),
            or a pandas DataFrame.
        y : ignored
            Not used, present for sklearn API consistency.
        """
        if self.mode not in ("full", "lite"):
            raise ValueError(f"mode must be 'full' or 'lite', got {self.mode!r}")

        # --- Resolve defaults (raw params stored in __init__ for get_params) ---
        self._precomputed = self.metric == "precomputed"
        self._cfg = self.config or GaugingDeltaConfig()
        self._prox = self.proximity or DefaultProximity()

        # Precomputed constraints
        if self._precomputed:
            if self.mode == "lite":
                raise ValueError(
                    "metric='precomputed' requires mode='full' "
                    "(lite mode needs recomputable centroid distances)"
                )
            if self.continuity is not None and self.continuity is not False:
                raise ValueError(
                    "continuity requires feature coordinates and cannot be "
                    "used with metric='precomputed'"
                )

        # Continuity: None → mode default, False → disabled, instance → custom
        if self._precomputed or self.continuity is False:
            self._cont: ContinuityMetric | None = None
        elif self.continuity is None:
            self._cont = DefaultContinuity(self._cfg) if self.mode == "full" else None
        else:
            self._cont = self.continuity  # type: ignore[assignment]

        # Linkage resolution
        if self.linkage is None:
            self._lnk: LinkageMetric | None = DefaultLinkage() if self.mode == "full" else None
        else:
            self._lnk = self.linkage

        # --- Input validation ---
        if self._precomputed:
            X_arr = np.asarray(X, dtype=float)
            self._X = self._validate_precomputed(X_arr)
            self.n_features_in_ = self._X.shape[0]
        else:
            from sklearn.utils.validation import validate_data

            self._X = validate_data(self, X, accept_sparse=False, dtype="numeric", reset=True)

        n = len(self._X)

        # Trivial: 0 or 1 sample
        if n <= 1:
            self._clusters = {
                i: Cluster(label=i, point_indices=[i], center=self._X[i].copy())
                for i in range(n)
            }
            self._merge_log = []
            self._build_labels()
            return self

        # Validate metric satisfies triangle inequality
        if not self._precomputed and self.metric != "euclidean" and n >= 3:
            self._validate_metric()

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
        if self.mode == "full":
            self._dist_matrix = np.full((n, n), np.inf)
            self._near_ref = np.empty((n, n), dtype=int)
        self._init_distances()

        # Precompute rho rejection bound for lite mode (max possible threshold)
        if self.mode == "lite":
            cfg = self._cfg
            self._rho_reject_bound = (
                (cfg.vision_scale_coeff / 2 + cfg.vision_scale_offset)
                * max(cfg.t_stat_numerator / 2 + cfg.t_stat_offset, cfg.t_stat_fallback)
            )

        # --- Merge log for hierarchical outputs ---
        self._merge_log: list[tuple[int, int, float]] = []

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
                if self.mode == "lite":
                    if c1 not in self._clusters or c2 not in self._clusters:
                        i += 1
                        continue
                elif np.isinf(self._dist_matrix[c1, c2]):
                    i += 1
                    continue

                # last_merged heuristic (perception.py L104-113)
                if last_merged is not None:
                    _c = self._get_nearest_cluster(last_merged)
                    if _c is not None:
                        if self.mode == "lite":
                            d_lm = self._point_dist(
                                self._clusters[last_merged].center, self._clusters[_c].center
                            )
                            d_pair = self._point_dist(
                                self._clusters[c1].center, self._clusters[c2].center
                            )
                            closer = d_lm < d_pair
                        else:
                            closer = self._dist_matrix[last_merged, _c] < self._dist_matrix[c1, c2]
                        if closer:
                            c1, c2 = last_merged, _c
                        else:
                            i += 1
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

    # ----- Distance initialisation -----------------------------------------

    def _init_distances(self) -> None:
        """Initialise distance matrix and supporting structures."""
        if self.mode == "lite":
            self._init_distances_lite()
        else:
            self._init_distances_full()

    def _init_distances_full(self) -> None:
        """Full mode: single-linkage distances + sorted neighbor arrays."""
        n = len(self._X)

        # Pairwise distances: precomputed → use directly, else compute via cdist
        if self._precomputed:
            pairwise = self._X  # already a distance matrix
        else:
            pairwise = self._cdist(self._X, self._X)  # (N, N)

        self._dist_matrix = pairwise.copy()
        np.fill_diagonal(self._dist_matrix, np.inf)

        # For singletons: nearest point in cluster i to cluster j = point i itself
        # _near_ref[i, j] = i for all j (vectorized init)
        idx = np.arange(n)
        self._near_ref[:] = idx[:, np.newaxis]

        # Sorted neighbor arrays: only needed when continuity is enabled
        if self._cont is not None:
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
        # Row-wise minimum tracking: per-row argmin column + value arrays
        self._row_argmins = np.argmin(self._dist_matrix, axis=1).astype(int)
        self._row_mins = self._dist_matrix[np.arange(n), self._row_argmins].copy()
        upper = pairwise[np.triu_indices(n, k=1)]
        upper.sort()
        self._fallback_dist = float(upper[int(len(upper) * self._cfg.min_dist_percentile)])

    def _init_distances_lite(self) -> None:
        """Lite mode: build KDTree over initial singleton centroids."""
        n = len(self._X)
        D = self._X.shape[1] if self._X.ndim > 1 else 1

        if n <= 1:
            self._fallback_dist = 0.0
            self._lite_tree = None
            self._lite_ids = list(self._clusters.keys())
            self._lite_centers = self._X.copy() if n == 1 else np.empty((0, D))
            self._lite_id_to_pos = {cid: i for i, cid in enumerate(self._lite_ids)}
            self._lite_K = n
            return

        # Build arrays: _lite_centers is (N, D) with room for shrinking via _lite_K
        self._lite_ids = list(range(n))
        self._lite_centers = self._X.copy()  # singletons → centers = data points
        self._lite_id_to_pos = {i: i for i in range(n)}
        self._lite_K = n

        # KDTree for euclidean; brute-force cdist otherwise
        if self.metric == "euclidean":
            from scipy.spatial import cKDTree

            self._lite_tree = cKDTree(self._lite_centers)
            nn_dists, _ = self._lite_tree.query(self._lite_centers, k=2)
            nn_sorted = np.sort(nn_dists[:, 1])
        else:
            self._lite_tree = None
            pw = self._cdist(self._lite_centers, self._lite_centers)
            np.fill_diagonal(pw, np.inf)
            nn_sorted = np.sort(np.min(pw, axis=1))

        self._fallback_dist = float(
            nn_sorted[max(0, int(len(nn_sorted) * self._cfg.min_dist_percentile))]
        )

    # ----- Sorted pairs ----------------------------------------------------

    def _get_sorted_pairs(self) -> np.ndarray:
        """Port of ``get_indices_of_k_smallest`` (perception.py L80, L247-255).

        Returns (2, k) array of (row, col) pairs sorted by distance.
        """
        if self.mode == "lite":
            return self._get_sorted_pairs_lite()
        k = len(self._X)
        mat = self._dist_matrix
        flat = mat.ravel()
        k = min(k, flat.size)
        idx = np.argpartition(flat, k)[:k]
        ind = np.array(np.unravel_index(idx, mat.shape))
        values = mat[tuple(ind)]
        order = np.argsort(values)
        return ind[:, order]

    def _get_sorted_pairs_lite(self) -> np.ndarray:
        """Sorted candidate pairs (KDTree for euclidean, brute-force otherwise)."""
        K = self._lite_K
        if K <= 1:
            return np.empty((2, 0), dtype=int)

        centers = self._lite_centers[:K]

        # Adaptive k: match original's coverage
        entries_per_cluster = min(K - 1, max(1, len(self._X) // K))
        k_query = entries_per_cluster + 1  # +1 for self

        if self.metric == "euclidean":
            from scipy.spatial import cKDTree

            self._lite_tree = cKDTree(centers)
            dists_arr, idxs_arr = self._lite_tree.query(centers, k=min(k_query, K))
        else:
            self._lite_tree = None
            pw = self._cdist(centers, centers)
            np.fill_diagonal(pw, np.inf)
            k_nn = min(k_query, K)
            idxs_arr = np.argpartition(pw, k_nn - 1, axis=1)[:, :k_nn]
            dists_arr = np.take_along_axis(pw, idxs_arr, axis=1)
            order = np.argsort(dists_arr, axis=1)
            idxs_arr = np.take_along_axis(idxs_arr, order, axis=1)
            dists_arr = np.take_along_axis(dists_arr, order, axis=1)

        # Ensure 2D
        if dists_arr.ndim == 1:
            dists_arr = dists_arr.reshape(1, -1)
            idxs_arr = idxs_arr.reshape(1, -1)

        # Update fallback_dist
        if K >= 2:
            current_min = float(np.min(dists_arr[:, 1]))
            if not np.isinf(current_min):
                self._fallback_dist = max(current_min, self._fallback_dist)

        # Build unique pairs sorted by distance
        # Skip self-pairs (nn == i) which can occur when KDTree returns the
        # queried point itself due to tie-breaking with duplicate points.
        seen: set[tuple[int, int]] = set()
        pairs_list: list[tuple[int, int, float]] = []
        for i in range(K):
            for j in range(1, idxs_arr.shape[1]):
                nn = int(idxs_arr[i, j])
                if nn == i:  # skip self-pair from KDTree tie-breaking
                    continue
                d = float(dists_arr[i, j])
                c1, c2 = self._lite_ids[i], self._lite_ids[nn]
                pair_key = (min(c1, c2), max(c1, c2))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs_list.append((c1, c2, d))

        pairs_list.sort(key=lambda x: x[2])
        if not pairs_list:
            return np.empty((2, 0), dtype=int)
        return np.array([[p[0] for p in pairs_list], [p[1] for p in pairs_list]])

    def _get_nearest_cluster(self, cid: int) -> int | None:
        """Nearest active cluster to *cid*."""
        if self.mode == "lite":
            return self._get_nearest_cluster_lite(cid)
        # Full mode: O(1) lookup from cached row mins
        if np.isinf(self._row_mins[cid]):
            return None
        return int(self._row_argmins[cid])

    def _get_nearest_cluster_lite(self, cid: int) -> int | None:
        """Nearest active cluster via brute-force centroid scan (O(K·D))."""
        K = self._lite_K
        if K < 2:
            return None
        if cid not in self._lite_id_to_pos:
            return None
        center = self._clusters[cid].center
        pos = self._lite_id_to_pos[cid]
        dists = self._row_dists(self._lite_centers[:K], center)
        dists[pos] = np.inf
        nearest_pos = int(np.argmin(dists))
        return self._lite_ids[nearest_pos]

    # ----- Mergeability pipeline -------------------------------------------

    def _try_merge(self, c1: int, c2: int) -> int | None:
        """Run mergeability pipeline. Dispatches based on *mode*."""
        if self.mode == "lite":
            return self._try_merge_lite(c1, c2)
        return self._try_merge_full(c1, c2)

    def _try_merge_full(self, c1: int, c2: int) -> int | None:
        """Full mode: proximity → threshold → continuity → merge."""
        C_i = self._clusters[c1]
        C_j = self._clusters[c2]
        d_ij: float = float(self._dist_matrix[c1, c2])

        # Step 1: Proximity (perception.py L310)
        prox = self._prox.compute(C_i, C_j, d_ij, self._fallback_dist)
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
            self._cfg,
        )

        # Gate: proximity > threshold → reject (perception.py L318)
        if rho > thr.T_i or rho > thr.T_j:
            return None

        # Step 3: Continuity (perception.py L323-325)
        if self._cont is not None:
            lead.ref_point = int(self._near_ref[lead_id, child_id])
            child.ref_point = int(self._near_ref[child_id, lead_id])

            cont_threshold = self._cfg.threshold_continuity / thr.xi_s
            d_ij_norm = d_ij / rho if rho != 0 else d_ij
            smoothness = self._cont.compute(
                lead,
                child,
                cont_threshold,
                d_ij_norm,
                thr.adp_prox,
                self._X,
                (self._pd_indices, self._pd_dists),
            )

            # Gate: smoothness > threshold → accept merge (perception.py L328)
            if smoothness <= cont_threshold:
                return None

            lead.density_history.append(smoothness)

        # Record and merge
        lead.merge_history.append(d_ij)
        lead.merging_dists.append(d_ij)

        self._merge_log.append((lead_id, child_id, d_ij))
        self._do_merge(lead_id, child_id)
        return lead_id

    def _try_merge_lite(self, c1: int, c2: int) -> int | None:
        """Lite mode: proximity → threshold → merge (no continuity)."""
        from gauging_delta.threshold import compute_adaptive_threshold_lite

        C_i = self._clusters[c1]
        C_j = self._clusters[c2]
        d_ij = self._point_dist(C_i.center, C_j.center)

        # Step 1: Proximity
        prox = self._prox.compute(C_i, C_j, d_ij, self._fallback_dist)
        rho = prox.rho

        # Short-circuit: rho exceeds maximum possible threshold → skip threshold
        if rho > self._rho_reject_bound:
            return None

        lead_id, child_id = prox.lead_id, prox.child_id
        lead = self._clusters[lead_id]

        # Step 2: Adaptive threshold (brute-force centers array, no KDTree)
        thr = compute_adaptive_threshold_lite(
            lead,
            self._clusters[child_id],
            d_ij,
            rho,
            self._clusters,
            self._lite_centers[: self._lite_K],
            self._lite_ids,
            self._lite_id_to_pos,
            self._cfg,
            metric=self.metric,
        )

        # Gate: proximity > threshold → reject
        if rho > thr.T_i or rho > thr.T_j:
            return None

        # No continuity gate — merge directly
        lead.merge_history.append(d_ij)
        lead.merging_dists.append(d_ij)

        self._merge_log.append((lead_id, child_id, d_ij))
        self._do_merge(lead_id, child_id)
        return lead_id

    # ----- Merge execution -------------------------------------------------

    def _do_merge(self, lead_id: int, child_id: int) -> None:
        """Merge *child* into *lead* and update distances."""
        if self.mode == "lite":
            self._do_merge_lite(lead_id, child_id)
        else:
            self._do_merge_full(lead_id, child_id)

    def _do_merge_full(self, lead_id: int, child_id: int) -> None:
        """Full mode: Lance-Williams single-linkage update."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # --- Merge data (perception.py L298-306) ---
        p1 = int(self._near_ref[lead_id, child_id])
        p2 = int(self._near_ref[child_id, lead_id])
        lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = self._row_dists(self._X[lead.point_indices], lead.center)
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
        # Row-wise minimum tracking: maintain per-row min value + position.
        # Only rescan rows that actually changed instead of full O(N²) scan.

        # 1. Identify stale rows (argmin pointed to the now-dead child)
        stale_mask = self._row_argmins == child_id

        # 2. Invalidate child row
        self._row_mins[child_id] = np.inf

        # 3. Rescan lead row (always needed: lead distances changed)
        lead_min_col = int(np.argmin(self._dist_matrix[lead_id, :]))
        self._row_argmins[lead_id] = lead_min_col
        self._row_mins[lead_id] = self._dist_matrix[lead_id, lead_min_col]

        # 4. Non-stale winning rows: lead column decreased, might be new row min
        if len(winning) > 0:
            non_stale_winning = winning[~stale_mask[winning]]
            if len(non_stale_winning) > 0:
                new_dists = self._dist_matrix[non_stale_winning, lead_id]
                improved = new_dists < self._row_mins[non_stale_winning]
                update_idx = non_stale_winning[improved]
                self._row_mins[update_idx] = new_dists[improved]
                self._row_argmins[update_idx] = lead_id

        # 5. Stale rows: full row rescan (expected ~1-5 rows)
        stale_mask[child_id] = False
        stale_mask[lead_id] = False
        stale_rows = np.where(stale_mask)[0]
        for c in stale_rows:
            c_min_col = int(np.argmin(self._dist_matrix[c, :]))
            self._row_argmins[c] = c_min_col
            self._row_mins[c] = self._dist_matrix[c, c_min_col]

        # 6. Global min → update fallback_dist
        current_min = float(np.min(self._row_mins))
        if not np.isinf(current_min):
            self._fallback_dist = max(current_min, self._fallback_dist)

    def _do_merge_lite(self, lead_id: int, child_id: int) -> None:
        """Lite mode: merge cluster data, swap-and-pop centers array (O(D))."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # --- Merge data ---
        lead.merge_edges.extend(child.merge_edges)
        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = self._row_dists(self._X[lead.point_indices], lead.center)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)
        lead.merge_history.extend(child.merge_history)

        # --- Delete child ---
        del self._clusters[child_id]

        # --- Swap-and-pop child from arrays (O(D)) ---
        child_pos = self._lite_id_to_pos[child_id]
        self._lite_K -= 1
        last_pos = self._lite_K

        if child_pos != last_pos:
            last_id = self._lite_ids[last_pos]
            self._lite_ids[child_pos] = last_id
            self._lite_centers[child_pos] = self._lite_centers[last_pos]
            self._lite_id_to_pos[last_id] = child_pos

        self._lite_ids.pop()
        del self._lite_id_to_pos[child_id]

        # Update lead's center in-place
        lead_pos = self._lite_id_to_pos[lead_id]
        self._lite_centers[lead_pos] = lead.center

        # Update fallback_dist from lead's nearest neighbor (O(K·D))
        K = self._lite_K
        if K >= 2:
            dists = self._row_dists(self._lite_centers[:K], lead.center)
            dists[lead_pos] = np.inf
            current_min = float(np.min(dists))
            if not np.isinf(current_min):
                self._fallback_dist = max(current_min, self._fallback_dist)

        # Mark tree as stale (will be rebuilt in _get_sorted_pairs_lite)
        self._lite_tree = None

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

        for cid, cluster in remaining:
            if self.mode == "lite":
                c_center = cluster.center
                dists = [
                    (tid, self._point_dist(self._clusters[tid].center, c_center))
                    for tid in top_k_ids
                ]
            else:
                dists = [(tid, self._dist_matrix[tid, cid]) for tid in top_k_ids]
            nearest = min(dists, key=lambda x: x[1])
            merging_pairs.append([nearest[0], cid])

        for lead_id, child_id in merging_pairs:
            self._force_merge(lead_id, child_id)

    def _force_merge(self, lead_id: int, child_id: int) -> None:
        """Unconditional merge for post-processing (perception.py L172)."""
        lead = self._clusters[lead_id]
        child = self._clusters[child_id]

        # Record merge distance for hierarchical outputs
        if self.mode == "full":
            d = float(self._dist_matrix[lead_id, child_id])
        else:
            d = self._point_dist(lead.center, child.center)
        if np.isinf(d):
            d = self._point_dist(lead.center, child.center)
        self._merge_log.append((lead_id, child_id, d))

        if self.mode == "full" and not np.isinf(self._dist_matrix[lead_id, child_id]):
            p1 = int(self._near_ref[lead_id, child_id])
            p2 = int(self._near_ref[child_id, lead_id])
            lead.merge_edges.append((p1, p2))
        lead.merge_edges.extend(child.merge_edges)

        lead.point_indices.extend(child.point_indices)
        lead.center = np.mean(self._X[lead.point_indices], axis=0)

        dist_data = self._row_dists(self._X[lead.point_indices], lead.center)
        lead.mu_dist = float(np.mean(dist_data))
        lead.sigma_dist = float(np.std(dist_data))
        lead.sigma_history.append(lead.sigma_dist)
        lead.merge_history.extend(child.merge_history)

        del self._clusters[child_id]
        if self.mode == "full":
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

        if not self._precomputed:
            ordered = sorted(self._clusters.items())
            self.cluster_centers_ = np.array([c.center for _, c in ordered])

        # Build hierarchical outputs from merge log
        n = len(self._X)
        self.n_leaves_ = n

        id_map = dict(enumerate(range(n)))
        children_list: list[list[int]] = []
        distances_list: list[float] = []
        for step, (lead_id, child_id, d_ij) in enumerate(self._merge_log):
            new_id = n + step
            children_list.append([id_map[lead_id], id_map[child_id]])
            distances_list.append(d_ij)
            id_map[lead_id] = new_id

        # Complete the hierarchy for scipy compatibility (needs n-1 rows).
        # If the algorithm stopped at k > 1 clusters, merge remaining clusters
        # at increasing distances above the last recorded merge.
        remaining_ids = sorted(self._clusters.keys())
        if len(remaining_ids) > 1:
            pad_dist = max(distances_list) * 1.5 if distances_list else 1.0
            while len(remaining_ids) > 1:
                a, b = remaining_ids[0], remaining_ids[1]
                step = len(children_list)
                new_id = n + step
                children_list.append([id_map[a], id_map[b]])
                distances_list.append(pad_dist)
                id_map[a] = new_id
                remaining_ids = [a] + remaining_ids[2:]
                pad_dist *= 1.5

        self.children_ = np.array(children_list, dtype=int).reshape(-1, 2)
        self.distances_ = np.array(distances_list)

    @property
    def linkage_matrix_(self) -> np.ndarray:
        """Scipy-compatible (n_merges, 4) linkage matrix Z.

        Rows: ``[child_1, child_2, distance, sample_count]``.
        Compatible with :func:`scipy.cluster.hierarchy.dendrogram`.
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        sizes: list[int] = []
        for (c1, c2), d in zip(self.children_, self.distances_):
            s1 = 1 if c1 < self.n_leaves_ else sizes[c1 - self.n_leaves_]
            s2 = 1 if c2 < self.n_leaves_ else sizes[c2 - self.n_leaves_]
            sizes.append(s1 + s2)
        return np.column_stack([self.children_, self.distances_, sizes])
