"""
Sparse neighbor graph with priority queue for Gauging-δ algorithm.

CRITICAL: This replaces the O(N²) dense distance matrix with O(N×k) sparse storage.

Memory comparison:
    - Dense matrix at 50k points: ~10 GB RAM (crashes)
    - Sparse graph at 50k points: ~200 MB RAM (manageable)

Uses:
    - KDTree for efficient spatial queries
    - Min-heap for O(log E) merge candidate extraction
    - Lazy deletion for O(1) cluster invalidation
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree


@dataclass
class NeighborGraph:
    """
    Sparse neighbor graph with priority queue for merge candidates.

    Space Complexity: O(N × k) where k = n_neighbors (default 50)

    Instead of storing all N² distances, we only store edges to k-nearest
    neighbors. The merge heap uses lazy deletion: when clusters merge,
    we mark the absorbed cluster as "dead" rather than searching the heap.

    Attributes:
        X: Original data points
        kdtree: KDTree for spatial queries
        n_neighbors: Number of neighbors per point in initial graph
        neighbors: Adjacency list (cluster_id -> set of neighbor cluster_ids)
        merge_heap: Min-heap of (distance, cluster_i, cluster_j) tuples
        dead_clusters: Set of cluster IDs that have been merged away
        cluster_points: Mapping from cluster_id to point indices
        cluster_centers: Mapping from cluster_id to centroid
    """

    X: np.ndarray
    n_neighbors: int = 50

    kdtree: KDTree | None = field(default=None, init=False)
    neighbors: dict[int, set[int]] = field(default_factory=dict, init=False)
    merge_heap: list[tuple[float, int, int]] = field(default_factory=list, init=False)
    dead_clusters: set[int] = field(default_factory=set, init=False)
    cluster_points: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    cluster_centers: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    point_to_cluster: np.ndarray | None = field(default=None, init=False)  # Reverse lookup

    def __post_init__(self):
        """Build KDTree after initialization."""
        self.kdtree = KDTree(self.X)

    def initialize(self) -> None:
        """
        Build initial neighbor graph from k-nearest neighbors.

        Each point starts as its own cluster. We query KDTree for
        k-nearest neighbors and push all edges to the heap.

        Time: O(N × k × log(N×k))
        Space: O(N × k)
        """
        n_points = len(self.X)
        k = min(self.n_neighbors + 1, n_points)  # +1 because query includes self

        # Initialize each point as its own cluster
        self.point_to_cluster = np.arange(n_points)  # point i -> cluster i initially
        for i in range(n_points):
            self.cluster_points[i] = np.array([i])
            self.cluster_centers[i] = self.X[i].copy()
            self.neighbors[i] = set()

        # Handle single point case
        if n_points <= 1:
            return

        # Query k-nearest neighbors for all points at once (vectorized)
        distances, indices = self.kdtree.query(self.X, k=k)

        # Ensure 2D arrays for single k case
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # Build adjacency list and heap
        seen_edges = set()
        for i in range(n_points):
            for j_idx in range(k):
                j = indices[i, j_idx]
                if i == j:
                    continue

                # Avoid duplicate edges (i,j) and (j,i)
                edge = (min(i, j), max(i, j))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)

                dist = distances[i, j_idx]
                self.neighbors[i].add(j)
                self.neighbors[j].add(i)

                # Push to heap: (distance, smaller_id, larger_id)
                heapq.heappush(self.merge_heap, (dist, edge[0], edge[1]))

    def pop_closest_pair(self) -> tuple[int, int, float] | None:
        """
        Extract the closest pair of clusters from the heap.

        Uses lazy deletion: if either cluster in the popped edge is dead,
        skip it and pop the next one.

        Returns:
            Tuple of (cluster_i, cluster_j, distance) or None if heap empty

        Time: O(log E) amortized (dead edges are O(1) to skip)
        """
        while self.merge_heap:
            dist, c_i, c_j = heapq.heappop(self.merge_heap)

            # Lazy deletion: skip if either cluster is dead
            if c_i in self.dead_clusters or c_j in self.dead_clusters:
                continue

            return (c_i, c_j, dist)

        return None

    def merge_clusters(self, keep_id: int, remove_id: int) -> None:
        """
        Merge two clusters: keep one, mark other as dead.

        1. Mark remove_id as dead (lazy deletion)
        2. Update cluster_points[keep_id] to include removed points
        3. Recompute center for keep_id
        4. Find new neighbors for merged cluster
        5. Push new edges to heap

        Args:
            keep_id: ID of cluster to keep (absorbs the other)
            remove_id: ID of cluster to remove (marked dead)

        Time: O(k × log E) for pushing new edges
        """
        # Mark removed cluster as dead
        self.dead_clusters.add(remove_id)

        # Merge point indices
        keep_points = self.cluster_points[keep_id]
        remove_points = self.cluster_points[remove_id]
        merged_points = np.concatenate([keep_points, remove_points])
        self.cluster_points[keep_id] = merged_points

        # Recompute centroid
        self.cluster_centers[keep_id] = np.mean(self.X[merged_points], axis=0)

        # Update point_to_cluster for all merged points
        if self.point_to_cluster is not None:
            self.point_to_cluster[merged_points] = keep_id

        # Transfer neighbors from removed cluster
        for neighbor in self.neighbors.get(remove_id, set()):
            if neighbor != keep_id and neighbor not in self.dead_clusters:
                self.neighbors[keep_id].add(neighbor)
                self.neighbors[neighbor].add(keep_id)  # Ensure bidirectional
                # Push new edge to heap
                dist, _, _ = self.get_cluster_distance(keep_id, neighbor)
                edge = (min(keep_id, neighbor), max(keep_id, neighbor))
                heapq.heappush(self.merge_heap, (dist, edge[0], edge[1]))

        # Remove from neighbors adjacency
        if remove_id in self.neighbors:
            del self.neighbors[remove_id]

    def get_cluster_distance(self, c_i: int, c_j: int) -> tuple[float, int, int]:
        """
        Compute distance between two clusters.

        Returns distance between nearest points (not centroids).

        Args:
            c_i: First cluster ID
            c_j: Second cluster ID

        Returns:
            Tuple of (distance, point_i, point_j) where points are the
            closest pair between the clusters
            
        OPTIMIZED: Uses vectorized distance computation with tie-breaking correction.
        When multiple pairs have distances within epsilon of minimum, resolves
        ties using legacy iteration order.
        """
        points_i = self.cluster_points[c_i]
        points_j = self.cluster_points[c_j]
        
        n_i, n_j = len(points_i), len(points_j)
        
        # For small clusters, use direct loop (overhead of vectorization not worth it)
        if n_i * n_j <= 16:
            min_dist = float("inf")
            best_i, best_j = points_i[0], points_j[0]
            for pi in points_i:
                for pj in points_j:
                    d = np.linalg.norm(self.X[pi] - self.X[pj])
                    if d < min_dist:
                        min_dist = d
                        best_i, best_j = pi, pj
            return (min_dist, best_i, best_j)
        
        # Vectorized distance computation using np.linalg.norm for exact parity
        # NOTE: scipy cdist uses different FP accumulation that causes ~15% of
        # distances to differ at the last bit vs np.linalg.norm. We must use
        # np.linalg.norm for bit-exact parity with legacy.
        Xi = self.X[points_i]  # shape (n_i, dim)
        Xj = self.X[points_j]  # shape (n_j, dim)
        
        # Compute pairwise distances: dists[i,j] = ||Xi[i] - Xj[j]||
        diff = Xi[:, np.newaxis, :] - Xj[np.newaxis, :, :]  # (n_i, n_j, dim)
        dists = np.linalg.norm(diff, axis=2)  # (n_i, n_j)
        
        # Find minimum and potential ties
        min_dist = dists.min()
        
        # Epsilon for tie detection - must catch all numerical near-ties
        eps = max(1e-12, min_dist * 1e-9)
        
        # Find all pairs within epsilon of minimum
        tie_mask = dists <= min_dist + eps
        tie_count = np.count_nonzero(tie_mask)
        
        if tie_count == 1:
            # No ties - use vectorized result directly
            idx = np.argmin(dists)
            idx_i, idx_j = divmod(idx, n_j)
            return (float(dists[idx_i, idx_j]), points_i[idx_i], points_j[idx_j])
        
        # Multiple ties - resolve using legacy iteration order
        # Legacy order: outer loop over points_i, inner loop over points_j
        for local_i in range(n_i):
            for local_j in range(n_j):
                if tie_mask[local_i, local_j]:
                    # First tie in legacy order wins
                    return (float(dists[local_i, local_j]), points_i[local_i], points_j[local_j])

    def get_active_clusters(self) -> set[int]:
        """Return set of cluster IDs that are still active (not dead)."""
        return set(self.cluster_points.keys()) - self.dead_clusters

    def query_radius(self, point: np.ndarray, radius: float) -> list[int]:
        """
        Find all points within radius of query point.

        Uses KDTree for O(log N) query.

        Args:
            point: Query point coordinates
            radius: Search radius

        Returns:
            List of point indices within radius
        """
        return self.kdtree.query_ball_point(point, radius)

    def query_k_nearest_points(self, point: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest points to query point.

        Args:
            point: Query point coordinates
            k: Number of neighbors

        Returns:
            Tuple of (distances, indices) arrays
        """
        k = min(k, len(self.X))
        distances, indices = self.kdtree.query(point, k=k)
        return np.atleast_1d(distances), np.atleast_1d(indices)

    def get_nearby_clusters(self, cluster_id: int, k: int) -> list[tuple[int, float]]:
        """
        Find k nearest active clusters to given cluster.

        Used for computing environmental factor β_ij.

        Args:
            cluster_id: Query cluster ID
            k: Number of neighbors

        Returns:
            List of (cluster_id, distance) tuples, sorted by distance
        """
        center = self.cluster_centers[cluster_id]
        active = self.get_active_clusters() - {cluster_id}

        if not active:
            return []

        # Compute distances to all active cluster centers
        cluster_dists = []
        for cid in active:
            dist = np.linalg.norm(center - self.cluster_centers[cid])
            cluster_dists.append((cid, dist))

        # Sort by distance and return top k
        cluster_dists.sort(key=lambda x: x[1])
        return cluster_dists[:k]
