"""
Merge Trace Tests - Detailed step-by-step comparison of merge decisions.

This file traces each merge decision in both legacy and new implementations
to identify exactly where divergence occurs.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/mohan/Desktop/Gauging-delta/legacy')
sys.path.insert(0, '/Users/mohan/Desktop/Gauging-delta/src')

from perception_original import Perception
from gauging_delta import GaugingDelta


def trace_legacy_merges(X, verbose=True):
    """Run legacy and collect merge trace."""
    legacy = Perception()
    
    # Patch to collect merge info
    original_merge = legacy.merge_clusters
    merge_trace = []
    
    def traced_merge(c1, c2):
        result = original_merge(c1, c2)
        is_merged = result[0]
        if is_merged:
            merged_id = result[1]
            sizes = (len(legacy.initial_clusters[c1]['data']) if c1 in legacy.initial_clusters else 0,
                    len(legacy.initial_clusters[c2]['data']) if c2 in legacy.initial_clusters else 0)
            merge_trace.append({
                'pair': (c1, c2),
                'merged_into': merged_id,
                'sizes': sizes,
                'result': 'MERGED'
            })
        else:
            merge_trace.append({
                'pair': (c1, c2),
                'result': 'REJECTED'
            })
        return result
    
    legacy.merge_clusters = traced_merge
    labels, _ = legacy.fit(X)
    
    if verbose:
        print(f"Legacy: {len(set(labels))} clusters")
        print(f"Legacy merge trace ({len([m for m in merge_trace if m['result'] == 'MERGED'])} merges):")
        for i, m in enumerate(merge_trace[:20]):  # Limit output
            if m['result'] == 'MERGED':
                print(f"  {i}: {m['pair']} -> {m['merged_into']} (sizes={m['sizes']})")
    
    return labels, merge_trace


def trace_new_merges(X, verbose=True):
    """Run new implementation and collect merge trace."""
    model = GaugingDelta()
    
    # We'll use the debug output mechanism
    labels = model.fit(X)
    
    if verbose:
        print(f"New: {len(set(labels))} clusters")
    
    return labels, []


def compare_first_divergence(X, name="test"):
    """Find where legacy and new first diverge."""
    print(f"\n{'='*60}")
    print(f"TRACE: {name}")
    print(f"{'='*60}")
    
    legacy_labels, legacy_trace = trace_legacy_merges(X, verbose=True)
    new_labels, new_trace = trace_new_merges(X, verbose=True)
    
    n_legacy = len(set(legacy_labels))
    n_new = len(set(new_labels))
    
    if n_legacy == n_new:
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(legacy_labels, new_labels)
        print(f"\nResult: SAME cluster count ({n_legacy}), ARI={ari:.4f}")
    else:
        print(f"\nResult: DIFFERENT - legacy={n_legacy}, new={n_new}")
    
    return n_legacy == n_new


def test_linear_chain_trace():
    """Trace linear chain - legacy=1, new=3."""
    np.random.seed(42)
    n_points = 20
    X = np.column_stack([
        np.linspace(0, 10, n_points),
        np.random.randn(n_points) * 0.1
    ])
    compare_first_divergence(X, "linear_chain")


def test_grid_points_trace():
    """Trace grid points - legacy=2, new=1."""
    np.random.seed(42)
    points = []
    for i in range(5):
        for j in range(5):
            noise = np.random.randn(2) * 0.01
            points.append([i + noise[0], j + noise[1]])
    X = np.array(points)
    compare_first_divergence(X, "grid_points_5x5")


def test_two_blobs_trace():
    """Trace two clear blobs."""
    np.random.seed(42)
    c1 = np.random.randn(10, 2) * 0.5
    c2 = np.random.randn(10, 2) * 0.5 + [10, 0]
    X = np.vstack([c1, c2])
    compare_first_divergence(X, "two_blobs")


def detailed_comparison(X, name="test"):
    """Detailed side-by-side comparison of specific merge decisions."""
    from gauging_delta.core.cluster import Cluster
    from gauging_delta.core.neighbor_graph import NeighborGraph
    from gauging_delta.mergeability.continuity import compute_continuity
    
    print(f"\n{'='*70}")
    print(f"DETAILED COMPARISON: {name}")
    print(f"{'='*70}")
    
    # Initialize legacy
    legacy = Perception()
    legacy.X = X
    legacy.DIMENSION = X.shape[1]
    legacy.X_RANGE = (min(X[:, 0]), max(X[:, 0]))
    legacy.Y_RANGE = (min(X[:, 1]), max(X[:, 1]))
    legacy.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
    legacy.FAIL_MATRIX = np.empty((len(X), len(X), 2))
    legacy.initial_clusters = {
        id: {
            'label': id,
            'data': [id],
            'center': x,
            'mean_dist': 0,
            'std_dist': 0,
            'past_dists': [],
            'merging_dists': [],
            'past_densities': [],
            'past_std': [],
            'traces': []
        }
        for id, x in enumerate(X)
    }
    legacy.clusters_dist, legacy.points_dist = legacy.initiate_dists()
    
    # Initialize new
    model = GaugingDelta()
    model._X = X
    model._graph = NeighborGraph(X=X, n_neighbors=min(50, len(X) - 1))
    model._graph.initialize()
    model.clusters_ = {
        i: Cluster(label=i, point_indices=[i], center=X[i].copy())
        for i in range(len(X))
    }
    
    # Get MIN_BTN_CLUSTER_DIST
    DISTS = sorted(legacy.DIST_MATRIX.flatten())
    DISTS = [d for d in DISTS if not np.isinf(d)]
    MIN_BTN = DISTS[int(len(DISTS)/100)] if len(DISTS) > 100 else DISTS[0]
    model._fallback_dist = MIN_BTN
    
    # Compare first few pairs
    print(f"\nMIN_BTN_CLUSTER_DIST = {MIN_BTN:.6f}")
    print(f"\nComparing first 5 closest pairs:")
    print("-" * 70)
    
    # Get sorted pairs from legacy
    n = len(X)
    k = min(n * n - 1, 100)
    k_nearest = legacy.get_indices_of_k_smallest(k=k, matrix=legacy.DIST_MATRIX, sorted=True)
    
    seen_pairs = set()
    count = 0
    for i in range(k_nearest.shape[1]):
        if count >= 5:
            break
        c1, c2 = int(k_nearest[0, i]), int(k_nearest[1, i])
        if np.isinf(legacy.DIST_MATRIX[c1, c2]) or c1 == c2:
            continue
        pair_key = (min(c1, c2), max(c1, c2))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        count += 1
        
        d_ij = legacy.DIST_MATRIX[c1, c2]
        
        # Legacy continuity (simplified - just checking vision_generic)
        print(f"\nPair ({c1}, {c2}): distance = {d_ij:.6f}")
        
        # New continuity
        C_i = model.clusters_[c1]
        C_j = model.clusters_[c2]
        
        T_continuity = 0.15
        T_adaptive = 2.0  # Approximate
        
        new_cont = compute_continuity(
            C_i, C_j, c1, c2, X, T_continuity, d_ij, T_adaptive, _debug=False
        )
        print(f"  New continuity: {new_cont:.4f}")


def test_simple_linear():
    """Test simple linear arrangement."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])
    detailed_comparison(X, "simple_linear_5pts")


def trace_linear_chain_divergence():
    """Deep trace of linear chain divergence."""
    np.random.seed(42)
    n_points = 20
    X = np.column_stack([
        np.linspace(0, 10, n_points),
        np.random.randn(n_points) * 0.1
    ])
    
    print("\n" + "="*70)
    print("LINEAR CHAIN DIVERGENCE ANALYSIS")
    print("="*70)
    
    # Run legacy with full output
    legacy = Perception()
    legacy_labels, _ = legacy.fit(X)
    n_legacy = len(set(legacy_labels))
    print(f"Legacy: {n_legacy} clusters")
    
    # Run new
    model = GaugingDelta()
    new_labels = model.fit(X)
    n_new = len(set(new_labels))
    print(f"New: {n_new} clusters")
    
    if n_legacy != n_new:
        print(f"\nDIVERGENCE: legacy={n_legacy}, new={n_new}")
        
        # Print final cluster assignments
        print("\nLegacy cluster assignments:")
        for i, lbl in enumerate(legacy_labels):
            print(f"  Point {i}: cluster {lbl}")
        
        print("\nNew cluster assignments:")
        for i, lbl in enumerate(new_labels):
            print(f"  Point {i}: cluster {lbl}")


def trace_failing_case():
    """Trace the exact failing hypothesis case."""
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("FAILING HYPOTHESIS CASE ANALYSIS")
    print("="*70)
    print(f"Data shape: {X.shape}")
    
    # Run legacy
    legacy = Perception()
    legacy_labels, _ = legacy.fit(X)
    n_legacy = len(set(legacy_labels))
    
    # Run new  
    model = GaugingDelta()
    new_labels = model.fit(X)
    n_new = len(set(new_labels))
    
    print(f"\nLegacy: {n_legacy} clusters")
    print(f"New: {n_new} clusters")
    
    # Show cluster assignments
    print("\nCluster assignments comparison:")
    print("Point | X coord  | Legacy | New")
    print("-" * 40)
    for i in range(len(X)):
        print(f"  {i:2d}  | {X[i,0]:7.2f} | {int(legacy_labels[i]):6d} | {int(new_labels[i]):3d}")


def trace_bridge_merge():
    """Trace why point 13 (the bridge) fails to merge."""
    
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("BRIDGE MERGE ANALYSIS - Point 13")
    print("="*70)
    
    # Check distance from point 13 to nearby points
    print("\nDistances from point 13 (X=2.53):")
    for i in range(len(X)):
        if i != 13:
            d = np.linalg.norm(X[13] - X[i])
            print(f"  to point {i:2d} (X={X[i,0]:6.2f}): {d:.4f}")
    
    # Point 13's nearest neighbor is likely point 3 (X=1.51) or point 12 (X=3.68)
    d_to_3 = np.linalg.norm(X[13] - X[3])
    d_to_12 = np.linalg.norm(X[13] - X[12])
    print(f"\nClosest points to 13:")
    print(f"  Point 3 (X=1.51): d={d_to_3:.4f}")
    print(f"  Point 12 (X=3.68): d={d_to_12:.4f}")
    
    # Run legacy and check its merge decisions for point 13
    print("\n--- LEGACY MERGE TRACE ---")
    legacy = Perception()
    
    # Patch merge_clusters to trace
    original_merge = legacy.merge_clusters
    def traced_merge(c1, c2):
        result = original_merge(c1, c2)
        if result[0] and (13 in legacy.initial_clusters.get(c1, {}).get('data', []) or 
                          13 in legacy.initial_clusters.get(c2, {}).get('data', [])):
            print(f"  Merge involving point 13: ({c1}, {c2}) -> {result[1]}")
        return result
    legacy.merge_clusters = traced_merge
    
    legacy_labels, _ = legacy.fit(X)
    print(f"Legacy final: {len(set(legacy_labels))} clusters")


def compare_continuity_values():
    """Compare continuity calculation for the critical merge."""
    from gauging_delta.core.cluster import Cluster
    from gauging_delta.core.neighbor_graph import NeighborGraph
    from gauging_delta.mergeability.continuity import compute_continuity
    
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("CONTINUITY COMPARISON FOR CRITICAL MERGE")
    print("="*70)
    
    # Legacy reports: (2, 13) with cont=0.157
    # Let's compute continuity for point 13 merging with a cluster
    
    # Initialize minimal new model state
    model = GaugingDelta()
    model._X = X
    model._graph = NeighborGraph(X=X, n_neighbors=min(50, len(X) - 1))
    model._graph.initialize()
    
    # Create cluster with point 13 only
    C_13 = Cluster(label=13, point_indices=[13], center=X[13].copy())
    
    # Create cluster with point 3 (nearest to 13)
    C_3 = Cluster(label=3, point_indices=[3], center=X[3].copy())
    
    d_ij = np.linalg.norm(X[13] - X[3])
    T_continuity = 0.15
    T_adaptive = 2.0  # Approximate
    
    print(f"\nTest merge: point 13 with point 3")
    print(f"Distance: {d_ij:.4f}")
    
    # Compute continuity with debug
    cont = compute_continuity(
        C_13, C_3, 13, 3, X, T_continuity, d_ij, T_adaptive, _debug=True
    )
    print(f"\nNew continuity: {cont:.4f}")
    print(f"Threshold: {T_continuity}")
    print(f"Would merge: {cont > T_continuity}")


def compare_transition_state():
    """Compare transition state calculation between legacy and new."""
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("TRANSITION STATE COMPARISON")
    print("="*70)
    
    # Initialize legacy
    legacy = Perception()
    legacy.X = X
    legacy.DIMENSION = X.shape[1]
    legacy.X_RANGE = (min(X[:, 0]), max(X[:, 0]))
    legacy.Y_RANGE = (min(X[:, 1]), max(X[:, 1]))
    legacy.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
    legacy.FAIL_MATRIX = np.empty((len(X), len(X), 2))
    legacy.initial_clusters = {
        id: {
            'label': id,
            'data': [id],
            'center': x,
            'mean_dist': 0,
            'std_dist': 0,
            'past_dists': [],
            'merging_dists': [],
            'past_densities': [],
            'past_std': [],
            'traces': []
        }
        for id, x in enumerate(X)
    }
    legacy.clusters_dist, legacy.points_dist = legacy.initiate_dists()
    
    # Compute base parameters
    p_i, p_j = 13, 3
    middle_point = (X[p_i] + X[p_j]) / 2.0
    points_dist = np.linalg.norm(X[p_i] - X[p_j])
    
    # Get MIN_BTN for base_length
    DISTS = sorted(legacy.DIST_MATRIX.flatten())
    DISTS = [d for d in DISTS if not np.isinf(d)]
    MIN_BTN = DISTS[0]
    base_length = MIN_BTN * 2
    
    print(f"\nPoints: p_i={p_i}, p_j={p_j}")
    print(f"Middle point: {middle_point}")
    print(f"Points distance: {points_dist:.4f}")
    print(f"MIN_BTN: {MIN_BTN:.4f}")
    print(f"Base length: {base_length:.4f}")
    
    # Compare transition state at different radii
    print("\n--- LEGACY transition state ---")
    for i, r_rate in enumerate([2, 2.5, 3]):
        r = r_rate * points_dist
        
        # Legacy's find_local_points
        local_i = legacy.find_local_points(p_i, r, X, exclude_point=-1, all=True)
        local_j = legacy.find_local_points(p_j, r, X, exclude_point=-1, all=True)
        
        N_i = len([idx for idx in local_i if idx in legacy.initial_clusters[p_i]['data']])
        N_j = len([idx for idx in local_j if idx in legacy.initial_clusters[p_j]['data']])
        
        # Legacy's compute_transition_state
        trans = legacy.compute_transition_state(local_i, local_j, N_i, N_j)
        print(f"  i={i}: r={r:.2f}, local_i={len(local_i)}, local_j={len(local_j)}, N_i={N_i}, N_j={N_j}, trans={trans:.4f}")


def direct_continuity_comparison():
    """Directly compare continuity values for the failing case."""
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("DIRECT CONTINUITY COMPARISON")
    print("="*70)
    
    # Run legacy to completion, then extract continuity for merge (2, 13)
    # We need to catch when legacy evaluates this merge
    legacy = Perception()
    
    # Patch vision_generic to capture the continuity value
    original_vision = legacy.vision_generic
    captured_continuity = {}
    
    def patched_vision(cluster1, cluster2):
        result = original_vision(cluster1, cluster2)
        # Capture if this involves cluster containing point 13
        c1_data = legacy.initial_clusters.get(cluster1, {}).get('data', [])
        c2_data = legacy.initial_clusters.get(cluster2, {}).get('data', [])
        if 13 in c1_data or 13 in c2_data:
            captured_continuity[(cluster1, cluster2)] = {
                'result': result,
                'c1_data': c1_data,
                'c2_data': c2_data
            }
        return result
    
    legacy.vision_generic = patched_vision
    legacy_labels, _ = legacy.fit(X)
    
    print(f"\nLegacy: {len(set(legacy_labels))} clusters")
    print(f"\nCaptured merges involving point 13:")
    for (c1, c2), info in captured_continuity.items():
        print(f"  ({c1}, {c2}): result={info['result']}")
        print(f"    c1_data={info['c1_data'][:5]}... c2_data={info['c2_data'][:5]}...")


def trace_legacy_transition_state():
    """Trace legacy's exact transition state values."""
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("LEGACY TRANSITION STATE TRACE")
    print("="*70)
    
    # Initialize legacy
    legacy = Perception()
    
    # Patch compute_transition_state to capture values
    original_trans_state = legacy.compute_transition_state
    
    def traced_trans_state(p1, p2, middle_point, N1, N2, radius, points_set=None):
        result = original_trans_state(p1, p2, middle_point, N1, N2, radius, points_set)
        print(f"    [legacy trans] p1={p1}, p2={p2}, N1={N1}, N2={N2}, radius={radius:.4f}")
        print(f"    [legacy trans] r_e={result[0]}, r_i={result[1]}, g_i={result[2]}, g_e={result[3]}")
        return result
    
    legacy.compute_transition_state = traced_trans_state
    
    # Patch compute_local_transition to see when point 13 is involved
    original_local_trans = legacy.compute_local_transition
    
    def traced_local_trans(cluster1, cluster2, THRESHOLD_CONTINUATION, mean_dist, prox_threshold):
        c1_data = legacy.initial_clusters.get(cluster1, {}).get('data', [])
        c2_data = legacy.initial_clusters.get(cluster2, {}).get('data', [])
        if 13 in c1_data or 13 in c2_data:
            print(f"\n  [legacy] compute_local_transition({cluster1}, {cluster2})")
            print(f"    c1_data={c1_data}, c2_data={c2_data}")
        result = original_local_trans(cluster1, cluster2, THRESHOLD_CONTINUATION, mean_dist, prox_threshold)
        if 13 in c1_data or 13 in c2_data:
            print(f"    result smoothness={result}")
        return result
    
    legacy.compute_local_transition = traced_local_trans
    
    legacy_labels, _ = legacy.fit(X)
    print(f"\nLegacy final: {len(set(legacy_labels))} clusters")


def compare_radius_computation():
    """Compare radius/base_length between legacy and new."""
    from gauging_delta.core.cluster import Cluster
    from gauging_delta.core.neighbor_graph import NeighborGraph
    
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("RADIUS COMPARISON")
    print("="*70)
    
    # Initialize legacy
    legacy = Perception()
    legacy.X = X
    legacy.DIMENSION = X.shape[1]
    legacy.X_RANGE = (min(X[:, 0]), max(X[:, 0]))
    legacy.Y_RANGE = (min(X[:, 1]), max(X[:, 1]))
    legacy.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
    legacy.FAIL_MATRIX = np.empty((len(X), len(X), 2))
    legacy.initial_clusters = {
        id: {
            'label': id,
            'data': [id],
            'center': X[id],
            'mean_dist': 0,
            'std_dist': 0,
            'past_dists': [],
            'merging_dists': [],
            'past_densities': [],
            'past_std': [],
            'traces': []
        }
        for id in range(len(X))
    }
    legacy.clusters_dist, legacy.points_dist = legacy.initiate_dists()
    
    # Simulate the merge scenario: cluster 2 has points [2,6,1,3,0,4], cluster 13 has [13]
    # Reference points for this merge would be p1=3 (closest in cluster 2 to 13), p2=13
    p1, p2 = 3, 13
    points_dist = np.linalg.norm(X[p1] - X[p2])
    middle_point = (X[p1] + X[p2]) / 2
    
    print(f"\nReference points: p1={p1}, p2={p2}")
    print(f"Points distance: {points_dist:.4f}")
    print(f"Middle point: {middle_point}")
    
    # Legacy base_length computation (simplified for singleton merge)
    # For initial merges, past_dists is empty, so c_past_dists = [points_dist]
    c_past_dists = [points_dist]
    compact = 1.0  # Both singletons: 0.5 + 0.5 = 1.0
    center_dist = np.linalg.norm(X[p1] - X[p2]) / compact
    rate = max(center_dist / points_dist, 3) - 3
    enlarge_rate = 2 / (1 + np.exp(-rate / 20))
    legacy_base_length = np.mean(c_past_dists) * enlarge_rate
    
    print(f"\nLegacy base_length computation:")
    print(f"  c_past_dists: {c_past_dists}")
    print(f"  compact: {compact}")
    print(f"  center_dist: {center_dist:.4f}")
    print(f"  rate: {rate:.4f}")
    print(f"  enlarge_rate: {enlarge_rate:.4f}")
    print(f"  base_length: {legacy_base_length:.4f}")
    
    # Radii for explore_range [2, 2.5, 3]
    for i, r_mult in enumerate([2, 2.5, 3]):
        r = r_mult * legacy_base_length
        print(f"  radius[{i}] (mult={r_mult}): {r:.4f}")


def trace_new_algorithm_debug():
    """Trace new algorithm with debug enabled."""
    X = np.array([[ 1.18522267, -0.10169312],
           [ 0.26885562, -0.06935017],
           [ 0.65758958,  0.27587087],
           [ 1.50560012,  0.09677927],
           [ 1.25476552,  0.97709001],
           [-0.65660858,  0.51132222],
           [ 0.63834066,  0.0817504 ],
           [ 4.54822061,  0.43914999],
           [ 4.47418744,  0.58079307],
           [ 5.25383438, -0.49864212],
           [ 4.11215929,  1.52499139],
           [ 4.46034236, -0.97715194],
           [ 3.67615443,  0.030744  ],
           [ 2.53470997, -0.12576415]])
    
    print("\n" + "="*70)
    print("NEW ALGORITHM DEBUG TRACE")
    print("="*70)
    
    # Patch _compute_mergeability to trace ALL calls
    from gauging_delta import GaugingDelta as GD
    original_mergeability = GD._compute_mergeability
    
    first_call = [True]  # Track first call only
    
    def traced_mergeability(self, C_i, C_j, d_ij):
        if 13 in C_i.point_indices or 13 in C_j.point_indices:
            if first_call[0]:
                # Enable debug for first call involving point 13
                from gauging_delta.mergeability import continuity as cont_mod
                original_cont = cont_mod.compute_continuity
                
                def debug_cont(*args, **kwargs):
                    kwargs['_debug'] = True
                    return original_cont(*args, **kwargs)
                
                cont_mod.compute_continuity = debug_cont
                result = original_mergeability(self, C_i, C_j, d_ij)
                cont_mod.compute_continuity = original_cont
                first_call[0] = False
                
                status = "ACCEPT" if result.is_mergeable else "REJECT"
                print(f"  [mergeability] {status}: C_i={C_i.label}(pts={C_i.point_indices}), "
                      f"C_j={C_j.label}(pts={C_j.point_indices})")
                print(f"    rho={result.rho:.4f}, cont={result.continuity:.4f}, T_cont=0.15")
                print(f"    merge_hist_i={C_i.merge_history}, merge_hist_j={C_j.merge_history}")
                return result
        
        return original_mergeability(self, C_i, C_j, d_ij)
    
    GD._compute_mergeability = traced_mergeability
    
    model = GaugingDelta()
    labels = model.fit(X)
    
    print(f"\nNew final: {len(set(labels))} clusters")
    print(f"Cluster assignments for point 13: {labels[13]}")
    
    # Show which cluster contains point 13
    for i, lbl in enumerate(labels):
        if lbl == labels[13]:
            print(f"  Point {i} in same cluster as 13")
    
    # Restore
    GD._compute_mergeability = original_mergeability


if __name__ == "__main__":
    trace_new_algorithm_debug()
