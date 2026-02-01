"""
Gauging-δ: A Non-Parametric Hierarchical Clustering Algorithm

This package implements the Gauging-δ clustering algorithm as described in:
    Yao, Pan, Zeng. "Gauging-δ: A Non-Parametric Hierarchical Clustering Algorithm"
    IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 47, No. 6, June 2025

The algorithm employs a hierarchical merging process guided by an adaptive
mergeability function that considers proximity and continuity statistics.
"""

from gauging_delta.core.algorithm import GaugingDelta
from gauging_delta.core.cluster import Cluster, ClusterPairInfo, MergeabilityResult


__version__ = "0.1.0"
__all__ = [
    "Cluster",
    "ClusterPairInfo",
    "GaugingDelta",
    "MergeabilityResult",
]
