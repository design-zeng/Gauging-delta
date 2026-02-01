"""
Visualization utilities for Gauging-δ algorithm.

This module is optional and requires matplotlib.
Install with: pip install gauging-delta[viz]
"""

try:
    from gauging_delta.visualization.plotter import ClusterPlotter, plot_clusters

    __all__ = ["ClusterPlotter", "plot_clusters"]
except ImportError:
    __all__ = []
