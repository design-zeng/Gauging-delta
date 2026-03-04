"""Gauging-delta: non-parametric hierarchical clustering."""

from gauging_delta.algorithm import GaugingDelta
from gauging_delta.cluster import Cluster
from gauging_delta.config import GaugingDeltaConfig
from gauging_delta.continuity import DefaultContinuity
from gauging_delta.linkage import DefaultLinkage
from gauging_delta.proximity import DefaultProximity


__all__ = [
    "Cluster",
    "DefaultContinuity",
    "DefaultLinkage",
    "DefaultProximity",
    "GaugingDelta",
    "GaugingDeltaConfig",
]
