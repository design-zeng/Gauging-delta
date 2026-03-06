"""Shared fixtures for Gauging-delta tests.

Session-scoped legacy results to avoid re-running Perception.fit() per test.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest


# Ensure the legacy directory (containing perception.py) is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "legacy"))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BENCHMARK_DATASETS = {
    "flame": DATA_DIR / "flame.txt",
    "3_blobs": DATA_DIR / "3_blobs.txt",
    "pathbased": DATA_DIR / "pathbased.txt",
    "3-spiral": DATA_DIR / "3-spiral.txt",
    "jain": DATA_DIR / "jain.txt",
    "compound": DATA_DIR / "compound.txt",
}


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y_true) from a comma-separated txt file."""
    data = np.loadtxt(str(path), delimiter=",")
    return data[:, :2], data[:, 2].astype(int)


def _run_legacy(X: np.ndarray) -> np.ndarray:
    """Run Perception.fit and extract raw cluster labels."""
    from perception import Perception

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        p = Perception(k=None)
        p.fit(X)
    finally:
        sys.stdout = old_stdout

    labels = np.full(len(X), -1, dtype=int)
    for cid, cl in p.initial_clusters.items():
        for pt in cl["data"]:
            labels[pt] = cid
    return labels


# ---------------------------------------------------------------------------
# Session-scoped fixtures: run legacy ONCE per dataset per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All benchmark datasets as {name: (X, y_true)}."""
    return {name: _load_dataset(path) for name, path in BENCHMARK_DATASETS.items()}


@pytest.fixture(scope="session")
def legacy_labels(datasets: dict) -> dict[str, np.ndarray]:
    """Legacy labels for all benchmark datasets, computed once per session."""
    return {name: _run_legacy(X) for name, (X, _) in datasets.items()}
