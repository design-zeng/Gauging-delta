"""
Pytest configuration and shared fixtures for Gauging-δ tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add project root for perception.py (parity tests)
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_2d_data():
    """Simple 2D dataset with 3 clear clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([5, 0])
    cluster3 = np.random.randn(20, 2) + np.array([2.5, 5])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_3d_data():
    """Simple 3D dataset with 2 clear clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(15, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(15, 3) + np.array([5, 5, 5])
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def collinear_points():
    """Points arranged in a line (for angle tests)."""
    return np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
        ]
    )


@pytest.fixture
def right_angle_points():
    """Points forming a right angle."""
    return np.array(
        [
            [0, 0],  # origin
            [1, 0],  # along x-axis
            [0, 1],  # along y-axis
        ]
    )


@pytest.fixture
def load_test_datasets():
    """Load all test datasets from data/ directory."""
    data_dir = Path(__file__).parent.parent / "data"
    datasets = {}
    for file in data_dir.glob("*.txt"):
        # Try comma delimiter first, then whitespace
        try:
            data = np.loadtxt(file, delimiter=",")
        except ValueError:
            data = np.loadtxt(file)
        # Assume last column is label if present
        if data.shape[1] > 2:
            X = data[:, :-1]
            y = data[:, -1]
        else:
            X = data
            y = None
        datasets[file.stem] = {"X": X, "y": y}
    return datasets
