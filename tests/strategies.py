"""
Custom Hypothesis strategies for Gauging-δ tests.

Provides generators for:
    - Valid point clouds and clusters
    - Merge history sequences
    - Geometric configurations
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# Basic numeric strategies
positive_floats = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
small_positive_floats = st.floats(
    min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False
)
angles = st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False)


@st.composite
def points_2d(draw, min_points=3, max_points=50):
    """Generate 2D point arrays."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    return draw(
        arrays(
            dtype=np.float64,
            shape=(n, 2),
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        )
    )


@st.composite
def points_nd(draw, min_dim=2, max_dim=10, min_points=3, max_points=50):
    """Generate N-dimensional point arrays."""
    n = draw(st.integers(min_value=min_points, max_value=max_points))
    d = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return draw(
        arrays(
            dtype=np.float64,
            shape=(n, d),
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        )
    )


@st.composite
def vector_pair(draw, dim=2):
    """Generate pair of vectors for angle computation."""
    v1 = draw(
        arrays(
            dtype=np.float64,
            shape=(dim,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    v2 = draw(
        arrays(
            dtype=np.float64,
            shape=(dim,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    # Ensure non-zero vectors
    if np.linalg.norm(v1) < 0.001:
        v1 = np.array([1.0] + [0.0] * (dim - 1))
    if np.linalg.norm(v2) < 0.001:
        v2 = np.array([0.0, 1.0] + [0.0] * (dim - 2))
    return v1, v2


@st.composite
def merge_history(draw, min_length=0, max_length=20):
    """Generate valid merge history (list of positive distances)."""
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(
        st.lists(
            positive_floats,
            min_size=n,
            max_size=n,
        )
    )


@st.composite
def sigma_history(draw, min_length=0, max_length=20):
    """Generate valid sigma history (list of non-negative std devs)."""
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    return draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )


@st.composite
def cluster_data(draw, min_points=1, max_points=30):
    """Generate data for a single cluster."""
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    point_indices = list(range(n_points))
    merge_hist = draw(merge_history(max_length=n_points))
    sigma_hist = draw(sigma_history(max_length=n_points))

    return {
        "label": draw(st.integers(min_value=0, max_value=1000)),
        "point_indices": point_indices,
        "merge_history": merge_hist,
        "sigma_history": sigma_hist,
    }


@st.composite
def three_points_2d(draw):
    """Generate three 2D points for angle tests (origin, p1, p2)."""
    origin = draw(
        arrays(
            dtype=np.float64,
            shape=(2,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    p1 = draw(
        arrays(
            dtype=np.float64,
            shape=(2,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    p2 = draw(
        arrays(
            dtype=np.float64,
            shape=(2,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )

    # Ensure points are distinct
    while np.allclose(origin, p1) or np.allclose(origin, p2) or np.allclose(p1, p2):
        p1 = origin + np.array([1.0, 0.0])
        p2 = origin + np.array([0.0, 1.0])

    return origin, p1, p2
