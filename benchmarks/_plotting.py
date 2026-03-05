"""Shared plotting utilities for benchmarks."""

from __future__ import annotations

import matplotlib.pyplot as plt


def use_science_style() -> None:
    """Activate science-style plots if scienceplots is installed."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except (ImportError, OSError):
        plt.style.use("default")
