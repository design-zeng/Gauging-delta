"""Shared plotting utilities for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


def use_science_style() -> None:
    """Activate science-style plots if scienceplots is installed."""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex", "grid"])
    except (ImportError, OSError):
        plt.style.use("default")


# ---------------------------------------------------------------------------
# Catppuccin theme infrastructure (Mocha = dark, Latte = light)
# Hex values from https://catppuccin.com/palette — no pip dependency needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatppuccinTheme:
    """Color palette for one Catppuccin flavor."""

    # Chrome (text, axes, annotations)
    text: str
    spine: str
    ref_line: str
    ref_text: str
    footnote: str
    # Data series
    full: str  # Full mode / primary series
    lite: str  # Lite mode / secondary series
    dist: str  # Distance init phase
    ceil: str  # Ceiling line / accent
    # Scatter cluster cycle
    cluster_colors: tuple[str, ...]
    # Input/neutral points
    input_points: str


MOCHA = CatppuccinTheme(
    text="#cdd6f4",
    spine="#cdd6f4",
    ref_line="#45475a",
    ref_text="#9399b2",
    footnote="#9399b2",
    full="#74c7ec",
    lite="#fab387",
    dist="#89b4fa",
    ceil="#f38ba8",
    cluster_colors=("#89b4fa", "#fab387", "#a6e3a1", "#f38ba8", "#cba6f7", "#94e2d5"),
    input_points="#7f849c",
)

LATTE = CatppuccinTheme(
    text="#4c4f69",
    spine="#4c4f69",
    ref_line="#bcc0cc",
    ref_text="#7c7f93",
    footnote="#7c7f93",
    full="#209fb5",
    lite="#fe640b",
    dist="#1e66f5",
    ceil="#d20f39",
    cluster_colors=("#1e66f5", "#fe640b", "#40a02b", "#d20f39", "#8839ef", "#179299"),
    input_points="#8c8fa1",
)


def apply_theme(theme: str) -> CatppuccinTheme:
    """Apply Catppuccin Mocha (dark) or Latte (light) to rcParams.

    Returns the theme object so callers can use data-series and
    annotation colors that are not covered by rcParams.
    """
    tc = MOCHA if theme == "dark" else LATTE
    plt.rcParams.update(
        {
            "text.color": tc.text,
            "axes.edgecolor": tc.spine,
            "axes.labelcolor": tc.text,
            "xtick.color": tc.text,
            "ytick.color": tc.text,
        }
    )
    return tc
