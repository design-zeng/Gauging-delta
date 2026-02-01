"""
Mathematical utility functions for Gauging-δ algorithm.
"""

import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that returns default when denominator is zero.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero
    """
    if denominator == 0:
        return default
    return numerator / denominator


def sigmoid(x: float, scale: float = 1.0, shift: float = 0.0) -> float:
    """
    Sigmoid function: 1 / (1 + e^(-scale * (x - shift)))

    Args:
        x: Input value
        scale: Steepness of sigmoid
        shift: Horizontal shift

    Returns:
        Sigmoid output in (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-scale * (x - shift)))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range [min_val, max_val].

    Args:
        value: Input value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))
