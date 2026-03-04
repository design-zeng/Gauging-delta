"""Configuration dataclass holding all algorithm constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaugingDeltaConfig:
    """All magic numbers from the Gauging-delta algorithm.

    Every constant from perception.py is extracted here so nothing is
    hardcoded in the algorithm modules.  Override any field via::

        GaugingDelta(config=GaugingDeltaConfig(threshold_continuity=0.2))
    """

    # --- Thresholds ---
    threshold_continuity: float = 0.15  # T_cont (perception.py L44)
    num_nearest_clusters: int = 5  # K neighbors (perception.py L45)

    # --- Distance initialization ---
    min_dist_percentile: float = 0.01  # index = len/100 (perception.py L213)
    mean_wtn_fallback: float = 0.0001  # MEAN_WTN_CLUSTER_DIST (perception.py L215)

    # --- Vision scale sigmoid: coeff / (1 + e^(exp_rate * x)) + offset ---
    vision_scale_coeff: float = 2.0
    vision_scale_exp: float = 5.0
    vision_scale_offset: float = 1.0

    # --- T_stat sigmoid: numerator / (1 + e^(exp_coeff * mu/sigma)) + offset ---
    t_stat_numerator: float = 3.0  # (perception.py L405)
    t_stat_exp_coeff: float = 0.3  # (perception.py L405)
    t_stat_offset: float = 1.4  # (perception.py L405)
    t_stat_fallback: float = 2.7  # when mu=0 or sigma=0
    t_stat_min_history: int = 3  # need > 3 past_dists for T_stat (perception.py L402)

    # --- Shape similarity (xi_s) ---
    xi_s_offset: float = 0.5  # diff/(1+diff) + 0.5 (perception.py L466)
    xi_s_min_history: int = 4  # returns 1.0 when N <= 4 (perception.py L466)

    # --- Continuity radii ---
    explore_radii: tuple[float, ...] = (2.0, 2.5, 3.0)  # (perception.py L755)

    # --- Enlarge rate: 2 / (1 + e^(-rate / divisor)) ---
    enlarge_rate_divisor: float = 20.0  # (perception.py L752)
    enlarge_rate_floor: float = 3.0  # max(x, 3) - 3 (perception.py L751)

    # --- Rounding precision ---
    angle_rounding: int = 4  # round(..., 4) in to_find_angle (perception.py L1444,1447)
    std_ratio_rounding: int = 3  # round(..., 3) in shape_diff (perception.py L449,456)

    # --- Force computation ---
    n_contextual_clusters: int = 2  # N_rc in adaptive threshold (perception.py L375)

    # --- Outlier detection ---
    outlier_min_points: int = 3  # need > 3 local points to check (perception.py L1039)
    max_angle_zero_threshold: float = 0.2  # (perception.py L812)

    # --- Transition smoothness ---
    boundary_threshold: float = 2.0  # transition > 2 = boundary (perception.py L840)
    mass_ratio_jump: float = 2.718281828  # e ~ np.e (perception.py L857)
