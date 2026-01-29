---
kind: conceptual
title: "Local Compactness Measure"
description: "Quantifies internal cluster cohesion and determines exploration radius"
complexity: "O(n_local × |C|)"
---

# Local Compactness Measure

> [!TIP] Quick Summary
> Measures how **tightly packed** a cluster's points are, then uses this to determine the **exploration radius** for continuity analysis.

> [!WARNING] Paper Divergence
> The Paper (Eq. 13) defines compactness ($\tau$) as the ratio of the latest merging distance to the average historical distance ($d_{merge}/d_{avg}$). The Code (`perception.py` Lines 734-752) implements a **Log-Normalized Coefficient of Variation** and converts it to an adaptive radius via sigmoid scaling.

---

## Core Formula

$$
\tau_i = \underbrace{\frac{\sigma_i}{\mu_i \cdot \ln|C_i|}}_{dispersion} \cdot \underbrace{\frac{|C_i|}{|C_1| + |C_2|}}_{size\ weight}
$$

| Symbol | Meaning | Code Reference |
|--------|---------|----------------|
| $\sigma_i$ | Std dev of past merge distances | `past_std[-1]` |
| $\mu_i$ | Mean of past merge distances | `np.mean(past_dists)` |
| $\|C_i\|$ | Current cluster size | `size1`, `size2` |

---

## Combined Compactness

The total compactness is the sum of both clusters' contributions:

$$
\tau_{combined} = \tau_1 + \tau_2
$$

---

## Conversion to Exploration Radius

> [!IMPORTANT] Code Flow
> The compactness value is **not used directly** in the continuity score. Instead, it determines the **radius** of the local exploration area via a multi-step transformation.

### Step 1: Normalize Center Distance

$$
d_{norm} = \frac{d_{center}}{\tau_{combined}}
$$

### Step 2: Compute Rate

$$
\text{rate} = \max\left(\frac{d_{norm}}{d_{points}}, 3\right) - 3
$$

- Rate ≥ 0 (floor at 3 then subtract 3)
- Higher rate = clusters are more separated relative to closest points

### Step 3: Sigmoid Scaling (enlarge_rate)

$$
\text{enlarge\_rate} = \frac{2}{1 + e^{-\text{rate}/20}}
$$

| rate | enlarge_rate |
|------|--------------|
| 0 | 1.0 |
| 10 | ~1.3 |
| 20 | ~1.5 |
| ∞ | → 2.0 |

### Step 4: Base Length

$$
\text{base\_length} = \bar{d}_{past} \times \text{enlarge\_rate}
$$

### Step 5: Exploration Radii

Multiple radii are explored as multiples of the base length. In code, this corresponds to the `explore_range` used to define the local neighborhood radius.

---

## Interpretation

| Compactness | Effect on Radius |
|-------------|------------------|
| **Low τ** | Tight clusters → smaller normalized center_dist → smaller rate → radius ≈ base |
| **High τ** | Loose clusters → larger normalized center_dist → larger rate → radius expands |

This ensures **loose clusters get larger exploration areas** to find compatible neighbors.

---

## See Also

- [[Density Continuation Calculator]] — Uses the exploration radius
- [[Orientation Similarity Calculator]] — Computed within each radius
