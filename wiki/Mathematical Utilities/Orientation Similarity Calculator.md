---
kind: conceptual
title: "Orientation Similarity Calculator"
description: "Measures orientation similarity between clusters via mass smoothness (ξ_o)"
complexity: "O(n_local)"
---

# Orientation Similarity Calculator

> [!TIP] Quick Summary
> Evaluates **orientation consistency** between neighboring clusters by combining local density balance (mass smoothness) with angular alignment (angle smoothness) using a geometric mean.

> [!WARNING] Paper Divergence
> The Paper defines $\xi_o$ as a **density ratio**: $\rho_1 / \max(\rho_1, \rho_2)$.
> The Code (`perception.py` Lines 820-834) implements a **Mass Smoothness** metric that combines local mass balance with angular transition quality.

---

## Code Implementation

### Step 1: Compute Local Mass

For each cluster's local region around the closest point pair:

$$
\text{mass}_i = N_i \times \text{area}_i
$$

| Symbol | Definition | Code Reference |
|--------|------------|----------------|
| $N_i$ | Number of points in local region + 1 | `perception.py` Line 791 |
| $\text{area}_i$ | Angular spread (max angle from `compute_max_angle`) | `perception.py` Line 811 |

### Step 2: Mass Smoothness

Ratio of local masses (min/max normalization):

$$
\text{mass\_smoothness} = \frac{\min(\text{mass}_1, \text{mass}_2)}{\max(\text{mass}_1, \text{mass}_2)}
$$

```python
# perception.py Line 820
locality_info[i]['mass_smoothness'] = min(mass1, mass2) / max(mass1, mass2)
```

### Step 3: Orientation Smoothness (Combined)

Geometric mean of mass smoothness and angle transition smoothness:

$$
\xi_o = \sqrt{\text{mass\_smoothness} \times \text{angle\_smoothness}}
$$

```python
# perception.py Lines 832-833
orientation_smoothness = math.sqrt(locality_info[i]['mass_smoothness'] *
                                   locality_info[i]['angle_smoothness'])
```

---

## Interpretation

| Score | Meaning |
|-------|---------|
| 0.9 - 1.0 | Highly compatible — balanced mass and smooth angle transition |
| 0.6 - 0.9 | Moderately similar — minor density or angular differences |
| 0.3 - 0.6 | Significant differences — structural asymmetry |
| < 0.3 | Poor compatibility — likely different cluster types |

---

## Why Geometric Mean?

The geometric mean ensures that **both factors must be good** for a high score:

| mass_smoothness | angle_smoothness | orientation_smoothness |
|-----------------|------------------|------------------------|
| 1.0 | 1.0 | 1.0 |
| 1.0 | 0.25 | 0.5 |
| 0.25 | 1.0 | 0.5 |
| 0.25 | 0.25 | 0.25 |

This implements a **"weakest link"** philosophy — a cluster pair with good density balance but poor angular transition (or vice versa) receives a reduced score.

---

## See Also

- [[Angle Transition Calculator]] — Provides `angle_smoothness` component
- [[Density Continuation Calculator]] — Provides `transition_smoothness` for final continuity
- [[Shape Similarity Calculator]] — Separate shape-based filtering
