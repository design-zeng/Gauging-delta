---
kind: conceptual
title: "Shape Similarity Calculator"
description: "Compares cluster shape evolution over time (ξ_s)"
complexity: "O(N) where N = history length"
---

# Shape Similarity Calculator

> [!TIP] Quick Summary
> Prevents merging clusters with **radically different shape evolutions** (e.g., long chain vs. compact ball).

---

## Core Formula

$$
\xi_s = \frac{\phi}{1 + \phi} + 0.5
$$

where

$$
\phi = \frac{\min(\sigma_{is}, \sigma_{js})}{\max(\sigma_{is}, \sigma_{js})}
$$

| Symbol | Meaning |
|--------|---------|
| $\sigma_{is}$ | Std dev of shape metric history for cluster $i$ |
| $\phi$ | Ratio of shape metric std devs (always ≤ 1) |

---

## Shape Metric

| Term | Definition |
|------|------------|
| $s_i$ | Std dev of distances from points to centroid (Eq 19) |
| $\sigma_s$ | Std dev of $s_i$ values accumulated during merging |

---

## Integration

Shape similarity is used in **two phases** of the pipeline:

### 1. Proximity Phase

Scales the adaptive threshold for proximity filtering:

$$
\text{Threshold}_{\text{proximity}} = \text{Threshold}_{\text{adaptive}} \cdot \xi_s
$$

### 2. Continuity Phase

Also used to adjust the continuity threshold:

$$
\text{Threshold}_{\text{continuity}} = \frac{\text{Threshold}_{\text{base}}}{\xi_s}
$$

> [!NOTE] Dual Usage
> Unlike other metrics, `shape_diff` influences **both** the proximity filtering (multiplicative) and the continuity analysis (divisive). This creates a symmetric effect: dissimilar shapes make proximity harder to pass AND continuity easier to reject.

> [!WARNING] Paper Divergence
> The Paper (Eq. 8) describes Shape Similarity only as a component of the **Continuity** function. The Code uses it in **both** phases for a more comprehensive shape-aware filtering.

---

## Interpretation

- **Similar shapes** → higher $\xi_s$ → easier merge
- **Different shapes** → lower $\xi_s$ → requires stronger factors

---

## See Also

- [[Density Continuation Calculator]] — $\xi_d$
- [[Angle Transition Calculator]] — $\xi_a$
- [[Orientation Similarity Calculator]] — $\xi_o$
