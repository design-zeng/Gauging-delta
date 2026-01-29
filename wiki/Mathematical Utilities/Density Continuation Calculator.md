---
kind: conceptual
title: "Density Continuation Calculator"
description: "Evaluates density transition quality at cluster boundaries (ξ_d)"
complexity: "O(n_boundary × n_total)"
---

# Density Continuation Calculator

> [!TIP] Quick Summary
> Measures **density uniformity** at cluster boundaries using variance-based analysis.


> [!WARNING] Paper Divergence
> The Paper defines $\xi_d$ as a **ratio of point counts** in overlapping vs. non-overlapping regions:
> $$\xi_d^{\text{(paper)}} = \frac{\min(|P_{i \cap j}|, |P_{j \cap i}|)}{\max(|P_{i-j}|, |P_{j-i}|)}$$
> The Code (`perception.py` Lines 1176-1188) implements a **Case-based Branching Logic** that compares "internal" vs "external" point counts ($r_i, g_i, r_e, g_e$) in the transition region.

---

## Code Implementation

### Step 1: Transition State Analysis

For each cluster (Red and Green), points in the local transition region are classified as:
- **Internal ($N_i$)**: Points belonging to the cluster that are within the transition radius.
- **External ($N_e$)**: Points belonging to the cluster that are *outside* the transition radius but relevant.

### Step 2: Smoothness Calculation

The smoothness score is determined by branching logic designed to handle boundary asymmetry:

**Case A: Missing External Points**
If $r_e = 0$ or $g_e = 0$, the structure is considered incomplete or a hard boundary:
$$ \xi_d = \max\left(2, \frac{\max(N_1, N_2)}{\min(N_1, N_2) \cdot \text{rate}}\right) $$

**Case B: Asymmetric Density Drop**
If internal density is significantly lower than external density (indicating a gap):
$$ \xi_d = \min\left( \frac{\min(r_i, g_i)}{\max(g_i, g_e)}, \frac{\min(r_i, g_i)}{\max(r_i, r_e)} \right) $$

**Case C: Standard Continuity**
Otherwise, checks if ratios are below a threshold (0.1):
- If $\frac{r_e}{\max(g_i, g_e, r_i)} \le 0.1$ or reciprocal: $\xi_d = 1.5$ (Boosted)
- Else: $\xi_d = 1.0$ (Stable)

### Integration

The result is combined with Angle and Shape smoothness:
$$ \xi_{final} = \min(\xi_d \cdot \xi_{orientation}, 1) $$

---

## See Also

- [[Angle Transition Calculator]] — $\xi_a$
- [[Orientation Similarity Calculator]] — $\xi_o$ (statistical similarity)
- [[Shape Similarity Calculator]] — threshold modifier
