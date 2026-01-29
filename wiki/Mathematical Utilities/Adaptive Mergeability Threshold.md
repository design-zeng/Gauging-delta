---
kind: conceptual
title: "Adaptive Mergeability Threshold"
description: "Dynamic thresholding for cluster mergeability assessment"
complexity: "O(k) per calculation where k is neighborhood size"
---

# Adaptive Mergeability Threshold

> [!TIP] Quick Summary
> This threshold dynamically adjusts merge criteria based on **historical patterns**, **local forces**, and **shape similarity**—preventing both over-merging and under-merging.

---

## Core Formula

$$
T_{\text{adaptive}} = \underbrace{\beta_{ij}}_{\text{environment}} \cdot \underbrace{T_{\text{stat}}}_{\text{history}} \cdot \underbrace{\xi_s}_{\text{shape}}
$$

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| $\beta_{ij}$ | Force-based scaling from nearby clusters | 1.0 – 2.0 |
| $T_{\text{stat}}$ | Statistical threshold from merge history | 1.4 – 4.4 |
| $\xi_s$ | Shape similarity adjustment | 0.5 – 1.5 |

---

## Component Details

### 1. Statistical Threshold $T_{\text{stat}}$

Captures how consistent past merges have been:

$$
T_{\text{stat}} = \frac{3}{1 + e^{0.3 \cdot \mu/\sigma}} + 1.4
$$

| Variable | Definition |
|----------|------------|
| $\mu$ | Mean of historical merge distances |
| $\sigma$ | Std dev of historical merge distances |

> [!WARNING] Paper Divergence
> The Paper (Eq. 4) uses a sigmoid based on the coefficient of variation ($\sigma/\mu$), whereas the Code (`perception.py` Line 405) uses the inverse ($\mu/\sigma$) and different constants.

> [!NOTE] Edge Case
> Returns **2.7** when $\sigma = 0$, $\mu = 0$, or history has ≤3 samples.

**Intuition:** High variance → lower threshold → more permissive merging.

---

### 2. Environment Factor $\beta_{ij}$

Accounts for forces from neighboring clusters using a **Weighted Average** model:

$$
\beta_{ij} = \sum w_k \cdot S_k
$$

Where for each contextual cluster $C_k$ selected by interaction force:

| Formula | Purpose |
|---------|---------|
| $S_k = \frac{2}{1 + e^{5 \cdot d_{\text{cand}}/d_{k}}} + 1$ | Distance ratio scaling |
| $w_k = F_{\text{interaction}}(C_k) / \sum F_{\text{total}}$ | Normalized force weight |

> [!WARNING] Paper Divergence
> The Paper describes selecting a single "most influential" cluster. The Code (`perception.py` Lines 392-396) implements a weighted average of multiple influential clusters.

---

### 3. Shape Similarity $\xi_s$

Adjusts for geometric compatibility:

$$
\xi_s = \frac{r}{1 + r} + 0.5 \quad \text{where} \quad r = \max\left(\frac{\sigma_{\text{recent}}}{\sigma_{\text{hist}}}\right)
$$

> [!NOTE] Small Sample Protection
> $\xi_s = 1$ if fewer than 5 statistical samples available.

---

## Dual Threshold System

Both clusters must pass independently:

$$
\text{merge\_allowed} = (\rho \le T_1) \land (\rho \le T_2)
$$

Where $T_1$ and $T_2$ use each cluster's own statistical history.

---

## Tuning Reference

| Constant | Role | Adjustment Tip |
|----------|------|----------------|
| `0.3` | Variance sensitivity | ↑ for denser data |
| `3`, `1.4` | Threshold range bounds | — |
| `5` | Distance ratio sensitivity | ↓ for noisy data |
| `0.5` | Shape penalty baseline | ↑ for complex shapes |

---

## See Also


