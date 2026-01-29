---
kind: conceptual
title: "Perceptual Proximity Statistic"
description: "Scale-invariant distance assessment for merge decisions"
complexity: "O(n) where n = history length"
---

# Perceptual Proximity Statistic

> [!TIP] Quick Summary
> Normalizes current distance against **historical merge patterns** to create a scale-invariant merge criterion.

---

## Core Formula

$$
\rho = \frac{d_{\text{current}}}{\mu_{\text{historical}}}
$$

| Symbol | Meaning |
|--------|---------|
| $d_{\text{current}}$ | Euclidean distance between reference points |
| $\mu_{\text{historical}}$ | Mean of recent merge distances from both clusters |

---

## History Length

$$
n = \begin{cases}
\max(\lfloor n_1/2 \rfloor, n_2) & \text{if } n_1 > n_2 \\
\max(\lfloor n_2/2 \rfloor, n_1) & \text{otherwise}
\end{cases}
$$

> [!NOTE] Edge Cases
> - **No history** → Use `MIN_BTN_CLUSTER_DIST = 0.0001`
> - **< 3 samples** → Insufficient for reliable mean

---

## Properties

| Property | Description |
|----------|-------------|
| **Scale Invariant** | Divides by historical context |
| **Adaptive** | Accounts for cluster evolution |
| **Context-Aware** | Incorporates local density characteristics |

---

## Interpretation

Clusters should merge if their current distance is **unusually small** compared to historical patterns.

| Ratio | Meaning |
|-------|---------|
| $\rho \ll 1$ | Much closer than typical → likely merge |
| $\rho \approx 1$ | Typical distance |
| $\rho \gg 1$ | Farther than typical → unlikely merge |

---

## Integration

- **Adaptive Thresholding** — Provides normalized input
- **Merge Decisions** — Primary proximity filter
- **Continuity Analysis** — Input for smoothness evaluation

---

## See Also

- [[Adaptive Mergeability Threshold]] — uses this as input
