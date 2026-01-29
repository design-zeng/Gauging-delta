---
kind: conceptual
title: "Interaction Force Calculator"
description: "Quantifies attraction between cluster pairs (Paper Eq. 7)"
complexity: "O(1)"
---

# Interaction Force Calculator

> [!TIP] Quick Summary
> An **inverse-square law** inspired by gravity—used to calculate the influence of environmental clusters on the mergeability of a candidate pair.

---

## Core Formula

The interaction force is calculated as the **Geometric Mean** of the forces exerted by a third cluster $C$ on the two clusters being considered for merging ($C_1, C_2$).

$$
F_{\text{interaction}}(C, C_1, C_2) = \sqrt{F(C, C_1) \cdot F(C, C_2)}
$$

Where the base force $F(A, B)$ is given by:

$$
F(A, B) = \frac{|A| \cdot |B|}{d_{AB}^2}
$$

| Symbol | Meaning |
|--------|---------|
| $\|A\|, \|B\|$ | Cluster sizes (point counts) |
| $d_{AB}$ | Distance between centroids |

---

## Integration

Used to compute the **Environment Factor** $\beta_{ij}$ via a **Weighted Average** (Code: `perception.py` Lines 393-396):

$$
\beta_{ij} = \sum_{k} \left( \frac{w_k}{\sum w} \cdot S_k \right) \quad \text{where} \quad S_k = \frac{2}{1 + e^{5 \cdot d_{\text{cand}}/d_{k}}} + 1
$$

The weights $w_k$ are derived from the relative interaction forces.

---

## Interpretation

A large, nearby cluster exerts high "force," influencing the mergeability of its neighbors by altering the adaptive proximity threshold. The geometric mean ensures that a third cluster must have a significant relationship with **both** merging candidates to strongly influence the decision.

---

## See Also

- [[Adaptive Mergeability Threshold]] — uses $\beta_{ij}$
