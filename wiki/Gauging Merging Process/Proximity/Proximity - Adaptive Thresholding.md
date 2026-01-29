---
kind: conceptual
title: "Proximity - Adaptive Thresholding"
description: "Orchestrates the dynamic barrier for cluster merging"
---

# Proximity - Adaptive Thresholding

> [!TIP] Concept
> The **Adaptive Threshold** is a dynamic "Bar Height" that the [[Proximity - Distance Context|Proximity Ratio]] must clear. If the ratio is lower than this bar, the merge is **rejected**.

## The Orchestra

The threshold isn't a single number; it's a dynamic barrier derived from the **Adaptive Mergeability Function** (referenced as **Eq. 5** in the *Gauging-delta* paper). The algorithm calculates a specific threshold for *each* of the two clusters attempting to merge, primarily driven by:

1.  **Internal Statistics ($\mathcal{S}$)**: The **Coefficient of Variation ($CV$)** of the edge lengths within the cluster.
    *   *Paper Insight*: The paper defines the threshold $T(C_i)$ as proportional to the inverse of the coefficient of variation ($CV^{-1}$).
    *   *Meaning*: Clusters with **homogeneous density** (low variance/stable structure) have higher thresholds (easier to merge into), while noisy or sparse clusters have lower thresholds.
2.  **Environmental Factor ($\mathcal{E}$)**: The local density context surrounding the cluster.
3.  **Interaction Force ($\mathcal{F}$)**: The gravitational-like pull between the two clusters.

$$
\text{Threshold} = \text{History} \times \text{Environment} \times \text{Shape}
$$

### 1. [[Proximity - Historical Statistics|Historical Statistics]]
*Does this cluster have a history of consistent, tight merges?*
- High consistency $\to$ Lower Threshold (Stricter).
- High variance $\to$ Higher Threshold (More Permissive).

### 2. [[Proximity - Environmental Force|Environmental Force]]
*Are there other strong clusters nearby pulling on this one?*
- Strong competition $\to$ Lower Threshold (Harder to merge).
- Isolation $\to$ Higher Threshold (Easier to merge).

### 3. [[Proximity - Shape Scaling|Shape Scaling]]
*Does the potential merge partner look compatible?*
- Similar shape $\to$ Higher Threshold (Bonus).
- Dissimilar shape $\to$ Lower Threshold (Penalty).

## Decision Logic

For a merge between Cluster A and Cluster B to proceed to the **Continuity** phase, the Proximity Ratio usually must be lower than the Adaptive Thresholds of **both** clusters (or satisfy a combined condition).

## Mathematical Details

For the master formula combining these factors:
👉 [[Adaptive Mergeability Threshold]]
