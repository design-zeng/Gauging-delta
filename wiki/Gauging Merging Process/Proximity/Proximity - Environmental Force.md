---
kind: conceptual
title: "Proximity - Environmental Force"
description: "Adjusts mergeability based on competition from neighboring clusters"
---

# Proximity - Environmental Force

> [!TIP] Concept
> A cluster doesn't exist in a vacuum. It is surrounded by **Neighbors**. If a neighbor is "strong" (massive and close), it exerts a "gravitational pull" that distracts the cluster from merging with its current target.

## The Logic

We model this as **Interaction Force** (Gravity), formally defined as the **Environmental Factor** ($\mathcal{E}$) in **Eq. 6** of the *Gauging-delta* paper.

1.  **Calculate Forces**: Compute the sum of interaction forces from *all* surrounding clusters to quantify the total "environmental pressure".
2.  **Assess Competition**:
    -   The environmental factor $\mathcal{E}(C_i)$ acts as a penalty term in the mergeability threshold.
    -   If a cluster is in a "dense gravitational field" (many strong neighbors), its threshold drops, making it "pickier" about who it merges with. It prevents a cluster from randomly merging with a weak neighbor when a strong neighbor is nearby.

## Effect on Threshold

- **High External Competition**: The threshold **Decreases**. It becomes harder to merge with the target because the "environment" suggests you should be merging with someone else (the stronger neighbor).
- **Low Competition**: The threshold remains **Neutral**.

## Mathematical Details

For the gravity-based force calculation and weighted averaging:
👉 [[Interaction Force Calculator]]
👉 [[Adaptive Mergeability Threshold#2-environment-factor-beta_ij|Adaptive Mergeability Threshold (Environment Section)]]
