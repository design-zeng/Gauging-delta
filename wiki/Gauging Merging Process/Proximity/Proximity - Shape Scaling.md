---
kind: conceptual
title: "Proximity - Shape Scaling"
description: "Modulates threshold based on morphological similarity"
---

# Proximity - Shape Scaling

> [!TIP] Concept
> Merging should preserve structure. It is safer to merge two linear structures than to merge a tight ball into a long line. **Shape Scaling** penalizes merges between structurally incompatible clusters.

> [!WARNING] Discrepancy / Proximity vs. Continuity
> **Paper vs. Implementation**: The reference paper (*Gauging-delta*) primarily addresses shape preservation through **Continuity** checks (smoothness of density and direction) rather than filtering it at the **Proximity** stage via historical shape evolution.
> *   **Paper**: Uses $CV$ (Coefficient of Variation) in the Proximity Threshold (Eq. 5) to account for local density homogeneity, but does not explicitly describe a "Shape Evolution" history comparison during the proximity check.
> *   **Wiki/Code**: This distinct "Shape Scaling" component appears to be a custom extension or a specific interpretation of the "Interaction Force" that is not explicitly detailed in the standard algorithm formulation.

## The Logic

This component compares the **Shape Evolution** of the two clusters.

- **Check History**: How has the "spread" of points relative to the centroid changed over time?
- **Compare Trends**:
    - **Similar Evolution**: Both clusters are growing in a similar way (e.g., both expanding spherically).
    - **Divergent Evolution**: One is expanding linearly, the other spherically.

## Effect on Threshold

- **Similar Shapes**: The threshold is **Boosted** ($\times 1.0$ to $1.5$). We encourage the merge.
- **Dissimilar Shapes**: The threshold is **Penalized** ($\times 0.5$ to $1.0$). We require the clusters to be much closer (lower Proximity Ratio) to justify the merge.

## Mathematical Details

For the calculation of Shape Deviation ($\sigma_s$) and the similarity ratio ($\xi_s$):
👉 [[Shape Similarity Calculator]]
