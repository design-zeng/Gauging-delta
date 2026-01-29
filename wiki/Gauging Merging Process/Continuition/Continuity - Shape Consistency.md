---
kind: conceptual
title: "Continuity - Shape Consistency"
description: "Ensures the merge preserves established shape properties"
---

# Continuity - Shape Consistency

> [!TIP] Concept
> **Clustering is Morphological.** If you have spent 100 steps building a perfect line, you shouldn't suddenly merge with a point that turns you into a blob.

## The Logic

This component checks if the **Resulting Shape** (after merge) would contradict the **Historical Shape** (before merge).

- **Input**: The standard deviation of point distributions (the "spread") of the individual clusters.
- **Projected**: The spread of the combined cluster.
- **Check**:
    - If the "Spread Ratio" stays consistent, the shape is preserved.
    - If the "Spread Ratio" jumps wildly (e.g., standard deviation doubles instantly), the shape is broken.

## Dual Usage

Note that **Shape Similarity** is so fundamental it is used twice:
1. **[[Proximity - Shape Scaling|In Proximity]]**: As a "barrier" to even considering the merge.
2. **In Continuity**: As a "validity check" for the merge quality.

## Mathematical Details

For the definition of $\xi_s$ and how shape deviation is tracked:
👉 [[Shape Similarity Calculator]]
