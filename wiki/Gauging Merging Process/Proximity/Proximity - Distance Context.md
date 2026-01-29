---
kind: conceptual
title: "Proximity - Distance Context"
description: "Evaluates merge distance relative to historical merging behavior"
---

# Proximity - Distance Context

> [!TIP] Concept
> A "short" distance for a sparse cluster might be a "long" distance for a dense one. **Distance Context** normalizes the current range by comparing it to the cluster's own history.

## The Logic

Instead of asking "Is distance $D < 5.0$?", the system asks:
**"Is distance $D$ typical for these clusters?"**

It calculates a **Proximity Ratio**:

$$
\text{Proximity} = \frac{\text{Current Distance}}{\text{Historical Mean Distance}}
$$

- **Ratio $\approx 1.0$**: The clusters are merging at a distance consistent with their past behavior.
- **Ratio $\gg 1.0$**: The clusters are reaching far across a void to merge (suspicious).
- **Ratio $\ll 1.0$**: The clusters are extremely close compared to their usual spacing.

## Why It Matters

This normalization makes the algorithm **Scale-Invariant**. It works equally well for:
- **Dense Cores**: Merging at $0.1$ units.
- **Sparse Outliers**: Merging at $10.0$ units.

## Mathematical Details

For the exact formula on how history is weighted and selected to compute the denominator, see:
👉 [[Perceptual Proximity Statistic]]
