---
kind: conceptual
title: "Continuity - Density Transition"
description: "Detects density gaps at the merge interface"
---

# Continuity - Density Transition

> [!TIP] Concept
> A seamless object should have relatively uniform density. If two dense clusters are separated by a "valley" of sparse points, they likely belong to different objects.

## The Logic

This component compares density **Inside** the clusters vs. **At the Boundary**.

1. **Internal Density**: Count points within a local radius *inside* Cluster A.
2. **Boundary Density**: Count points within the same radius *at the midpoint* between clusters.
3. **Comparison**:
    - If Boundary Density $\approx$ Internal Density, the transition is **Smooth**.
    - If Boundary Density $\ll$ Internal Density, there is a **Gap**.

## The "Dip" Test

Think of it like driving over a bridge.
- A **Smooth Merge** is a flat bridge. You don't notice when you cross from side A to side B.
- A **Gap** is a deep canyon. You have to dip far down to get across. The algorithm detects this "dip" in point density.

## Mathematical Details

For the `compute_transition_smoothness` calculations and outlier removal:
👉 [[Density Continuation Calculator]]
