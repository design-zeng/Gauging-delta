---
kind: conceptual
title: "Continuity - Orchestration"
description: "High-level orchestration of the smoothness verification process"
---

# Continuity - Orchestration

> [!TIP] Concept
> Being "close enough" (Proximity) is necessary but not sufficient. **Continuity** asks: "Does this merge create a seamless object, or does it stitch together two incompatible parts?"

## The Logic

This phase enforces the **Gestalt Principle of Good Continuation** (as cited in the *Gauging-delta* paper). The system orchestrates a **Multi-Factor Smoothness Check** to ensure that the merged entity is perceptually consistent. It doesn't just look at one metric; it looks for a consensus to detect anomalies like "necks" (density drops) or "kinks" (directional breaks):

1.  **[[Continuity - Density Transition]]**: Checks for sudden drops in point density at the merge interface.
2.  **[[Continuity - Angle Transition]]** (Directional Continuity): Checks for abrupt changes in the cluster's principal growth direction.
3.  **[[Continuity - Shape Consistency]]**: Is the overall manifold shape preserved?
4.  **[[Continuity - Orientation Consistency]]**: Is the mass aligned with the direction?

## The Local Search

To ensure robust detection, the algorithm performs a **Local Radius Expansion**:
- It doesn't just check the immediate boundary.
- It expands a search radius ($r$) outwards from the potential merge interface.
- It calculates smoothness scores at multiple scales and aggregates them.

## The Final Verdict

The orchestration logic combines these factors (often multiplicatively) into a single **Smoothness Score** ($0.0$ to $1.0$).
- If $\text{Score} > \text{Threshold}$, the merge is approved.
- If $\text{Score} < \text{Threshold}$, the clusters remain separate (for now).

## Mathematical Details

For the exact implementation of the `compute_continuation` loop:
👉 [[Narrative - Continuity-Based Mergeability]] (Overview)
👉 [[Density Continuation Calculator]] (Core math)
