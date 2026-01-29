---
kind: conceptual
title: "Continuity - Angle Transition"
description: "Detects unnatural directional changes (kinks) at the merge interface"
---

# Continuity - Angle Transition

> [!TIP] Concept
> Natural clusters often follow smooth curves (lines, arcs). They rarely make sharp, jagged 90° turns instantly. **Angle Transition** detects these unnatural "kinks".

## The Logic

This component analyzes point triplets to measure **Angular Change**.

- **Vector A**: Direction of flow inside Cluster A.
- **Vector B**: Direction of flow inside Cluster B.
- **Transition**: The angle required to connect Vector A to Vector B.

## The "Kink" Test

- **Smooth Curve**: The transition angle is small (e.g., $10^\circ, 20^\circ$). The clusters flow into each other.
- **Sharp Kink**: The transition angle is large (e.g., $90^\circ, 120^\circ$). This looks like two separate objects colliding (e.g., a 'T' junction), not one single object.

## Mathematical Details

For the min-max max-angle calculations logic:
👉 [[Angle Transition Calculator]]
