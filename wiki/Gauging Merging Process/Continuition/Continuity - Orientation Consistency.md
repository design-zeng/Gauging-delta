---
kind: conceptual
title: "Continuity - Orientation Consistency"
description: "Verifies alignment between mass distribution and directional flow"
---

# Continuity - Orientation Consistency

> [!TIP] Concept
> A robust cluster should have its "weight" (mass) aligned with its "flow" (direction). **Orientation Consistency** checks this alignment.

## The Logic

This is a higher-order check that combines **Mass** and **Angle**.

$$
\text{Orientation} = \sqrt{\text{Mass Smoothness} \times \text{Angle Smoothness}}
$$

- **Mass Smoothness**: Are the two clusters roughly equal in density/importance? (from [[Continuity - Density Transition]])
- **Angle Smoothness**: Do they align directionally? (from [[Continuity - Angle Transition]])

## Why "Orientation"?

"Orientation" implies both a direction and a magnitude.
- A strong vector pointing East has **Orientation**.
- A weak vector pointing East has less significance.

This metric ensures we don't let a "ghost" cluster (very low mass) dictate the direction of a "solid" cluster (high mass), nor do we allow two solid clusters to merge if they point in wrong directions.

## Mathematical Details

For the geometric mean formula usage:
👉 [[Orientation Similarity Calculator]]
