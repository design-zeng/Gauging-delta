---
kind: conceptual
title: "Proximity - Historical Statistics"
description: "Sets baseline expectations based on internal cluster consistency"
---

# Proximity - Historical Statistics

> [!TIP] Concept
> **Predictability implies strictness.** If a cluster has been growing very predictably (low variance in merge distances), we expect that trend to continue. We are skeptical of sudden jumps.

## The Logic

This component looks at the **Standard Deviation** of the cluster's past merges.

- **Low Variance ($\sigma \approx 0$)**: The cluster is "crystallized". It expects neighbors at very specific distances.
    - **Result**: The threshold tightens. Only very close neighbors are accepted.
- **High Variance**: The cluster is "messy" or "exploratory". It has merged effectively at random distances.
    - **Result**: The threshold relaxes. We give it the benefit of the doubt.

## Intuition

Imagine a person who always arrives exactly at 8:00 AM. If they arrive at 8:15, you worry.
Now imagine a person who arrives anytime between 8:00 and 9:00. If they arrive at 8:15, you don't even blink.

## Mathematical Details

For the sigmoid function mapping Mean/Std statistics to a threshold multiplier:
👉 [[Adaptive Mergeability Threshold#1-statistical-threshold-t_stat|Adaptive Mergeability Threshold (Agile Section)]]
