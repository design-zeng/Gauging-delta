---
kind: conceptual
title: "Angle Transition Calculator"
description: "Measures angular smoothness at cluster boundaries (ξ_a)"
complexity: "O(n_boundary log n_boundary)"
---

# Angle Transition Calculator

> [!TIP] Quick Summary
> Evaluates how **smoothly** boundary points transition angularly — low variance indicates continuous structure.


> [!WARNING] Paper Divergence
> The Paper describes finding specific "boundary angles" ($\theta_L$, $\theta_R$) as discrete limits.
> The Code (`perception.py` Lines 1071-1121) calculates the **Mean Normalized Angular Deviation** based on specific transition angles between cluster centroids and boundary points.

---

## Code Implementation

### Step 1: Angle Identification

The algorithm identifies 4 specific "transition angles" ($\alpha_1, \alpha_2, \alpha_3, \alpha_4$) by finding the smallest angles formed between:
- The midpoint of the two clusters.
- The reference boundary points ($p_1, p_2, p_3, p_4$) derived from local angle info.
- The cluster centroids.

### Step 2: Deviation Calculation

For each transition angle $\alpha$, a deviation score is computed favoring orthogonality ($\frac{\pi}{2}$):

$$ s_\alpha = \frac{2 \cdot \left| \min(\alpha, |\alpha - \pi|) - \frac{\pi}{2} \right|}{\pi} $$

- If $\alpha \approx \frac{\pi}{2}$ (Orthogonal), $s_\alpha \approx 0$.
- If $\alpha \approx 0$ or $\pi$ (Collinear), $s_\alpha \approx 1$.

### Step 3: Aggregation

The final Shape Smoothness component uses the mean of these deviations:

$$ \xi_a = \frac{1}{4} \sum s_\alpha $$

### Integration

This angle transition score is combined with mass smoothness to form the **Orientation Smoothness**:
$$ \xi_{orientation} = \sqrt{\xi_{mass} \cdot \xi_{angle}} $$

---

## Edge Cases

| Condition | Behavior |
|-----------|----------|
| < 3 boundary points | Returns 0.0 (insufficient data) |
| Zero-length vectors | Skipped in computation |
| Circular wrapping | Points wrap around using modulo indexing |

---

## See Also

- [[Density Continuation Calculator]] — $\xi_d$
- [[Orientation Similarity Calculator]] — $\xi_o$ (statistical similarity)
- [[Local Compactness Measure]] — internal cohesion
