<div align="center">

# Gauging-δ

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![IEEE TPAMI 2025](https://img.shields.io/badge/IEEE_TPAMI-2025-blue.svg)](https://doi.org/10.1109/TPAMI.2025.3545573)
[![tests](https://img.shields.io/badge/tests-77_passed-brightgreen.svg)](#)
[![sklearn](https://img.shields.io/badge/API-sklearn--compatible-F7931E?logo=scikit-learn&logoColor=white)](#api-reference)

</div>

**Gauging-δ** is a hierarchical agglomerative clustering algorithm that automatically
determines the number of clusters. Instead of fixed distance thresholds, it evaluates
merge candidates through *relative proximity* — inter-cluster distance normalized by
historical merge patterns — and validates boundaries with adaptive thresholds and
angle-based continuity analysis. 

<div align="center">

![Gauging-δ clustering on jain, flame, and 3-spiral datasets](assets/readme_plot.png)

</div>

## Install

```bash
pip install gauging-delta
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add gauging-delta
```

From source:

```bash
git clone https://github.com/design-zeng/Gauging-delta.git
cd Gauging-delta && uv sync
```

## Quick Start

```python
from gauging_delta import GaugingDelta

labels = GaugingDelta().fit_predict([
    [0.0, 0.0], [0.5, 0.3], [0.2, 0.1],
    [8.0, 8.0], [8.5, 8.3], [8.2, 8.1],
])
# array([0, 0, 0, 1, 1, 1])
```

## Usage

### Automatic clustering

```python
import numpy as np
from gauging_delta import GaugingDelta

# X is any array-like: list, numpy array, or pandas DataFrame
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [10, 0],
])

model = GaugingDelta().fit(X)
print(model.labels_)       # cluster label per sample
print(model.n_clusters_)   # number of clusters found
```

### Target a specific number of clusters

```python
model = GaugingDelta(n_clusters=3).fit(X)
assert model.n_clusters_ == 3
```

### Lite mode for large datasets

```python
model = GaugingDelta(mode="lite").fit(X)
```

Lite mode uses centroid linkage and skips continuity analysis.
Linear memory — scales to millions of points on commodity hardware.
See [Scaling](#scaling) for projections.

### Works with any array-like input

`fit(X)` accepts anything NumPy can convert — lists, arrays, DataFrames:

```python
import pandas as pd

df = pd.DataFrame({"x": [1, 3, 10, 12], "y": [2, 4, 11, 13]})
labels = GaugingDelta().fit_predict(df)
```

## Scaling

<div align="center">

![Runtime and memory scaling for full and lite modes](assets/scaling_benchmark.png)

</div>

| N | Full time | Full memory | Lite time | Lite memory |
|---:|---:|---:|---:|---:|
| 5,000 | 13 s | 1.4 GB | 3 s | 5 MB |
| 10,000 | 27 s | 5.8 GB | 8 s | 10 MB |
| 20,000 | ~1 min | 22 GB | 17 s | 21 MB |
| 35,000 | ~1.8 min | **64 GB** | 35 s | 37 MB |
| 100,000 | ~6 min | 543 GB* | 2.1 min | 110 MB |
| 500,000 | ~40 min | — | 13 min | 570 MB |
| 1,000,000 | ~1.4 hr | — | 36 min | 1.2 GB |

Runtime is projected from measured O(N^1.1) and O(N^1.2) power-law fits.
Full mode memory is O(N^2) — exceeds 64 GB at N ≈ 35K.
Lite mode memory is O(N) — exceeds 64 GB at N ≈ 48M.

*\* Full mode runtime projections beyond 35K assume infinite memory; in practice, memory is the bottleneck.*

## Modes

|  | `"full"` (default) | `"lite"` |
|---|---|---|
| **Linkage** | Single-link (point-to-point) | Centroid (center-to-center) |
| **Merge gates** | Proximity + threshold + continuity | Proximity + threshold |
| **Quality** | Paper-exact (ARI = 1.000 on benchmarks) | Lower — tends to under-merge |
| **Memory** | O(N^2) | O(N) |
| **Practical limit** | ~35K samples (64 GB) | Millions |
| **Use when** | Quality matters | Scale or memory matters |

> Lite mode trades clustering quality for O(N) memory. It is best suited for
> large-scale exploratory analysis where approximate clusters are acceptable.
> For publication-quality results, use full mode.

## API Reference

### Constructor

```python
GaugingDelta(
    *,
    n_clusters=None,
    mode="full",
    config=None,
    proximity=None,
    continuity=None,
    linkage=None,
    preserve_labels=False,
)
```

All parameters are keyword-only.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int \| None` | `None` | Target cluster count. `None` = automatic. |
| `mode` | `str` | `"full"` | `"full"` or `"lite"`. See [Modes](#modes). |
| `config` | `GaugingDeltaConfig \| None` | `None` | Algorithm constants. Defaults are the paper's values. |
| `proximity` | `ProximityMetric \| None` | `None` | Custom proximity metric (full mode only). |
| `continuity` | `ContinuityMetric \| None` | `None` | Custom continuity metric (full mode only). |
| `linkage` | `LinkageMetric \| None` | `None` | Custom linkage metric (full mode only). |
| `preserve_labels` | `bool` | `False` | Keep internal cluster IDs instead of renumbering to 0..k-1. |

### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X)` | `self` | Fit the model. *X* is array-like of shape *(n_samples, n_features)*. |
| `fit_predict(X)` | `np.ndarray` | Fit and return cluster labels. |

### Attributes (available after `fit`)

| Attribute | Type | Description |
|---|---|---|
| `labels_` | `np.ndarray` | Cluster label for each sample, shape *(n_samples,)*. |
| `n_clusters_` | `int` | Number of clusters found. |

## Citation

If you use Gauging-δ in your research, please cite:

```bibtex
@article{yao2025gauging,
  title     = {Gauging-$\delta$: A Non-Parametric Hierarchical Clustering Algorithm},
  author    = {Yao, Jinli and Pan, Jie and Zeng, Yong},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2025},
  volume    = {47},
  number    = {6},
  pages     = {4897--4907},
  doi       = {10.1109/TPAMI.2025.3545573},
  publisher = {IEEE}
}
```

## License

MIT — see [LICENSE](LICENSE).
