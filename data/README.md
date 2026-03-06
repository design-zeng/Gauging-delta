# data/

Benchmark datasets for Gauging-delta. All files are comma-separated text with
the last column as the integer ground-truth cluster label.

---

## Data Format

```
x1, x2, ..., xD, label
```

- Columns 0 through D-1 are the feature coordinates (float64).
- The final column is the ground-truth cluster assignment (integer).
- No header row.

Load with:

```python
import numpy as np
data = np.loadtxt("data/flame.txt", delimiter=",")
X, y = data[:, :-1], data[:, -1].astype(int)
```

---

## Core Parity Datasets

These six datasets are used in `tests/test_parity.py` to verify ARI = 1.000
between `GaugingDelta` and the legacy `perception.py`:

| Dataset | N | Dimensions | Clusters | Description |
|---|---|---|---|---|
| `flame.txt` | 240 | 2 | 2 | Two interleaved flame-shaped clusters |
| `3_blobs.txt` | 300 | 2 | 3 | Three well-separated Gaussian blobs |
| `pathbased.txt` | 300 | 2 | 3 | Path-based non-convex clusters |
| `3-spiral.txt` | 312 | 2 | 3 | Three interlocked spirals |
| `jain.txt` | 373 | 2 | 2 | Crescent + circular cluster pair |
| `compound.txt` | 399 | 2 | 6 | Compound shapes with varying density |

---

## Extended Benchmark Datasets

These datasets are used for broader performance and quality evaluation but are
not part of the parity test suite:

| Dataset | N | Dimensions | Clusters | Description |
|---|---|---|---|---|
| `aggregation.txt` | 788 | 2 | 7 | Seven clusters with irregular shapes |
| `atom.txt` | 800 | 3 | 2 | 3-D atom-shaped structure |
| `chainlink.txt` | 1000 | 3 | 2 | Two interlocked 3-D rings |
| `impossible.txt` | 3673 | 2 | 8 | Large multi-shape dataset |

---

## Notes

- All datasets are read-only baselines. Do not modify the files in this directory.
- The `data/` directory is protected by the pre-commit hooks configured in
  `.claude/settings.json`.
