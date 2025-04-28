# Gauging-delta

This repository contains a python implementation of the clustering algorithm presented in:

Jinli Yao, Jie Pan, and Yong Zeng. Gauging-$\delta$: A Non-parametric Hierarchical Clustering Algorithm, IEEE Transactions on Pattern Analysis and Machine Intelligence, 47 (6), 4897-4907, June 2025.

If you use this implementation in your work, please add a reference/citation to the paper:

```bibtex
@article{
  title={Gauging-$\delta$: A Non-parametric Hierarchical Clustering Algorithm},
  author={Jinli, Yao and Jie, Pan and Yong, Zeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  volume={47},
  number={6},
  pages={4897-4907},
  doi={10.1109/TPAMI.2025.3545573}
}
```

## Usage instructions

In order to run Gauging-$\delta$, run the file perception_clustering.py using python.

## Input and output files format

The expected input file format is a comma separated file, where each row represents a different multi-dimensional data point. If ground truth is provided, the ground truth labels should be placed in the last column of each row.

The output file format is a multi-line file, where each line contains the label that was assigned to the data point. (The index of the data points correspond to the index in the input file). If the input data are 2-d or 3-d, the clustering results will be plotted.

The ten synthetic datasets that were evaluated in the main text of the paper are available under the "data" folder.
