# Data-driven-Model-Discovery-Neural-Operators
Code for OneMath World Summer School

## Setup

### Training data generation (`data_gen/`)

To generate the training dataset, you need **legacy FEniCS 2019.1.0** installed. The scripts in `data_gen/` (`FEM_KPP.py`, `FEM_GS1d.py`) use the classic `from fenics import *` API and are **not** compatible with FEniCSx (DOLFINx).

Install via [conda-forge](https://anaconda.org/conda-forge/fenics):

```bash
conda create -n fenics2019 -c conda-forge fenics=2019.1.0
conda activate fenics2019
```

See the [legacy FEniCS download page](https://fenicsproject.org/download/archive/) for other installation options (Docker, apt on Ubuntu).

Note: `gen_initial_GS_1d.py` only requires NumPy and does not need FEniCS.

### Neural network training

To train the neural networks (e.g. `FNO-KPP.py`, `DeepONet-KPP.py`, `LENO-KPP.py`, `LENO-GS1d.py`), [PyTorch](https://pytorch.org/) is required.
