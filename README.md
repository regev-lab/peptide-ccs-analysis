# peptide-ccs-analysis

**A repository containing data, models, and scripts for investigating collisional cross-section (CCS) bimodality and absolute CCS of peptides, as described in our paper *Analysis of Large Peptide Collisional Cross
Section Dataset Reveals Structural Origin of
Bimodal Behavior*.**

This repository contains code for training a Generative Additive Model (GAM) with our custom designed shared-spline architecture, which enables GAM's to work on sequential data.

---

## Table of Contents

- [Quickstart (TL;DR)](#quickstart-tldr)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Make + Conda (recommended)](#option-a-make--conda-recommended)
  - [Option B: Conda (manual, no make)](#option-b-conda-manual-no-make)
  - [Option C: Pure pip/venv (no conda)](#option-c-pure-pipvenv-no-conda)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [Data](#data)
- [Citing](#citing)
- [License](#license)

---

## Quickstart (TL;DR)

```bash
# make + conda (recommended)
make env           # create conda env
make install       # install package and deps
```

If you don’t want to use `make`, see [Option B](#option-b-conda-manual-no-make) or [Option C](#option-c-pure-pipvenv-no-conda).

---

## Requirements

- Python ≥ 3.11 (tested with 3.11)
  - numpy
  - matplotlib
  - pandas
  - pyarrow
  - pygam = 0.8.1 (our Shared Spline GAM requires patching on PyGAM)
  - scipy
  - seaborn
- Conda (Miniconda/Anaconda) **or** any Python 3 with `venv`
- Optional: `make` (GNU make)

---

## Installation

### Option A: Make + Conda (recommended)
```bash
make env ENV=<your-env> PYTHON=3.11
make install ENV=<your-env>
```

---

### Option B: Conda (manual, no make)

```bash
conda create -y -n <your-env> python=3.11
conda activate <your-env>
# core dependencies declared in pyproject.toml
pip install -e .
```

---

### Option C: Pure pip/venv (no conda)

```bash
python -m venv .venv
source .venv/bin/activate
# core dependencies declared in pyproject.toml
pip install -e .
```

---

## Reproducing Results

```bash
# after installation
conda activate <your-env>
python scripts/01_train_all_models.py # Trains all shared spline GAM models, and verifies that their trained coefficients match those in the `saved_models/` directory.
python scripts/02_plot_all_figures.py # Plots all figures shown in the paper to matplotlib's default back-end.
```

---

## Project Structure

```
.
├── pyproject.toml
├── Makefile
├── README.md
├── scripts/					# sample scripts
│   ├── 01_train_all_models.py
│   ├── 02_plot_all_figures.py
├── peptide_ccs_analysis/		# source code
│   ├── __init__.py
│   ├── constants.py
│   ├── figures
│   │   ├── plotting.py
│   │   └── utils.py
│   ├── load_data.py
│   └── models
│       ├── custom_shared_spline_gam.py
│       ├── loading.py
│       └── utils.py
├── datasets/
│   ├── external/				# Meier et al.'s datasets
│   └── raw/					# our MD simulations datasets
├── tests/						# pytest code for dev only
└── docs/

```
---

## Data

### External data
The datasets in `datasets/external/` are sourced from
> Meier, F., Köhler, N.D., Brunner, A.D., et al. Deep learning the collisional cross sections of the peptide universe from a million training samples. Nature Communications 12, 1185 (2021).
[https://doi.org/10.1038/s41467-021-21352-8](https://doi.org/10.1038/s41467-021-21352-8)

### Raw data
The datasets in `datasets/raw/` are outputs from our MD simulations.

---

## Citing

If you use this code, please cite:


> Xu, A.M., Szöllősi, D., Grubmüller, H., and Regev, O. Analysis of Large Peptide Collisional Cross Section Dataset Reveals Structural Origin of Bimodal Behavior. Journal TBD, (2025). DOI: to be assigned.


---

## License

This code is released under the MIT License. See the `LICENSE` file for details.  
The datasets in `datasets/external/` are provided by Meier et al. (2021) under their original terms.  
The datasets in `datasets/raw/` are from our own simulations and are released under CC BY 4.0.
