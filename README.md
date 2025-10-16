# Traveling Waves in Kuramoto Networks

This repository contains the simulation and analysis code for the Bachelor's thesis, "Traveling waves in Kuramoto networks," from the University of Padua. This project is an adaptation and extension of the code from the 2024 *Nature Communications* paper by Koller, Schirner, and Ritter, "Human connectome topology directs cortical traveling waves and shapes frequency gradients."

The project investigates how spatiotemporal wave patterns emerge in spatially structured networks of coupled oscillators, with a focus on a novel mechanism for wave guidance: spatial gradients of network instrength. The framework simulates oscillator dynamics using the Kuramoto model on 2D grids, analyzes wave phenomena using discrete differential geometry, and detects significant wave events with a robust statistical pipeline.

The original work provided the following context:

> Traveling waves and neural oscillation frequency gradients are pervasive in the human cortex. While the direction of traveling waves has been linked to brain function and dysfunction, the factors that determine this direction remain elusive. We hypothesized that structural connectivity instrength gradients — defined as the gradually varying sum of incoming connection strengths across the cortex — could shape both traveling wave direction and frequency gradients. We confirm the presence of instrength gradients in the human connectome across diverse cohorts and parcellations. Using a cortical network model, we demonstrate how these instrength gradients direct traveling waves and shape frequency gradients. Our model fits resting-state MEG functional connectivity best in a regime where instrength-directed traveling waves and frequency gradients emerge. We further show how structural subnetworks of the human connectome generate opposing wave directions and frequency gradients observed in the alpha and beta bands. Our findings suggest that structural connectivity instrength gradients affect both traveling wave direction and frequency gradients.

---

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [Project Goals](#project-goals)
3. [Repository Structure](#repository-structure)
4. [Experiment Naming Convention](#experiment-naming-convention)
5. [Current Work](#current-work)
6. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Dependencies](#dependencies)
7. [Workflow](#workflow)
   - [1. Configuration](#1-configuration)
   - [2. Running Simulations](#2-running-simulations)
   - [3. Running Analysis](#3-running-analysis)
   - [4. Visualizing Results](#4-visualizing-results)
8. [Future Work](#future-work)
9. [Original Work and Authors](#original-work-and-authors)
10. [References](#references)

---

## Key Concepts

This project is grounded in several key theoretical concepts from the thesis:

* **The Kuramoto Model**: A mathematical model used to describe synchronization in networks of coupled oscillators. Each oscillator is defined by a phase that evolves based on its intrinsic frequency and the influence of its coupled neighbors. Our simulations use a version with distance-dependent time delays.
* **Instrength Gradients**: The central mechanism explored in this work. The instrength of a node is the sum of its incoming connection weights ($s_{i}=\sum_{j}w_{ij}$). By imposing a smooth spatial gradient on this property, we can systematically guide traveling waves, which propagate from regions of low instrength to high instrength.
* **Wave Flow Potentials**: To identify the sources and sinks of wave activity, we analyze the phase gradient vector field ($g = \nabla\phi$). Using the Helmholtz-Hodge decomposition, we can derive a scalar potential field $U$ by solving the Poisson equation $\nabla^2U = \nabla \cdot g$. The peaks and troughs of this potential map directly to wave sources and sinks.
* **Discrete Differential Geometry**: Since our simulations are on discrete meshes, we use tools from discrete differential geometry to compute operators like gradient, divergence, and the Laplacian. This is implemented using a Finite Element Method (FEM) approach on a triangulated mesh, providing a robust and accurate way to calculate derivatives.
* **Statistical Wave Detection**: To identify significant diverging or rotating wave patterns, we compare the empirical phase gradients against idealized wave templates. The significance of this match is assessed using non-parametric permutation testing to create an empirical null distribution.

## Project Goals

The main contributions of this thesis project are:

1. **Theoretical Framing**: A clear explanation and implementation of the Kuramoto model where instrength gradients guide wave propagation.
2. **Wave Detection**: Implementation of angular similarity methods on triangulated meshes to identify diverging and rotating wave episodes in simulation data.
3. **Wave Potential Analysis**: Construction of scalar divergence potentials using a discrete Helmholtz-Hodge decomposition to map wave sources and sinks.
4. **Reproducibility**: A modular and well-documented Python implementation that mirrors published methods and facilitates future research.

## Repository Structure

```
.
├── README.md                   # This file
├── brain_waves.ipynb          # Main notebook with current thesis results
├── setup.py                   # Package installation setup
├── environment.yml            # Conda environment specification
├── requirements.txt           # Python package requirements
├── configuration/             # YAML configs and shell scripts for running experiments
├── data/                      # Raw simulation outputs and processed analysis data
│   ├── 01_simulations/       # Experiment 01 simulation data
│   ├── 01_analysis_potentials/ # Experiment 01 potential analysis results
│   ├── 02_simulations/       # Experiment 02 simulation data
│   ├── 02_analysis_potentials/ # Experiment 02 potential analysis results
│   └── ...                   # Additional experiment data
├── modules/                   # Reusable Python modules
│   ├── __init__.py
│   ├── helpers.py            # Helper functions
│   ├── visualization.py      # Visualization utilities
│   └── wave_detection_methods.py # Wave detection algorithms
└── scripts/                   # Core scripts for simulation, analysis, and processing
    ├── simulation/           # Scripts to run Kuramoto simulations
    ├── analysis/             # Scripts for wave detection, potential analysis, etc.
    ├── processing/           # Scripts for data preprocessing (e.g., HCP surfaces)
    └── results/              # Jupyter notebooks for original Koller et al. results
        ├── 01_results.ipynb
        ├── 03_results.ipynb
        ├── 20_results.ipynb
        ├── 30_results.ipynb
        ├── 40_results.ipynb
        └── 48_results.ipynb
```

### Key Directories

* **`brain_waves.ipynb`**: **Main thesis notebook** containing current work on potential calculation methods and simulation visualization.
* **`configuration/`**: Contains simulation and analysis configurations (`.yaml`) and wrapper scripts (`.sh`) for running experiments.
* **`data/`**: Organized by experiment number, contains both raw simulation outputs and processed analysis results.
* **`modules/`**: Contains helper functions, wave detection algorithms, and visualization utilities used throughout the project.
* **`scripts/simulation/`**: Python scripts for running Kuramoto model simulations with various configurations.
* **`scripts/analysis/`**: Analysis scripts including:
  - `wave_potential_analysis.py`: Main potential field calculation
  - `wave_potential_analysis_fd.py`: Finite difference implementation
  - `flow_analysis_2d.py`: 2D flow field analysis
  - `wave_detection_2d.py`: Wave pattern detection on 2D grids
  - `compare_potentials.py`: Comparison of different potential calculation methods
* **`scripts/results/`**: Jupyter notebooks containing results from the original Koller et al. study (for reference).

## Experiment Naming Convention

The numbered prefixes on files in the `configuration`, `data`, and `scripts` directories correspond to specific experiments from the original study. This project focuses on adapting and replicating the initial 2D models.

* **`01_*`**: 2D network model with an in-strength gradient (primary focus)
* **`02_*`**: 2D control model with uniform in-strength
* **`03_*`**: 2D network model for instrength-gradient vs. intrinsic-frequency-gradient interactions
* **`06_*`**: 2D network model with instrength-directed wave flow potentials but differing effective frequency gradients
* **`11_*`-`12_*`**: Additional 2D model variations
* **`20_*`**: Identifying in-strength gradients in the human connectome
* **`30_*`**: Simulation and analysis of a cortical network model based on the human connectome
* **`31_*`-`39_*`**: Various control simulations for the cortical network model
* **`40_*`**: Exploration of cortical network model with varying parameters
* **`48_*`**: Exploration of cortical network models based on putative alpha- and beta-subnetworks
* **`50_*`-`51_*`**: Control simulations with added noise or frequency dispersion

## Current Work

The current thesis work, documented in **`brain_waves.ipynb`**, focuses on:

1. **Potential Calculation Methods**: Comparing different computational approaches for calculating wave flow potentials:
   - Finite Element Method (FEM) implementation
   - Finite Difference (FD) implementation
   - Validation and comparison of accuracy between methods

2. **Simulation Visualization**: Developing visualizations for:
   - Phase dynamics on 2D grids
   - Instrength gradient fields
   - Wave propagation patterns
   - Potential field distributions

3. **Method Validation**: Quantitative comparison of potential calculation methods through:
   - Correlation analysis
   - Error metrics (L2 error, max error)
   - Residual energy analysis
   - Visual comparisons at different time points

The results from experiments 01 and 02 are stored in `data/01_analysis_potentials/` and `data/02_analysis_potentials/` respectively, with comparison metrics in the corresponding `*_comparison/` directories.

## Getting Started

### Installation

To set up the necessary environment, you will need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ginca03/Traveling-waves-in-Kuramoto-networks.git
   cd travelingwaves_code_rev1
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate travelingwaves
   ```

3. **Install the local modules:**
   Many scripts use the modules within this repository. Make them accessible by running:
   ```bash
   pip install -e .
   ```

4. **Launch Jupyter to view results:**
   ```bash
   jupyter notebook brain_waves.ipynb
   ```

### Dependencies

#### Required Data

Running the full pipeline, especially the cortical surface models, requires the following external datasets:

| DATA | SOURCE | IDENTIFIER |
|------|--------|------------|
| Human connectome project S900 data set | Connectome Coordination Facility | [RRID:SCR_008749](https://www.humanconnectome.org/) |
| Schaefer atlas parcellation (1000 regions) | Schaefer et al. 2018 | https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal |
| Structural connectome (Lausanne) | Griffa et al. | https://zenodo.org/record/2872624 |
| Structural connectome (Schaefer 400 regions) | EBRAINS | https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a |
| Structural connectome (Random) | Arnatkeviciute et al. 2021 | https://zenodo.org/record/4733297 |

**Note**: The 2D grid simulations (experiments 01-12) do not require these external datasets and can be run immediately after installation.

#### Key Software

The analysis relies on several key scientific software packages:

| SOFTWARE | SOURCE | IDENTIFIER |
|----------|--------|------------|
| Python 3.8.12 | Python | RRID:SCR_008394 |
| The Virtual Brain 2.3 | [The Virtual Brain](https://www.thevirtualbrain.org/tvb/zwei) | RRID:SCR_002249 |
| LibIGL-Python-bindings 2.2.1 | Jacobson et al. | https://libigl.github.io/ |
| MNE-Python 1.0.3 | [MNE-Python Project](https://mne.tools/stable/index.html) | RRID:SCR_005972 |
| MNE-HCP 0.1.dev12 | Engemann et al. | https://mne.tools/mne-hcp |
| BrainSpace 0.1.4 | Vos de Wael et al. 2020 | https://brainspace.readthedocs.io/ |
| FreeSurfer 7.1.1 | Martinos Center for Biomedical Imaging | [RRID:SCR_001847](https://surfer.nmr.mgh.harvard.edu/) |
| MRtrix 3.0.2 | Tournier et al. 2019 | [RRID:SCR_006971](https://www.mrtrix.org/) |
| RcppML 0.5.6 | DeBruine et al., 2021 | https://github.com/zdebruine/RcppML |

## Workflow

The project is designed around a reproducible, configuration-driven workflow.

### 1. Configuration

- Navigate to the `configuration/` directory.
- Open a configuration file (e.g., `01_configuration.yaml`) to inspect or modify simulation parameters.

### 2. Running Simulations

- Execute the corresponding wrapper script from the `configuration/` directory:
  ```bash
  cd configuration/
  ./01_run_parallel.sh
  ```
- This will generate simulation data in the corresponding `data/` directory (e.g., `data/01_simulations/`).

### 3. Running Analysis

- Once simulations are complete, run the analysis script:
  ```bash
  cd configuration/
  ./01_analysis_run_parallel.sh
  ```
- Results will be saved in the `data/` directory (e.g., `data/01_analysis_potentials/`).

### 4. Visualizing Results

- **Current thesis work**: Open and run `brain_waves.ipynb` in the root directory
- **Original Koller et al. results**: Explore the Jupyter notebooks in `scripts/results/`
  ```bash
  jupyter notebook scripts/results/01_results.ipynb
  ```

## Future Work

The primary goal is to extend the 2D analyses to realistic brain models:

- **Cortical Surface Analysis**: Adapt the wave detection and potential analysis pipelines to work with triangulated cortical surface meshes from the Human Connectome Project (HCP).
- **Whole-Brain Simulation**: Run large-scale simulations on the HCP-derived connectome using a platform like The Virtual Brain, as was done for the original study.
- **Validate Against Empirical Data**: Compare simulated wave dynamics to source-reconstructed MEG data to test hypotheses about anatomical-functional relationships.
- **Method Optimization**: Further optimize and validate the potential calculation methods based on current comparison results.

## Original Work and Authors

This project is an implementation and adaptation of the methods and code developed for the following paper:

> Koller, D. P., Schirner, M., & Ritter, P. (2024). Human connectome topology directs cortical traveling waves and shapes frequency gradients. *Nature Communications, 15*(3570).

The original authors of the code and study are:

* [Dominik Koller](https://scholar.google.com/citations?user=skNLEIwAAAAJ&hl=it&oi=ao)
* [Michael Schirner](https://scholar.google.com/citations?user=rDGc-f4AAAAJ&hl=it&oi=ao)
* [Petra Ritter](https://scholar.google.com/citations?hl=it&user=njHFeAsAAAAJ)

## References

> [1] Koller, D. P., Schirner, M., & Ritter, P. (2024). Human connectome topology directs cortical traveling waves and shapes frequency gradients. *Nature Communications, 15*(3570).
>
> [2] Muller, L., Piantoni, G., Koller, D., et al. (2016). Rotating waves during human sleep spindles organize global patterns of activity that repeat precisely through the night. *eLife, 5*, e17267.
>
> [3] Crane, K., Weischedel, C., & Wardetzky, M. (2013). Geometry processing with libigl. *ACM Transactions on Graphics (TOG), 32*(4), 1–10.
