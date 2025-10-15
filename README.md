# Human connectome topology directs cortical traveling waves and shapes frequency gradients

Traveling waves and neural oscillation frequency gradients are pervasive in the human cortex. While the direction of traveling waves has been linked to brain function and dysfunction, the factors that determine this direction remain elusive. We hypothesized that structural connectivity instrength gradients — defined as the gradually varying sum of incoming connection strengths across the cortex — could shape both traveling wave direction and frequency gradients. We confirm the presence of instrength gradients in the human connectome across diverse cohorts and parcellations. Using a cortical network model, we demonstrate how these instrength gradients direct traveling waves and shape frequency gradients. Our model fits resting-state MEG functional connectivity best in a regime where instrength-directed traveling waves and frequency gradients emerge. We further show how structural subnetworks of the human connectome generate opposing wave directions and frequency gradients observed in the alpha and beta bands. Our findings suggest that structural connectivity instrength gradients affect both traveling wave direction and frequency gradients.

#### Folder structure
```
    travelingwaves_code
    ├── configuration
    ├── data
    ├── modules
    └── scripts
        ├── analysis
        ├── processing
        │   └── schaefer_parcellation
        ├── results
        └── simulation
```

* configuration: contains simulation and analysis configurations and wrapper scripts.
* modules: contains helper, wave detection, and visualization functions used in scripts.
* scripts: contains the main code for simulations, analyses, and results.
* scripts/analysis: contains analyses scripts.
* scripts/processing: contains scripts for processing data.
* scripts/processing/schaefer_parcellation: contains scripts for processing human connectome project data to obtain tract weights and lengths based on Schaefer parcellation.
* scripts/results: contains notebooks that aggregate results and visualize simulation and analysis results.
* scripts/simulations: contains simulation scripts.

#### File structure

* 01_*: 2D network model with in-strength gradient.
* 02_*: 2D control model with uniform in-strength.
* 03_*: 2D network model for instrength-gradient vs. intrinsic-frequency-gradient interactions
* 06_*: 2D network model with instrength-directed wave flow potentials but differing effective frequency gradients
* 20_*: identifying in-strength gradients in the human connectome.
* 30_*: simulation and analysis of a cortical network model based on the human connectome.
* 31_*: control simulations and analysis of a cortical network model with permuted structural connectivity weights while preserving the topology.
* 32_*: control simulations and analysis of a cortical network model with synthetic structural connectivity generated with an exponential model of the connection strength - euclidean distance relationship.
* 33_*: control simulations and analysis of a cortical network model with Jansen-Rit neural masses.
* 34_*: control simulations and analysis of a cortical network model with instrength-normalized structural connectivity
* 38_*: control simulations and analysis of a cortical network model that removes the time delays while preserving the connection strengths and topology.
* 39_*: control simulations and analysis of a cortical network model with homogenous constant time delays.
* 40_*: exploration of cortical network model with varying intrinsic frequency, global coupling scaling and conduction speed.
* 48_*: exploration of cortical network models based on putative alpha- and beta-subnetworks.
* 50_*: control simulations and analysis of a cortical network model with added gaussian noise.
* 51_*: control simulations and analysis of a cortical network model with random gaussian intrinsic frequency dispersion.

### Usage

Our study relied heavily on Python and other key software (see table below). To install the Python libraries required to simulate the models and analyze the data you can either use conda

```
conda env create --name envname --file=environment.yml
```

or pip (not tested)

```
pip install -r requirements.txt
```

Many scripts use the modules within this repository, these are made accessible by

```
pip install -e .
```

Running the simulations and analyses requires the following data:

| DATA | SOURCE | IDENTIFIER |
| ------------- | ------------- | ----- |
| Human connectome project S900 data set | Connectome Coordination Facility	| [RRID:SCR_008749](https://www.humanconnectome.org/) |
| Schaefer atlas parcellation (1000 regions) | Schaefer et al. 2018 | https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal |
| Structural connectome (Lausanne) | Griffa et al. | https://zenodo.org/record/2872624 |
| Structural connectome (Schaefer 400 regions) | EBRAINS | https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a |
| Structural connectome (Random) | Arnatkeviciute et al. 2021 | https://zenodo.org/record/4733297 |

Key software to simulate and analyze the data:

| SOFTWARE | SOURCE | IDENTIFIER |
| ------------- | ------------- | ----- |
| Python 3.8.12	| Python | RRID:SCR_008394 |
| The Virtual Brain 2.3	| [The Virtual Brain](https://www.thevirtualbrain.org/tvb/zwei) | RRID:SCR_002249 |
| LibIGL-Python-bindings 2.2.1 | Jacobson et al. | https://libigl.github.io/ |
| MNE-Python 1.0.3 | [MNE-Python Project](https://mne.tools/stable/index.html) | RRID:SCR_005972 |
| MNE-HCP 0.1.dev12 | Engemann et al. | https://mne.tools/mne-hcp |
| BrainSpace 0.1.4 | Vos de Wael et al. 2020 | https://brainspace.readthedocs.io/ |
| FreeSurfer 7.1.1 | Martinos Center for Biomedical Imaging | [RRID:SCR_001847](https://surfer.nmr.mgh.harvard.edu/) |
| MRtrix 3.0.2 | Tournier et al. 2019 | [RRID:SCR_006971](https://www.mrtrix.org/) |
| RcppML 0.5.6 | DeBruine et al., 2021 | https://github.com/zdebruine/RcppML |


#### Authors
[Dominik Koller](https://www.researchgate.net/profile/Dominik_Koller)
[Michael Schirner](https://www.brainsimulation.org/bsw/)
[Petra Ritter](https://www.brainsimulation.org/bsw/)