#!/usr/bin/env python
""" Compute wave flow potential for parameter exploration of cortical network simulations and detect instrength-guided waves.

This script is used to analyze the parameter exploration results of a cortical network model based on Kuramoto oscillators.
Differential operators are computed on the 2D cortical mesh determined by the positions of the regions of the Schaefer 
brain atlas. These operators are used to compute the phase gradients. The phase gradients are then decomposed by the 
Helmholtz-Hodge Decomposition. The curl-free potential (representing the diverging phase flow) is correlated with the 
structural connectivity instrength. Their correlation is used to identify waves guided by structural connectivity 
instrength gradients.


Usage
-----
    python 40_analysis_potentials.py "simulation_configuration_path" "analysis_configuration_path" process_id

Arguments
---------
simulation_configuration_path : String
    Path to configuration file of the simulations.
analysis_configuration_path : String
    Path to configuration file of the analysis.
process_id : int
    ID of the process.
"""


import sys
import os
import gc
import pickle
import yaml
import itertools
import numpy as np
import pandas as pd
import logzero
from logzero import logger

from scipy.stats import spearmanr
from brainspace.null_models import SpinPermutations
from tvb.simulator.lab import *


from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "18. January 2023"
__status__ = "Prototype"


# Prepare Script
# --------------
# Read Configuration
try:
    configuration_path = sys.argv[1]
    with open(configuration_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.BaseLoader)
except BaseException as e:
    print("Error: Specify correct path to yaml configuration file.")
    raise

# Read Configuration
try:
    configuration_path = sys.argv[2]
    with open(configuration_path, "r") as ymlfile:
        config_analysis = yaml.load(ymlfile, Loader=yaml.BaseLoader)
except BaseException as e:
    print("Error: Specify correct path to yaml configuration file.")
    raise


# Simulation Parameters
# ---------------------
data_path = config["data_path"]
save_path = config["save_path"]
save_path_analysis = config_analysis["save_path_analysis"]
experiment_id = config["experiment_id"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms
initial_transient = float(config["initial_transient"])  # ms
initial_transient_samples = int(initial_transient/integration_step_size)  # samples

downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
integration_step_size_downsampled = integration_step_size * downsampling_factor
number_of_timesteps_downsampled = int((simulation_duration-initial_transient)/integration_step_size_downsampled)


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[3])

simulation_idx, frequency_idx, coupling_strength_idx, conduction_speed_idx = list(itertools.product(*[range(0,10), range(0,4), range(0,39), range(0,10)]))[process_id]

sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]

intrinsic_frequency = float(np.array(config["intrinsic_frequency"])[frequency_idx])  # Hz
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz - Kuramoto oscillator intrinsic frequency standard deviation

coupling_strength_start, coupling_strength_end, coupling_strength_steps = np.array(config["coupling_strength"], dtype=float)  # a.u. [start_id, end_id, number_of_steps]
coupling_strength = np.logspace(coupling_strength_start, coupling_strength_end, int(coupling_strength_steps))[coupling_strength_idx]

conduction_speed_start, conduction_speed_end, conduction_speed_steps = np.array(config["conduction_speed"], dtype=float)
conduction_speed = np.linspace(conduction_speed_start, conduction_speed_end, int(conduction_speed_steps))[conduction_speed_idx]

logger_local = logzero.setup_logger(logfile=os.path.join(data_path, save_path_analysis, f"logs/{experiment_id}_analysis_potentials_{simulation_id}.log"))
logger_local.info("Begin processing of simulation %s", simulation_id)
print(f"Simulation {simulation_id}: coupling_strength={coupling_strength}, frequency={intrinsic_frequency}, v={conduction_speed}")

# create random number generator with distinct seed for each process
seed = int(pd.read_csv(os.path.join(data_path, save_path_analysis, f'{experiment_id}_random_numbers.csv'), header=None)[0].values[process_id])
rng = np.random.default_rng(seed)


# Parameters for Wave Detection
# -----------------------------
n_permutations = int(config_analysis["number_of_permutations"])  # number of permutations for surrogate data testing
significance_level = float(config_analysis["significance_level"])


# Load Data
# ---------
# load schaefer surface mesh
with open(os.path.join(data_path, "connectomes/Schaefer2018_HCP_S900/schaefer_surface_mesh.pkl"), 'rb') as f:
    surface_mesh = pickle.load(f)
    
v_lh = surface_mesh['vertices_lh']
f_lh = surface_mesh['faces_lh']
v_rh = surface_mesh['vertices_rh']
f_rh = surface_mesh['faces_rh']

number_of_regions_per_hemi = v_lh.shape[0]

# load connectome 
avg_weights = np.load(os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900', 'avg_weights.npy'))
avg_weights = avg_weights + avg_weights.T - np.diag(np.diag(avg_weights))  # symmetrize matrix to create bidirectional connectivity

# compute region in-strength
# sum weights across all incoming connections
instrength = np.sum(avg_weights, axis=0)
instrength_lh = instrength[:number_of_regions_per_hemi]
instrength_rh = instrength[number_of_regions_per_hemi:]

## load phase
phase = np.load(os.path.join(data_path, save_path, f"{experiment_id}_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

# remove initial transient and downsample timesteps 
phase_lh = phase[:number_of_regions_per_hemi, np.arange(initial_transient_samples, phase.shape[1], integration_step_size_downsampled).astype(int)]
phase_rh = phase[number_of_regions_per_hemi:, np.arange(initial_transient_samples, phase.shape[1], integration_step_size_downsampled).astype(int)]


# Compute Spatial Gradients - Left Hemisphere
# -------------------------------------------
logger.info("Simulation %s - Compute phase gradients (left hemi)", simulation_id)

# pre-compute gradient operator
gradient_operator_lh = igl.grad(v_lh, f_lh)

# pre-compute barycenters
bc_coords_lh = compute_barycentric_coords(v_lh, f_lh)

# compute spatial gradient
phase_grad_lh = compute_phase_gradient(phase_lh, f_lh, bc_coords_lh, gradient_operator_lh)

# compute normalized spatial gradient    
phase_grad_norm_lh = (phase_grad_lh / np.linalg.norm(phase_grad_lh, axis=0)).T 


# Compute Helmholtz-Hodge Decomposition - Left Hemisphere
# -------------------------------------------------------
U_lh = compute_helmholtz_hodge_decomposition(-phase_grad_norm_lh, v_lh, f_lh)  # negative phase gradient to get flow direction

# compute instrength-potential correlation
corr_U_lh = np.array([spearmanr(instrength_lh, U_lh[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])


# Compute Spatial Gradients - Right Hemisphere
# -------------------------------------------
logger.info("Simulation %s - Compute phase gradients (right hemi)", simulation_id)

# pre-compute gradient operator
gradient_operator_rh = igl.grad(v_rh, f_rh)

# pre-compute barycenters
bc_coords_rh = compute_barycentric_coords(v_rh, f_rh)

# compute spatial gradient
phase_grad_rh = compute_phase_gradient(phase_rh, f_rh, bc_coords_rh, gradient_operator_rh)

# compute normalized spatial gradient    
phase_grad_norm_rh = (phase_grad_rh / np.linalg.norm(phase_grad_rh, axis=0)).T 


# Compute Helmholtz-Hodge Decomposition - Right Hemisphere
# --------------------------------------------------------
U_rh = compute_helmholtz_hodge_decomposition(-phase_grad_norm_rh, v_rh, f_rh)  # negative phase gradient to get flow direction

# compute instrength-potential correlation
corr_U_rh = np.array([spearmanr(instrength_rh, U_rh[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])


# Prepare spin permutations
# -------------------------
pos_sphere = np.load(os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900/positions_sphere.npy'))
pos_sphere_lh = pos_sphere[:number_of_regions_per_hemi]
pos_sphere_rh = pos_sphere[number_of_regions_per_hemi:]

# fit hemispheres separately (can also be done together)
sp_lh = SpinPermutations(n_rep=n_permutations, random_state=seed)
sp_lh.fit(pos_sphere_lh)
sp_rh = SpinPermutations(n_rep=n_permutations, random_state=seed)
sp_rh.fit(pos_sphere_rh)

# randomize location indices
rotated_idx_lh = sp_lh.randomize(np.arange(0,number_of_regions_per_hemi))
rotated_idx_rh = sp_rh.randomize(np.arange(0,number_of_regions_per_hemi))

assert np.invert(np.all(rotated_idx_lh==rotated_idx_rh))


# Generate surrogate data - left hemi
# -----------------------------------
logger.info("Simulation %s - Conduct permutation testing", simulation_id)

# shuffle original timeseries
corr_exceed_lh = np.zeros(number_of_timesteps_downsampled, dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Left hemi - Permutation %s", pt)

    # shuffle timeseries along spatial dimension
    phase_shuffle_lh = phase_lh[rotated_idx_lh[pt],:]

    # compute phase gradient
    phase_grad_shuffle_lh = compute_phase_gradient(phase_shuffle_lh, f_lh, bc_coords_lh, gradient_operator_lh)
    
    # normalize gradient
    phase_grad_norm_shuffle_lh = (phase_grad_shuffle_lh / np.linalg.norm(phase_grad_shuffle_lh, axis=0)).T

    # compute helmholtz-hodge decomposition
    U_lh_shuffle = compute_helmholtz_hodge_decomposition(-phase_grad_norm_shuffle_lh, v_lh, f_lh)  # negative phase gradient to get flow direction
    
    # compute potential-instrength correlation
    corr_U_shuffle_lh = np.array([spearmanr(instrength_lh, U_lh_shuffle[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])
    
    # Assess significance of correlation
    corr_exceed_lh += corr_U_shuffle_lh <= corr_U_lh

# Wave detection
# compute p-values for instrength-potential correlation
p_corr_lh = corr_exceed_lh / n_permutations

del phase_lh, phase_grad_lh, phase_shuffle_lh, phase_grad_shuffle_lh, U_lh_shuffle
gc.collect()


# Generate surrogate data - right hemi
# -----------------------------------
logger.info("Simulation %s - Conduct permutation testing", simulation_id)

# shuffle original timeseries
corr_exceed_rh = np.zeros(number_of_timesteps_downsampled, dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Right hemi - Permutation %s", pt)

    # shuffle timeseries along spatial dimension
    phase_shuffle_rh = phase_rh[rotated_idx_rh[pt],:]

    # compute phase gradient
    phase_grad_shuffle_rh = compute_phase_gradient(phase_shuffle_rh, f_rh, bc_coords_rh, gradient_operator_rh)
    
    # normalize gradient
    phase_grad_norm_shuffle_rh = (phase_grad_shuffle_rh / np.linalg.norm(phase_grad_shuffle_rh, axis=0)).T

    # compute helmholtz-hodge decomposition
    U_rh_shuffle = compute_helmholtz_hodge_decomposition(-phase_grad_norm_shuffle_rh, v_rh, f_rh)  # negative phase gradient to get flow direction
    
    # compute potential-instrength correlation
    corr_U_shuffle_rh = np.array([spearmanr(instrength_rh, U_rh_shuffle[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])
    
    # Assess significance of correlation
    corr_exceed_rh += corr_U_shuffle_rh <= corr_U_rh

# Wave detection
# compute p-values for instrength-potential correlation
p_corr_rh = corr_exceed_rh / n_permutations

del phase_rh, phase_grad_rh, phase_shuffle_rh, phase_grad_shuffle_rh, U_rh_shuffle
gc.collect()


# Statistics
# ----------
significant_guidedwaves_rh = p_corr_rh <= significance_level 
significant_guidedwaves_lh = p_corr_lh <= significance_level

# create wave mask
guidedwaves_mask_div_rh = significant_guidedwaves_rh.astype(bool)
guidedwaves_mask_div_lh = significant_guidedwaves_lh.astype(bool)


# Save Results
# ------------
logger.info(f"Simulation %s - Save data", simulation_id)

# save wave masks
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_guidedwaves_mask_div_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), guidedwaves_mask_div_lh)
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_guidedwaves_mask_div_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), guidedwaves_mask_div_rh)

# save instrength-potential correlation
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_corr_div_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_U_lh)
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_corr_div_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_U_rh)