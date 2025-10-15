#!/usr/bin/env python
""" Detect singularities in parameter exploration of cortical network simulations on Schaefer parcellation

This script is used to analyze the simulation results of a cortical network model based on Kuramoto oscillators.
Differential operators are computed on the 2D cortical mesh determined by the positions of the regions of the Schaefer 
brain atlas. These operators are used to compute the phase gradients. These empirical phase gradients are compared
(angular similarity) to idealized diverging vector fields within a local neighbourhood around a region to identify 
strong systematic phase changes across space that are consistent with diverging wave organization.


Usage
-----
    python 40_analysis.py "simulation_configuration_path" "analysis_configuration_path" process_id

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

from tvb.simulator.lab import *


from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "10. June 2022"
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
experiment_id = config["experiment_id"]
data_path = config["data_path"]
save_path = config["save_path"]
save_path_analysis = config_analysis["save_path_analysis"]

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

logger_local = logzero.setup_logger(logfile=os.path.join(data_path, save_path_analysis, f"logs/{experiment_id}_analysis_{simulation_id}.log"))
logger_local.info("Begin processing of simulation %s", simulation_id)
print(f"Simulation {simulation_id}: coupling_strength={coupling_strength}, frequency={intrinsic_frequency}, v={conduction_speed}")

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, save_path_analysis, f'{experiment_id}_random_numbers.csv'), header=None)[0].values[process_id]
rng = np.random.default_rng(int(seed))


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


# Compute Angular Similarity - Left Hemisphere
# ---------------------------------------------
# Calculate the angular similarity between phase gradients and the ideal divergent vector field
# compute wave template
div_template_lh, neighbours_faces_lh = compute_wave_template(v_lh, f_lh)

# compute angular similarity
angular_similarity_div_lh = compute_angular_similarity(phase_grad_lh, div_template_lh, neighbours_faces_lh)


# Compute Spatial Gradients - Right Hemisphere
# -------------------------------------------
logger.info("Simulation %s - Compute phase gradients (right hemi)", simulation_id)

# pre-compute gradient operator
gradient_operator_rh = igl.grad(v_rh, f_rh)

# pre-compute barycenters
bc_coords_rh = compute_barycentric_coords(v_rh, f_rh)

# compute spatial gradient
phase_grad_rh = compute_phase_gradient(phase_rh, f_rh, bc_coords_rh, gradient_operator_rh)


# Compute Angular Similarity - Right Hemisphere
# ---------------------------------------------
# Calculate the angular similarity between phase gradients and the ideal divergent vector field
# compute wave template
div_template_rh, neighbours_faces_rh = compute_wave_template(v_rh, f_rh)

# compute angular similarity
angular_similarity_div_rh = compute_angular_similarity(phase_grad_rh, div_template_rh, neighbours_faces_rh)


# Generate surrogate data - left hemi
# -----------------------------------
# shuffle original timeseries
singularity_exceed_lh = np.zeros((number_of_regions_per_hemi, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Left hemi - Permutation %s", pt)
    
    # shuffle timeseries along spatial dimension
    phase_shuffle_lh = phase_lh[rng.permutation(number_of_regions_per_hemi),:]

    # compute phase gradient
    phase_grad_shuffle_lh = compute_phase_gradient(phase_shuffle_lh, f_lh, bc_coords_lh, gradient_operator_lh)
    
    # compute angular similarity
    angular_similarity_div_shuffle_lh = compute_angular_similarity(phase_grad_shuffle_lh, div_template_lh, neighbours_faces_lh)
    
    singularity_exceed_lh += (np.max(abs(angular_similarity_div_shuffle_lh), axis=0) >= abs(angular_similarity_div_lh))

# Wave detection
# compute p-values for singularity statistic
p_singularities_lh = singularity_exceed_lh / n_permutations

del phase_lh, phase_grad_lh, phase_shuffle_lh, phase_grad_shuffle_lh, angular_similarity_div_shuffle_lh
gc.collect()


# Generate surrogate data - right hemi
# ------------------------------------
# shuffle original timeseries
singularity_exceed_rh = np.zeros((number_of_regions_per_hemi, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Right hemi - Permutation %s", pt)
    
    # shuffle timeseries along spatial dimension
    phase_shuffle_rh = phase_rh[rng.permutation(number_of_regions_per_hemi),:]

    # compute phase gradient
    phase_grad_shuffle_rh = compute_phase_gradient(phase_shuffle_rh, f_rh, bc_coords_rh, gradient_operator_rh)
    
    # compute angular similarity
    angular_similarity_div_shuffle_rh = compute_angular_similarity(phase_grad_shuffle_rh, div_template_rh, neighbours_faces_rh)
    
    singularity_exceed_rh += (np.max(abs(angular_similarity_div_shuffle_rh), axis=0) >= abs(angular_similarity_div_rh))

# Wave detection
# compute p-values for singularity statistic
p_singularities_rh = singularity_exceed_rh / n_permutations

del phase_rh, phase_grad_rh, phase_shuffle_rh, phase_grad_shuffle_rh, angular_similarity_div_shuffle_rh
gc.collect()


# Statistics
# ----------
significant_singularity_rh = p_singularities_rh <= significance_level
significant_singularity_lh = p_singularities_lh <= significance_level

# create wave mask
wave_mask_div_rh = np.sum(significant_singularity_rh, axis=0, dtype=bool)
wave_mask_div_lh = np.sum(significant_singularity_lh, axis=0, dtype=bool)


# Save Results
# ------------
logger.info(f"Simulation %s - Save data", simulation_id)

# save wave masks
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_wave_mask_div_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), wave_mask_div_lh)
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_wave_mask_div_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), wave_mask_div_rh)