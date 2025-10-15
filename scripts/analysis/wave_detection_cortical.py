#!/usr/bin/env python
""" Detect singularities in cortical network simulations on Schaefer parcellation

This script is used to analyze the simulation results of a cortical network model based on Kuramoto oscillators.
Differential operators are computed on the 2D cortical mesh determined by the positions of the regions of the Schaefer 
brain atlas. These operators are used to compute the phase gradients. These empirical phase gradients are compared
(angular similarity) to idealized diverging vector fields within a local neighbourhood around a region to 
identify strong systematic phase changes across space that are consistent with diverging wave organization.

Usage
-----
    python wave_detection_cortical.py "simulation_configuration_path" "analysis_configuration_path" simulation_idx

Arguments
---------
simulation_configuration_path : String
    Path to configuration file of the simulations.
    
analysis_configuration_path : String
    Path to configuration file of the analysis.
    
simulation_idx : int
    Index for the simulation ID.
"""


import sys
import os
import gc
import pickle
import yaml
import numpy as np
import pandas as pd
import logzero
from logzero import logger

from tvb.simulator.lab import *
from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "25. July 2023"
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
simulation_idx = int(sys.argv[3])  # get idx
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]

print(f"Simulation {simulation_id}")
logzero.logfile(os.path.join(data_path, f"{experiment_id}_analysis/{experiment_id}_analysis.log"))
logger.info("Begin processing of simulation %s", simulation_id)
logger_local = logzero.setup_logger(logfile=os.path.join(data_path, f"{experiment_id}_analysis/logs/{experiment_id}_analysis_{simulation_id}.log"))
logger_local.info("Begin processing of simulation %s", simulation_id)

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, f'{experiment_id}_analysis/{experiment_id}_random_numbers.csv'), header=None)[0].values[simulation_id]
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

# get boundary mask
boundary_k_ring = int(config['boundary_k_ring'])
boundary_mask_lh = k_ring_boundary(v_lh, f_lh, k=boundary_k_ring)
boundary_mask_rh = k_ring_boundary(v_rh, f_rh, k=boundary_k_ring)

## load phase
phase = np.load(os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{simulation_id}.npy")).T

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
angular_similarity_div_lh = compute_angular_similarity(phase_grad_lh, div_template_lh, neighbours_faces_lh, boundary_mask=boundary_mask_lh)


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
angular_similarity_div_rh = compute_angular_similarity(phase_grad_rh, div_template_rh, neighbours_faces_rh, boundary_mask=boundary_mask_rh)


# Generate surrogate data - left hemi
# -----------------------------------
# shuffle original timeseries
div_exceed_lh = np.zeros((number_of_regions_per_hemi, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Left hemi - Permutation %s", pt)
    
    # shuffle timeseries along spatial dimension
    phase_shuffle_lh = phase_lh[rng.permutation(number_of_regions_per_hemi),:]

    # compute phase gradient
    phase_grad_shuffle_lh = compute_phase_gradient(phase_shuffle_lh, f_lh, bc_coords_lh, gradient_operator_lh)
    
    # compute angular similarity
    angular_similarity_div_shuffle_lh = compute_angular_similarity(phase_grad_shuffle_lh, div_template_lh, neighbours_faces_lh, boundary_mask=boundary_mask_lh)
    
    div_exceed_lh[np.invert(boundary_mask_lh)] += (np.max(abs(angular_similarity_div_shuffle_lh), axis=0) >= abs(angular_similarity_div_lh))

# Wave detection
# compute p-values for singularity statistic
p_div_lh = div_exceed_lh / n_permutations

del phase_lh, phase_grad_lh, phase_shuffle_lh, phase_grad_shuffle_lh, angular_similarity_div_shuffle_lh, angular_similarity_rot_shuffle_lh
gc.collect()


# Generate surrogate data - right hemi
# ------------------------------------
# shuffle original timeseries
div_exceed_rh = np.zeros((number_of_regions_per_hemi, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Left hemi - Permutation %s", pt)
    
    # shuffle timeseries along spatial dimension
    phase_shuffle_rh = phase_rh[rng.permutation(number_of_regions_per_hemi),:]

    # compute phase gradient
    phase_grad_shuffle_rh = compute_phase_gradient(phase_shuffle_rh, f_rh, bc_coords_rh, gradient_operator_rh)
    
    # compute angular similarity
    angular_similarity_div_shuffle_rh = compute_angular_similarity(phase_grad_shuffle_rh, div_template_rh, neighbours_faces_rh, boundary_mask=boundary_mask_rh)
    angular_similarity_rot_shuffle_rh = compute_angular_similarity(phase_grad_shuffle_rh, rot_template_rh, neighbours_faces_rh, boundary_mask=boundary_mask_rh)
    
    div_exceed_rh[np.invert(boundary_mask_rh)] += (np.max(abs(angular_similarity_div_shuffle_rh), axis=0) >= abs(angular_similarity_div_rh))

# Wave detection
# compute p-values for singularity statistic
p_div_rh = div_exceed_rh / n_permutations

del phase_rh, phase_grad_rh, phase_shuffle_rh, phase_grad_shuffle_rh, angular_similarity_div_shuffle_rh, angular_similarity_rot_shuffle_rh
gc.collect()


# Save Results
# ------------
logger.info(f"Simulation %s - Save data", simulation_id)

# save p-values
np.save(os.path.join(data_path, f"{experiment_id}_analysis", f"{experiment_id}_p_div_lh_{simulation_id}.npy"), p_div_lh)
np.save(os.path.join(data_path, f"{experiment_id}_analysis", f"{experiment_id}_p_div_rh_{simulation_id}.npy"), p_div_rh)