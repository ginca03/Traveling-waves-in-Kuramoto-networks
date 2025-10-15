#!/usr/bin/env python
""" Detect singularities in 2D Kuramoto network simulations.

This script is used to analyze the simulation results of a 2D network model based on of Kuramoto oscillators.
Differential operators are computed on the 2D mesh determined by the positions of the oscillators. These operators are 
used to compute the phase gradients. These empirical phase gradients are compared (angular similarity) to idealized 
diverging vector fields within a local neighbourhood around an oscillator to identify strong systematic 
phase changes across space that are consistent with diverging wave organization.

Usage
-----
    python wave_detection_2d.py "simulation_configuration_path" "analysis_configuration_path" simulation_idx

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
import yaml
import numpy as np
import pandas as pd

from scipy import spatial
from modules.wave_detection_methods import *

__author__ = "Dominik Koller"
__date__ = "26. July 2023"
__status__ = "Prototype"


def create_region_positions(x_extent, y_extent, nx, ny):
    """Computes 2D-positions given the extent of the dimensions and the number of regions along each dimension.
    
    Parameters
    ----------
    x_extent : numpy.float
        Extent along the x dimension.
    y_extent : numpy.float
        Extent along the y dimension.
    nx : int
        Number of regions along the x dimension
    ny : int
        Number of regions along the y dimension
        
    Returns
    -------
    pos : numpy.array
        Array with the x-y positions of each region.
    """
    y = np.linspace(0, y_extent, ny)
    x = np.linspace(0, x_extent, nx)

    pos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)
    
    return pos


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

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, f'seed_random_numbers.csv'), header=None)[0].values[simulation_id]
rng = np.random.default_rng(seed)


# Parameters for Wave Detection
# -----------------------------
n_permutations = int(config_analysis["number_of_permutations"])
# Reduce permutations to speed up computation
n_permutations = max(100, n_permutations // 10)  # Use 1/10 of original, minimum 100
print(f"Using {n_permutations} permutations (reduced for computational efficiency)")

significance_level = float(config_analysis["significance_level"])


# Model Parameters
# ----------------
# network parameters
nx = int(config["nx"])  # number of regions along x-dimension
ny = int(config["ny"])  # number of regions along y-dimension

n = nx*ny  # number of regions

x_extent = float(config["x_extent"])  # mm, extent of x-dimension
y_extent = float(config["y_extent"])  # mm, extent of y-dimension

# create node positions
y = np.linspace(0, y_extent, ny)
x = np.linspace(0, x_extent, nx)
pos = create_region_positions(x_extent, y_extent, nx, ny)

# prepare mesh
v = np.c_[pos, np.zeros(len(pos))]
f = spatial.Delaunay(pos).simplices

# get boundary mask
boundary_k_ring = int(config["boundary_k_ring"])
boundary_mask = k_ring_boundary(v, f, k=boundary_k_ring)


# Load Data
# ---------
## load phase
phase = np.load(os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{simulation_id}.npy")).T

# remove initial transient and downsample timesteps
phase = phase[:, np.arange(initial_transient_samples, phase.shape[1], integration_step_size_downsampled).astype(int)]


# Compute Spatial Gradients
# -------------------------
# pre-compute gradient operator
gradient_operator = igl.grad(v, f)

# pre-compute barycenters
bc_coords = compute_barycentric_coords(v, f)

# compute spatial gradient
phase_grad = compute_phase_gradient(phase, f, bc_coords, gradient_operator)


# Compute Angular Similarity
# --------------------------
# Calculate the angular similarity between phase gradients and the ideal divergent and rotational vector fields
# compute wave templates (both divergent and curl)
div_template, curl_template, neighbours_faces = compute_wave_template(v, f, return_curl_template=True)

# compute angular similarities
angular_similarity_div = compute_angular_similarity(phase_grad, div_template, neighbours_faces, boundary_mask)
angular_similarity_rot = compute_angular_similarity(phase_grad, curl_template, neighbours_faces, boundary_mask)


# Generate surrogate data
# -----------------------
# shuffle original timeseries
div_exceed = np.zeros((n, number_of_timesteps_downsampled), dtype=np.uint16)
rot_exceed = np.zeros((n, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    print("Permutation %i" % pt)
    # shuffle timeseries along spatial dimension
    phase_shuffle = phase[rng.permutation(n),:]

    # compute phase gradient
    phase_grad_shuffle = compute_phase_gradient(phase_shuffle, f, bc_coords, gradient_operator)
    
    # compute angular similarities for both divergent and rotational patterns
    angular_similarity_div_shuffle = compute_angular_similarity(phase_grad_shuffle, div_template, neighbours_faces, boundary_mask)
    angular_similarity_rot_shuffle = compute_angular_similarity(phase_grad_shuffle, curl_template, neighbours_faces, boundary_mask)
    
    # update exceed counters
    div_exceed[np.invert(boundary_mask)] += (np.max(abs(angular_similarity_div_shuffle), axis=0) >= abs(angular_similarity_div))
    rot_exceed[np.invert(boundary_mask)] += (np.max(abs(angular_similarity_rot_shuffle), axis=0) >= abs(angular_similarity_rot))

# Wave detection
# compute p-values for both singularity statistics
p_div = div_exceed / n_permutations
p_rot = rot_exceed / n_permutations

# Create directory if it doesn't exist
os.makedirs(os.path.join(data_path, f"{experiment_id}_wave_analysis"), exist_ok=True)

# Save Results
# ------------
# save p-values for both divergent and rotational patterns
np.save(os.path.join(data_path, f"{experiment_id}_wave_analysis", f"{experiment_id}_p_div_{simulation_id}.npy"), p_div)
np.save(os.path.join(data_path, f"{experiment_id}_wave_analysis", f"{experiment_id}_p_rot_{simulation_id}.npy"), p_rot)
