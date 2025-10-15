#!/usr/bin/env python
""" Intrinsic frequency vs. instrength gradient in 2D network simulations.

This script analyzes the effect of superimposing an intrinsic frequency gradient to
the instrength gradient.

Usage
-----
    python {experiment_id}_analysis.py "simulation_configuration_path" "analysis_configuration_path" simulation_idx

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
import itertools
import yaml
import numpy as np
import pandas as pd

from scipy import spatial
from scipy.stats import spearmanr, multivariate_normal


from modules.wave_detection_methods import *
from modules.helpers import *


__author__ = "Dominik Koller"
__date__ = "31. July 2023"
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
number_of_timesteps = int((simulation_duration-initial_transient)/integration_step_size)


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[3])

sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
gradient_scaling_start, gradient_scaling_end, gradient_scaling_steps = np.array(config["intrinsic_frequency_gradient_scaling"], dtype=float)

simulation_idx, gradient_scaling_idx = list(itertools.product(*[range(0,len(np.arange(sim_id_start, sim_id_end, sim_id_step))), range(0,int(gradient_scaling_steps))]))[process_id]

simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]
gradient_scaling = np.linspace(gradient_scaling_start, gradient_scaling_end, int(gradient_scaling_steps))[gradient_scaling_idx]

print("gradient scaling %.2f" % gradient_scaling)
print(f"Simulation {simulation_id}")

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, save_path_analysis, f'{experiment_id}_random_numbers.csv'), header=None)[0].values[simulation_id]
rng = np.random.default_rng(seed)


# Parameters for Wave Detection
# -----------------------------
n_permutations = int(config_analysis["number_of_permutations"])  # number of permutations for surrogate data testing
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

# create spatial in-strength gradient
sink_pos = np.array(config["sink_pos"])
source_pos = np.array(config["source_pos"])
singularity_width = float(config["singularity_width"])

sink = multivariate_normal(sink_pos, cov=singularity_width).pdf(pos)  # generate sink gaussian
source = multivariate_normal(source_pos, cov=singularity_width).pdf(pos)  # generate source gaussian

gradient = (sink - source)  # combine and gaussians to create gradient
gradient = (gradient - gradient.min()) * 2 / (gradient.max() - gradient.min()) - 1  # min-max normalized gradient


# Load Data
# ---------
## load phase
phase = np.load(os.path.join(data_path, save_path, f"{experiment_id}_simulation_{simulation_idx}_{gradient_scaling_idx}.npy"))[initial_transient_samples:].T

phase_downsampled = phase[:, np.arange(0, phase.shape[1], integration_step_size_downsampled).astype(int)]
print(phase.shape)

# Compute Spatial Gradients
# -------------------------
# pre-compute gradient operator
gradient_operator = igl.grad(v, f)

# pre-compute barycenters
bc_coords = compute_barycentric_coords(v, f)

# compute spatial gradient
phase_grad = compute_phase_gradient(phase_downsampled, f, bc_coords, gradient_operator)

# compute normalized spatial phase gradient
phase_grad_norm = (phase_grad / np.linalg.norm(phase_grad, axis=0)).T 


# Compute Helmholtz-Hodge Decomposition
# -------------------------------------
U = compute_helmholtz_hodge_decomposition(-phase_grad_norm, v, f)  # negative phase gradient to get wave flow direction

# compute gradient-potential correlation
corr_U = np.array([spearmanr(gradient, U[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])


# Compute effective frequency
# ---------------------------
effective_frequency = compute_instantaneous_frequency(np.exp(1j*phase), integration_step_size*1e-3)


# Compute Angular Similarity
# --------------------------
# Calculate the angular similarity between phase gradients and the ideal divergent vector field
# compute wave template
div_template, neighbours_faces = compute_wave_template(v, f)

# compute angular similarity
angular_similarity_div = compute_angular_similarity(phase_grad, div_template, neighbours_faces, boundary_mask)


# Generate surrogate data
# -----------------------
# shuffle original timeseries
div_exceed = np.zeros((n, number_of_timesteps_downsampled), dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):    
    print("Permutation %i" % pt)
    # shuffle timeseries along spatial dimension
    phase_shuffle = phase_downsampled[rng.permutation(n)]

    # compute phase gradient
    phase_grad_shuffle = compute_phase_gradient(phase_shuffle, f, bc_coords, gradient_operator)

    # compute angular similarity
    angular_similarity_div_shuffle = compute_angular_similarity(phase_grad_shuffle, div_template, neighbours_faces, boundary_mask)
    
    div_exceed[np.invert(boundary_mask)] += (np.max(abs(angular_similarity_div_shuffle), axis=0) >= abs(angular_similarity_div))

# Wave detection
# compute p-values for singularity statistic
p_div = div_exceed / n_permutations


# Save Results
# ------------
# save instrength-potential correlation
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_corr_div_{simulation_idx}_{gradient_scaling_idx}.npy"), corr_U)

# save wave flow potential
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_potential_div_{simulation_idx}_{gradient_scaling_idx}.npy"), U)

# save effective frequency
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_effective_frequency_{simulation_idx}_{gradient_scaling_idx}.npy"), effective_frequency)

# save p-values
np.save(os.path.join(data_path, save_path_analysis, f"{experiment_id}_p_div_{simulation_idx}_{gradient_scaling_idx}.npy"), p_div)