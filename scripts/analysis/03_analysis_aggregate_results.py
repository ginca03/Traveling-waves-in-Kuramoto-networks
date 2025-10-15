#!/usr/bin/env python
""" Intrinsic frequency vs. instrength gradient in 2D network simulations.

This script analyzes the effect of superimposing an intrinsc frequency gradient to
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
save_path_analysis = config_analysis["save_path_analysis"]
save_path = config["save_path"]
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
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
gradient_scaling_start, gradient_scaling_end, gradient_scaling_steps = np.array(config["intrinsic_frequency_gradient_scaling"], dtype=float)

parameter_ids = list(itertools.product(*[range(0,len(np.arange(sim_id_start, sim_id_end, sim_id_step))), range(0,int(gradient_scaling_steps))]))


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


# Aggregate results
# -----------------
proportion_waves = np.zeros((sim_id_end, int(gradient_scaling_steps)))
gradient_effective_frequency_corr = np.zeros((sim_id_end,int(gradient_scaling_steps)))
gradient_potential_corr = np.zeros((sim_id_end,int(gradient_scaling_steps)))

for sim_id, scale_id in parameter_ids:
    # Find significant singularities
    p_div = np.load(os.path.join(data_path, save_path_analysis, f'{experiment_id}_p_div_{sim_id}_{scale_id}.npy'))[np.invert(boundary_mask)]

    significant_div = p_div <= significance_level

    # Create wave mask
    wave_mask_div = np.nansum(significant_div, axis=0, dtype=bool)
    proportion_waves[sim_id,scale_id] = np.nansum(wave_mask_div) / number_of_timesteps_downsampled
        
    # Gradient - potential correlation
    gradient_potential_corr[sim_id,scale_id] = np.mean(np.load(os.path.join(data_path, save_path_analysis, f"{experiment_id}_corr_div_{sim_id}_{scale_id}.npy")))
    
    # Gradient - effective frequency correlation
    effective_frequency = np.median(np.load(os.path.join(data_path, save_path_analysis, f'{experiment_id}_effective_frequency_{sim_id}_{scale_id}.npy')), axis=1)
    gradient_effective_frequency_corr[sim_id,scale_id] = spearmanr(gradient, effective_frequency)[0]


# Save Results
# ------------
# save gradient - potential correlation
np.save(os.path.join(data_path, f"{save_path_analysis}_aggregated", f"{experiment_id}_gradient_potential_corr_div.npy"), gradient_potential_corr)

# save gradient - effective frequency 
np.save(os.path.join(data_path, f"{save_path_analysis}_aggregated", f"{experiment_id}_gradient_effective_frequency_corr_div.npy"), gradient_effective_frequency_corr)

# save proportion of waves
np.save(os.path.join(data_path, f"{save_path_analysis}_aggregated", f"{experiment_id}_proportion_div_waves.npy"), proportion_waves)