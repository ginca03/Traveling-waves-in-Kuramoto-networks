#!/usr/bin/env python
""" Compute wave flow potential in 2D network simulations and detect instrength-guided waves - CONTROL.

This script is used to analyze the control simulation results of a 2D network model based on of Kuramoto oscillators.
The control simulation was created from a network with uniform instrength as opposed to gradually changing instrength.
The phase gradients are decomposed by the Helmholtz-Hodge Decomposition. The curl-free potential (representing the 
diverging phase flow) is correlated with the structural connectivity instrength. Their correlation is used to 
identify waves guided by structural connectivity instrength gradients.

Usage
-----
    python {experiment_id}_analysis_potentials.py "simulation_configuration_path" "analysis_configuration_path" simulation_idx

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
import cv2 
import numpy as np
import pandas as pd
import logzero
from logzero import logger

from scipy import spatial
from scipy.stats import spearmanr, multivariate_normal


from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "23. January 2023"
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
logzero.logfile(os.path.join(data_path, f"{experiment_id}_analysis_potentials/{experiment_id}_analysis_potentials.log"))
logger.info("Begin processing of simulation %s", simulation_id)
logger_local = logzero.setup_logger(logfile=os.path.join(data_path, f"{experiment_id}_analysis_potentials/logs/{experiment_id}_analysis_potentials_{simulation_id}.log"))
logger_local.info("Begin processing of simulation %s", simulation_id)

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, f'{experiment_id}_analysis/{experiment_id}_random_numbers.csv'), header=None)[0].values[simulation_id]
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

# create spatial in-strength gradient
sink_pos = np.array(config["sink_pos"])
source_pos = np.array(config["source_pos"])
singularity_width = float(config["singularity_width"])

sink = multivariate_normal(sink_pos, cov=singularity_width).pdf(pos)  # generate sink gaussian
source = multivariate_normal(source_pos, cov=singularity_width).pdf(pos)  # generate source gaussian

gradient = (sink - source)  # combine gaussians to create gradient
instrength = (gradient - gradient.min()) * 2 / (gradient.max() - gradient.min()) - 1  # min-max normalized gradient


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

# compute normalized spatial phase gradient
phase_grad_norm = (phase_grad / np.linalg.norm(phase_grad, axis=0)).T 


# Compute Helmholtz-Hodge Decomposition
# -------------------------------------
U = compute_helmholtz_hodge_decomposition(-phase_grad_norm, v, f)  # negative phase gradient to get wave flow direction

# compute instrength-potential correlation
corr_U = np.array([spearmanr(instrength, U[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])


# Prepare spin permutations
# -------------------------
# random rotation angle and translation
ang = np.random.uniform(0, 360)
tx = np.random.uniform(0, nx)
ty = np.random.uniform(0, ny)

# get the center coordinates of the image to create the 2D rotation matrix
center = (nx/2, ny/2)
 
# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=ang, scale=1)
 
# build the translation matrix
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)

rot_hom = np.append(rotate_matrix, [[0, 0, 1]], axis=0)
tr_hom = np.append(translation_matrix, [[0, 0, 1]], axis=0)
rt_mat = rot_hom @ tr_hom


# Generate surrogate data
# -----------------------
logger.info("Simulation %s - Conduct permutation testing", simulation_id)

# shuffle original timeseries
corr_exceed = np.zeros(number_of_timesteps_downsampled, dtype=np.uint16)

# iterate over permutations
for pt in range(n_permutations):
    logger_local.info("Left hemi - Permutation %s", pt)

    # random rotation angle and translation
    ang = rng.uniform(0, 360)
    tx = rng.uniform(0, nx)
    ty = rng.uniform(0, ny)
    
    # create rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=(nx/2, ny/2), angle=ang, scale=1)
    
    # create the translation matrix
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=np.float32)

    rot_hom = np.append(rotate_matrix, [[0, 0, 1]], axis=0)
    tr_hom = np.append(translation_matrix, [[0, 0, 1]], axis=0)
    rt_mat = rot_hom @ tr_hom
    
    phase_2d = np.mod(phase.T.reshape(number_of_timesteps_downsampled, nx, ny).transpose((2,1,0)), np.pi*2)
    
    # rotate and translate phases
    phase_shuffle = np.array([cv2.warpAffine(src=phase_2d[:,:,tp], M=rt_mat[:2,:], dsize=(nx, ny), borderMode=cv2.BORDER_REFLECT) for tp in range(number_of_timesteps_downsampled)]).reshape(number_of_timesteps_downsampled, nx*ny).T

    # compute phase gradient
    phase_grad_shuffle = compute_phase_gradient(phase_shuffle, f, bc_coords, gradient_operator)
    
    # normalize gradient
    phase_grad_norm_shuffle = (phase_grad_shuffle / np.linalg.norm(phase_grad_shuffle, axis=0)).T

    # compute helmholtz-hodge decomposition
    U_shuffle = compute_helmholtz_hodge_decomposition(-phase_grad_norm_shuffle, v, f)  # negative phase gradient to get wave flow direction
    
    # compute potential-instrength correlation
    corr_U_shuffle = np.array([spearmanr(instrength, U_shuffle[:,tp])[0] for tp in range(number_of_timesteps_downsampled)])
    
    # Assess significance of correlation
    corr_exceed += corr_U_shuffle <= corr_U

# Detection of guided waves
# compute p-values for instrength-potential correlation
p_corr = corr_exceed / n_permutations


# Save Results
# ------------
logger.info(f"Simulation %s - Save data", simulation_id)

# save p-values
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_p_corr_{simulation_id}.npy"), p_corr)

# save instrength-potential correlation
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_corr_div_{simulation_id}.npy"), corr_U)

# save wave flow potential
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_potential_div_{simulation_id}.npy"), U)
