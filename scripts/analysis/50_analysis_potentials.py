#!/usr/bin/env python
""" CONTROL - Compute wave flow potential in cortical network simulations and detect instrength-guided waves.

This script is used to analyze the simulation results of the cortical network model with added noise.
Differential operators are computed on the 2D cortical mesh determined by the positions of the regions of the Schaefer 
brain atlas. These operators are used to compute the phase gradients. The phase gradients are then decomposed by the 
Helmholtz-Hodge Decomposition. The curl-free potential (representing the diverging phase flow) is correlated with the 
structural connectivity instrength. Their correlation is used to identify waves guided by structural connectivity 
instrength gradients.

Usage
-----
    python {experiment_id}_analysis_potentials.py simulation_configuration_path analysis_configuration_path simulation_idx

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

from brainspace.null_models import SpinPermutations
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.stats import spearmanr
from tvb.simulator.lab import *

from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "27. July 2023"
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
final_transient = float(config["final_transient"])  # ms
final_transient_samples = int(final_transient/integration_step_size)  # samples

downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
integration_step_size_downsampled = integration_step_size * downsampling_factor
number_of_timesteps_downsampled = int((simulation_duration-initial_transient-final_transient)/integration_step_size_downsampled)


# Parameters to Explore
# ---------------------
simulation_idx = int(sys.argv[3])  # get idx
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]

print(f"Simulation {simulation_id}")
logzero.logfile(os.path.join(data_path, "30_analysis_potentials/30_analysis_potentials.log"))
logger.info("Begin processing of simulation %s", simulation_id)
logger_local = logzero.setup_logger(logfile=os.path.join(data_path, f"30_analysis_potentials/logs/30_analysis_potentials_{simulation_id}.log"))
logger_local.info("Begin processing of simulation %s", simulation_id)

# create random number generator with distinct seed for each process
seed = pd.read_csv(os.path.join(data_path, f'{experiment_id}_analysis_potentials/{experiment_id}_random_numbers.csv'), header=None)[0].values[simulation_id]
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
phase = np.load(os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{simulation_id}.npy")).T
activity = np.cos(phase)

# bandpass filter data
f_low = float(config_analysis["f_low"])  # Hz
f_high = float(config_analysis["f_high"])  # Hz
filt_order = 8
sos = butter(int(filt_order/2), [f_low, f_high], btype='band', output='sos', fs=1/(integration_step_size*1e-3))
activity_filt = sosfiltfilt(sos, activity, axis=1)

# remove initial and final transient and downsample timesteps 
activity_lh = activity_filt[:number_of_regions_per_hemi, initial_transient_samples:-final_transient_samples]
activity_rh = activity_filt[number_of_regions_per_hemi:, initial_transient_samples:-final_transient_samples]

# extract instantaneous phases and downsample
analytic_signal_lh = hilbert(activity_lh, axis=1)
phase_lh = np.angle(analytic_signal_lh)[:, np.arange(0, activity_lh.shape[1], integration_step_size_downsampled).astype(int)]
analytic_signal_rh = hilbert(activity_rh, axis=1)
phase_rh = np.angle(analytic_signal_rh)[:, np.arange(0, activity_rh.shape[1], integration_step_size_downsampled).astype(int)]


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


# Save Results
# ------------
logger.info(f"Simulation %s - Save data", simulation_id)

# save p-values
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_p_corr_rh_{simulation_id}.npy"), p_corr_rh)
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_p_corr_lh_{simulation_id}.npy"), p_corr_lh)

# save instrength-potential correlation
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_corr_div_lh_{simulation_id}.npy"), corr_U_lh)
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_corr_div_rh_{simulation_id}.npy"), corr_U_rh)

# save angular_similarity
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_potential_div_lh_{simulation_id}.npy"), U_lh)
np.save(os.path.join(data_path, f"{experiment_id}_analysis_potentials", f"{experiment_id}_potential_div_rh_{simulation_id}.npy"), U_rh)