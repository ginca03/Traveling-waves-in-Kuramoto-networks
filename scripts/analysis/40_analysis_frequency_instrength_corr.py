#!/usr/bin/env python
""" Compute effective frequency - instrength correlation.

This script is used to analyze the effective frequency - instrength correlation for the parameter exploration of
the cortical network model.


Usage
-----
    python 40_analysis_frequency_instrength_corr.py "simulation_configuration_path" "analysis_configuration_path" process_id

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
import yaml
import itertools
import numpy as np

from tvb.simulator.lab import *


from modules.wave_detection_methods import *
from modules.helpers import *


__author__ = "Dominik Koller"
__date__ = "28. November 2022"
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

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms
initial_transient = float(config["initial_transient"])  # ms
number_of_timesteps = int((simulation_duration-initial_transient)/integration_step_size)

downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
integration_step_size_downsampled = integration_step_size * downsampling_factor
number_of_timesteps_downsampled = int((simulation_duration-initial_transient)/integration_step_size_downsampled)

proportion_waves_threshold = float(config_analysis["proportion_waves_threshold"])


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

print(f"Simulation {simulation_id}: coupling_strength={coupling_strength}, frequency={intrinsic_frequency}, v={conduction_speed}")


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


# Prepare spin permutations
# -------------------------
sphere_path = np.os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900/positions_sphere.npy')


# Load Data
# ---------
# load instrength
avg_weights = np.load(os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900', 'avg_weights.npy'))
avg_weights = avg_weights + avg_weights.T - np.diag(np.diag(avg_weights))  # symmetrize matrix to create bidirectional connectivity

# compute region in-strength
# sum the SIFT2 weights across all incoming connections
instrength = np.sum(avg_weights, axis=0)
instrength_lh = instrength[:500]
instrength_rh = instrength[500:]

# load wave mask
wave_mask_div_lh = np.load(os.path.join(data_path, f'40_analysis/40_wave_mask_div_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy'))
wave_mask_div_rh = np.load(os.path.join(data_path, f'40_analysis/40_wave_mask_div_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy'))
#print('Proportion waves %f' % (wave_mask_div_lh.sum()/number_of_timesteps_downsampled))

# compute wave speed only if waves are present
proportion_waves_lh = wave_mask_div_lh.sum()/number_of_timesteps_downsampled
proportion_waves_rh = wave_mask_div_rh.sum()/number_of_timesteps_downsampled


# Process Left Hemisphere
# -----------------------
if (proportion_waves_lh > proportion_waves_threshold):
    ## load phase
    phase = np.load(os.path.join(data_path, f"40_simulations/40_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # downsampling of timesteps
    phase_lh = phase[:500, np.arange(initial_transient, phase.shape[1], integration_step_size).astype(int)]

    # complex exponential phase
    ce_phase_lh = np.exp(1j*phase_lh)
    

    # Compute frequency - instrength correlation
    # ------------------------------------------
    frequency_tmp_lh = compute_instantaneous_frequency(ce_phase_lh, integration_step_size*1e-3)  # compute inst freq
    frequency_tmp_lh = frequency_tmp_lh[:, np.arange(0, number_of_timesteps-1, integration_step_size_downsampled).astype(int)]  # downsampling
    frequency_median_lh = np.median(frequency_tmp_lh[:,wave_mask_div_lh], axis=1)  # compute median inst frequency
        
    corr_lh, corr_p_lh = correlation_spin_permutation_testing(frequency_median_lh, instrength_lh, sphere_path=sphere_path, hemi='lh')
 
 
    # Save Results
    # ------------
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_frequency-instrength_corr_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_lh)
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_frequency-instrength_corr_p_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_p_lh)
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_instantaneous_frequency_median_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), frequency_median_lh)


# Process Right Hemisphere
# ------------------------
if (proportion_waves_rh > proportion_waves_threshold):
    ## load phase
    phase = np.load(os.path.join(data_path, f"40_simulations/40_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # downsampling of timesteps
    phase_rh = phase[500:, np.arange(initial_transient, phase.shape[1], integration_step_size).astype(int)]

    # complex exponential phase
    ce_phase_rh = np.exp(1j*phase_rh)
    

    # Compute frequency - instrength correlation
    # ------------------------------------------
    frequency_tmp_rh = compute_instantaneous_frequency(ce_phase_rh, integration_step_size*1e-3)  # compute inst freq
    frequency_tmp_rh = frequency_tmp_rh[:, np.arange(0, number_of_timesteps-1, integration_step_size_downsampled).astype(int)]  # downsampling
    frequency_median_rh = np.median(frequency_tmp_rh[:,wave_mask_div_rh], axis=1)  # compute median inst frequency
    
    corr_rh, corr_p_rh = correlation_spin_permutation_testing(frequency_median_rh, instrength_rh, sphere_path=sphere_path, hemi='rh')
 
 
    # Save Results
    # ------------
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_frequency-instrength_corr_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_rh)
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_frequency-instrength_corr_p_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), corr_p_rh)
    np.save(os.path.join(data_path, "40_analysis_frequency_instrength_corr", f"40_instantaneous_frequency_median_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), frequency_median_rh)