#!/usr/bin/env python
""" Compute wave speed

This script is used to analyze the simulation results of a full brain network model based on of Kuramoto oscillators.

Usage
-----
    python 40_analysis_wave_speed.py "simulation_configuration_path" "analysis_configuration_path" process_id

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
import pickle
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
n = v_lh.shape[0]

# load wave mask
wave_mask_div_lh = np.load(os.path.join(data_path, f'40_analysis/40_wave_mask_div_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy'))
wave_mask_div_rh = np.load(os.path.join(data_path, f'40_analysis/40_wave_mask_div_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy'))

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


    # Compute Spatial Gradients
    # -------------------------
    # pre-compute gradient operator
    gradient_operator_lh = igl.grad(v_lh, f_lh)

    # pre-compute barycenters
    bc_coords_lh = compute_barycentric_coords(v_lh, f_lh)
    
    # interpolate vertex values on triangle barycenters
    ce_phase_faces_lh = np.sum(ce_phase_lh.T[:,f_lh] * bc_coords_lh, axis=2).T

    # compute spatial gradient
    phase_grad_tmp = gradient_operator_lh.dot(ce_phase_lh).reshape(3, f_lh.shape[0], phase_lh.shape[1])
    phase_grad_lh = np.real(-1j*ce_phase_faces_lh.conj()*phase_grad_tmp)


    # Compute wave speed
    # ------------------
    # compute phase gradient magnitude
    phase_grad_mag_lh = np.linalg.norm(phase_grad_lh, axis=0)

    # compute instantaneous frequency on triangle barycenters
    instantaneous_frequency_lh = compute_instantaneous_frequency(ce_phase_faces_lh, integration_step_size*1e-3)  # compute inst freq

    # compute wave speeds
    wave_speed_lh = (instantaneous_frequency_lh*2*np.pi) / phase_grad_mag_lh[:,:-1]
    wave_speed_lh = wave_speed_lh[:, np.arange(0, number_of_timesteps, integration_step_size_downsampled).astype(int)]  # downsampling
    wave_speed_lh = np.median(wave_speed_lh[:,wave_mask_div_lh], axis=1)  # median of wave speed
    
 
    # Save Results
    # ------------
    # save wave masks
    np.save(os.path.join(data_path, "40_analysis_wave_speed", f"40_wave_speed_lh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), wave_speed_lh)


# Process Right Hemisphere
# ------------------------
if (proportion_waves_rh > proportion_waves_threshold):
    
    ## load phase
    phase = np.load(os.path.join(data_path, f"40_simulations/40_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # downsampling of timesteps
    phase_rh = phase[500:, np.arange(initial_transient, phase.shape[1], integration_step_size).astype(int)]

    # complex exponential phase
    ce_phase_rh = np.exp(1j*phase_rh)


    # Compute Spatial Gradients
    # -------------------------
    # pre-compute gradient operator
    gradient_operator_rh = igl.grad(v_rh, f_rh)

    # pre-compute barycenters
    bc_coords_rh = compute_barycentric_coords(v_rh, f_rh)
    
    # interpolate vertex values on triangle barycenters
    ce_phase_faces_rh = np.sum(ce_phase_rh.T[:,f_rh] * bc_coords_rh, axis=2).T

    # compute spatial gradient
    phase_grad_tmp = gradient_operator_rh.dot(ce_phase_rh).reshape(3, f_rh.shape[0], phase_rh.shape[1])
    phase_grad_rh = np.real(-1j*ce_phase_faces_rh.conj()*phase_grad_tmp)


    # Compute wave speed
    # ------------------
    # compute phase gradient magnitude
    phase_grad_mag_rh = np.linalg.norm(phase_grad_rh, axis=0)

    # compute instantaneous frequency on triangle barycenters
    instantaneous_frequency_rh = compute_instantaneous_frequency(ce_phase_faces_rh, integration_step_size*1e-3)  # compute inst freq

    # compute wave speeds
    wave_speed_rh = (instantaneous_frequency_rh*2*np.pi) / phase_grad_mag_rh[:,:-1]
    wave_speed_rh = wave_speed_rh[:, np.arange(0, number_of_timesteps, integration_step_size_downsampled).astype(int)]  # downsampling
    wave_speed_rh = np.median(wave_speed_rh[:,wave_mask_div_rh], axis=1)  # median of wave speed
    
 
    # Save Results
    # ------------
    # save wave masks
    np.save(os.path.join(data_path, "40_analysis_wave_speed", f"40_wave_speed_rh_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), wave_speed_rh)