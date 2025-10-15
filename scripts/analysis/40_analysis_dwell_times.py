#!/usr/bin/env python
""" Compute dwell times in parameter exploration of cortical network simulations on Schaefer parcellation

This script is used to compute the dwell times of activity in the simulation results of a cortical network model 
based on Kuramoto oscillators. We computed dwell times based on the inter-hemispheric cross-correlation as
introduced in Roberts et al. (2019) "Metastable brain waves".


Usage
-----
    python 40_analysis_dwell_times.py "simulation_configuration_path" "analysis_configuration_path" process_id

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

from scipy.signal import correlate
from tvb.simulator.lab import *

from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "11. July 2023"
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
initial_transient_samples = int(initial_transient/integration_step_size)  # samples


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[3])

frequency_idx, coupling_strength_idx, conduction_speed_idx = list(itertools.product(*[range(0,4), range(0,39), range(0,10)]))[process_id]

sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_ids = np.arange(sim_id_start, sim_id_end, sim_id_step)

intrinsic_frequency = float(np.array(config["intrinsic_frequency"])[frequency_idx])  # Hz
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz - Kuramoto oscillator intrinsic frequency standard deviation

coupling_strength_start, coupling_strength_end, coupling_strength_steps = np.array(config["coupling_strength"], dtype=float)  # a.u. [start_id, end_id, number_of_steps]
coupling_strength = np.logspace(coupling_strength_start, coupling_strength_end, int(coupling_strength_steps))[coupling_strength_idx]

conduction_speed_start, conduction_speed_end, conduction_speed_steps = np.array(config["conduction_speed"], dtype=float)
conduction_speed = np.linspace(conduction_speed_start, conduction_speed_end, int(conduction_speed_steps))[conduction_speed_idx]


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


# Compute dwell times
# -------------------
dwell_times = []
for simulation_id in simulation_ids:
    ## load phase
    phase = np.load(os.path.join(data_path, f"40_simulations/40_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # remove initial transient timesteps 
    phase_lh = phase[:number_of_regions_per_hemi, np.arange(initial_transient_samples, phase.shape[1], integration_step_size).astype(int)]
    phase_rh = phase[number_of_regions_per_hemi:, np.arange(initial_transient_samples, phase.shape[1], integration_step_size).astype(int)]

    ce_phase_lh = np.exp(1j*phase_lh)
    ce_phase_rh = np.exp(1j*phase_rh)


    # Compute dwell-times
    # -------------------
    # compute sync order parameter
    sync_lh = abs(np.mean(ce_phase_lh, axis=0))
    sync_rh = abs(np.mean(ce_phase_rh, axis=0))

    # compute cross-correlation
    window_size = int(config_analysis["window_size"])  # samples
    step_size = int(config_analysis["xcorr_step_size"])  # samples

    start = 0
    end = start+window_size
    xcorr = []
    for i in range(0, sync_lh.shape[0], step_size):
        if end <= sync_lh.shape[0]:
            xcorr.append(correlate(sync_lh[start:end], sync_rh[start:end], mode='same'))
        start += step_size
        end = start+window_size

    xcorr = np.array(xcorr)

    # compute reciprocal variance of cross correlation
    xcorr_recvar = 1/np.var(xcorr,axis=1)
    xcorr_recvar_mean = np.mean(xcorr_recvar)

    # threshold reciprocal variance
    thresholded = (xcorr_recvar > xcorr_recvar_mean).astype(int)
    crossing_indicator = np.diff(thresholded)  # 1: from stable state to transition; -1: from transition to stable state

    # identify transitions between states
    stable_to_trans = np.where(crossing_indicator==1)[0]
    trans_to_stable = np.where(crossing_indicator==-1)[0]

    # compute dwell-times
    # excludes first and last bit    
    for i in range(len(stable_to_trans)-1):
        dwell_times.append(stable_to_trans[i+1]-trans_to_stable[i])

dwell_times = np.array(dwell_times)*step_size*integration_step_size


# Save Results
# ------------
np.save(os.path.join(data_path, "40_analysis_dwell_times", f"40_dwell_times_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), dwell_times)