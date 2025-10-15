#!/usr/bin/env python
""" Compute Kuramoto Order Parameter Variability in parameter exploration of cortical network simulations on Schaefer parcellation

This script is used to compute assess metastability in the simulation results of a cortical network model based on Kuramoto oscillators.
We use the Kuramoto order parameter variability as a measure for metastability.


Usage
-----
    python 40_analysis_metastability.py "simulation_configuration_path" "analysis_configuration_path" process_id

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

sync_mean = np.zeros(len(simulation_ids))
sync_sd = np.zeros(len(simulation_ids))
for simulation_id in simulation_ids:
    ## load phase
    phase = np.load(os.path.join(data_path, f"40_simulations/40_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # remove initial transient timesteps 
    phase = phase[:, np.arange(initial_transient_samples, phase.shape[1], integration_step_size).astype(int)]

    ce_phase = np.exp(1j*phase)
    
    # compute sync order parameter
    sync = abs(np.mean(ce_phase, axis=0))
    sync_mean[simulation_id] = np.mean(sync)
    sync_sd[simulation_id] = np.std(sync)


# Save Results
# ------------
np.save(os.path.join(data_path, "40_analysis_metastability", f"40_sync_mean_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), sync_mean)
np.save(os.path.join(data_path, "40_analysis_metastability", f"40_sync_sd_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), sync_sd)