#!/usr/bin/env python
""" Compute concordance correlation coefficient (CCC) between simulated and empirical effective frequency

This script is used to compute the ccc between simulated and empirical effective frequency.


Usage
-----
    python 48_analysis_ef_ccc.py "simulation_configuration_path" process_id

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
import pandas as pd
import numpy as np

from logzero import logger
from tvb.simulator.lab import *

from modules.wave_detection_methods import *
from modules.helpers import *


__author__ = "Dominik Koller"
__date__ = "29. November 2023"
__status__ = "Prototype"


def concordance_correlation(x, y):
    ### Compute concordance correlation coefficient ###
    return (2 * np.corrcoef(x, y)[0][1] * np.std(x) * np.std(y)) / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)


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
experiment_id = config["experiment_id"]
data_path = config["data_path"]
save_path = config["save_path"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms
initial_transient = float(config["initial_transient"])  # ms
initial_transient_samples = int(initial_transient/integration_step_size)  # samples
number_of_timesteps = int((simulation_duration-initial_transient)/integration_step_size)

downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
integration_step_size_downsampled = integration_step_size * downsampling_factor
number_of_timesteps_downsampled = int((simulation_duration-initial_transient)/integration_step_size_downsampled)


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[3])

frequency_idx, coupling_strength_idx, conduction_speed_idx = list(itertools.product(*[range(1,3), range(0,39), range(0,10)]))[process_id]

sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)

intrinsic_frequency = float(np.array(config["intrinsic_frequency"])[frequency_idx])  # Hz

coupling_strength_start, coupling_strength_end, coupling_strength_steps = np.array(config["coupling_strength"], dtype=float)  # a.u. [start_id, end_id, number_of_steps]
coupling_strength = np.logspace(coupling_strength_start, coupling_strength_end, int(coupling_strength_steps))[coupling_strength_idx]

conduction_speed_start, conduction_speed_end, conduction_speed_steps = np.array(config["conduction_speed"], dtype=float)
conduction_speed = np.linspace(conduction_speed_start, conduction_speed_end, int(conduction_speed_steps))[conduction_speed_idx]

logger.info(f"coupling_strength={coupling_strength}, frequency={intrinsic_frequency}, v={conduction_speed}")


# Load Data
# ---------
# load empirical FC
if intrinsic_frequency == 10:
    ef_emp = np.load(os.path.join(data_path, f"hcp_meg/avg_effective_frequency_alpha_IF.npy"))
elif intrinsic_frequency == 20:
    ef_emp = np.load(os.path.join(data_path, f"hcp_meg/avg_effective_frequency_beta_IF.npy"))

number_of_regions = ef_emp.shape[0]


# Compute simulated ef
# --------------------
ef_sim = []
for sid in simulation_id:
    print(f"Processing simulation {sid}")
    
    ## load phase
    phase = np.load(os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{sid}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T

    # downsampling of timesteps
    phase = phase[:, np.arange(initial_transient, phase.shape[1], integration_step_size).astype(int)]

    # complex exponential phase
    ce_phase = np.exp(1j*phase)
    

    # Compute instantaneous frequency
    frequency_tmp = compute_instantaneous_frequency(ce_phase, integration_step_size*1e-3)  # compute inst freq
    frequency_tmp = frequency_tmp[:, np.arange(0, number_of_timesteps-1, integration_step_size_downsampled).astype(int)]  # downsampling
    frequency_median = np.median(frequency_tmp, axis=1)  # compute median inst frequency
    
    ef_sim.append(frequency_median)

ef_avg = np.nanmean(ef_sim, axis=0)


# Empirical - simulated effective frequency ccc
# ---------------------------------------------
logger.info(f"Compute effective frequency ccc")
ccc = concordance_correlation(ef_emp, ef_avg)


# Save Results
# ------------
logger.info("Save data.")
np.save(os.path.join(data_path, f"{experiment_id}_analysis_ef_ccc", f"{experiment_id}_ef_ccc_all_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), ccc)