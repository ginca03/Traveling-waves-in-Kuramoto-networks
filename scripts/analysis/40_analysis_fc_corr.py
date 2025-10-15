#!/usr/bin/env python
""" Compute correlation between simulated and empirical functional connectivity

This script is used to compute the correlation between simulated and empirical functional connectivity.


Usage
-----
    python 40_analysis_fc_corr.py "simulation_configuration_path" process_id

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
from scipy.stats import pearsonr
from tvb.simulator.lab import *

from modules.wave_detection_methods import *


__author__ = "Dominik Koller"
__date__ = "22. December 2022"
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
experiment_id = config["experiment_id"]
data_path = config["data_path"]
save_path = config["save_path"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms
initial_transient = float(config["initial_transient"])  # ms
initial_transient_samples = int(initial_transient/integration_step_size)  # samples


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[3])

frequency_idx, coupling_strength_idx, conduction_speed_idx = list(itertools.product(*[range(1,4), range(0,39), range(0,10)]))[process_id]

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
    FC_plv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_alpha_plv_avg.npy"))
    FC_iplv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_alpha_iplv_avg.npy"))
    FC_pli_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_alpha_pli_avg.npy"))
elif intrinsic_frequency == 20:
    FC_plv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_beta_plv_avg.npy"))
    FC_iplv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_beta_iplv_avg.npy"))
    FC_pli_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_beta_pli_avg.npy"))
elif intrinsic_frequency == 40:
    FC_plv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_gamma_plv_avg.npy"))
    FC_iplv_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_gamma_iplv_avg.npy"))
    FC_pli_emp = np.load(os.path.join(data_path, f"hcp_meg/FC_gamma_pli_avg.npy"))
    
number_of_regions = FC_plv_emp.shape[0]


# Compute simulated FC
# --------------------
FC_plv_sim = np.zeros((number_of_regions,number_of_regions))
FC_iplv_sim = np.zeros((number_of_regions,number_of_regions))
FC_pli_sim = np.zeros((number_of_regions,number_of_regions))

for sid in simulation_id:
    print(f"Processing simulation {sid}")
    ## load phase
    phase = np.load(os.path.join(data_path, save_path, f"{experiment_id}_simulation_{sid}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy")).T
    
    # remove initial transient timesteps 
    phase = phase[:, np.arange(initial_transient_samples, phase.shape[1], integration_step_size).astype(int)]
    
    # compute phase locking value (Ghuman et al., 2011; Palva et al., 2018; Lachaux et al., 1999; Bastos and Schoffelen, 2016)
    plv_tmp = np.exp(1j*phase) @ np.exp(1j*phase).conj().T / np.shape(phase)[1]
    plv = abs(plv_tmp)
    iplv = abs(np.imag(plv_tmp))
    
    # compute phase lag index
    FC_pli_tmp = np.zeros((number_of_regions,number_of_regions))
    for i in range(number_of_regions):
        FC_pli_tmp[i,i:] = abs(np.mean(np.sign(np.sin(phase[i:]-phase[i])), axis=1))
    
    FC_pli_sim += (FC_pli_tmp + FC_pli_tmp.T)
    FC_plv_sim += plv
    FC_iplv_sim += iplv

FC_plv_avg = FC_plv_sim / len(simulation_id)
FC_iplv_avg = FC_iplv_sim / len(simulation_id)
FC_pli_avg = FC_pli_sim / len(simulation_id)


# Empirical - Simulated FC Correlation
# ------------------------------------
# correlate off-diagonal elements of FC matrices
out = np.ones(FC_plv_avg.shape, dtype=bool)
np.fill_diagonal(out,0)

FC_plv_corr = pearsonr(FC_plv_avg[out].flatten(), FC_plv_emp[out].flatten())[0]
FC_iplv_corr = pearsonr(FC_iplv_avg[out].flatten(), FC_iplv_emp[out].flatten())[0]
FC_pli_corr = pearsonr(FC_pli_avg[out].flatten(), FC_pli_emp[out].flatten())[0]

logger.info("FC correlations computed.")


# Save Results
# ------------
logger.info("Save data.")
np.save(os.path.join(data_path, f"{experiment_id}_analysis_fc_corr", f"{experiment_id}_fc_plv_corr_all_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), FC_plv_corr)
np.save(os.path.join(data_path, f"{experiment_id}_analysis_fc_corr", f"{experiment_id}_fc_iplv_corr_all_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), FC_iplv_corr)
np.save(os.path.join(data_path, f"{experiment_id}_analysis_fc_corr", f"{experiment_id}_fc_pli_corr_all_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), FC_pli_corr)