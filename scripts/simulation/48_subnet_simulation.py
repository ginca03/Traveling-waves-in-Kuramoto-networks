#!/usr/bin/env python
""" Cortical network model based on Schaefer parcellation

This script is used to simulate a cortical network model consisting of Kuramoto oscillators that are connected
via subnet-band specific SC inferred from HCP subjects total SC using nonnegative matrix factorization.
The parameters of this model were chosen according to the best fit to resting-state MEG effective frequency.

Usage
-----
    python 48_subnet_simulation.py "configuration/your_configuration_file.yaml" simulation_idx

Arguments
---------
configuration_path : String
    Path to configuration file.
    
simulation_idx : int
    Index for the simulation ID.
"""


import numpy as np
import yaml
from tvb.simulator.lab import *

__author__ = "Dominik Koller"
__date__ = "14. December 2023"
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


# Simulation Parameters
# ---------------------
data_path = config["data_path"]
weights_path = config["weights_path"]
lengths_path = config["lengths_path"]
labels_path = config["labels_path"]
positions_path = config["positions_path"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms


# Parameters to Explore
# ---------------------
simulation_idx = int(sys.argv[2])  # get idx
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]


# Model Parameters
# ----------------
# model parameters
intrinsic_frequency_mean = float(config["intrinsic_frequency_mean"])  # Hz - Kuramoto oscillator intrinsic frequency mean
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz - Kuramoto oscillator intrinsic frequency standard deviation

# network parameters
coupling_strength = np.array(float(config["coupling_strength"]))  # global scaling of all connections
conduction_speed = float(config["conduction_speed"])  # mm/ms

# load connectome 
lengths = np.load(os.path.join(data_path, lengths_path))
pos = np.load(os.path.join(data_path, positions_path))
labels = np.load(os.path.join(data_path, labels_path))

band = config["band"]
weights = np.load(os.path.join(data_path, f"hcp_meg/nmf_weights_{band}.npy"))

print(f"Simulation {simulation_id}, {band}-band")


# Create Network
# --------------
# build connectivity
sc = connectivity.Connectivity()
sc.weights = weights
sc.tract_lengths = lengths
sc.centres = pos
sc.region_labels = labels
sc.undirected = True

# configure structural connectivity of network
sc.speed = np.array(conduction_speed)  # set the conduction speed of connections
sc.configure()
N = sc.number_of_regions

# Kuramoto Model
# --------------
# specify local dynamics model
model = models.Kuramoto()

# specify global coupling
global_coupling = coupling.Kuramoto(a=coupling_strength)

# compute intrinsic frequency distribution
intrinsic_frequency_distribution = intrinsic_frequency_standard_deviation * np.random.randn(N) + intrinsic_frequency_mean

# convert intrinsic frequency to intrinsic angular frequency
# this has to be radians/ms for TVB
omega = 2*np.pi * intrinsic_frequency_distribution * 1e-3  # rad/ms
model.omega = omega


# Simulation
# ----------
# initialise simulator
sim = simulator.Simulator(model = model, 
                          connectivity = sc,
                          coupling = global_coupling, 
                          conduction_speed = conduction_speed,
                          integrator = integrators.RungeKutta4thOrderDeterministic(dt=integration_step_size), 
                          simulation_length = simulation_duration,
                          monitors = (monitors.Raw(), )
                         )
sim.configure()

# run simulation
(time, raw_data), = sim.run()
phase = np.squeeze(raw_data)

# save data
np.save(os.path.join(data_path, f"48_{band}_simulations/48_{band}_simulation_{simulation_id}.npy"), phase)