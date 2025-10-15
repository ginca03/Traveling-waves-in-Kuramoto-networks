#!/usr/bin/env python
""" Parameter exploration of cortical network model based on Schaefer parcellation

This script is used to explore the cortical network model consisting of Kuramoto oscillators that are connected
via the structural connectome inferred from HCP subjects based on the Schaefer et al. (2018) parcellation. The 
parameters explored are the intrinsic frequency, global coupling strength, and conduction speed.

Usage
-----
    python 40_simulation.py "configuration/40_configuration.yaml" process_id

Arguments
---------
configuration_path : String
    Path to configuration file.
    
process_id : int
    ID of the process.
"""


import numpy as np
import yaml
import itertools
from tvb.simulator.lab import *

__author__ = "Dominik Koller"
__date__ = "10. June 2022"
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
save_path = config["save_path"]
experiment_id = config["experiment_id"]

weights_path = config["weights_path"]
lengths_path = config["lengths_path"]
labels_path = config["labels_path"]
positions_path = config["positions_path"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms


# Parameters to Explore
# ---------------------
process_id = int(sys.argv[2])

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


# Network Parameters
# ------------------
# load connectome 
weights = np.load(os.path.join(data_path, weights_path))
lengths = np.load(os.path.join(data_path, lengths_path))
pos = np.load(os.path.join(data_path, positions_path))
labels = np.load(os.path.join(data_path, labels_path))

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
global_coupling = coupling.Kuramoto(a=np.array(coupling_strength))

# compute intrinsic frequency distribution
intrinsic_frequency_distribution = intrinsic_frequency_standard_deviation * np.random.randn(N) + intrinsic_frequency

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
np.save(os.path.join(data_path, save_path, f"{experiment_id}_simulation_{simulation_id}_{frequency_idx}_{coupling_strength_idx}_{conduction_speed_idx}.npy"), phase)