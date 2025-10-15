#!/usr/bin/env python
""" CONTROL - Cortical network model based on Schaefer parcellation

This script is used to simulate a cortical network model consisting of Kuramoto oscillators that are connected
via a surrogate distance-dependent structural connectivity. This connectome was constructed by fitting an exponential
model to the connection strength - euclidean distance relationship of the empirical connectome estimated by diffusion
weighted imaging followed by generating structural connectivity from this model. This model destroys the spatial
embedding of the network while preserving distance-strength relationships. Notably, the intrahemispheric connectivity
will be preserved from the empirical estimates.

Usage
-----
    python 32_simulation.py "configuration/your_configuration_file.yaml" simulation_idx

Arguments
---------
configuration_path : String
    Path to configuration file.
    
simulation_idx : int
    Index for the simulation ID.
"""


import yaml
import pickle
import sys
import os
import numpy as np
import potpourri3d as pp3d
from tvb.simulator.lab import *
from sklearn.metrics.pairwise import euclidean_distances


__author__ = "Dominik Koller"
__date__ = "30. May 2022"
__status__ = "Prototype"


def func_explaw(x, alpha, l):
    return alpha * np.exp(-x*l)


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

print(f"Simulation {simulation_id}")


# Model Parameters
# ----------------
# model parameters
intrinsic_frequency_mean = float(config["intrinsic_frequency_mean"])  # Hz - Kuramoto oscillator intrinsic frequency mean
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz - Kuramoto oscillator intrinsic frequency standard deviation

# network parameters
coupling_strength = np.array(float(config["coupling_strength"]))  # global scaling of all connections
conduction_speed = float(config["conduction_speed"])  # mm/ms

# load connectome 
weights = np.load(os.path.join(data_path, weights_path))
lengths = np.load(os.path.join(data_path, lengths_path))
pos = np.load(os.path.join(data_path, positions_path))
labels = np.load(os.path.join(data_path, labels_path))

# load schaefer surface mesh
with open(os.path.join(data_path, "connectomes/Schaefer2018_HCP_S900/schaefer_surface_mesh.pkl"), 'rb') as f:
    surface_mesh = pickle.load(f)
    
v_lh = surface_mesh['vertices_lh']
f_lh = surface_mesh['faces_lh']
v_rh = surface_mesh['vertices_rh']
f_rh = surface_mesh['faces_rh']
number_of_regions_per_hemi = v_lh.shape[0]
number_of_regions = number_of_regions_per_hemi * 2

# exponential model parameters
normalization_constant = float(config["normalization_constant"])
decay_rate = float(config["decay_rate"])


# Generate connectivity from exponential distance rule
# ----------------------------------------------------
# get empirical weights
weights_lh = np.triu(weights[:number_of_regions_per_hemi,:number_of_regions_per_hemi])  # take upper triangular because of symmetry
weights_rh = np.triu(weights[number_of_regions_per_hemi:number_of_regions,number_of_regions_per_hemi:number_of_regions])
weights_ih = weights[:number_of_regions_per_hemi, number_of_regions_per_hemi:number_of_regions]  # unidirectional intrahemispheric connection strengths

# compute euclidean distance matrix
euclidean_distance_lh = np.triu(euclidean_distances(v_lh)) * 1e3  # convert m to mm
euclidean_distance_rh = np.triu(euclidean_distances(v_rh)) * 1e3  # convert m to mm

# compute surrogate connectivity
weights_surrogate_lh = func_explaw(euclidean_distance_lh, normalization_constant, decay_rate)
weights_surrogate_lh = np.triu(weights_surrogate_lh)  # only use upper triangular
np.fill_diagonal(weights_surrogate_lh, 0)  # remove self-connections
weights_surrogate_lh /= weights_surrogate_lh.sum()/weights_lh.sum()  # equalize sum of weights

weights_surrogate_rh = func_explaw(euclidean_distance_rh, normalization_constant, decay_rate)
weights_surrogate_rh = np.triu(weights_surrogate_rh)  # only use upper triangular
np.fill_diagonal(weights_surrogate_rh, 0)  # remove self-connections
weights_surrogate_rh /= weights_surrogate_rh.sum()/weights_rh.sum()  # equalize sum of weights

# combine weights matrices
weights_surrogate = np.zeros_like(weights)
weights_surrogate[:number_of_regions_per_hemi,:number_of_regions_per_hemi] = weights_surrogate_lh
weights_surrogate[number_of_regions_per_hemi:,number_of_regions_per_hemi:] = weights_surrogate_rh
weights_surrogate[:number_of_regions_per_hemi, number_of_regions_per_hemi:] = weights_ih
weights_surrogate = weights_surrogate + weights_surrogate.T - np.diag(np.diag(weights_surrogate))  # symmetrize

# create surrogate tracts
lengths_surrogate = np.zeros_like(lengths)
lengths_surrogate[:number_of_regions_per_hemi,:number_of_regions_per_hemi] = euclidean_distance_lh
lengths_surrogate[number_of_regions_per_hemi:,number_of_regions_per_hemi:] = euclidean_distance_rh
lengths_surrogate[:number_of_regions_per_hemi, number_of_regions_per_hemi:] = lengths[:number_of_regions_per_hemi, number_of_regions_per_hemi:]
lengths_surrogate = lengths_surrogate + lengths_surrogate.T - np.diag(np.diag(lengths_surrogate))  # symmetrize


# Create Network
# --------------
# build connectivity
sc = connectivity.Connectivity()
sc.weights = weights_surrogate
sc.tract_lengths = lengths_surrogate
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
np.save(os.path.join(data_path, f"32_simulations/32_simulation_{simulation_id}.npy"), phase)