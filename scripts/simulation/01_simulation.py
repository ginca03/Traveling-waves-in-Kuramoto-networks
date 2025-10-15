 #!/usr/bin/env python
""" Traveling waves follow a connectivity instrength gradient

This script is used to simulate a 2D network of Kuramoto oscillators. We constructed random networks with 
connection strength and probability decreasing exponentially with distance from each oscillator. First, the 
connection strengths were normalized to create a uniform instrength distribution. Then, we created an 
instrength gradient by weighting the incoming connection strengths of each oscillator with two gaussians 
placed on the upper right and lower left of the network, respectively.

Usage
-----
    python 01_simulation.py "configuration/your_configuration_file.yaml" simulation_idx

Arguments
---------
configuration_path : String
    Path to configuration file.
    
simulation_idx : int
    Index for the simulation ID.
"""


import numpy as np
import yaml
from scipy import spatial
from scipy.stats import multivariate_normal
from tvb.simulator.lab import *

__author__ = "Dominik Koller"
__date__ = "26. July 2021"
__status__ = "Prototype"


def exp_kernel(distance, sigma):
    """ Exponential kernel function for distance-dependent connection strength"""
    f = 1/(2*sigma) * np.exp(-distance/sigma)
    return f


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

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms


# Parameters to Explore
# ---------------------
simulation_idx = int(sys.argv[2])  # get simulation idx
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]

print(f"Simulation {simulation_id}")


# Model Parameters
# ----------------
# model parameters
intrinsic_frequency_mean = float(config["intrinsic_frequency_mean"])  # Hz - Kuramoto oscillator intrinsic frequency mean
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz - Kuramoto oscillator intrinsic frequency standard deviation

# network parameters
nx = int(config["nx"])  # number of regions along x-dimension
ny = int(config["ny"])  # number of regions along y-dimension

N = nx*ny  # number of regions

x_extent = float(config["x_extent"])  # mm, extent of x-dimension
y_extent = float(config["y_extent"])  # mm, extent of y-dimension

# create node positions
y = np.linspace(0, y_extent, ny)
x = np.linspace(0, x_extent, nx)
pos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

coupling_strength = np.array(float(config["coupling_strength"]))  # global scaling of all connections
conduction_speed = float(config["conduction_speed"])  # mm/ms
connection_strength_scaling = float(config["connection_strength_scaling"])  # scaling factor for distance-dependent connection strength
connection_probability_scale = float(config["connection_probability_scale"])  # scale parameter for distance-dependent connection probability


# Create Network
# --------------
sc = connectivity.Connectivity()
sc.centres = pos

# Create distance-dependent connectivity
distance_matrix = spatial.distance.cdist(pos, pos)
np.fill_diagonal(distance_matrix, np.inf)  # avoids self-connections

# distance-dependent connection strength
A = exp_kernel(distance_matrix, connection_strength_scaling)

# calculate distance-dependent connection probability mask
connection_mask = distance_matrix <= abs(np.random.exponential(scale=connection_probability_scale, size=(N,N)))

#connection_mask = np.zeros((N,N), dtype=bool)
#node_degree = np.ones(N, dtype=int)*100
#for i in range(N):
#    connection_idxs = np.argsort(distance_matrix[i])[:node_degree[i]]
#    connection_mask[connection_idxs,i] = True

A[connection_mask==0] = 0
A /= A.sum(axis=0)  # normalize connections

# create spatial in-strength gradient
instrength_gradient_offset = float(config["instrength_gradient_offset"])
gradient_scaling = float(config['gradient_scaling'])
sink_pos = np.array(config["sink_pos"])
source_pos = np.array(config["source_pos"])
singularity_width = float(config["singularity_width"])

sink = multivariate_normal(sink_pos, cov=singularity_width).pdf(pos)  # generate sink gaussian
source = multivariate_normal(source_pos, cov=singularity_width).pdf(pos)  # generate source gaussian

gradient = (sink - source)  # combine gaussians to create gradient
gradient = (gradient - gradient.min()) * 2 / (gradient.max() - gradient.min()) - 1  # min-max normalized gradient
instrength_gradient_field = gradient_scaling * gradient + instrength_gradient_offset

# create connection strength distribution based on in-strength gradient field
sc.weights = (A * instrength_gradient_field).T
assert np.all(sc.weights>=0)

# configure structural connectivity of network
sc.tract_lengths = distance_matrix.T
sc.speed = np.array(conduction_speed)  # set the conduction speed of connections
sc.create_region_labels(mode='alphabetic')
sc.configure()


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

# Ensure the output directory exists
os.makedirs(os.path.join(data_path, "01_simulations"), exist_ok=True)

# save data
os.makedirs(os.path.join(data_path, "01_simulations"), exist_ok=True)
np.save(os.path.join(data_path, f"01_simulations/01_simulation_{simulation_id}.npy"), phase)
