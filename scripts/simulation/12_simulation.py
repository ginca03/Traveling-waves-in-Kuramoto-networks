#!/usr/bin/env python
""" Uniform in-strength Kuramoto network with left edge forcing

This script simulates a 2D network of Kuramoto oscillators with uniform in-strength
and an external periodic forcing applied to the left edge of the network.

Usage
-----
    python 12_simulation_uniform_forcing.py "configuration/12_config.yaml" simulation_idx

Arguments
---------
configuration_path : String
    Path to configuration file.
    
simulation_idx : int
    Index for the simulation ID.
"""

import numpy as np
import yaml
import sys
import os
from scipy import spatial
from scipy.stats import expon
from tvb.simulator.lab import *
from tvb.basic.neotraits.api import NArray

__author__ = "Assistant"
__date__ = "October 2025"
__status__ = "Prototype"


def exp_kernel(distance, sigma):
    """Exponential kernel function for distance-dependent connection strength"""
    return np.exp(-distance/sigma) / (2 * sigma)


class KuramotoWithForcing(models.Kuramoto):
    """
    Kuramoto model with spatially localized external forcing.
    
    The forcing is applied as an additional term to specific nodes:
    dθ/dt = ω + K * Σ sin(θ_j - θ_i) + A * sin(2πf*t - θ_i) * mask
    """
    
    forcing_amplitude = NArray(
        label="Forcing Amplitude",
        default=np.array([0.0]),
        doc="Amplitude of the external periodic forcing"
    )
    
    forcing_frequency = NArray(
        label="Forcing Frequency (Hz)",
        default=np.array([10.0]),
        doc="Frequency of external forcing in Hz"
    )
    
    forcing_mask = NArray(
        label="Forcing Mask",
        default=np.array([1.0]),
        doc="Spatial mask defining which nodes receive forcing (0 or 1)"
    )
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        Compute the derivatives of the Kuramoto model with forcing.
        """
        theta = state_variables[0, :]
        c_0 = coupling[0, :]
        
        # Base Kuramoto dynamics
        derivative = self.omega + c_0
        
        return np.array([derivative])


class HeunWithForcing(integrators.HeunDeterministic):
    """Heun integrator that applies forcing at each time step"""
    
    def __init__(self, forcing_amplitude=0.0, forcing_frequency=10.0, forcing_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.current_time = 0.0
        self.forcing_omega = 2 * np.pi * forcing_frequency * 1e-3  # rad/ms
        self.forcing_amplitude = forcing_amplitude
        self.forcing_mask = forcing_mask if forcing_mask is not None else np.array([0.0])
    
    def scheme(self, X, dfun, coupling, local_coupling, stimulus):
        """Heun integration scheme with forcing"""
        dt = self.dt
        
        # Extract current phases - ensure it's 1D
        current_phases = np.squeeze(X[0, :])  # Shape (N,)
        
        # Heun method - FIRST STEP
        dX = dfun(X, coupling, local_coupling)
        
        # Apply forcing term - use phase difference (forcing_phase - current_phase)
        forcing_phase = self.forcing_omega * self.current_time
        
        # Forcing term: A * sin(forcing_phase - current_phase) * mask
        # All operations on 1D arrays to avoid broadcasting issues
        forcing_scalar = np.sin(forcing_phase - current_phases)
        forcing_vector = self.forcing_amplitude * forcing_scalar * self.forcing_mask
        
        # Reshape to match dX[0] shape and add
        dX[0] += forcing_vector.reshape(-1, 1)
        
        X_pred = X + dt * dX
        
        # Heun method - SECOND STEP
        dX_pred = dfun(X_pred, coupling, local_coupling)
        
        # Apply forcing term for prediction step
        forcing_phase_pred = self.forcing_omega * (self.current_time + dt)
        predicted_phases = np.squeeze(X_pred[0, :])  # Shape (N,)
        
        forcing_scalar_pred = np.sin(forcing_phase_pred - predicted_phases)
        forcing_vector_pred = self.forcing_amplitude * forcing_scalar_pred * self.forcing_mask
        
        dX_pred[0] += forcing_vector_pred.reshape(-1, 1)
        
        X_next = X + dt/2 * (dX + dX_pred)
        
        # Update time
        self.current_time += dt
        
        return X_next


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
experiment_id = config["experiment_id"]

integration_step_size = float(config["integration_step_size"])  # ms
simulation_duration = float(config["simulation_duration"])  # ms


# Parameters to Explore
# ---------------------
simulation_idx = int(sys.argv[2])  # get simulation idx
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[simulation_idx]

print(f"Experiment {experiment_id}, Simulation {simulation_id}")


# Model Parameters
# ----------------
# model parameters
intrinsic_frequency_mean = float(config["intrinsic_frequency_mean"])  # Hz
intrinsic_frequency_standard_deviation = float(config["intrinsic_frequency_standard_deviation"])  # Hz

# network parameters
nx = int(config["nx"])  # number of regions along x-dimension
ny = int(config["ny"])  # number of regions along y-dimension
N = nx*ny  # number of regions

x_extent = float(config["x_extent"])  # mm
y_extent = float(config["y_extent"])  # mm

# create node positions
y = np.linspace(0, y_extent, ny)
x = np.linspace(0, x_extent, nx)
pos = np.array(np.meshgrid(x, y)).T.reshape(-1,2)

coupling_strength = np.array(float(config["coupling_strength"]))  # global scaling
conduction_speed = float(config["conduction_speed"])  # mm/ms
connection_strength_scaling = float(config["connection_strength_scaling"])  # mm
connection_probability_scale = float(config["connection_probability_scale"])

# forcing parameters
forcing_amplitude = float(config["forcing_amplitude"])
forcing_frequency = float(config["forcing_frequency"])  # Hz


# Create Network - CORRECTED CONNECTIVITY
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

A[connection_mask==0] = 0

# FIX: Don't normalize by in-strength if you want forcing to work properly
# Instead, scale by coupling strength and ensure connectivity is symmetric
sc.weights = A.T * coupling_strength  # Apply coupling strength directly

# Configure uniform in-strength field separately
instrength_gradient_offset = float(config["instrength_gradient_offset"])
instrength_gradient_field = np.ones(N) * instrength_gradient_offset

# configure structural connectivity of network
sc.tract_lengths = distance_matrix.T
sc.speed = np.array(conduction_speed)
sc.create_region_labels(mode='alphabetic')
sc.configure()


# Create Forcing Mask - LEFT EDGE
# --------------------------------
forcing_mask = np.zeros(N)

# Find all nodes on the left edge (minimum x-coordinate)
x_coords = pos[:, 0]
min_x = np.min(x_coords)
tolerance = 1e-10
left_edge_nodes = np.abs(x_coords - min_x) < tolerance
forcing_mask[left_edge_nodes] = 1.0

num_forced_nodes = np.sum(left_edge_nodes)
print(f"Forcing applied to {num_forced_nodes} nodes along the left edge")
print(f"Forcing: {forcing_frequency} Hz, amplitude {forcing_amplitude}")


# Kuramoto Model with Forcing
# ---------------------------
# specify local dynamics model
model = KuramotoWithForcing()
model.forcing_amplitude = np.array([forcing_amplitude])
model.forcing_frequency = np.array([forcing_frequency])
model.forcing_mask = forcing_mask

# Use Kuramoto coupling
global_coupling = coupling.Kuramoto(a=np.array([1.0]))

# compute intrinsic frequency distribution
intrinsic_frequency_distribution = intrinsic_frequency_standard_deviation * np.random.randn(N) + intrinsic_frequency_mean

# convert intrinsic frequency to intrinsic angular frequency
# this has to be radians/ms for TVB
omega = 2*np.pi * intrinsic_frequency_distribution * 1e-3  # rad/ms
model.omega = omega


# Simulation
# ----------
# initialise simulator
sim = simulator.Simulator(
    model=model,
    connectivity=sc,
    coupling=global_coupling,
    conduction_speed=conduction_speed,
    integrator=HeunWithForcing(
        dt=integration_step_size,
        forcing_amplitude=forcing_amplitude,
        forcing_frequency=forcing_frequency,
        forcing_mask=forcing_mask
    ),
    simulation_length=simulation_duration,
    monitors=(monitors.Raw(), )
)
sim.configure()

# run simulation
print("Running simulation...")
(time, raw_data), = sim.run()
phase = np.squeeze(raw_data)

# Ensure the output directory exists
output_dir = os.path.join(data_path, f"{experiment_id}_simulations")
os.makedirs(output_dir, exist_ok=True)

# save data
np.save(os.path.join(output_dir, f"{experiment_id}_simulation_{simulation_id}.npy"), phase)
np.save(os.path.join(output_dir, f"{experiment_id}_forcing_mask_{simulation_id}.npy"), forcing_mask)
np.save(os.path.join(output_dir, f"{experiment_id}_positions_{simulation_id}.npy"), pos)
np.save(os.path.join(output_dir, f"{experiment_id}_instrength_field_{simulation_id}.npy"), instrength_gradient_field)

print(f"Simulation complete. Data saved to {output_dir}")
print(f"Phase shape: {phase.shape}")
print(f"Uniform in-strength field: constant value = {instrength_gradient_offset}")
