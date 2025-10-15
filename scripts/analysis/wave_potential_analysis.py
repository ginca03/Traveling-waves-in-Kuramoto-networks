#!/usr/bin/env python
""" Compute divergence potentials and correlations in 2D Kuramoto network simulations.

This script computes the scalar divergence potential for each timestep
of a 2D Kuramoto simulation and optionally analyzes correlations with network instrength.
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from scipy import sparse, spatial
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from scipy.stats import pearsonr, multivariate_normal
from sksparse.cholmod import cholesky
from modules.wave_detection_methods import compute_phase_gradient
import igl
import argparse

__author__ = "Dominik Koller, extended by Giancarlo Venturato"
__date__ = "05. September 2025"
__status__ = "Prototype"

# ---------------------- FUNCTIONS ----------------------

def compute_divergence_potential(phase_grad, vertices, faces, verbose=True):
    """Compute the scalar divergence potential for each timestep of a 2D phase field."""
    N = vertices.shape[0]
    F = faces.shape[0]
    T = phase_grad.shape[2]

    if verbose:
        print(f"Computing divergence potential for {N} vertices, {F} faces, {T} timesteps")

    # Precompute gradient and Laplace operators
    grad_op = igl.grad(vertices, faces)        # shape: (3*F, N)
    laplace_op = igl.cotmatrix(vertices, faces)  # shape: (N,N)
    
    if verbose:
        print(f"Laplace operator shape: {laplace_op.shape}, nnz: {laplace_op.nnz}")
        print(f"Laplace operator range: [{laplace_op.data.min():.3e}, {laplace_op.data.max():.3e}]")

    # Regularized Cholesky factorization
    try:
        chol_factor = cholesky(-laplace_op + 1e-10 * sparse.eye(N))
        if verbose:
            print("Cholesky factorization successful")
    except Exception as e:
        chol_factor = cholesky(-laplace_op + 1e-6 * sparse.eye(N))
        if verbose:
            print(f"Cholesky failed with small regularization, retrying. Error: {e}")

    # Divergence operator (vertex-based)
    d_area = igl.doublearea(vertices, faces)
    ta = sparse.diags(np.hstack([d_area, d_area, d_area]) * 0.5)
    divergence_op = -grad_op.T.dot(ta)
    
    if verbose:
        print(f"Divergence operator shape: {divergence_op.shape}")
        print(f"Double areas range: [{d_area.min():.3e}, {d_area.max():.3e}]")

    potential_div = np.zeros((N, T))
    non_zero_count = 0

    # Process all timesteps
    for t in range(T):
        if verbose and t % 500 == 0:
            print(f"  Timestep {t}/{T}")
        vector_field = phase_grad[:, :, t].T  # shape: (F,3)
        div_t = divergence_op.dot(vector_field.flatten('F'))

        if np.max(np.abs(div_t)) > 1e-12:
            non_zero_count += 1
            try:
                potential_div[:, t] = chol_factor(div_t)
            except Exception as e:
                if verbose and t < 5:
                    print(f"Solver failed at timestep {t}: {e}")
                potential_div[:, t] = 0.0
        else:
            potential_div[:, t] = 0.0

    if verbose:
        print(f"Non-zero divergence timesteps: {non_zero_count}/{T}")
        print(f"Final potential range: [{np.min(potential_div):.3e}, {np.max(potential_div):.3e}]")
        print(f"Final potential std: {np.std(potential_div):.3e}")

    return potential_div


def create_region_positions(x_extent, y_extent, nx, ny):
    """Compute 2D positions for nx*ny regions in the specified extent."""
    y = np.linspace(0, y_extent, ny)
    x = np.linspace(0, x_extent, nx)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def exp_kernel(distance, sigma):
    """Exponential kernel for distance-dependent connectivity."""
    return 1/(2*sigma) * np.exp(-distance/sigma)


def compute_instrength_field(config, pos, simulation_id):
    """Recreate the instrength field as in the simulation."""
    N = len(pos)
    distance_matrix = spatial.distance.cdist(pos, pos)
    np.fill_diagonal(distance_matrix, np.inf)
    
    scaling = float(config["connection_strength_scaling"])
    A = exp_kernel(distance_matrix, scaling)

    prob_scale = float(config["connection_probability_scale"])
    np.random.seed(simulation_id + 42)
    mask = distance_matrix <= np.random.exponential(scale=prob_scale, size=(N, N))
    A[mask==0] = 0
    A /= A.sum(axis=0)

    # In-strength gradient
    offset = float(config["instrength_gradient_offset"])
    grad_scale = float(config["gradient_scaling"])
    sink_pos = np.array(config["sink_pos"], dtype=float)
    source_pos = np.array(config["source_pos"], dtype=float)
    width = float(config["singularity_width"])

    sink = multivariate_normal(sink_pos, cov=width).pdf(pos)
    source = multivariate_normal(source_pos, cov=width).pdf(pos)
    gradient = sink - source
    gradient = (gradient - gradient.min())*2 / (gradient.max()-gradient.min()) - 1
    instrength_field = grad_scale * gradient + offset

    weights = (A * instrength_field).T
    instrength = np.sum(weights, axis=1)
    return instrength


def compute_face_barycentric_coords(vertices, faces):
    """Compute barycentric coordinates of triangle centroids (uniform 1/3,1/3,1/3)."""
    return np.full((len(faces), 3), 1.0/3.0)

# ---------------------- MAIN SCRIPT ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("sim_cfg")
parser.add_argument("analysis_cfg")
parser.add_argument("simulation_idx", type=int)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

verbose = args.verbose and not args.quiet

# Load YAML configs safely (numbers → float/int automatically)
with open(args.sim_cfg, "r") as f:
    config = yaml.safe_load(f)
with open(args.analysis_cfg, "r") as f:
    config_analysis = yaml.safe_load(f)

# Simulation parameters
data_path = config["data_path"]
experiment_id = str(config["experiment_id"])
integration_step_size = float(config["integration_step_size"])
simulation_duration = float(config["simulation_duration"])
initial_transient = float(config["initial_transient"])
initial_transient_samples = int(initial_transient / integration_step_size)

# REMOVED: No downsampling
# downsampling_factor = float(config_analysis.get("downsampling_factor", 1))
# step_ds = int(downsampling_factor)

sim_id_start, sim_id_end, sim_id_step = map(int, config["simulation_id"])
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[args.simulation_idx]

if verbose:
    print(f"Processing simulation {simulation_id}")

# Load phase data - NO DOWNSAMPLING
phase_path = os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{simulation_id}.npy")
phase = np.load(phase_path).T

# NO DOWNSAMPLING - use all timesteps including transient
# phase = phase[:, :]  # This is implicit, just keep all data

if verbose:
    print(f"Loaded phase data: {phase.shape} (full resolution, includes transient)")

# Mesh setup
nx, ny = int(config["nx"]), int(config["ny"])
x_extent, y_extent = float(config["x_extent"]), float(config["y_extent"])
pos = create_region_positions(x_extent, y_extent, nx, ny)
vertices = np.c_[pos, np.zeros(len(pos))]
faces = Delaunay(pos).simplices
if verbose:
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

# Phase gradient
bc_coords = compute_face_barycentric_coords(vertices, faces)
grad_op = igl.grad(vertices, faces)
phase_grad = compute_phase_gradient(phase, faces, bc_coords, grad_op)
if verbose:
    print(f"Phase gradient shape: {phase_grad.shape}")

# Divergence potential
potential_div = compute_divergence_potential(phase_grad, vertices, faces, verbose=verbose)

# Save potential
output_dir = os.path.join(data_path, f"{experiment_id}_analysis_potentials")
os.makedirs(output_dir, exist_ok=True)
potential_path = os.path.join(output_dir, f"{experiment_id}_potential_div_{simulation_id}.npy")
np.save(potential_path, potential_div)
if verbose:
    print(f"Saved potential: {potential_path}, shape: {potential_div.shape}")

# ---------------------- CORRELATION ANALYSIS ----------------------
if experiment_id not in ["02", "12"]:
    instrength = compute_instrength_field(config, pos, simulation_id)
    if np.std(instrength) < 1e-12:
        if verbose:
            print("Instrength uniform, skipping correlation analysis")
    else:
        if verbose:
            print("Running correlation analysis...")
        # Safe numeric cast — fixes the crash
        n_perm = max(100, int(float(config_analysis.get("number_of_permutations", 1000))) // 10)
        significance = float(config_analysis.get("significance_level", 0.05))
        seed_file = os.path.join(data_path, 'seed_random_numbers.csv')
        seed = pd.read_csv(seed_file, header=None)[0].values[simulation_id]
        rng = np.random.default_rng(seed)

        n_timesteps = potential_div.shape[1]
        corr_over_time = np.zeros(n_timesteps)
        p_corr_over_time = np.zeros(n_timesteps)
        sig_count = 0

        for t in range(n_timesteps):
            if t % 500 == 0 and verbose:
                print(f"  Timestep {t}/{n_timesteps}")
            pot_t = potential_div[:, t]
            if np.std(pot_t) < 1e-12:
                corr_over_time[t] = 0.0
                p_corr_over_time[t] = 1.0
                continue
            corr_over_time[t], _ = pearsonr(instrength, pot_t)
            corr_null = np.zeros(n_perm)
            for perm in range(n_perm):
                corr_null[perm], _ = pearsonr(rng.permutation(instrength), pot_t)
            p_corr_over_time[t] = np.sum(corr_null <= corr_over_time[t]) / n_perm
            if p_corr_over_time[t] <= significance:
                sig_count += 1

        # Save correlation results
        corr_path = os.path.join(output_dir, f"{experiment_id}_corr_div_{simulation_id}.npy")
        p_corr_path = os.path.join(output_dir, f"{experiment_id}_p_corr_{simulation_id}.npy")
        np.save(corr_path, corr_over_time)
        np.save(p_corr_path, p_corr_over_time)

        if verbose:
            print(f"Correlation analysis complete. Significant timesteps: {sig_count}/{n_timesteps}")
else:
    if verbose:
        print("Experiment ID 02 detected: skipping correlation analysis.")

if verbose:
    print("Analysis complete!")
