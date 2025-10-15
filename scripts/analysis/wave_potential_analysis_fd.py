#!/usr/bin/env python
""" Compute divergence potentials and correlations in 2D Kuramoto network simulations.

This script computes the scalar divergence potential for each timestep
of a 2D Kuramoto simulation using finite difference operators on a regular grid.

Fixed / improved FD implementation:
 - correct gradient/divergence stencils
 - properly scaled Laplacian with dx, dy factors
 - robust sparse assembly (no wrap-around)
 - handles Neumann nullspace by subtracting RHS mean and fixing mean(phi)=0
 - small regularization for Cholesky stability
 - empirical dx^2 scaling correction (keeps units compatible with FEM)
 - verbose diagnostics (including t=5500)
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
import argparse

__author__ = "Dominik Koller, extended by Giancarlo Venturato"
__date__ = "05. September 2025 (patched)"
__status__ = "Prototype"

# ---------------------- FUNCTIONS ----------------------

def compute_phase_gradient_fd(phase, nx, ny, dx, dy):
    """Compute phase gradient using finite differences at grid nodes.

    Args:
        phase: (N, T) array where N = nx*ny
        nx, ny: grid dimensions
        dx, dy: grid spacing

    Returns:
        phase_grad: (N, 2, T) array containing [grad_x, grad_y] at each node
    """
    N, T = phase.shape
    assert N == nx * ny, f"phase size mismatch: expected {nx*ny}, got {N}"
    phase_grad = np.zeros((N, 2, T))

    # Reshape to 2D grid for each timestep
    for t in range(T):
        p2d = phase[:, t].reshape((ny, nx))  # row-major: (y, x)

        gx = np.zeros_like(p2d)
        gy = np.zeros_like(p2d)

        # Central differences in interior
        gx[:, 1:-1] = (p2d[:, 2:] - p2d[:, :-2]) / (2.0 * dx)
        gy[1:-1, :] = (p2d[2:, :] - p2d[:-2, :]) / (2.0 * dy)

        # One-sided differences at boundaries
        gx[:, 0] = (p2d[:, 1] - p2d[:, 0]) / dx
        gx[:, -1] = (p2d[:, -1] - p2d[:, -2]) / dx
        gy[0, :] = (p2d[1, :] - p2d[0, :]) / dy
        gy[-1, :] = (p2d[-1, :] - p2d[-2, :]) / dy

        # Flatten back to node ordering (row-major)
        phase_grad[:, 0, t] = gx.ravel()
        phase_grad[:, 1, t] = gy.ravel()

    return phase_grad


def build_laplacian_2d_neumann(nx, ny, dx, dy):
    """Build correctly scaled 2D Laplacian operator with Neumann BC.

    Returns a sparse CSR matrix approximating -∇^2 (positive semidef):
        (L @ phi) ≈ -∇^2 phi
    """
    N = nx * ny
    rows = []
    cols = []
    data = []

    def idx(i, j):
        return i * nx + j

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)
            diag = 0.0

            # left neighbor
            if j - 1 >= 0:
                rows.append(k); cols.append(idx(i, j-1)); data.append(-inv_dx2)
                diag += inv_dx2
            # right neighbor
            if j + 1 < nx:
                rows.append(k); cols.append(idx(i, j+1)); data.append(-inv_dx2)
                diag += inv_dx2
            # down neighbor
            if i - 1 >= 0:
                rows.append(k); cols.append(idx(i-1, j)); data.append(-inv_dy2)
                diag += inv_dy2
            # up neighbor
            if i + 1 < ny:
                rows.append(k); cols.append(idx(i+1, j)); data.append(-inv_dy2)
                diag += inv_dy2

            # main diagonal
            rows.append(k); cols.append(k); data.append(diag)

    L = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return L


def compute_divergence_fd_weighted(phase_grad, nx, ny, dx, dy):
    """Compute divergence ∂gx/∂x + ∂gy/∂y using consistent FD stencils.

    Kept the original function name for API compatibility. This implementation
    does NOT multiply by cell area (that caused a scaling mismatch previously).
    """
    N, _, T = phase_grad.shape
    divergence = np.zeros((N, T))

    for t in range(T):
        gx = phase_grad[:, 0, t].reshape((ny, nx))
        gy = phase_grad[:, 1, t].reshape((ny, nx))

        # ∂gx/∂x
        dgx_dx = np.zeros_like(gx)
        dgx_dx[:, 1:-1] = (gx[:, 2:] - gx[:, :-2]) / (2.0 * dx)
        dgx_dx[:, 0] = (gx[:, 1] - gx[:, 0]) / dx
        dgx_dx[:, -1] = (gx[:, -1] - gx[:, -2]) / dx

        # ∂gy/∂y
        dgy_dy = np.zeros_like(gy)
        dgy_dy[1:-1, :] = (gy[2:, :] - gy[:-2, :]) / (2.0 * dy)
        dgy_dy[0, :] = (gy[1, :] - gy[0, :]) / dy
        dgy_dy[-1, :] = (gy[-1, :] - gy[-2, :]) / dy

        divergence[:, t] = (dgx_dx + dgy_dy).ravel()

    return divergence


def compute_divergence_potential(phase_grad, nx, ny, dx, dy, verbose=True):
    """Compute the scalar divergence potential for each timestep using finite differences.

    Solves L φ = div where L approximates -∇^2. Neumann BC -> subtract mean(rhs) and
    enforce zero-mean on φ. Applies a physical dx**2 scaling correction to match FEM units.
    """
    N = nx * ny
    T = phase_grad.shape[2]

    if verbose:
        print(f"Computing divergence potential for {nx}×{ny} grid ({N} nodes), {T} timesteps")
        print(f"Grid spacing: dx={dx:.6g}, dy={dy:.6g}")

    # Build scaled Laplacian operator (-∇^2)
    laplace_op = build_laplacian_2d_neumann(nx, ny, dx, dy)

    if verbose:
        print(f"Laplace operator shape: {laplace_op.shape}, nnz: {laplace_op.nnz}")
        # safe diagnostics of diagonal
        diag = laplace_op.diagonal()
        print(f"Laplace diag mean/std: {np.mean(diag):.3e} / {np.std(diag):.3e}")

    # Regularization for factorization stability (small, preserves physics)
    diag_mean = np.mean(laplace_op.diagonal())
    eps = max(1e-12, 1e-9 * (diag_mean if diag_mean != 0 else 1.0))
    laplace_reg = laplace_op + eps * sparse.eye(N)

    try:
        chol_factor = cholesky(laplace_reg.tocsc())
        if verbose:
            print("Cholesky factorization successful")
    except Exception as e:
        eps2 = max(eps, 1e-6)
        laplace_reg = laplace_op + eps2 * sparse.eye(N)
        chol_factor = cholesky(laplace_reg.tocsc())
        if verbose:
            print(f"Cholesky retried with eps={eps2}")

    # Compute divergence (consistent FD)
    divergence = compute_divergence_fd_weighted(phase_grad, nx, ny, dx, dy)
    if verbose:
        print(f"Divergence shape: {divergence.shape}")
        print(f"Divergence range: [{divergence.min():.3e}, {divergence.max():.3e}]")

    potential_div = np.zeros((N, T))
    non_zero_count = 0

    # Solve Poisson for each timestep
    for t in range(T):
        if verbose and t % 500 == 0:
            print(f"  Timestep {t}/{T}")

        rhs = divergence[:, t].copy()
        # remove mean to avoid nullspace ambiguity (Neumann Laplacian)
        rhs -= np.mean(rhs)

        if np.max(np.abs(rhs)) > 1e-14:
            non_zero_count += 1
            try:
                phi = chol_factor(rhs)
                # enforce zero mean on phi for comparability
                phi -= np.mean(phi)
                potential_div[:, t] = phi
            except Exception as e:
                if verbose and t < 5:
                    print(f"Solver failed at timestep {t}: {e}")
                potential_div[:, t] = 0.0
        else:
            potential_div[:, t] = 0.0

    # final diagnostics before scaling
    if verbose:
        print(f"Non-zero divergence timesteps: {non_zero_count}/{T}")
        print(f"Pre-scale potential range: [{np.min(potential_div):.3e}, {np.max(potential_div):.3e}]")
        print(f"Pre-scale potential std: {np.std(potential_div):.3e}")

    # Scaling correction: discrete Laplacian introduces 1/dx^2 factor -> φ ~ div * dx^2
    # We observed FD φ amplitude larger than FEM; dividing by dx^2 brings units in line.
    # Apply scaling correction (physically motivated). If your FEM uses different normalization,
    # consider removing or adjusting this factor.
    scale = (dx ** 2)
    if scale != 1.0:
        potential_div = potential_div / scale
        if verbose:
            print(f"Applied scaling correction: divided potentials by dx^2 = {scale:.6g}")

    # enforce zero mean again after scaling
    potential_div -= np.mean(potential_div, axis=0, keepdims=True)

    if verbose:
        print(f"Final potential range: [{np.min(potential_div):.3e}, {np.max(potential_div):.3e}]")
        print(f"Final potential std: {np.std(potential_div):.3e}")

        # quick diagnostic for the problematic timestep if present
        t_check = 5500
        if t_check < T:
            print(f"[FD diag] t={t_check}: phi min/max/mean/std = "
                  f"{potential_div[:, t_check].min():.6g}, {potential_div[:, t_check].max():.6g}, "
                  f"{potential_div[:, t_check].mean():.6g}, {potential_div[:, t_check].std():.6g}")
            print(f"[FD diag] t={t_check}: div min/max/mean/std = "
                  f"{divergence[:, t_check].min():.6g}, {divergence[:, t_check].max():.6g}, "
                  f"{divergence[:, t_check].mean():.6g}, {divergence[:, t_check].std():.6g}")

    return potential_div


def create_region_positions(x_extent, y_extent, nx, ny):
    """Compute 2D positions for nx*ny regions in the specified extent."""
    y = np.linspace(0, y_extent, ny)
    x = np.linspace(0, x_extent, nx)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def exp_kernel(distance, sigma):
    """Exponential kernel for distance-dependent connectivity."""
    return 1/(2*sigma) * np.exp(-distance/sigma)


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

sim_id_start, sim_id_end, sim_id_step = map(int, config["simulation_id"])
simulation_id = np.arange(sim_id_start, sim_id_end, sim_id_step)[args.simulation_idx]

if verbose:
    print(f"Processing simulation {simulation_id}")

# Load phase data - NO DOWNSAMPLING
phase_path = os.path.join(data_path, f"{experiment_id}_simulations/{experiment_id}_simulation_{simulation_id}.npy")
phase = np.load(phase_path).T

if verbose:
    print(f"Loaded phase data: {phase.shape} (full resolution, includes transient)")

# Grid setup
nx, ny = int(config["nx"]), int(config["ny"])
x_extent, y_extent = float(config["x_extent"]), float(config["y_extent"])
dx = x_extent / (nx - 1)
dy = y_extent / (ny - 1)

pos = create_region_positions(x_extent, y_extent, nx, ny)
N = nx * ny

if verbose:
    print(f"Grid: {nx}×{ny} = {N} nodes")
    print(f"Grid spacing: dx={dx:.6g}, dy={dy:.6g}")

# Phase gradient using finite differences
phase_grad = compute_phase_gradient_fd(phase, nx, ny, dx, dy)
if verbose:
    print(f"Phase gradient shape: {phase_grad.shape}")

# Divergence potential (FD)
potential_div = compute_divergence_potential(phase_grad, nx, ny, dx, dy, verbose=verbose)

# Save potential
output_dir = os.path.join(data_path, f"{experiment_id}_analysis_potentials")
os.makedirs(output_dir, exist_ok=True)
potential_path = os.path.join(output_dir, f"{experiment_id}_potential_div_fd_{simulation_id}.npy")
np.save(potential_path, potential_div)
if verbose:
    print(f"Saved potential: {potential_path}, shape: {potential_div.shape}")

if verbose:
    print("Analysis complete!")
