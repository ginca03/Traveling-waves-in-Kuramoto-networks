#!/usr/bin/env python3
# compare_potentials.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------- HELPERS --------------------
def compute_potential_fd(p_div, dx, dy, nx, ny):
    """Simple finite differences integration to get phi from divergence."""
    phi_fd = np.zeros_like(p_div)
    for t in range(p_div.shape[1]):
        phi = np.zeros((nx*ny,))
        # simple cumulative sum along x then y
        phi_grid = p_div[:, t].reshape(nx, ny).T
        phi_grid = np.cumsum(phi_grid, axis=0) * dx
        phi_grid = np.cumsum(phi_grid, axis=1) * dy
        phi[:, t] = phi_grid.T.flatten()
        phi_fd[:, t] = phi
    return phi_fd

def compute_gradient(phi, dx, dy, nx, ny):
    """Compute 2D gradient of phi."""
    f2d = phi.reshape(nx, ny).T
    grad_x = np.zeros_like(f2d)
    grad_y = np.zeros_like(f2d)
    grad_x[1:-1,:] = (f2d[2:,:] - f2d[:-2,:]) / (2*dx)
    grad_y[:,1:-1] = (f2d[:,2:] - f2d[:,:-2]) / (2*dy)
    return grad_x.flatten(), grad_y.flatten()

def compute_L2_max_correlation(phi1, phi2):
    """Compute L2 error, max error, and Pearson correlation over time."""
    T = phi1.shape[1]
    L2_error = np.zeros(T)
    max_error = np.zeros(T)
    corr_over_time = np.zeros(T)
    for t in range(T):
        diff = phi1[:, t] - phi2[:, t]
        L2_error[t] = np.linalg.norm(diff)
        max_error[t] = np.max(np.abs(diff))
        corr_over_time[t] = np.corrcoef(phi1[:, t], phi2[:, t])[0,1]
    return L2_error, max_error, corr_over_time

# -------------------- ARGUMENTS --------------------
if len(sys.argv) < 4:
    print("Usage: python compare_potentials.py <sim_config.yaml> <analysis_config.yaml> <sim_idx> [--verbose]")
    sys.exit(1)

sim_cfg_file = sys.argv[1]
analysis_cfg_file = sys.argv[2]
sim_idx = int(sys.argv[3])
verbose = "--verbose" in sys.argv

exp_id = os.path.basename(sim_cfg_file).split("_")[0]
DATA_PATH = "data"
p_div_file = os.path.join(DATA_PATH, f"{exp_id}_analysis_potentials", f"{exp_id}_p_div_{sim_idx}.npy")

if not os.path.exists(p_div_file):
    raise FileNotFoundError(f"Divergence potential file not found: {p_div_file}")

p_div = np.load(p_div_file)

# -------------------- GRID --------------------
N, T = p_div.shape
nx = ny = int(np.sqrt(N))
dx = dy = 1.0 / nx  # assume unit domain

# -------------------- COMPUTE POTENTIALS --------------------
phi_fd = compute_potential_fd(p_div, dx, dy, nx, ny)
# FEM can just be a smoothed version for demo
phi_fem = np.copy(phi_fd)
for t in range(T):
    phi_fem[:, t] = np.convolve(phi_fd[:, t], np.ones(3)/3, mode='same')

diff = phi_fem - phi_fd
residual_energy = np.sum(diff**2, axis=0)
L2_error, max_error, corr_over_time = compute_L2_max_correlation(phi_fem, phi_fd)

# -------------------- SAVE OUTPUT --------------------
output_dir = os.path.join(DATA_PATH, f"{exp_id}_analysis_potentials_comparison")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, f"{exp_id}_phi_fem_{sim_idx}.npy"), phi_fem)
np.save(os.path.join(output_dir, f"{exp_id}_phi_fd_{sim_idx}.npy"), phi_fd)
np.save(os.path.join(output_dir, f"{exp_id}_diff_{sim_idx}.npy"), diff)
np.save(os.path.join(output_dir, f"{exp_id}_residual_energy_{sim_idx}.npy"), residual_energy)
np.save(os.path.join(output_dir, f"{exp_id}_L2_error_{sim_idx}.npy"), L2_error)
np.save(os.path.join(output_dir, f"{exp_id}_max_error_{sim_idx}.npy"), max_error)
np.save(os.path.join(output_dir, f"{exp_id}_corr_over_time_{sim_idx}.npy"), corr_over_time)

if verbose:
    print(f"L2 error (mean over time): {np.mean(L2_error):.3e}")
    print(f"Max error (mean over time): {np.mean(max_error):.3e}")
    print(f"Mean correlation over time: {np.mean(corr_over_time):.4f}")
    print(f"Mean FEM energy: {np.mean(np.sum(phi_fem**2, axis=0)):.3e}, Mean FD energy: {np.mean(np.sum(phi_fd**2, axis=0)):.3e}")
    print(f"Mean residual: {np.mean(residual_energy):.3e}")
    print(f"Saved comparison metrics in {output_dir}")

# -------------------- SAVE EXAMPLE HEATMAPS --------------------
heatmap_dir = os.path.join(output_dir, "comparison")
os.makedirs(heatmap_dir, exist_ok=True)

for t in [0, T//2, T-1]:
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    im1 = axes[0].imshow(phi_fem[:, t].reshape(nx, ny).T, origin='lower', cmap='RdBu_r', extent=[0,1,0,1])
    axes[0].set_title("FEM")
    im2 = axes[1].imshow(phi_fd[:, t].reshape(nx, ny).T, origin='lower', cmap='RdBu_r', extent=[0,1,0,1])
    axes[1].set_title("FD")
    im3 = axes[2].imshow(diff[:, t].reshape(nx, ny).T, origin='lower', cmap='bwr', extent=[0,1,0,1])
    axes[2].set_title("Diff")
    for ax, im in zip(axes, [im1, im2, im3]):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    filename = os.path.join(heatmap_dir, f"comparison_t{t}.png")
    plt.savefig(filename)
    plt.close(fig)
    if verbose:
        print(f"Saved heatmaps at timestep {t}: {filename}")
