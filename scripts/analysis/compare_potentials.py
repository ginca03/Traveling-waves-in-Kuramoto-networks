#!/usr/bin/env python3
# compare_potentials.py (enhanced diagnostics)

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==================== ARGUMENTS ====================
if len(sys.argv) < 4:
    print("Usage: python compare_potentials.py <sim_config.yaml> <analysis_config.yaml> <sim_idx> [--verbose]")
    sys.exit(1)

sim_cfg_file = sys.argv[1]
analysis_cfg_file = sys.argv[2]
sim_idx = int(sys.argv[3])
verbose = "--verbose" in sys.argv

exp_id = os.path.basename(sim_cfg_file).split("_")[0]
DATA_PATH = "data"

# ==================== LOAD POTENTIALS ====================

phi_fem_file = os.path.join(DATA_PATH, f"{exp_id}_analysis_potentials", f"{exp_id}_potential_div_{sim_idx}.npy")
phi_fd_file  = os.path.join(DATA_PATH, f"{exp_id}_analysis_potentials", f"{exp_id}_potential_div_fd_{sim_idx}.npy")

if not os.path.exists(phi_fem_file):
    raise FileNotFoundError(f"FEM potential file not found: {phi_fem_file}")
if not os.path.exists(phi_fd_file):
    raise FileNotFoundError(f"FD potential file not found: {phi_fd_file}")

phi_fem = np.load(phi_fem_file)
phi_fd  = np.load(phi_fd_file)

if phi_fem.shape != phi_fd.shape:
    raise ValueError(f"Shape mismatch: FEM {phi_fem.shape}, FD {phi_fd.shape}")

N, T = phi_fem.shape

# ==================== METRICS ====================

diff = phi_fem - phi_fd
residual_energy = np.sum(diff**2, axis=0)
L2_error = np.linalg.norm(diff, axis=0)
max_error = np.max(np.abs(diff), axis=0)

corr_over_time = np.zeros(T)
slopes = np.zeros(T)
intercepts = np.zeros(T)

for t in range(T):
    a = phi_fem[:, t]
    b = phi_fd[:, t]
    corr_over_time[t] = np.corrcoef(a, b)[0,1]
    s, c, _, _, _ = stats.linregress(b, a)
    slopes[t] = s
    intercepts[t] = c

# ==================== SUMMARY STATS ====================

mean_corr = np.nanmean(corr_over_time)
mean_L2   = np.mean(L2_error)
mean_slope = np.nanmean(slopes)
mean_int   = np.nanmean(intercepts)

if verbose:
    print(f"L2 error (mean over time): {mean_L2:.3e}")
    print(f"Max error (mean over time): {np.mean(max_error):.3e}")
    print(f"Mean correlation over time: {mean_corr:.4f}")
    print(f"Mean slope (a ≈ s*b + c): {mean_slope:.4f}")
    print(f"Mean intercept: {mean_int:.3e}")
    print(f"Mean residual: {np.mean(residual_energy):.3e}")

# ==================== EXTRA DIAGNOSTICS ====================

t_diag = min(5500, T-1)
a = phi_fem[:, t_diag]
b = phi_fd[:, t_diag]
s, c, r, _, _ = stats.linregress(b, a)
if verbose:
    print(f"\n[Diagnostics @ t={t_diag}]")
    print(f"FEM: min/max/mean/std = {a.min():.4f}, {a.max():.4f}, {a.mean():.3e}, {a.std():.4f}")
    print(f"FD : min/max/mean/std = {b.min():.4f}, {b.max():.4f}, {b.mean():.3e}, {b.std():.4f}")
    print(f"Linear fit (a ≈ s*b + c): s={s:.4f}, c={c:.3e}, corr={r:.4f}")
    print(f"L2 before/after mapping: {np.linalg.norm(a-b):.4f}, {np.linalg.norm(a-(s*b+c)):.4f}")

# ==================== OPTIONAL NORMALIZATION CHECK ====================

apply_norm = False  # toggle to True to test normalization correction
if apply_norm:
    phi_fem -= phi_fem.mean(axis=0, keepdims=True)
    phi_fd  -= phi_fd.mean(axis=0, keepdims=True)
    phi_fd  /= np.nanmean(slopes)  # global scale correction
    if verbose:
        print("Applied mean-centering + global scaling correction.")

# ==================== SAVE OUTPUTS ====================

output_dir = os.path.join(DATA_PATH, f"{exp_id}_analysis_potentials_comparison")
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, f"{exp_id}_phi_fem_{sim_idx}.npy"), phi_fem)
np.save(os.path.join(output_dir, f"{exp_id}_phi_fd_{sim_idx}.npy"), phi_fd)
np.save(os.path.join(output_dir, f"{exp_id}_diff_{sim_idx}.npy"), diff)
np.save(os.path.join(output_dir, f"{exp_id}_residual_energy_{sim_idx}.npy"), residual_energy)
np.save(os.path.join(output_dir, f"{exp_id}_L2_error_{sim_idx}.npy"), L2_error)
np.save(os.path.join(output_dir, f"{exp_id}_max_error_{sim_idx}.npy"), max_error)
np.save(os.path.join(output_dir, f"{exp_id}_corr_over_time_{sim_idx}.npy"), corr_over_time)
np.save(os.path.join(output_dir, f"{exp_id}_slopes_{sim_idx}.npy"), slopes)
np.save(os.path.join(output_dir, f"{exp_id}_intercepts_{sim_idx}.npy"), intercepts)

if verbose:
    print(f"Saved comparison metrics in {output_dir}")

# ==================== OPTIONAL HEATMAPS ====================

nx = ny = int(np.sqrt(phi_fem.shape[0]))  # assume square grid
heatmap_dir = os.path.join(output_dir, "comparison")
os.makedirs(heatmap_dir, exist_ok=True)

for t in [0, T//2, T-1, t_diag]:
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    im1 = axes[0].imshow(phi_fem[:, t].reshape(nx, ny).T, origin='lower', cmap='RdBu_r', extent=[0,1,0,1])
    axes[0].set_title(f"FEM (t={t})")
    im2 = axes[1].imshow(phi_fd[:, t].reshape(nx, ny).T, origin='lower', cmap='RdBu_r', extent=[0,1,0,1])
    axes[1].set_title("FD")
    im3 = axes[2].imshow((phi_fem[:, t]-phi_fd[:, t]).reshape(nx, ny).T, origin='lower', cmap='bwr', extent=[0,1,0,1])
    axes[2].set_title("Diff")
    for ax, im in zip(axes, [im1, im2, im3]):
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    filename = os.path.join(heatmap_dir, f"comparison_t{t}.png")
    plt.savefig(filename)
    plt.close(fig)
    if verbose:
        print(f"Saved heatmaps at timestep {t}: {filename}")

# ==================== ORDER CHECK (optional) ====================

check_order = False  # set True to test alternate ravel order
if check_order:
    nx = ny = int(np.sqrt(N))
    b_alt = phi_fd.reshape((ny, nx, T), order='F').reshape((N, T), order='C')
    corr_alt = np.corrcoef(phi_fem[:, t_diag], b_alt[:, t_diag])[0,1]
    print(f"Correlation with alternate flattening (t={t_diag}): {corr_alt:.4f}")
