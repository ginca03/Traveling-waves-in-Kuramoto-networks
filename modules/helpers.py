#!/usr/bin/env python
""" Helper functions """


import os
import yaml
import igl
import cv2
import pickle
import numpy as np
import brainspace.mesh as bsmesh
import scipy as sp

from modules.wave_detection_methods import k_ring_boundary
from brainspace.null_models import MoranRandomization, SpinPermutations
from brainspace.mesh import mesh_elements as me
from scipy.stats import spearmanr, pearsonr

__author__ = "Dominik Koller"
__date__ = "16. February 2021"
__status__ = "Prototype"


def correlation_spin_permutation_testing(x, y, n_permutations = 10000, sphere_path='../data/connectomes/Schaefer2018_HCP_S900/positions_sphere.npy', hemi=None):
    """ Correlation spin permutation testing
    
    Parameters
    ----------
    x : numpy.array
        1D array of independent variable.
    y : numpy.array
        1D array of dependent variable.
    n_permutations : int
        Number of permutations.
    sphere_path : string
        Path to region positions on sphere.
    hemi: string
        String indicating hemisphere for stats.
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    rho_p : float
        p-value of permutation test. 
    """
    # load sphere
    pos_sphere = np.load(sphere_path)
    N = len(pos_sphere)
    
    if hemi=='lh':
        pos_sphere = pos_sphere[:int(N/2)]
    elif hemi=='rh':
        pos_sphere = pos_sphere[int(N/2):]

    # fit brain
    spins = SpinPermutations(n_rep=n_permutations)
    spins.fit(pos_sphere)

    # randomize location indices
    rotated_idx = spins.randomize(np.arange(0,N))

    x_permuted = x[rotated_idx]

    # Compute original spearman correlation
    rho = spearmanr(y, x)[0]

    # Run permutation test
    rho_permuted = np.zeros(n_permutations)
    for pt in range(n_permutations):
        rho_permuted[pt] = spearmanr(y, x_permuted[pt])[0]

    # Calculate p-values
    rho_p = np.sum(np.abs(rho_permuted) >= np.abs(rho)) / n_permutations

    print(f"\tcorrelation = {rho:.2f} (p < {rho_p})\n")
    
    return rho, rho_p


def pearsonr_spin_permutation_testing(x, y, n_permutations = 10000, sphere_path='../data/connectomes/Schaefer2018_HCP_S900/positions_sphere.npy'):
    """ Pearson correlation spin permutation testing
    
    Parameters
    ----------
    x : numpy.array
        1D array of independent variable.
    y : numpy.array
        1D array of dependent variable.
    n_permutations : int
        Number of permutations.
    sphere_path : string
        Path to region positions on sphere
        
    Returns
    -------
    rho : float
        Pearson correlation coefficient.
    rho_p : float
        p-value of permutation test. 
    """
    # load sphere
    pos_sphere = np.load(sphere_path)

    # fit brain
    spins = SpinPermutations(n_rep=n_permutations)
    spins.fit(pos_sphere)

    # randomize location indices
    rotated_idx = spins.randomize(np.arange(0,len(pos_sphere)))

    x_permuted = x[rotated_idx]

    # Compute original spearman correlation
    rho = pearsonr(y, x)[0]

    # Run permutation test
    rho_permuted = np.zeros(n_permutations)
    for pt in range(n_permutations):
        rho_permuted[pt] = pearsonr(y, x_permuted[pt])[0]

    # Calculate p-values
    rho_p = np.sum(np.abs(rho_permuted) >= np.abs(rho)) / n_permutations

    print(f"\tcorrelation = {rho:.2f} (p < {rho_p})\n")
    
    return rho, rho_p


def correlation_spin_permutation_testing_2d(x, y, n_permutations = 10000, return_rho_permuted=False):
    """ Correlation spin permutation testing
    
    Computes non-parametric correlation statistics based on rotated and translated 2d
    images of the original image (x) and thus preserves the spatial autocorrelation.
    
    Parameters
    ----------
    x : numpy.array
        2D array of independent variable (image to rotate and translate).
    y : numpy.array
        2D array of dependent variable.
    n_permutations : int
        Number of permutations.
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    rho_p : float
        p-value of permutation test. 
    """
    nx = np.shape(x)[0]
    ny = np.shape(x)[1]
    
    # Compute original spearman correlation
    rho = spearmanr(y.flatten(), x.flatten())[0]

    # Run permutation test
    rho_permuted = np.zeros(n_permutations)
    for pt in range(n_permutations):
        # pick random rotation angle and translation
        ang = np.random.uniform(0, 360)
        tx = np.random.uniform(0, nx)
        ty = np.random.uniform(0, ny)
        
        # create rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center=(nx/2, ny/2), angle=ang, scale=1)
        
        # create the translation matrix
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty]
        ], dtype=np.float32)
        
        rot_hom = np.append(rotate_matrix, [[0, 0, 1]], axis=0)
        tr_hom = np.append(translation_matrix, [[0, 0, 1]], axis=0)
        rt_mat = rot_hom @ tr_hom
        
        # rotate and translate data
        x_permuted = cv2.warpAffine(src=x, M=rt_mat[:2,:], dsize=(nx, ny), borderMode=cv2.BORDER_REFLECT)
        
        # compute correlation
        rho_permuted[pt] = spearmanr(y.flatten(), x_permuted.flatten())[0]

    # Calculate p-values
    rho_p = np.sum(np.abs(rho_permuted) >= np.abs(rho)) / n_permutations

    print(f"\tcorrelation = {rho:.2f} (p < {rho_p})\n")
    
    if return_rho_permuted==True:
        return rho, rho_p, rho_permuted
    else:
        return rho, rho_p



def correlation_moran_randomization_testing(x, y, v, f, n_permutations = 10000):
    """ Correlation permutation testing
    
    Parameters
    ----------
    x : numpy.array
        1D array of independent variable.
    y : numpy.array
        1D array of dependent variable.
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    n_permutations : int
        Number of permutations.
        
    Returns
    -------
    rho : float
        Spearman correlation coefficient.
    rho_p : float
        p-value of permutation test. 
    """
    # compute spatial weight matrix
    surf = bsmesh.mesh_creation.build_polydata(v, f)

    w = me.get_ring_distance(surf, n_ring=1)
    w.data **= -1  # inverse geodesic distance

    # Moran Spectral Randomization
    msr = MoranRandomization(n_rep=n_permutations, procedure='singleton', tol=1e-6)
    msr.fit(w)

    # Generate randomizations
    x_permuted = msr.randomize(x)

    # Compute original spearman correlation
    rho = spearmanr(y, x)[0]

    # Run permutation test
    rho_permuted = np.zeros(n_permutations)
    for pt in range(n_permutations):
        rho_permuted[pt] = spearmanr(y, x_permuted[pt])[0]

    # Calculate p-values
    rho_p = np.sum(np.abs(rho_permuted) >= np.abs(rho)) / n_permutations

    print(f"\tcorrelation = {rho:.2f} (p < {rho_p})\n")
    
    return rho, rho_p


def compute_instantaneous_frequency(analytic_signal, time_step):
    """ Compute temporal phase gradient / instantaneous frequency
    
    The algorithm is based on Algorithm 3, p.18, Feldman (2011; Hilbert Transform Applications in Mechanical Vibration)
    
    Parameters
    ----------
    analytic_signal : numpy.complex128
        #vertices x time complex array of the analytic signal.
    time_step : float
        time step in seconds.
        
    Returns
    -------
    instantaneous_frequency : numpy.array
        time-1 x #vertices array of time phase gradients / instantaneous frequencies.
    """  
    instantaneous_frequency = np.angle(analytic_signal[:,1:]*np.conj(analytic_signal[:,:-1])) / (2*np.pi*time_step)
    
    return instantaneous_frequency


def spectral_decomposition(v, f, k):
    """ Eigendecomposition of Laplacian
        
    This code was inspired by https://libigl.github.io/tutorial/#eigen-decomposition and 
    Vallet and Lévy (2007; 'Spectral Geometry Processing with Manifold Harmonics')
    Iivanainen et al. (2021; 'Spatial sampling of MEG and EEG based on generalized spatial-frequency analysis and optimal design')
    
    Parameters
    ----------
    v : numpy.array
        #vertices x 3 array of vertex coordinates in 3D.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
    k : int
        # of eigenmodes
        
    Returns
    -------
    eigenvalues : numpy.array
        k eigenvalues corresponding to the eigenmodes.
    eigenmodes : numpy.array
        #v x k eigenmodes.
    massmatrix : numpy.array
        #v by #v mass matrix.
    """
    # Compute Laplacian
    cotlaplacian = -igl.cotmatrix(v, f).toarray()
    massmatrix = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI).toarray()
    
    # solve generalized eigenvalue problem
    eigenvalues, eigenmodes = sp.linalg.eigh(cotlaplacian, massmatrix)
    
    return eigenvalues[:k], eigenmodes[:,:k], massmatrix


def compute_parcel_topology(per_vertex_labels, f):
    """ Compute the topology between parcels.
    
    NOTE: Removes medial wall at index 0.
    
    Parameters
    ----------
    per_vertex_labels : numpy.array
        #vertices array of per vertex parcel labels. This usually comes from a high-resolution surface such as fsaverage.
    f : numpy.array
        #faces x 3 array of faces indexing the vertices of a triangle.
        
    Returns
    -------
    f_parcels : numpy.array
        #faces x 3 array of faces indexing the parcels of the given parcellation.
    """
    region_to_face_mapping = per_vertex_labels[f]
    
    # find faces that belong to three distinct parcels 
    f_idx = (region_to_face_mapping[:,0]!=region_to_face_mapping[:,1]) & (region_to_face_mapping[:,0]!=region_to_face_mapping[:,2]) & (region_to_face_mapping[:,2]!=region_to_face_mapping[:,1])
    f_parcels = region_to_face_mapping[np.where(f_idx==True)[0]].astype(int)
    f_parcels = np.delete(f_parcels, np.where(f_parcels==0)[0], axis=0)-1  # remove medial wall
    
    return f_parcels

        
def aggregate_results(data_path, experiment_id):
    """ Aggregate results of analysis
    
    Parameters
    ----------
    data_path : String
        Path to data.
    experiment_id : int
        ID of experiment to aggregate results.
        
    Returns
    -------
    wave_potential : numpy.array
        #regions array of average wave potentials.    
    effective_frequency :
        #regions array of average effective frequency.
    corr_mean : list of numpy.array
        hemisphere x time array of average instrength - potential correlation across simulations.
    corr_sd : list of umpy.array
        hemisphere x time array of instrength - potential correlation standard deviation across simulations.
    order_parameter : numpy.array
        #simulations synchronization order parameter across simulations.    
    """
    # Read Configuration
    configuration_path = f"../../configuration/{experiment_id}_configuration.yaml"
    configuration_analysis_path = f"../../configuration/{experiment_id}_analysis_configuration.yaml"
    try:
        with open(configuration_path, "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.BaseLoader)
            
        with open(configuration_analysis_path, "r") as ymlfile:
            config_analysis = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    except BaseException as e:
        print("Error: Specify correct path to yaml configuration file.")
        raise
        
    # load schaefer surface mesh
    with open(os.path.join(data_path, "connectomes/Schaefer2018_HCP_S900/schaefer_surface_mesh.pkl"), 'rb') as f:
        surface_mesh = pickle.load(f)
        
    v_lh = surface_mesh['vertices_lh']
    f_lh = surface_mesh['faces_lh']
    v_rh = surface_mesh['vertices_rh']
    f_rh = surface_mesh['faces_rh']

    number_of_regions_per_hemi = v_lh.shape[0]

    # get boundary mask
    boundary_mask_lh = k_ring_boundary(v_lh, f_lh, k=1)
    boundary_mask_rh = k_ring_boundary(v_rh, f_rh, k=1)


    # Simulation Parameters
    # ---------------------
    sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
    simulation_ids = np.arange(sim_id_start, sim_id_end, sim_id_step)
    number_of_simulations = len(simulation_ids)

    integration_step_size = float(config["integration_step_size"])  # ms
    initial_transient = int(config["initial_transient"])  # ms
    initial_transient_samples = int(initial_transient/integration_step_size)
    try:
        final_transient = int(config["final_transient"])  # ms
        final_transient_samples = int(final_transient/integration_step_size)
        simulation_duration = float(config["simulation_duration"])-initial_transient-final_transient  # ms
    except:
        simulation_duration = float(config["simulation_duration"])-initial_transient  # ms
    number_of_timesteps = int(simulation_duration/integration_step_size)

    downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
    integration_step_size_downsampled = integration_step_size * downsampling_factor
    number_of_timesteps_downsampled = int(simulation_duration/integration_step_size_downsampled)

    number_of_regions_per_hemi = 500

    # Analysis Parameters
    # -------------------
    significance_level = float(config_analysis["significance_level"])


    # Define variables
    # ----------------
    proportion_waves_lh = np.zeros(number_of_simulations)

    wave_potential_div_lh_sum = np.zeros(number_of_regions_per_hemi)

    proportion_corr_lh = np.zeros(number_of_simulations)

    corr_lh = np.zeros((number_of_simulations, number_of_timesteps_downsampled))
    corr_lh_sim = np.zeros(number_of_simulations)

    proportion_waves_rh = np.zeros(number_of_simulations)

    wave_potential_div_rh_sum = np.zeros(number_of_regions_per_hemi)

    proportion_corr_rh = np.zeros(number_of_simulations)

    corr_rh = np.zeros((number_of_simulations, number_of_timesteps_downsampled))
    corr_rh_sim = np.zeros(number_of_simulations)

    instantaneous_frequency_lh_sum = np.zeros(number_of_regions_per_hemi)
    instantaneous_frequency_rh_sum = np.zeros(number_of_regions_per_hemi)
    order_parameter = np.zeros(number_of_simulations)


    for sid in simulation_ids:
        print(f"Experiment ID: {experiment_id}; Simulation ID: {sid}", end='\r')
        
        # Frequency distribution
        # ----------------------
        # load phase
        phase = np.load(os.path.join(data_path, f'{experiment_id}_simulations', f'{experiment_id}_simulation_{sid}.npy')).T
        phase = phase[:, np.arange(initial_transient_samples, number_of_timesteps+initial_transient_samples, integration_step_size).astype(int)]  # remove transient
        ce_phase = np.exp(1j*phase)  # complex exponential phase
        
        instantaneous_frequency = compute_instantaneous_frequency(ce_phase, integration_step_size*1e-3).T
        instantaneous_frequency = instantaneous_frequency[np.arange(0, number_of_timesteps, integration_step_size_downsampled).astype(int)]  # downsampling
        
        
        # Synchronization
        # ---------------
        # compute kuramoto order parameter
        order_parameter[sid] = np.mean(abs(np.mean(ce_phase, axis=0)))
        
        
        # Process divergent activity - left hemi
        # --------------------------------------
        # Find significant singularities (based on angular similarity)
        p_div_lh = np.load(os.path.join(data_path, f'{experiment_id}_analysis', f'{experiment_id}_p_div_lh_{sid}.npy'))[np.invert(boundary_mask_lh)]
        
        significant_div_lh = p_div_lh <= significance_level

        # Create wave mask
        wave_mask_div_lh = np.nansum(significant_div_lh, axis=0, dtype=bool)

        # Load HHD potentials
        potential_div_lh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_potential_div_lh_{sid}.npy'))
        
        
        # Stats
        # -----
        if wave_mask_div_lh.sum():
            # compute proportion waves
            proportion_waves_lh[sid] = np.nansum(wave_mask_div_lh) / number_of_timesteps_downsampled

            # Sum up time-averaged potential across waves
            wave_potential_div_lh_sum_tmp = np.nanmean(potential_div_lh[:,wave_mask_div_lh], axis=1)
            wave_potential_div_lh_sum = np.nansum([wave_potential_div_lh_sum, wave_potential_div_lh_sum_tmp], axis=0)
        
            # Load instrength-potential correlation
            corr_lh[sid] = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_corr_div_lh_{sid}.npy'))
            
            # Assess significance of correlation
            p_corr_lh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_p_corr_lh_{sid}.npy'))
            
            significant_corr_lh = p_corr_lh <= significance_level  # one-sided hypothesis: instrength is negatively correlated to potentials

            # Proportion of significant correlation during waves
            proportion_corr_lh[sid] = np.nansum(significant_corr_lh & wave_mask_div_lh) / np.sum(wave_mask_div_lh)

            # Sum up correlation means
            corr_lh_sim[sid] = np.nanmean(corr_lh[sid][wave_mask_div_lh & significant_corr_lh])

            # instantaneous frequency median
            instantaneous_frequency_lh_sum += np.nanmedian(instantaneous_frequency[wave_mask_div_lh, :number_of_regions_per_hemi], axis=0)

            
        # Process divergent activity - right hemi
        # --------------------------------------
        # Find significant singularities (based on angular similarity)
        p_div_rh = np.load(os.path.join(data_path, f'{experiment_id}_analysis', f'{experiment_id}_p_div_rh_{sid}.npy'))[np.invert(boundary_mask_rh)]

        significant_div_rh = p_div_rh <= significance_level
        
        # Create wave mask
        wave_mask_div_rh = np.nansum(significant_div_rh, axis=0, dtype=bool)

        # Load HHD potentials
        potential_div_rh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_potential_div_rh_{sid}.npy'))
        
        
        # Stats
        # -----
        if wave_mask_div_rh.sum():
            # compute proportion waves
            proportion_waves_rh[sid] = np.nansum(wave_mask_div_rh) / number_of_timesteps_downsampled

            # Sum up time-averaged potential across waves
            wave_potential_div_rh_sum_tmp = np.nanmean(potential_div_rh[:,wave_mask_div_rh], axis=1)
            wave_potential_div_rh_sum = np.nansum([wave_potential_div_rh_sum, wave_potential_div_rh_sum_tmp], axis=0)
        
            # Load instrength-potential correlation
            corr_rh[sid] = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_corr_div_rh_{sid}.npy'))
            
            # Assess significance of correlation
            p_corr_rh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_p_corr_rh_{sid}.npy'))
            
            significant_corr_rh = p_corr_rh <= significance_level  # one-sided hypothesis: instrength is negatively correlated to potentials

            # Proportion of significant correlation during waves
            proportion_corr_rh[sid] = np.nansum(significant_corr_rh & wave_mask_div_rh) / np.sum(wave_mask_div_rh)

            # Sum up correlation means
            corr_rh_sim[sid] = np.nanmean(corr_rh[sid][wave_mask_div_rh & significant_corr_rh])

            # instantaneous frequency median
            instantaneous_frequency_rh_sum += np.nanmedian(instantaneous_frequency[wave_mask_div_rh, number_of_regions_per_hemi:], axis=0)


    # Aggregate stats - left hemi
    wave_potential_div_lh_mean = wave_potential_div_lh_sum / number_of_simulations

    proportion_waves_lh_median = np.nanmedian(proportion_waves_lh)
    proportion_corr_lh_median = np.nanmedian(proportion_corr_lh)

    proportion_waves_lh_quartiles = np.nanpercentile(proportion_waves_lh, [25, 75])
    proportion_corr_lh_quartiles = np.nanpercentile(proportion_corr_lh, [25, 75])

    corr_lh_time_avg_mean = np.nanmean(corr_lh_sim)
    corr_lh_time_avg_sd = np.nanstd(corr_lh_sim)

    corr_lh_mean = np.nanmean(corr_lh, axis=0)
    corr_lh_sd = np.nanstd(corr_lh, axis=0)

    instantaneous_frequency_lh_mean = instantaneous_frequency_lh_sum / number_of_simulations


    # Aggregate stats - right hemi
    wave_potential_div_rh_mean = wave_potential_div_rh_sum / number_of_simulations

    proportion_waves_rh_median = np.nanmedian(proportion_waves_rh)
    proportion_corr_rh_median = np.nanmedian(proportion_corr_rh)

    proportion_waves_rh_quartiles = np.nanpercentile(proportion_waves_rh, [25, 75])
    proportion_corr_rh_quartiles = np.nanpercentile(proportion_corr_rh, [25, 75])

    corr_rh_time_avg_mean = np.nanmean(corr_rh_sim)
    corr_rh_time_avg_sd = np.nanstd(corr_rh_sim)

    corr_rh_mean = np.nanmean(corr_rh, axis=0)
    corr_rh_sd = np.nanstd(corr_rh, axis=0)

    instantaneous_frequency_rh_mean = instantaneous_frequency_rh_sum / number_of_simulations

    wave_potential = np.concatenate([wave_potential_div_lh_mean, wave_potential_div_rh_mean])
    effective_frequency = np.concatenate([instantaneous_frequency_lh_mean, instantaneous_frequency_rh_mean])
    corr_mean = [corr_lh_mean, corr_rh_mean]
    corr_sd = [corr_lh_sd, corr_rh_sd]

    # Print stats
    print(f"Left Hemisphere:\n"
        f"\tMedian (IQR) proportion of source-sink wave episodes across simulation duration: {proportion_waves_lh_median:.2f} ({proportion_waves_lh_quartiles[0]:.2f} - {proportion_waves_lh_quartiles[1]:.2f})\n"
        f"\tMedian (IQR) proportion of instrength-guided waves across wave episodes: {proportion_corr_lh_median:.2f}  ({proportion_corr_lh_quartiles[0]:.2f} - {proportion_corr_lh_quartiles[1]:.2f})\n"
        f"\tAverage instrength-potential correlation: {corr_lh_time_avg_mean:.2f} \u00B1 {corr_lh_time_avg_sd:.2f}\n\n"
        f"Right Hemisphere:\n"
        f"\tMedian (IQR) proportion of source-sink wave episodes across simulation duration: {proportion_waves_rh_median:.2f} ({proportion_waves_rh_quartiles[0]:.2f} - {proportion_waves_rh_quartiles[1]:.2f})\n"
        f"\tMedian (IQR) proportion of instrength-guided waves across wave episodes: {proportion_corr_rh_median:.2f}  ({proportion_corr_rh_quartiles[0]:.2f} - {proportion_corr_rh_quartiles[1]:.2f})\n"
        f"\tAverage instrength-potential correlation: {corr_rh_time_avg_mean:.2f} \u00B1 {corr_rh_time_avg_sd:.2f}"
        )
    
    return wave_potential, effective_frequency, corr_mean, corr_sd, order_parameter


def aggregate_proportions(data_path, experiment_id):
    """ Aggregate proportion results of analysis
    
    Parameters
    ----------
    data_path : String
        Path to data.
    experiment_id : int
        ID of experiment to aggregate results.
        
    Returns
    -------
    proportion_waves_lh : numpy.float
        Proportion of waves in left hemisphere.    
    proportion_waves_rh : numpy.float
        Proportion of waves in right hemisphere.  
    proportion_corr_lh : numpy.float
        Proportion of instrength-guided waves in left hemisphere. 
    proportion_corr_rh : numpy.float
        Proportion of instrength-guided waves in right hemisphere.     
    """
    # Read Configuration
    configuration_path = f"../../configuration/{experiment_id}_configuration.yaml"
    configuration_analysis_path = f"../../configuration/{experiment_id}_analysis_configuration.yaml"
    try:
        with open(configuration_path, "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.BaseLoader)
            
        with open(configuration_analysis_path, "r") as ymlfile:
            config_analysis = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    except BaseException as e:
        print("Error: Specify correct path to yaml configuration file.")
        raise
        
    # load schaefer surface mesh
    with open(os.path.join(data_path, "connectomes/Schaefer2018_HCP_S900/schaefer_surface_mesh.pkl"), 'rb') as f:
        surface_mesh = pickle.load(f)
        
    v_lh = surface_mesh['vertices_lh']
    f_lh = surface_mesh['faces_lh']
    v_rh = surface_mesh['vertices_rh']
    f_rh = surface_mesh['faces_rh']

    number_of_regions_per_hemi = v_lh.shape[0]

    # get boundary mask
    boundary_mask_lh = k_ring_boundary(v_lh, f_lh, k=1)
    boundary_mask_rh = k_ring_boundary(v_rh, f_rh, k=1)


    # Simulation Parameters
    # ---------------------
    sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
    simulation_ids = np.arange(sim_id_start, sim_id_end, sim_id_step)
    number_of_simulations = len(simulation_ids)

    integration_step_size = float(config["integration_step_size"])  # ms
    initial_transient = int(config["initial_transient"])  # ms
    initial_transient_samples = int(initial_transient/integration_step_size)
    try:
        final_transient = int(config["final_transient"])  # ms
        final_transient_samples = int(final_transient/integration_step_size)
        simulation_duration = float(config["simulation_duration"])-initial_transient-final_transient  # ms
    except:
        simulation_duration = float(config["simulation_duration"])-initial_transient  # ms
    number_of_timesteps = int(simulation_duration/integration_step_size)

    downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
    integration_step_size_downsampled = integration_step_size * downsampling_factor
    number_of_timesteps_downsampled = int(simulation_duration/integration_step_size_downsampled)

    number_of_regions_per_hemi = 500

    # Analysis Parameters
    # -------------------
    significance_level = float(config_analysis["significance_level"])


    # Define variables
    # ----------------
    proportion_waves_lh = np.zeros(number_of_simulations)
    proportion_corr_lh = np.zeros(number_of_simulations)

    proportion_waves_rh = np.zeros(number_of_simulations)
    proportion_corr_rh = np.zeros(number_of_simulations)


    for sid in simulation_ids:
        # Process divergent activity - left hemi
        # --------------------------------------
        # Find significant singularities (based on angular similarity)
        p_div_lh = np.load(os.path.join(data_path, f'{experiment_id}_analysis', f'{experiment_id}_p_div_lh_{sid}.npy'))[np.invert(boundary_mask_lh)]
        
        significant_div_lh = p_div_lh <= significance_level

        # Create wave mask
        wave_mask_div_lh = np.nansum(significant_div_lh, axis=0, dtype=bool)

        
        # Stats
        # -----
        if wave_mask_div_lh.sum():
            # compute proportion waves
            proportion_waves_lh[sid] = np.nansum(wave_mask_div_lh) / number_of_timesteps_downsampled

            # Assess significance of correlation
            p_corr_lh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_p_corr_lh_{sid}.npy'))
            
            significant_corr_lh = p_corr_lh <= significance_level  # one-sided hypothesis: instrength is negatively correlated to potentials

            # Proportion of significant correlation during waves
            proportion_corr_lh[sid] = np.nansum(significant_corr_lh & wave_mask_div_lh) / np.sum(wave_mask_div_lh)

            
        # Process divergent activity - right hemi
        # --------------------------------------
        # Find significant singularities (based on angular similarity)
        p_div_rh = np.load(os.path.join(data_path, f'{experiment_id}_analysis', f'{experiment_id}_p_div_rh_{sid}.npy'))[np.invert(boundary_mask_rh)]

        significant_div_rh = p_div_rh <= significance_level
        
        # Create wave mask
        wave_mask_div_rh = np.nansum(significant_div_rh, axis=0, dtype=bool)

        
        # Stats
        # -----
        if wave_mask_div_rh.sum():
            # compute proportion waves
            proportion_waves_rh[sid] = np.nansum(wave_mask_div_rh) / number_of_timesteps_downsampled

            # Assess significance of correlation
            p_corr_rh = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_p_corr_rh_{sid}.npy'))
            
            significant_corr_rh = p_corr_rh <= significance_level  # one-sided hypothesis: instrength is negatively correlated to potentials

            # Proportion of significant correlation during waves
            proportion_corr_rh[sid] = np.nansum(significant_corr_rh & wave_mask_div_rh) / np.sum(wave_mask_div_rh)

    return proportion_waves_lh, proportion_waves_rh, proportion_corr_lh, proportion_corr_rh


def aggregate_proportions_2d(data_path, experiment_id):
    """ Aggregate proportion results of analysis
    
    Parameters
    ----------
    data_path : String
        Path to data.
    experiment_id : int
        ID of experiment to aggregate results.
        
    Returns
    -------
    proportion_waves : numpy.float
        Proportion of waves.  
    proportion_corr : numpy.float
        Proportion of instrength-guided waves.     
    """
    # Read Configuration
    configuration_path = f"../../configuration/{experiment_id}_configuration.yaml"
    configuration_analysis_path = f"../../configuration/{experiment_id}_analysis_configuration.yaml"
    try:
        with open(configuration_path, "r") as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.BaseLoader)
            
        with open(configuration_analysis_path, "r") as ymlfile:
            config_analysis = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    except BaseException as e:
        print("Error: Specify correct path to yaml configuration file.")
        raise


    # Simulation Parameters
    # ---------------------
    sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
    simulation_ids = np.arange(sim_id_start, sim_id_end, sim_id_step)
    number_of_simulations = len(simulation_ids)

    integration_step_size = float(config["integration_step_size"])  # ms
    initial_transient = int(config["initial_transient"])  # ms
    initial_transient_samples = int(initial_transient/integration_step_size)
    try:
        final_transient = int(config["final_transient"])  # ms
        final_transient_samples = int(final_transient/integration_step_size)
        simulation_duration = float(config["simulation_duration"])-initial_transient-final_transient  # ms
    except:
        simulation_duration = float(config["simulation_duration"])-initial_transient  # ms
    number_of_timesteps = int(simulation_duration/integration_step_size)

    downsampling_factor = float(config_analysis["downsampling_factor"])  # a.u.
    integration_step_size_downsampled = integration_step_size * downsampling_factor
    number_of_timesteps_downsampled = int(simulation_duration/integration_step_size_downsampled)


    # Analysis Parameters
    # -------------------
    significance_level = float(config_analysis["significance_level"])


    # Define variables
    # ----------------
    proportion_waves = np.zeros(number_of_simulations)
    proportion_corr = np.zeros(number_of_simulations)

    for sid in simulation_ids:
        # Process divergent activity - left hemi
        # --------------------------------------
        # Find significant singularities (based on angular similarity)
        p_div = np.load(os.path.join(data_path, f'{experiment_id}_analysis', f'{experiment_id}_p_div_{sid}.npy'))
        
        significant_div = p_div <= significance_level

        # Create wave mask
        wave_mask_div = np.nansum(significant_div, axis=0, dtype=bool)

        
        # Stats
        # -----
        if wave_mask_div.sum():
            # compute proportion waves
            proportion_waves[sid] = np.nansum(wave_mask_div) / number_of_timesteps_downsampled

            # Assess significance of correlation
            p_corr = np.load(os.path.join(data_path, f'{experiment_id}_analysis_potentials', f'{experiment_id}_p_corr_{sid}.npy'))
            
            significant_corr = p_corr <= significance_level  # one-sided hypothesis: instrength is negatively correlated to potentials

            # Proportion of significant correlation during waves
            proportion_corr[sid] = np.nansum(significant_corr & wave_mask_div) / np.sum(wave_mask_div)

    return proportion_waves, proportion_corr