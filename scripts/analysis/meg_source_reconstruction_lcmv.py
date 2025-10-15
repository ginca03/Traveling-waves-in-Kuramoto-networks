#!/usr/bin/env python
""" Preprocessing and source reconstruction of MEG data.

Preprocessing according to the HCP MEG processing pipeline followed by LCMV Beamformer source reconstruction of the MEG data.
The source activities are aggregated per region of the Schaefer parcellation for further analyses.

This script was largely inspired by MNE-HCP http://mne.tools/mne-hcp/:
Denis A. Engemann, jona-sassenhagen, Danilo Bzdok, Eric Larson, Mainak Jas, Alexandre Gramfort, & John Griffiths. (2016). 
mne-tools/mne-hcp: 0.1dev12 (0.1dev12). Zenodo. https://doi.org/10.5281/zenodo.159089

Usage
-----
    python meg_source_reconstruction_lcmv.py "configuration_path" subject_idx
    
Arguments
---------
configuration_path : String
    Path to configuration file.
subject_idx : int
    Index for subject ID.
"""


import sys
import os
import gc
import yaml
import mne
import numpy as np
import hcp
from hcp import preprocessing as preproc
from scipy.signal import butter, sosfiltfilt


__author__ = "Dominik Koller"
__date__ = "04. July 2022"
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

# get data paths
data_path = config["data_path"]

# path to connectome
regions_path = os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900/hcp_parcellation')

# path to store data
storage_dir = os.path.join(data_path, 'hcp_meg')

# path to downloaded HCP data
hcp_path = os.path.join(data_path, 'hcp_meg')

# path to subject specific data
subjects_dir = storage_dir + '/hcp-subjects_lcmv'

# get subject id
subject_idx = int(sys.argv[2])  # get from list
with open(os.path.join(storage_dir, 'subjects.txt')) as file:
    subs = file.readlines()
    subs = [s.rstrip() for s in subs]
subject = subs[subject_idx]

theta = np.array(config['theta_band'], dtype=float)
alpha = np.array(config['alpha_band'], dtype=float)
beta = np.array(config['beta_band'], dtype=float)
gamma = np.array(config['gamma_band'], dtype=float)
full_band = np.array(config['full_band'], dtype=float)


# Make Anatomy
# ------------
# creates the subfolders 'bem', 'mir', 'surf' and 'label'
try:
    hcp.make_mne_anatomy(
        subject=subject, 
        hcp_path=hcp_path,
        recordings_path=subjects_dir,
        subjects_dir=subjects_dir
        )
except:
    pass

# Build forward model
# -------------------
head_mri_t = mne.read_trans(os.path.join(subjects_dir, subject, '{}-head_mri-trans.fif'.format(subject)))

# create source space on fsaverage
src_fsaverage = mne.setup_source_space(
    subject='fsaverage', 
    subjects_dir=regions_path, 
    add_dist=False,
    spacing='oct6')

# morph it onto the subject
src_subject = mne.morph_source_spaces(
    src_fsaverage, subject, subjects_dir=subjects_dir)

# make BEM model
bems = mne.make_bem_model(subject, conductivity=(0.3,),  # single layer model
                          subjects_dir=subjects_dir,
                          ico=None)  # ico = None for morphed SP.
bem_sol = mne.make_bem_solution(bems)
bem_sol['surfs'][0]['coord_frame'] = 5  # Surface RAS code
bem_sol['surfs'][0]['tris'] = bem_sol['surfs'][0]['tris'].astype('i')  # this is a bug fix in mne-hcp - https://github.com/mne-tools/mne-python/issues/10564

# make forward solution
info = hcp.read_info(subject=subject, hcp_path=hcp_path, data_type='rest', run_index=0)
picks = mne.pick_types(info, meg=True, ref_meg=False)
info = mne.pick_info(info, picks)

fwd = mne.make_forward_solution(info, trans=head_mri_t, bem=bem_sol, src=src_subject)

del bem_sol
gc.collect()


# Data pre-processing
# -------------------
hcp_params = dict(
    hcp_path=hcp_path,
    subject=subject,
    data_type='rest')

# load data
raw = hcp.read_raw(**hcp_params)
raw.load_data()

# apply ref channel correction and drop ref channels
preproc.apply_ref_correction(raw)

# remove bad channels and segments
annots = hcp.read_annot(**hcp_params)
# construct MNE annotations
bad_seg = (annots['segments']['all']) / raw.info['sfreq']
annotations = mne.Annotations(
    bad_seg[:,0], (bad_seg[:, 1] - bad_seg[:, 0]), 
    description='bad')

raw.set_annotations(annotations)
raw.info['bads'].extend(annots['channels']['all'])
raw.pick_types(meg=True, ref_meg=False)  # removes bad channels

# check if artifact-free segment with min_duration exists
min_duration = float(config['min_duration'])  # s
edge_effect_duration = float(config['edge_effect_duration'])  # s

max_segment = np.argmax(bad_seg[1:,0]-bad_seg[0:-1,1])
segment = list([bad_seg[max_segment,1], bad_seg[max_segment+1,0]])
segment_duration = segment[1] - segment[0]

# check if segment is at least min_duration
if segment_duration < min_duration:
    print("WARNING: segment does not reach minimum duration")
    sys.exit(0)

# bandpass filter
# use the same settings as in the HCP-MEG processing pipeline
low_freq = float(config['low_frequency_cutoff'])  # Hz
high_freq = float(config['high_frequency_cutoff'])

raw.filter(low_freq, None, method='iir',
           iir_params=dict(order=4, ftype='butter'), n_jobs=1)
raw.filter(None, high_freq, method='iir',
           iir_params=dict(order=4, ftype='butter'), n_jobs=1)

# notch filter
notch_freq = np.array(config['notch_frequency'], dtype=float)
raw.notch_filter(notch_freq)  # Hz

# read ICA and remove EOG ECG
# note that the HCP ICA assumes that bad channels have already been removed
ica_mat = hcp.read_ica(**hcp_params)

# select the brain ICs only
exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
           if ii not in annots['ica']['brain_ic_vs']]
preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)

del ica_mat
gc.collect()

# Preprocess empty room noise
# NOTE: the ICAs are not applied here because there are no physiological signals
raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path, data_type='noise_empty_room')
raw_noise.load_data()

# apply ref channel correction and drop ref channels
preproc.apply_ref_correction(raw_noise)

# remove bad channels
annots = hcp.read_annot(**hcp_params)
raw_noise.info['bads'].extend(annots['channels']['all'])
raw_noise.pick_types(meg=True, ref_meg=False)  # removes bad channels

# bandpass filter
# use the same settings as in the HCP-MEG processing pipeline
raw_noise.filter(low_freq, None, method='iir',
           iir_params=dict(order=4, ftype='butter'), n_jobs=1)
raw_noise.filter(None, high_freq, method='iir',
           iir_params=dict(order=4, ftype='butter'), n_jobs=1)

# notch filter
raw_noise.notch_filter(notch_freq)


# Extract longest artifact-free segment
# -------------------------------------
# After bad segments are removed data might be discontinuous and induce filtering artifacts. To prevent filter artifact
# contamination, the longest artifact-free segment with a minimum duration is found.

# crop segment if it is affected by filtering artifacts
if (segment[0] - raw.times[0] - edge_effect_duration) < 0:
    segment[0] = raw.times[0] + edge_effect_duration
    print('remove filter edge effect - lower bound')
if (raw.times[-1] - segment[1] - edge_effect_duration) < 0:
    segment[1] = raw.times[-1] - edge_effect_duration
    print('remove filter edge effect - upper bound')
segment_duration = np.floor(segment[1]) - np.ceil(segment[0])

# extract longest artifact free segment
raw.crop(tmin=np.ceil(segment[0]), tmax=np.floor(segment[1]))

# remove filter edge effects from empty room data
raw_noise.crop(tmin=edge_effect_duration, tmax=raw_noise.times[-1]-edge_effect_duration)  # remove edge artifacts while keeping noise recording as long as possible


# Compute inverse solution
# ------------------------
# estimate data and noise covariances
noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')
data_cov = mne.compute_raw_covariance(raw, method='empirical')
rank = mne.compute_rank(data_cov, info=raw.info)

# create single epoch
epochs = mne.make_fixed_length_epochs(raw, duration=segment_duration)

# resample data (save memory and computational time)
resampling_frequency = float(config['resampling_frequency'])
epochs.load_data()
epochs.resample(resampling_frequency)
epochs.crop(tmin=edge_effect_duration, tmax=epochs.times[-1]-edge_effect_duration)  # remove edge artifacts

# save final duration of resting-state segment
segment_duration = epochs.times[-1] - 2*edge_effect_duration
np.save(os.path.join(subjects_dir, subject, f'sources_lcmv_{subject}_segment_duration.npy'), segment_duration)
    
# compute LCMV filters
filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain', rank=rank)

stcs = mne.beamformer.apply_lcmv_epochs(epochs[0], filters)

del noise_cov
gc.collect()

# extract sources for fsaverage
stc = stcs[0].to_original_src(src_fsaverage, subjects_dir=subjects_dir)

del stcs
gc.collect()


# Parcellate source time courses
# ------------------------------
# load region labels
regions = np.array(mne.read_labels_from_annot('fsaverage', 'Schaefer2018_1000Parcels_17Networks_order', 'both', surf_name='inflated', subjects_dir=regions_path, sort=False))
regions = list(regions[["background" not in r.name.lower() for r in regions]])  # remove background

# Aggregate sources within parcels
sources = mne.extract_label_time_course(stc, regions, src_fsaverage, mode='mean', allow_empty=True, return_generator=False)


# Extract frequency of interest
# -----------------------------
filt_order = 8
band = [full_band] #[theta, alpha, beta, gamma, full_band]
band_name = ['full'] #['theta', 'alpha', 'beta', 'gamma', 'full']

for frequency_idx in range(len(band_name)):
    sos = butter(int(filt_order/2), band[frequency_idx], btype='band', output='sos', fs=epochs.info['sfreq'])
    sources_band = sosfiltfilt(sos, sources, axis=1)

    # remove filter edge effects
    crop_samples = int(np.rint(edge_effect_duration/(1/epochs.info['sfreq'])))  # edge effects
    source_activity = sources_band[:,crop_samples:-crop_samples]

    # save source activity
    np.save(os.path.join(subjects_dir, subject, f'sources_lcmv_{subject}_{band_name[frequency_idx]}.npy'), source_activity)