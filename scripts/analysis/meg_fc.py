#!/usr/bin/env python
""" Compute functional connectivity for MEG

We compute the MEG functional connectivity using the phase locking value (PLV) and phase lag index (PLI) 
between all pairs of regions. (Ghuman et al., 2011; Schmidt et al., 2014; Palva et al., 2018; Lachaux et al., 1999; Bastos and Schoffelen, 2016)


Usage
-----
    python meg_fc.py "configuration_path"
    
Arguments
---------
"""


import sys
import os
import yaml
import itertools
import numpy as np

from scipy.signal import hilbert

__author__ = "Dominik Koller"
__date__ = "06. July 2022"
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


# get idxs
process_id = int(sys.argv[2])

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

number_of_regions = int(config["number_of_regions"])

band_name = ['alpha', 'beta', 'gamma']

# get subject ids
with open(os.path.join(storage_dir, 'subjects.txt')) as file:
    subs = file.readlines()
    subs = [s.rstrip() for s in subs]
n_subs = len(subs)

subject_idx, frequency_idx = list(itertools.product(*[range(0,n_subs), range(0,len(band_name))]))[process_id]

subject = subs[subject_idx]
band = band_name[frequency_idx]

print(f"Subject {subject}, {band} band")


# Compute Functional Connectivity
# -------------------------------
# load source activity
sources = np.load(os.path.join(subjects_dir, f'{subject}/sources_lcmv_{subject}_{band}.npy'))

analytic_signal = hilbert(sources, axis=1)
phase = np.angle(analytic_signal)

# compute phase locking value
FC_plv = abs(np.exp(1j*phase)@np.exp(1j*phase).conj().T / np.shape(phase)[1])

# compute phase lag index
FC_pli_tmp = np.zeros((number_of_regions,number_of_regions))
for i in range(number_of_regions):
    FC_pli_tmp[i,i:] = abs(np.mean(np.sign(np.sin(phase[i:]-phase[i])), axis=1))
    
FC_pli = FC_pli_tmp + FC_pli_tmp.T


# Save data
# ---------
np.save(os.path.join(subjects_dir, subject, f'FC_plv_{subject}_{band}.npy'), FC_plv)
np.save(os.path.join(subjects_dir, subject, f'FC_pli_{subject}_{band}.npy'), FC_pli)