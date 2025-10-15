#!/usr/bin/env python
""" Create surface mesh spanned by the schaefer parcels

This script creates the surface mesh spanned by the Schaefer et al. (2018) regions. A surface mesh is defined by vertex
positions and triangle faces (index of three vertices). Vertex positions are determined by Schaefer region locations in 3D.
Faces are determined by the topological neighbourhood of the Schaefer regions. This script uses the high-resolution surface
mesh of the Schaefer parcellation in Freesurfer and reduces the number of vertices to the actual number of parcels.


Usage
-----
    python create_schaefer_surface.py "data_path"
    
Arguments
---------
data_path : String
    Path to data. Directory where the connectome data is stored.
"""


import sys
import os
import pickle
import mne
import numpy as np


__author__ = "Dominik Koller"
__date__ = "09. February 2022"
__status__ = "Prototype"


# Prepare Script
# --------------
# Read data path
try:
    data_path = sys.argv[1]
except BaseException as e:
    print("Error: Specify correct path to data.")
    raise

regions_path = os.path.join(data_path, 'connectomes/Schaefer2018_HCP_S900/hcp_parcellation')

# load inflated surface 
regions_lh = mne.read_labels_from_annot('fsaverage5', 'Schaefer2018_1000Parcels_17Networks_order', 'lh', surf_name='inflated', subjects_dir=regions_path, sort=False)
regions_rh = mne.read_labels_from_annot('fsaverage5', 'Schaefer2018_1000Parcels_17Networks_order', 'rh', surf_name='inflated', subjects_dir=regions_path, sort=False)

# get region positions
rr_lh, tris_lh = mne.read_surface(os.path.join(regions_path, 'fsaverage5/surf/lh.inflated'))
rr_rh, tris_rh = mne.read_surface(os.path.join(regions_path, 'fsaverage5/surf/rh.inflated'))

pos_lh = np.array([np.mean(rr_lh[r_lh.vertices,:], axis=0) for r_lh in regions_lh[1:]])  # regions begin at index 1
pos_rh = np.array([np.mean(rr_rh[r_rh.vertices,:], axis=0) for r_rh in regions_rh[1:]])  # regions begin at index 1

pos = np.concatenate([pos_lh, pos_rh])/1e3 # convert millimeters to meter


# Construct Decimated Topology from High Resolution Topology
# ----------------------------------------------------------
# process left hemisphere
# remove all triangles of medial wall
tris_lh = np.delete(tris_lh, np.concatenate([np.where(tris_lh==v)[0] for v in regions_lh[0].vertices]), axis=0)
tris_lh_new = np.zeros_like(tris_lh)

# assign region label to each triangles vertices
for i, rr in enumerate(regions_lh[1:]):
    for v in rr.vertices:
        tris_lh_new[np.where(tris_lh==v)] = i
        
# get indices of faces with three unique entries (triangles)
tris_lh_new_idx = (tris_lh_new[:,0]!=tris_lh_new[:,1]) & (tris_lh_new[:,0]!=tris_lh_new[:,2]) & (tris_lh_new[:,2]!=tris_lh_new[:,1])

# get region topology
region_topology_lh = tris_lh_new[tris_lh_new_idx,:]

# process right hemisphere
# remove all triangles of medial wall
tris_rh = np.delete(tris_rh, np.concatenate([np.where(tris_rh==v)[0] for v in regions_rh[0].vertices]), axis=0)
tris_rh_new = np.zeros_like(tris_rh)

# assign region label to each triangles vertices
for i, rr in enumerate(regions_rh[1:]):
    for v in rr.vertices:
        tris_rh_new[np.where(tris_rh==v)] = i
        
# get indices of faces with three unique entries (triangles)
tris_rh_new_idx = (tris_rh_new[:,0]!=tris_rh_new[:,1]) & (tris_rh_new[:,0]!=tris_rh_new[:,2]) & (tris_rh_new[:,2]!=tris_rh_new[:,1])

# get region topology
region_topology_rh = tris_rh_new[tris_rh_new_idx,:]


# Save Results
# ------------
mesh = {'vertices_lh': pos_lh/1e3, 
        'vertices_rh': pos_rh/1e3, 
        'faces_lh': region_topology_lh.astype(np.int64), 
        'faces_rh': region_topology_rh.astype(np.int64)}

# save mesh
with open(os.path.join(data_path, "connectomes/Schaefer2018_HCP_S900/schaefer_surface_mesh.pkl"), 'wb') as f:
    pickle.dump(mesh, f)

# save positions
np.save(os.path.join(data_path,"connectomes/Schaefer2018_HCP_S900/positions.npy"), pos)