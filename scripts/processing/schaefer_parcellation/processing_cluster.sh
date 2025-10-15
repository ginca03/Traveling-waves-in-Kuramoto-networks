#!/bin/sh

# Compute Connectome Based on Schaefer Parcellation
#
# This script computes the connectomes (weights) and tract lengths based  on the Schaefer 
# parcellation (Schaefer et al. (2018), "Local-Global Parcellation of the Human 
# Cerebral Cortex from Intrinsic Functional Connectivity MRI"; retrieved from:
# https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).
#
# This script depends on the freesurfer and mrview software. Data from the S900 release of the
# Human Connectome Project were used.

export FREESURFER_HOME=$HOME/work/freesurfer
export SUBJECTS_DIR=/fast/users/kollerd_c/work/travelingwaves/data/connectomes/Schaefer2018_HCP_S900
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# read subject id argument
SUBJECTID=$1

# get subject code
SUBJECT=$(sed -n ${SUBJECTID}p subjects_list.txt)

# print processing status
echo "Processing subject ${SUBJECT}"

# resample cortical surface parcellation of fsaverage to subject space
mri_surf2surf --srcsubject fsaverage --trgsubject ${SUBJECT} --hemi lh --sval-annot Schaefer2018_1000Parcels_17Networks_order.annot --tval $SUBJECTS_DIR/${SUBJECT}/label/lh.Schaefer2018_1000Parcels_17Networks_order_indspace.annot
mri_surf2surf --srcsubject fsaverage --trgsubject ${SUBJECT} --hemi rh --sval-annot Schaefer2018_1000Parcels_17Networks_order.annot --tval $SUBJECTS_DIR/${SUBJECT}/label/rh.Schaefer2018_1000Parcels_17Networks_order_indspace.annot

# surface parcellation to volume parcellation
mri_aparc2aseg --s ${SUBJECT} --annot Schaefer2018_1000Parcels_17Networks_order_indspace --o $SUBJECTS_DIR/${SUBJECT}/mri/aparc+aseg_schaefer2018_1000parcels_17networks.mgz

# convert freesurfer volume parcellation to mrtrix format
mrconvert -force -datatype uint32 $SUBJECTS_DIR/${SUBJECT}/mri/aparc+aseg_schaefer2018_1000parcels_17networks.mgz $SUBJECTS_DIR/${SUBJECT}/mri/schaefer2018_1000parcels_17networks.mif

# convert parcellation image to comply with look-up-table
labelconvert -force $SUBJECTS_DIR/${SUBJECT}/mri/schaefer2018_1000parcels_17networks.mif $SUBJECTS_DIR/Schaefer2018_1000Parcels_17Networks_order_LUT.txt $SUBJECTS_DIR/Schaefer2018_1000Parcels_17Networks_order_LUT_modified.txt $SUBJECTS_DIR/${SUBJECT}/mri/schaefer2018_1000parcels_17networks_parcellated.mif

# create tract weights for parcellation
tck2connectome -force $HOME/work/HCP_tracks/${SUBJECT}/${SUBJECT}_25M_tracks.tck $SUBJECTS_DIR/${SUBJECT}/mri/schaefer2018_1000parcels_17networks_parcellated.mif $SUBJECTS_DIR/${SUBJECT}/connectome.csv -tck_weights_in $HOME/work/HCP_tracks_sift2/${SUBJECT}/${SUBJECT}_25M_tracks_weights.txt -out_assignments $SUBJECTS_DIR/${SUBJECT}/assignments.txt

# compute tract lengths for parcellation
tck2connectome -force $HOME/work/HCP_tracks/${SUBJECT}/${SUBJECT}_25M_tracks.tck $SUBJECTS_DIR/${SUBJECT}/mri/schaefer2018_1000parcels_17networks_parcellated.mif $SUBJECTS_DIR/${SUBJECT}/lengths_sift2.csv -scale_length -stat_edge mean -tck_weights_in $HOME/work/HCP_tracks_sift2/${SUBJECT}/${SUBJECT}_25M_tracks_weights.txt