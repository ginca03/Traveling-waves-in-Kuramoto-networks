#!/bin/sh

# Cluster Connectome Processing
# 
# This script is used for parallel processing of connectomes (weights) and tract lengths based 
# on the Schaefer parcellation (Schaefer et al. (2018), "Local-Global Parcellation of the Human 
# Cerebral Cortex from Intrinsic Functional Connectivity MRI"; retrieved from:
# https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal) 
# using a SLURM cluster.
#
# This script depends on the freesurfer and mrview software. Data from the S900 release of the
# Human Connectome Project were used.

#SBATCH --job-name=schaefer_parcellation
#SBATCH --output=logs/%x-%j.log
#SBATCH --ntasks=32
#SBATCH --exclusive
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=medium
#SBATCH --time=1-00:00:00

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate parc

# build srun command
srun="srun --exclusive -N1 -n1"

# build gnu parallel command
parallel="parallel --delay .2 -j $SLURM_NTASKS --joblog logs/schaefer_parcellation.log --resume"

# run parallel
$parallel "$srun ./processing_cluster.sh {1}" ::: {1..777}
