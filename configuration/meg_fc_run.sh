#!/bin/sh

#SBATCH --job-name=meg_fc
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-240
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=short
#SBATCH --time=4:00:00


# python script
PYTHONSCRIPT="scripts/analysis/meg_fc.py"
CONFIGPATH="configuration/meg_source_reconstruction_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $SLURM_ARRAY_TASK_ID