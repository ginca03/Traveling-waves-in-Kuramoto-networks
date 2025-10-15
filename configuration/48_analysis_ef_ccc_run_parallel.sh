#!/bin/sh

#SBATCH --job-name=48_analysis_ef_ccc
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-1170
#SBATCH --requeue
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=short
#SBATCH --time=04:00:00


# python script
PYTHONSCRIPT="scripts/analysis/48_analysis_ef_ccc.py"
CONFIGPATH="configuration/48_configuration.yaml"
CONFIGPATHANALYSIS="configuration/48_analysis_ef_configuration.yaml"

ROWINDEX=$((SLURM_ARRAY_TASK_ID+0))  # add the index from which the row index should start

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $ROWINDEX