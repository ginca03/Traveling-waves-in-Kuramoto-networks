#!/bin/sh

#SBATCH --job-name=40_analysis_potentials_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --requeue
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=04:00:00


# python script
PYTHONSCRIPT="scripts/analysis/40_analysis_potentials.py"
CONFIGPATH="configuration/40_configuration.yaml"
CONFIGPATHANALYSIS="configuration/40_analysis_potentials_configuration.yaml"

ROWINDEX=$((SLURM_ARRAY_TASK_ID+$1))  # add the index from which the row index should start

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $ROWINDEX