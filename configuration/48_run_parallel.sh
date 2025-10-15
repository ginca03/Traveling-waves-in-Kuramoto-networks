#!/bin/sh

#SBATCH --job-name=48_simulations
#SBATCH --output=logs/%x-%j.log
#SBATCH --requeue
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=04:00:00


# python script
PYTHONSCRIPT="scripts/simulation/48_simulation.py"
CONFIGPATH="configuration/48_configuration.yaml"

ROWINDEX=$((SLURM_ARRAY_TASK_ID+$1))  # add the index from which the row index should start

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $ROWINDEX