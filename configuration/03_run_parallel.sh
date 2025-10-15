#!/bin/sh

#SBATCH --job-name=03_sim_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=02:00:00

# python script
PYTHONSCRIPT="scripts/simulation/03_simulation.py"
CONFIGPATH="configuration/03_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

ROWINDEX=$((SLURM_ARRAY_TASK_ID+$1))  # add the index from which the row index should start

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $ROWINDEX
