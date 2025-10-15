#!/bin/sh

#SBATCH --job-name=48_alpha_sim_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-99
#SBATCH --requeue
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=04:00:00

# python script
PYTHONSCRIPT="scripts/simulation/48_subnet_simulation.py"
CONFIGPATH="configuration/48_alpha_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $SLURM_ARRAY_TASK_ID
