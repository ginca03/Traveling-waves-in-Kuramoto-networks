#!/bin/sh

#SBATCH --job-name=02_analysis_potentials_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-99
#SBATCH --requeue
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=standard
#SBATCH --time=20:00:00

# python script
PYTHONSCRIPT="scripts/analysis/flow_analysis_2d.py"
CONFIGPATH="configuration/02_configuration.yaml"
CONFIGPATHANALYSIS="configuration/02_analysis_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $SLURM_ARRAY_TASK_ID