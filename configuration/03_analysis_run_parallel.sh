#!/bin/sh

#SBATCH --job-name=03_analysis_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --requeue
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=standard
#SBATCH --time=24:00:00

# python script
PYTHONSCRIPT="scripts/analysis/03_analysis.py"
CONFIGPATH="configuration/03_configuration.yaml"
CONFIGPATHANALYSIS="configuration/03_analysis_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

ROWINDEX=$((SLURM_ARRAY_TASK_ID+$1))  # add the index from which the row index should start

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $ROWINDEX
