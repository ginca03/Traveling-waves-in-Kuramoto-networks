#!/bin/sh

#SBATCH --job-name=40_analysis_wave_speed
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-2000
#SBATCH --requeue
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=04:00:00


# python script
PYTHONSCRIPT="scripts/analysis/40_analysis_wave_speed.py"
CONFIGPATH="configuration/40_configuration.yaml"
CONFIGPATHANALYSIS="configuration/40_analysis_wave_speed_configuration.yaml"

ROWINDEX=$((SLURM_ARRAY_TASK_ID+0))  # add the index from which the row index should start

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $ROWINDEX