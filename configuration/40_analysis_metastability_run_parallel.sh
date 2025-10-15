#!/bin/sh

#SBATCH --job-name=40_analysis_metastability
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-1560
#SBATCH --requeue
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=short
#SBATCH --time=02:00:00

# python script
PYTHONSCRIPT="scripts/analysis/40_analysis_metastability.py"
CONFIGPATH="configuration/40_configuration.yaml"
CONFIGPATHANALYSIS="configuration/40_analysis_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $SLURM_ARRAY_TASK_ID