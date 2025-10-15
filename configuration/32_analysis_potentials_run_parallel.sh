#!/bin/sh

#SBATCH --job-name=32_analysis_potentials_waves
#SBATCH --output=logs/%x-%j.log
#SBATCH --array=0-99
#SBATCH --requeue
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard
#SBATCH --time=20:00:00

# python script
PYTHONSCRIPT="scripts/analysis/flow_analysis_cortical.py"
CONFIGPATH="configuration/32_configuration.yaml"
CONFIGPATHANALYSIS="configuration/32_analysis_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS $SLURM_ARRAY_TASK_ID