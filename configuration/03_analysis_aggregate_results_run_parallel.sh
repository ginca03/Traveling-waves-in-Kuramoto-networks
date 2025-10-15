#!/bin/sh

#SBATCH --job-name=03_analysis_aggregate
#SBATCH --output=logs/%x-%j.log
#SBATCH --requeue
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --time=01:00:00

# python script
PYTHONSCRIPT="scripts/analysis/03_analysis_aggregate_results.py"
CONFIGPATH="configuration/03_configuration.yaml"
CONFIGPATHANALYSIS="configuration/03_analysis_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build srun command
srun="srun --exclusive -N1 -n1"

$srun python $PYTHONSCRIPT $CONFIGPATH $CONFIGPATHANALYSIS
