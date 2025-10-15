#!/bin/sh

# python script
PYTHONSCRIPT="scripts/simulation/06_simulation.py"
CONFIGPATH="configuration/06_configuration.yaml"

# activate conda environment
eval "$($(which conda) shell.bash hook)"
conda activate tvb-waves

# build gnu parallel command
parallel="parallel --delay .2 -j 50% --memsuspend 12G"

# run parallel
$parallel "python $PYTHONSCRIPT $CONFIGPATH {1}" ::: {0..99}

# parallel --delay .2 -j 50% --memsuspend 12G python "scripts/simulation/06_simulation.py" "configuration/06_configuration.yaml" {1}" ::: {0..99}