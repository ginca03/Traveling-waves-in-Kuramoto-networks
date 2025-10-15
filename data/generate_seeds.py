import numpy as np
import pandas as pd
import os
import yaml
import sys

# --- Usage ---
# python generate_seeds.py path/to/01_configuration.yaml

if len(sys.argv) < 2:
    print("Usage: python generate_seeds.py path/to/config.yaml")
    sys.exit(1)

config_file = sys.argv[1]

# Load YAML config
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.BaseLoader)

data_path = config["data_path"]       # should already be an absolute path in your YAML
experiment_id = config["experiment_id"]

# Parse simulation id range [start, end, step]
sim_id_start, sim_id_end, sim_id_step = np.array(config["simulation_id"], dtype=int)
n_simulations = len(np.arange(sim_id_start, sim_id_end, sim_id_step))

# Build full path for output
outdir = os.path.join(data_path, f"{experiment_id}_analysis")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{experiment_id}_random_numbers.csv")

# Check if file exists
if os.path.exists(outfile):
    ans = input(f"⚠️  {outfile} already exists. Overwrite? [y/N] ").strip().lower()
    if ans != "y":
        print("❌ Aborted, file not overwritten.")
        sys.exit(0)

# Generate seeds
seeds = np.random.randint(0, 1e9, size=n_simulations)

# Save CSV (no header, one seed per line)
pd.DataFrame(seeds).to_csv(outfile, index=False, header=False)

print(f"✅ Created {outfile} with {n_simulations} seeds.")
