#!/bin/bash
# Submit consecutive job arrays

total_tasks=3100  # Total number of jobs
max_array_size=2000  # MaxSizeArray limit of cluster

for ((start_idx = 0; start_idx < total_tasks; start_idx += max_array_size)); do
    end_idx=$((start_idx + max_array_size - 1))
    if ((end_idx > total_tasks)); then
        end_idx=$((total_tasks-1))
    fi
    
    array_size=$((end_idx - start_idx + 1))
    
    sbatch --array=0-$array_size ./configuration/03_analysis_run_parallel.sh ${start_idx}
done