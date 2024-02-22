#!/bin/bash
#SBATCH --job-name=generate_ground_truth_simulation
#SBATCH --output=generate_ground_truth_simulation.out
#SBATCH --error=generate_ground_truth_simulation.err
#SBATCH --partition=compute
#SBATCH --account=mh0010
#SBATCH --time=7:00:00

# Load environment
source activate /work/mh0010/m300408/envs/cpd  # noqa: SC1091

# Change to the working directory
cd /work/mh0010/m300408/cold_pool_detection || exit

# Run the DVC stage
dvc repro -s -f generate_ground_truth_simulation

# END
