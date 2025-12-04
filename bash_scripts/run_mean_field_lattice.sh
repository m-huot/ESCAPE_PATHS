#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # Request one node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 23:00:00                   # Walltime per task
#SBATCH -o mf_lattice_outfile         # Standard output
#SBATCH -e mf_lattice_errfile         # Standard error
#SBATCH --array=0-23                  # 24 tasks (3 beta_ab × 2 T × 4 beta_phi)

# Activate the conda environment
source activate lantern

# Define parameter arrays
BETA_AB_VALUES=(0 0.5 1 3 5 10)
D_VALUES=(0.5 1 2 3)
BETA_PHI_VALUES=(1)

# Dimensions
NUM_D=${#D_VALUES[@]}
NUM_PHI=${#BETA_PHI_VALUES[@]}

# Compute parameter indices
beta_ab_index=$(( SLURM_ARRAY_TASK_ID / (NUM_D * NUM_PHI) ))
D_index=$(( (SLURM_ARRAY_TASK_ID / NUM_PHI) % NUM_D ))
beta_phi_index=$(( SLURM_ARRAY_TASK_ID % NUM_PHI ))

# Retrieve actual values
beta_ab=${BETA_AB_VALUES[$beta_ab_index]}
D=${D_VALUES[$D_index]}
beta_phi=${BETA_PHI_VALUES[$beta_phi_index]}

# Construct output folder name
folder_name="results_scripts/lattice/mean_f_lattice_D_${D}_beta_ab_${beta_ab}"

# Navigate to script directory
cd ../mean_field_theory

# Run the Python script
python3 script_mean_field_lattice.py --T 10 --beta_ab ${beta_ab} --beta_phi ${beta_phi} --folder ${folder_name} --D ${D}

echo "Task ${SLURM_ARRAY_TASK_ID} completed: D=${D}, beta_ab=${beta_ab}."

