#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # Request one node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 23:00:00                   # Walltime per task
#SBATCH -o mf_lattice_outfile         # Standard output
#SBATCH -e mf_lattice_errfile         # Standard error
#SBATCH --array=0-2                  # 24 tasks (3 beta_w × 2 T × 4 beta_phi)

# Activate the conda environment
source activate lantern

# Define parameter arrays
BETA_W_VALUES=(1 2 3)
T_VALUES=(6)
BETA_PHI_VALUES=(1)

# Dimensions
NUM_T=${#T_VALUES[@]}
NUM_PHI=${#BETA_PHI_VALUES[@]}

# Compute parameter indices
beta_w_index=$(( SLURM_ARRAY_TASK_ID / (NUM_T * NUM_PHI) ))
T_index=$(( (SLURM_ARRAY_TASK_ID / NUM_PHI) % NUM_T ))
beta_phi_index=$(( SLURM_ARRAY_TASK_ID % NUM_PHI ))

# Retrieve actual values
beta_w=${BETA_W_VALUES[$beta_w_index]}
T=${T_VALUES[$T_index]}
beta_phi=${BETA_PHI_VALUES[$beta_phi_index]}

# Construct output folder name
folder_name="mean_f_lattice_T_${T}_beta_w_${beta_w}_beta_phi_${beta_phi}"

# Navigate to script directory
cd ../mean_field_theory

# Run the Python script
python3 script_mean_field_lattice.py --T ${T} --beta_w ${beta_w} --beta_phi ${beta_phi} --folder ${folder_name} --D ${T}

echo "Task ${SLURM_ARRAY_TASK_ID} completed: T=${T}, beta_w=${beta_w}, beta_phi=${beta_phi}."

