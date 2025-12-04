#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # Request one node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 23:00:00                   # Walltime per task
#SBATCH -o mf_covid_outfile          # Standard output
#SBATCH -e mf_covid_errfile          # Standard error
#SBATCH --array=0-23                 # 16 tasks (4 beta_ab × 4 D values)

# Activate the environment
source activate lantern

# Define parameter arrays
BETA_AB_VALUES=(0 0.5 1 3 5 10)
D_VALUES=(0.5 1 2 3)

# Total D values
NUM_D=${#D_VALUES[@]}

# Compute indices
beta_ab_index=$(( SLURM_ARRAY_TASK_ID / NUM_D ))
D_index=$(( SLURM_ARRAY_TASK_ID % NUM_D ))

# Retrieve parameter values
beta_ab=${BETA_AB_VALUES[$beta_ab_index]}
D=${D_VALUES[$D_index]}

# Construct folder name
folder_name="results_scripts/covid_D_20/mean_f_free_ab_beta_ab_${beta_ab}_D_${D}"

# Change to the directory containing the Python script
cd ../mean_field_theory

# Run the Python script
python3 script_mean_field_covid.py --folder ${folder_name} --beta_ab ${beta_ab} --T 20 --D ${D} --init 'wt'
echo "Task ${SLURM_ARRAY_TASK_ID} completed: beta_ab=${beta_ab}, D=${D}."
