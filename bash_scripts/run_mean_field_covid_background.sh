#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # Request one node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 23:00:00                   # Walltime per task
#SBATCH -o mf_background_outfile.out  # Standard output
#SBATCH -e mf_background_errfile.err  # Standard error
#SBATCH --array=0-4                   # 7 backgrounds → 5 tasks

# Activate conda environment
source activate lantern

BACKGROUNDS=("WT" "Alpha" "Delta" "BA1" "BA2")

background=${BACKGROUNDS[$SLURM_ARRAY_TASK_ID]}

beta_ab=1
D=20
T=20

folder_name="mean_f_free_ab_beta_ab_${beta_ab}_D_${D}_bg_${background}"

cd ../mean_field_theory

python3 script_mean_field_covid_background.py \
    --folder ${folder_name} \
    --beta_ab ${beta_ab} \
    --T ${T} \
    --D ${D} \
    --background ${background}

echo "Task ${SLURM_ARRAY_TASK_ID} completed: beta_ab=${beta_ab}, D=${D}, background=${background}."
