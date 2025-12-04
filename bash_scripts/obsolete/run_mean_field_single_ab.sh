#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # Request one node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 23:00:00                   # Walltime per task
#SBATCH -o mf_single_outfile.out  # Standard output
#SBATCH -e mf_single_errfile.err  # Standard error
#SBATCH --array=0-28                  # One task per antibody (29 in total)

# Activate the environment
source activate lantern

# Define your list of antibodies
AB_LIST=("COV2-2050" "COV2-2082" "COV2-2094" "COV2-2096" "COV2-2130"
         "COV2-2165" "COV2-2196" "COV2-2479" "COV2-2499" "COV2-2677"
         "COV2-2832" "CR3022" "LY-CoV016" "LY-CoV555" "REGN10933"
         "REGN10987" "S2D106" "S2E12" "S2H13" "S2H14" "S2H58"
         "S2H97" "S2X16" "S2X227" "S2X259" "S2X35" "S2X58"
         "S304" "S309")

# Parameters
beta_ab=1
D=10

# Get the antibody for this task
ab=${AB_LIST[$SLURM_ARRAY_TASK_ID]}

# Folder name
folder_name="mean_f_free_ab_beta_ab_${beta_ab}_D_${D}_ab_${ab}"

# Move to script directory
cd ../mean_field_theory

# Run the script
python3 script_mean_field_cocktail_covid.py \
    --folder "${folder_name}" \
    --beta_ab "${beta_ab}" \
    --T "${D}" \
    --D "${D}" \
    --ab_list "${ab}"

echo "Task ${SLURM_ARRAY_TASK_ID} completed for antibody: ${ab}"
