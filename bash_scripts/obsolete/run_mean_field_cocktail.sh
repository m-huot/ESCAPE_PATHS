#!/bin/bash
#SBATCH -p shakhnovich,shared         # Partitions to submit to
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=5000                    # Memory in MB per task
#SBATCH -t 10:00:00                   # Walltime
#SBATCH -o mf_cocktail_outfile.out  # Stdout per task
#SBATCH -e mf_cocktail_errfile.err  # Stderr per task
#SBATCH --array=0-28                  # 3 beta × 1 D × 5 AB configs = 15 tasks


# Activate conda environment (robust method)
source activate lantern

# Define parameter arrays
BETA_AB_VALUES=(1)
D_VALUES=(10)
AB_CONFIGS=(
"COV2-2050"
"COV2-2082"
"COV2-2094"
"COV2-2096"
"COV2-2130"
"COV2-2165"
"COV2-2196"
"COV2-2479"
"COV2-2499"
"COV2-2677"
"COV2-2832"
"CR3022"
"LY-CoV016"
"LY-CoV555"
"REGN10933"
"REGN10987"
"S2D106"
"S2E12"
"S2H13"
"S2H14"
"S2H58"
"S2H97"
"S2X16"
"S2X227"
"S2X259"
"S2X35"
"S2X58"
"S304"
"S309"
)


# Dimensions
NUM_BETA=${#BETA_AB_VALUES[@]}
NUM_D=${#D_VALUES[@]}
NUM_AB=${#AB_CONFIGS[@]}

# Compute indices
beta_ab_index=$(( SLURM_ARRAY_TASK_ID / (NUM_D * NUM_AB) ))
rem=$(( SLURM_ARRAY_TASK_ID % (NUM_D * NUM_AB) ))
D_index=$(( rem / NUM_AB ))
ab_index=$(( rem % NUM_AB ))

# Extract values
beta_ab="${BETA_AB_VALUES[$beta_ab_index]}"
D="${D_VALUES[$D_index]}"
ab_string="${AB_CONFIGS[$ab_index]}"
ab_label=$(echo "$ab_string" | tr ' ' '_')

# Output folder
folder_name="results_scripts/cocktail/mean_f_free_ab_beta_ba1_ab_${beta_ab}_D_${D}_ab_${ab_label}"

# Change to script directory
cd ../mean_field_theory

# Run Python script
python3 script_mean_field_cocktail_covid.py \
    --folder "$folder_name" \
    --beta_ab "$beta_ab" \
    --T "$D" \
    --D "$D" \
    --ab_list $ab_string

echo "Task $SLURM_ARRAY_TASK_ID done: beta_ab=$beta_ab, D=$D, ab_list='$ab_string'"
