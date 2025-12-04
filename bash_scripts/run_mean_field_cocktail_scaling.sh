#!/bin/bash
#SBATCH -p shakhnovich,sapphire         # Partitions to submit to
#SBATCH -N 1                          # One node
#SBATCH -n 1                          # One task per job
#SBATCH -c 1                          # One core per task
#SBATCH --mem=1000                    # Memory in MB per task
#SBATCH -t 10:00:00                   # Walltime
#SBATCH -o mf_cocktail_outfile.out  # Stdout per task
#SBATCH -e mf_cocktail_errfile.err  # Stderr per task
#SBATCH --array=0-6                   # 7 cumulative steps (1 beta x 1 D x 7 configs)


# Activate conda environment (robust method)
source activate lantern

# Define parameter arrays
BETA_AB_VALUES=(1)
D_VALUES=(1)

# Defined Order for Cumulative Cocktail
RAW_AB_ORDER=("REGN10933" "COV2-2196" "S2X16" "S2E12" "S2H58" "LY-CoV016" "S2H13")

# Build cumulative configurations
# Result will be: 
# 0: "REGN10933"
# 1: "REGN10933 COV2-2196"
# 2: "REGN10933 COV2-2196 S2X16" ... etc
AB_CONFIGS=()
current_combo=""

for ab in "${RAW_AB_ORDER[@]}"; do
  if [ -z "$current_combo" ]; then
    current_combo="$ab"
  else
    current_combo="$current_combo $ab"
  fi
  AB_CONFIGS+=("$current_combo")
done

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
folder_name="results_scripts/cocktail_scaling/mean_f_free_ab_beta_ab_${beta_ab}_D_${D}_ab_${ab_label}"

# Change to script directory
cd ../mean_field_theory

# Run Python script

python3 script_mean_field_cocktail_covid.py \
    --folder "$folder_name" \
    --beta_ab "$beta_ab" \
    --T 10 \
    --D "$D" \
    --ab_list $ab_string

echo "Task $SLURM_ARRAY_TASK_ID done: beta_ab=$beta_ab, D=$D, ab_list='$ab_string'"