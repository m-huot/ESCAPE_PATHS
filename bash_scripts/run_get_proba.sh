#!/bin/sh
#SBATCH -p shakhnovich,shared   # Partition to submit to
#SBATCH -N 1     # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1     # cores requested
#SBATCH --mem=2000  # memory in Mb
#SBATCH -o proba_outfile  # send stdout to outfile
#SBATCH -e proba_errfile  # send stderr to errfile
#SBATCH -t 4:00:00  # time requested in hour:minute:second

source activate lantern
cd ../mean_field_theory

# Run the Python script with the specified parameters
python3 script_get_proba.py --folder 'results_beta_rbm_1' --beta_rbm 1.
echo 
