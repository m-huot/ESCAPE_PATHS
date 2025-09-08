#!/bin/sh
#SBATCH -p shakhnovich,shared   # Partition to submit to
#SBATCH -N 1     # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4     # cores requested
#SBATCH --mem=2000  # memory in Mb
#SBATCH -o RBM_outfile  # send stdout to outfile
#SBATCH -e RBM_errfile  # send stderr to errfile
#SBATCH -t 24:00:00  # time requested in hour:minute:second

source activate lantern
cd ../lattice

# Run the Python script with the specified parameters
python3 script_test_lattice_path.py --n_path 100

echo 
