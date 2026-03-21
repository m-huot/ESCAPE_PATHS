#!/bin/bash
#SBATCH --job-name=seq_scoring_esmif_gpu         # Job name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of cores per node
#SBATCH --gpus=1                             # No GPUs requested
#SBATCH --time=6:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=20000  # memory in Mb
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH --partition=gpu_test                # Partition name


# Set the library path to the known location of the libpython2.7.so.1.0

# Set the library path





echo "Activating esmfold environment..."
source activate esmfold 


echo "Running Python script..."

python3 score_log_likelihoods.py data/6m0j.pdb \
    data/desai_ace2.fasta --chain E \
    --outpath output/desai_ace2_esmif.csv 
    --multichain-backbone
    
=