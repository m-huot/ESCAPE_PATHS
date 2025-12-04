import argparse
import os
import sys
import numpy as np
import pickle
from importlib import reload, import_module
from Bio.Seq import Seq

# --- PATH CONFIGURATION ---
# Determine the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming the structure is script/../ so we go up one level for main_path
MAIN_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

sys.path.append(os.path.join(MAIN_PATH, "PGM/source/"))
sys.path.append(os.path.join(MAIN_PATH, "PGM/utilities/"))
sys.path.append(os.path.join(MAIN_PATH, "path/"))

# --- IMPORTS ---
# These imports rely on the paths added above
import utilities
import Proteins_utils
import sequence_logo
import plots_utils
import RBM_utils

from path_nucleotide import PathNT, str_to_nts

from global_variables import *
from utils_evaluate_seq import *


# --- CONSTANTS ---
DEFAULT_NT_SEQ_STR = "TCTGTTTATGCTTGGAACAGGAAGAGAATCAGCAACTGTGTTGCTGATTATTCTGTCCTATATAATTCCGCATCATTTTCCACTTTTAAGTGTTATGGAGTGTCTCCTACTAAATTAAATGATCTCTGCTTTACTAATGTCTATGCAGATTCATTTGTAATTAGAGGTGATGAAGTCAGACAAATCGCTCCAGGGCAAACTGGAAAGATTGCTGATTATAATTATAAATTACCAGATGATTTTACAGGCTGCGTTATAGCTTGGAATTCTAACAATCTTGATTCTAAGGTTGGTGGTAATTATAATTACCTGTATAGATTGTTTAGGAAGTCTAATCTCAAACCTTTTGAGAGAGATATTTCAACTGAAATCTATCAGGCCGGTAGCACACCTTGTAATGGTGTTGAAGGTTTTAATTGTTACTTTCCTTTACAATCATATGGTTTCCAACCCACTAATGGTGTTGGTTACCAACCATACAGAGTAGTAGTACTTTCTTTTGAACTTCTACATGCACCAGCAACTGTTTGTGGA"


# --- HELPER CLASSES ---
class CustomedCovidRBM:
    """
    Wrapper for the RBM model to include antibody energy in the score.
    """

    def __init__(self, rbm_model, beta_ab=0):
        self.rbm_model = rbm_model
        self.beta_ab = beta_ab

    def __call__(self, v):
        # Calculates free energy and subtracts the antibody energy weighted by beta_ab
        score = -self.rbm_model.free_energy(v)[0] - get_ab_energy(v) * self.beta_ab
        return score


# --- MAIN FUNCTION ---
def main(args):
    # 1. Setup Data
    print(f"Initializing sequence processing...")
    nt_seq = Seq(args.sequence.replace("\n", ""))
    PROT_INIT = str_to_nts(nt_seq)

    # Verify RBM availability (loaded from global_variables)
    if "RBM" not in globals():
        raise NameError(
            "The variable 'RBM' was not found. Ensure it is defined in 'global_variables.py'."
        )

    # 2. Set Random Seeds
    np.random.seed(args.seed)

    # 3. Iterate through Beta values
    for beta_ab in args.beta_abs:
        print(f"--- Processing beta_ab: {beta_ab} ---")

        # Initialize the custom wrapper
        rbm_nt = CustomedCovidRBM(RBM, beta_ab=beta_ab)

        # Initialize Sampler
        print(f"Initializing PathNT sampler (T={args.time_horizon})...")
        sampler = PathNT(
            T=args.time_horizon,
            rbm_energy=rbm_nt,
            beta=args.beta,
            v_start=PROT_INIT,
            v_end=None,
            extrem="free",
            seed=args.seed + 100,  # Offset seed for internal sampler logic
        )

        # Define Output Path
        output_dir = os.path.join(
            args.output_base, f"test_nt_covid_T_{args.time_horizon}_beta_ab_{beta_ab}/"
        )

        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Run Sampling
        print(
            f"Starting sampling: {args.sampling_steps} steps after {args.warming_steps} warming steps."
        )
        sampler.sample_paths(
            output_dir,
            warming_steps=args.warming_steps,
            sampling_steps=args.sampling_steps,
            paths_nb=args.paths_nb,
        )
        print(f"Completed sampling for beta_ab {beta_ab}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RBM Path Sampling on Nucleotide Sequences."
    )

    # Simulation Parameters
    parser.add_argument(
        "--beta_abs",
        nargs="+",
        type=float,
        default=[1, 3, 5, 10],
        help="List of beta_ab values to iterate over (default: 1 3 5 10).",
    )
    parser.add_argument(
        "--time_horizon",
        "-T",
        type=int,
        default=20,
        help="Time horizon T for PathNT (default: 20).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature beta (default: 1.0).",
    )

    # Sampling Steps
    parser.add_argument(
        "--warming_steps",
        type=int,
        default=10000,
        help="Number of warming steps (default: 10000).",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=3000,
        help="Number of sampling steps (default: 3000).",
    )
    parser.add_argument(
        "--paths_nb",
        type=int,
        default=100,
        help="Number of paths to sample (default: 100).",
    )

    # Config
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="paths_nt",
        help="Base directory for output files (default: 'paths_nt').",
    )

    # Input Data (Optional override)
    parser.add_argument(
        "--sequence",
        type=str,
        default=DEFAULT_NT_SEQ_STR,
        help="The nucleotide sequence to process (defaults to hardcoded Covid sequence).",
    )

    args = parser.parse_args()

    main(args)
