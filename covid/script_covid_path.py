import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import trange
from copy import deepcopy
import pandas as pd

import sys

sys.path.extend(["../PGM/source/", "../PGM/utilities/", "../path/"])

import Proteins_utils
from path import *
from global_variables import BEGIN, END, RBM, CODE_RBM
from utils_evaluate_seq import get_ab_energy


# ------------- MAIN SCRIPT ------------------
def main(args):
    PROT_INIT = Proteins_utils.load_FASTA("exp_data/wt_omicron.fasta")[0]
    PROT_INIT = PROT_INIT[BEGIN:-END]
    L = PROT_INIT.shape[0]

    class customed_covid_rbm:
        def __init__(self, rbm_model, beta_w=0):
            self.rbm_model = rbm_model
            self.beta_w = beta_w

        def __call__(self, v):
            score = -self.rbm_model.free_energy(v)[0] - get_ab_energy(v) * self.beta_w

            return score

    def P_fixation(v, v_prime):
        return 0

    path = Path(
        v_start=PROT_INIT,
        beta_pi=1,
        beta_rbm=1,
        Pi=P_fixation,
        Rbm=customed_covid_rbm(RBM, beta_w=args.beta_w),
        code=CODE_RBM,
        T=args.T,
    )

    output_dir = f"paths/test_covid_T{args.T}_beta_w_{args.beta_w}/"
    path.generate_paths(
        output_directory=output_dir,
        warming_steps=args.warming_steps,
        sampling_steps=args.sampling_steps,
        paths_nb=args.n_path,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampling_steps", type=int, default=3000, help="Recording interval"
    )
    parser.add_argument(
        "--warming_steps", type=int, default=10000, help="Number of MCMC warming steps"
    )
    parser.add_argument(
        "--n_path", type=int, default=100, help="Number of paths to generate"
    )
    parser.add_argument(
        "--T", type=int, default=19, help="Temperature for the path generation"
    )
    parser.add_argument(
        "--beta_w", type=float, default=1, help="Weight for the additional energy term"
    )

    args = parser.parse_args()
    main(args)
