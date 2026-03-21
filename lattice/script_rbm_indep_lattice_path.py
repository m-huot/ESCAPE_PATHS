import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import os, sys

main_path = "../"

sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
import utilities, Proteins_utils, sequence_logo, plots_utils, RBM_utils

sys.path.append(main_path + "path/")
from importlib import reload, import_module
from path import *
import lattice
from Site_indep_model import *

CODE_LATTICE = [
    "C",
    "M",
    "F",
    "I",
    "L",
    "V",
    "W",
    "Y",
    "A",
    "G",
    "T",
    "S",
    "N",
    "Q",
    "D",
    "E",
    "H",
    "R",
    "K",
    "P",
]

CODE_RBM = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def one_hot_encode_concat(s):
    num_categories = 20
    one_hot_matrix = np.zeros((len(s), num_categories))
    one_hot_matrix[np.arange(len(s)), s.astype(int)] = 1
    return one_hot_matrix.flatten()


def P_fixation(v, v_prime):
    return 0


RBM = RBM_utils.loadRBM("rbm/RBM_structure_0_beta_100")
msa = Proteins_utils.load_FASTA("msa/output_msa_structure_0_beta_100.fasta")
INDEP_MODEL = SiteIndependentModel()
INDEP_MODEL.fit(msa)


class customed_lattice_RBM:
    def __init__(self, Lattice_model, w=None, beta_ab=1, beta_lattice=1000):
        self.w = w
        self.beta_ab = beta_ab

    def __call__(self, v):
        score = RBM.likelihood(v)
        if self.w is not None:
            epsilon = 0.05  # min binding
            score += (
                np.log(1 - np.exp(-epsilon - np.dot(one_hot_encode_concat(v), self.w)))
            ) * self.beta_ab
        if isinstance(score, np.ndarray):
            score = score.item()
        return score


class customed_lattice_indep:
    def __init__(self, Lattice_model, w=None, beta_ab=1, beta_lattice=1000):
        self.w = w
        self.beta_ab = beta_ab

    def __call__(self, v):
        score = INDEP_MODEL.log_prob(v)
        if self.w is not None:
            epsilon = 0.05  # min binding
            score += (
                np.log(1 - np.exp(-epsilon - np.dot(one_hot_encode_concat(v), self.w)))
            ) * self.beta_ab
        if isinstance(score, np.ndarray):
            score = score.item()
        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_path", type=int, default=100, help="Number of paths to generate"
    )
    parser.add_argument(
        "--warming_steps", type=int, default=1000, help="Number of MCMC warming steps"
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=400,
        help="Number of steps between samples",
    )
    args = parser.parse_args()

    PROTEIN_INIT = Proteins_utils.load_FASTA(
        "msa/output_msa_structure_0_beta_1000.fasta"
    )[0]

    protein_lattice = lattice.ProteinLattice(structure_idx=0)
    lattice_model = lattice.LatticeModel(protein_lattice)

    POS_CHARGE = {"K", "R", "H"}
    NEG_CHARGE = {"D", "E"}

    sites = np.array([9, 10, 11, 12, 13, 16, 17, 25, 26]) - 1
    w = np.zeros((PROTEIN_INIT.shape[0] * 20))

    for site in sites:
        wt_idx = PROTEIN_INIT[site]
        wt_char = CODE_RBM[wt_idx]

        for i in range(20):
            mut_char = CODE_RBM[i]

            wt_is_pos = wt_char in POS_CHARGE
            wt_is_neg = wt_char in NEG_CHARGE
            mut_is_pos = mut_char in POS_CHARGE
            mut_is_neg = mut_char in NEG_CHARGE

            is_charge_flip = (wt_is_pos and mut_is_neg) or (wt_is_neg and mut_is_pos)

            # Check if mutation exists and if it is a charge flip
            if i != wt_idx and is_charge_flip:
                w[site * 20 + i] = 1

    beta_array = [10, 5, 1]

    for T in [10]:
        for beta_ab in beta_array:
            customed_model = customed_lattice_RBM(
                lattice_model, w=w, beta_ab=beta_ab, beta_lattice=1000
            )
            path = Path(
                v_start=PROTEIN_INIT,
                beta_pi=1,
                beta_rbm=1,
                Pi=P_fixation,
                Rbm=customed_model,
                code=CODE_RBM,
                T=T,
            )

            output_dir = f"paths/w_upper_T{T}_beta_ab_{beta_ab}/"
            path.generate_paths(
                output_directory=output_dir,
                warming_steps=args.warming_steps,
                sampling_steps=args.sampling_steps,
                paths_nb=args.n_path,
                verbose=False,
            )

            customed_model = customed_lattice_indep(
                lattice_model, w=w, beta_ab=beta_ab, beta_lattice=1000
            )
            path = Path(
                v_start=PROTEIN_INIT,
                beta_pi=1,
                beta_rbm=1,
                Pi=P_fixation,
                Rbm=customed_model,
                code=CODE_RBM,
                T=T,
            )

            output_dir = f"paths/indep_w_upper_T{T}_beta_ab_{beta_ab}/"
            path.generate_paths(
                output_directory=output_dir,
                warming_steps=args.warming_steps,
                sampling_steps=args.sampling_steps,
                paths_nb=args.n_path,
                verbose=False,
            )


if __name__ == "__main__":
    main()
