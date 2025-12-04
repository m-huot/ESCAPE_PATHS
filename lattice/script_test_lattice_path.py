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


class customed_lattice:
    def __init__(self, Lattice_model, w=None, beta_ab=1, beta_lattice=1000):
        self.Lattice_model = Lattice_model
        self.w = w
        self.beta_ab = beta_ab
        self.beta_lattice = beta_lattice

    def __call__(self, v):
        v_string = Proteins_utils.num2seq(v)
        v_transfo = lattice.string2seq(v_string)
        score = np.log(self.Lattice_model(v_transfo)) * self.beta_lattice
        if self.w is not None:
            epsilon = 0.01  # min binding
            score += (
                np.log(1 - np.exp(-epsilon - np.dot(one_hot_encode_concat(v), self.w)))
            ) * self.beta_ab
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
        default=100,
        help="Number of steps between samples",
    )
    args = parser.parse_args()

    PROTEIN_INIT = Proteins_utils.load_FASTA(
        "msa/output_msa_structure_0_beta_1000.fasta"
    )[0]

    protein_lattice = lattice.ProteinLattice(structure_idx=0)
    lattice_model = lattice.LatticeModel(protein_lattice)

    sites = np.array([9, 10, 11, 12, 13, 16, 17, 25, 26]) - 1
    w = np.zeros((PROTEIN_INIT.shape[0] * 20))
    for site in sites:
        wt_aa = PROTEIN_INIT[site]
        for i in range(20):
            if i != wt_aa:
                w[site * 20 + i] = 1

    beta_array = [0, 0.5, 1, 3, 5, 10]

    for T in [6]:
        for beta_ab in beta_array:
            customed_model = customed_lattice(
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

            output_dir = f"paths/test_w_upper_T{T}_beta_ab_{beta_ab}/"
            path.generate_paths(
                output_directory=output_dir,
                warming_steps=args.warming_steps,
                sampling_steps=args.sampling_steps,
                paths_nb=args.n_path,
                verbose=False,
            )


if __name__ == "__main__":
    main()
