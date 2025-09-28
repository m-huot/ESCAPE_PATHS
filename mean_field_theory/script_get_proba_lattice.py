# %% [markdown]
# # Path lattice
# A path = consecutive sequences

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

import os, sys

main_path = "../"

sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
import utilities, Proteins_utils, sequence_logo, plots_utils, RBM_utils

sys.path.append(main_path + "path/")
sys.path.append(main_path + "lattice/")

# importlib

from importlib import reload, import_module

from path import *

import lattice
from utils_rbm_mean_field import *


# %% [markdown]
# # Parameters


def main(args):
    # %%
    RBM = RBM_utils.loadRBM("../lattice/rbm/RBM_structure_0_beta_1000")
    results_root = args.folder

    # %%
    PROT_INIT = Proteins_utils.load_FASTA(
        "../lattice/msa/output_msa_structure_0_beta_1000.fasta"
    )[0]  # Load the protein sequence from the FASTA file.

    sites = np.array([9, 10, 11, 12, 13, 16, 17, 25, 26]) - 1
    print("sites upper face", sites)
    w_bias = np.zeros((PROT_INIT.shape[0], 20, 1))
    for site in sites:
        wt_aa = PROT_INIT[site]
        print("wt_aa", Proteins_utils.num2seq(np.array([wt_aa])))
        # put a 1 everywhere except for the wt_aa
        for i in range(20):
            if i != wt_aa:
                w_bias[site, i, 0] = 1

    L = PROT_INIT.shape[0]
    Q = 20

    beta_rbm = args.beta_rbm

    # RBM weights
    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))
    g = np.array(np.expand_dims(RBM.vlayer.fields[:, :], axis=-1))
    print("g", g.shape)
    gamma_f_list = create_gamma_functions(
        torch.tensor(RBM.hlayer.gamma_plus),
        torch.tensor(RBM.hlayer.gamma_minus),
        torch.tensor(RBM.hlayer.theta_plus),
        torch.tensor(RBM.hlayer.theta_minus),
        beta_rbm=beta_rbm,
    )

    all_s_functions = []
    w_components = []
    name_array = []

    w_components.append(wgamma)
    all_s_functions.extend(gamma_f_list)
    name_array.extend(["gamma " + str(i) for i in range(len(gamma_f_list))])

    w = torch.tensor(np.concatenate(w_components, axis=-1))

    # add escape
    def escape_function(G_t):
        epsilon = 0.01
        return -torch.log(1 - torch.exp(-epsilon - G_t))

    all_s_functions.append(escape_function)
    name_array.append("escape")
    w = torch.cat((w, torch.tensor(w_bias)), dim=-1)

    # Define folders to read from
    paths = [
        name
        for name in os.listdir(args.folder)
        if os.path.isdir(os.path.join(args.folder, name))
    ]

    rbm_betas = np.ones(len(paths)) * args.beta_rbm

    # Compute and save probabilities
    for j, folder in enumerate(paths):
        qhat_path = os.path.join(results_root, folder, "qhat_array.npy")
        mhat_path = os.path.join(results_root, folder, "mhat_array.npy")
        qhat_array = np.load(qhat_path)[-1]
        mhat_array = np.load(mhat_path)[-1]

        path_data = []
        for i in tqdm(range(PROT_INIT.shape[0]), desc=f"Processing {folder}"):
            f = compute_frequency(
                Q,
                qhat_array,
                mhat_array,
                w[i],
                g[i],
                v1_fixed=PROT_INIT[i],
                vT_fixed=None,
                beta_rbm=rbm_betas[j],
                free=True,
            )
            a = f.detach().numpy()[:, :, 0]
            path_data.append(a)

        path_data = np.array(path_data).T
        output_path = os.path.join(args.out_folder, f"{folder}.npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, path_data)

    print("All frequency paths saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean field AA frequency computation for COVID paths"
    )
    parser.add_argument(
        "--folder", type=str, default="results_scripts/lattice", help="Input folder"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="results_scripts/lattice",
        help="Output folder",
    )
    parser.add_argument("--beta_rbm", type=float, default=1, help="Beta rbm")

    args = parser.parse_args()
    main(args)
