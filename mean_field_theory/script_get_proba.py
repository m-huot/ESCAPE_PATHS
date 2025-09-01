import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_mean_field import *
from script_mean_field_covid import *
# from utils_analysis_mf import *

# Setup paths and imports
main_path = "../"
sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
sys.path.append(main_path + "covid/")

import utilities, Proteins_utils, sequence_logo, plots_utils
from global_variables import *
from utils_evaluate_seq import *


def main(args):
    results_root = args.folder
    # Set working directory to mean field
    os.chdir(main_path + "mean_field_theory")

    # Load WT sequence
    PROT_INIT = Proteins_utils.load_FASTA(
        main_path + "covid/exp_data/wt_omicron.fasta"
    )[0]
    PROT_INIT = PROT_INIT[BEGIN:-END]

    L = PROT_INIT.shape[0]
    Q = 20

    # Antibody escape vectors
    ab_names = list(ESCAPE_VECTORS.keys())
    wab = np.zeros((L, Q, len(ab_names)))
    for idx, ab in enumerate(ab_names):
        w_ab = ESCAPE_VECTORS[ab].reshape(L, Q)
        wab[:, :, idx] = w_ab

    # Validate escape vector signs
    for i in range(len(ab_names)):
        if np.any(wab[:, :, i] > 0):
            raise ValueError(f"Escape vector {ab_names[i]} has positive coeffs")

    # RBM weights and gamma functions
    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))
    g = np.expand_dims(RBM.vlayer.fields[:, :], axis=-1)
    gamma_f_list = create_gamma_functions(
        torch.tensor(RBM.hlayer.gamma_plus),
        torch.tensor(RBM.hlayer.gamma_minus),
        torch.tensor(RBM.hlayer.theta_plus),
        torch.tensor(RBM.hlayer.theta_minus),
    )

    all_s_functions = []
    w_components = []
    name_array = []

    ab_function_list = create_ab_functions(len(ab_names), 1)
    all_s_functions.extend(ab_function_list)
    w_components.append(wab)
    name_array.extend(ab_names)

    w_components.append(wgamma)
    all_s_functions.extend(gamma_f_list)
    name_array.extend(["gamma " + str(i) for i in range(len(gamma_f_list))])

    w = torch.tensor(np.concatenate(w_components, axis=-1))

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
        output_path = os.path.join(args.out_folder, f"{folder}a.npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, path_data)

    print("All frequency paths saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean field AA frequency computation for COVID paths"
    )
    parser.add_argument(
        "--folder", type=str, default="results_scripts/covid_D_20", help="Input folder"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="results_scripts/covid_D_20",
        help="Output folder",
    )
    parser.add_argument("--beta_rbm", type=float, default=1, help="Beta rbm")

    args = parser.parse_args()
    main(args)
