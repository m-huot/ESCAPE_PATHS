import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils_mean_field import *
from script_mean_field_covid import *

main_path = "../"
sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
sys.path.append(main_path + "covid/")

import utilities, Proteins_utils, sequence_logo, plots_utils
from global_variables import *
from utils_evaluate_seq import *


def main(args):
    results_root = args.folder
    os.chdir(main_path + "mean_field_theory")

    if args.init == "wt":
        PROT_INIT = Proteins_utils.load_FASTA(
            main_path + "covid/exp_data/wt_omicron.fasta"
        )[0][BEGIN:-END]
    elif args.init == "ba1":
        PROT_INIT = Proteins_utils.load_FASTA(
            main_path + "covid/exp_data/wt_omicron.fasta"
        )[1][BEGIN:-END]
    else:
        raise ValueError("Invalid init option. Choose 'wt' or 'ba1'.")

    L = PROT_INIT.shape[0]
    Q = 20
    g = np.expand_dims(RBM.vlayer.fields[:, :], axis=-1)

    ab_names = list(ESCAPE_VECTORS.keys())
    wab = np.zeros((L, Q, len(ab_names)))
    for idx, ab in enumerate(ab_names):
        w_ab = ESCAPE_VECTORS[ab].reshape(L, Q)
        wab[:, :, idx] = w_ab

    for i in range(len(ab_names)):
        if np.any(wab[:, :, i] > 0):
            raise ValueError(f"Escape vector {ab_names[i]} has positive coeffs")

    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))

    w_components = []

    w_components.append(wab)

    w_components.append(wgamma)

    w = torch.tensor(np.concatenate(w_components, axis=-1))

    paths = [
        name
        for name in os.listdir(args.folder)
        if os.path.isdir(os.path.join(args.folder, name))
    ]

    rbm_betas = np.ones(len(paths)) * args.beta_rbm

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
        "--folder", type=str, default="results_scripts/covid_D_20", help="Input folder"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="results_scripts/covid_D_20",
        help="Output folder",
    )
    parser.add_argument("--beta_rbm", type=float, default=1, help="Beta rbm")
    parser.add_argument("--init", type=str, default="wt", help="Init protein")

    args = parser.parse_args()
    main(args)
