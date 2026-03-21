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

from importlib import reload, import_module

from path import *

import lattice
from utils_rbm_mean_field import *
from utils_mean_field import *


def main(args):
    RBM = RBM_utils.loadRBM("../lattice/rbm/RBM_structure_0_beta_100")

    PROTEIN_INIT = Proteins_utils.load_FASTA(
        "../lattice/msa/output_msa_structure_0_beta_1000.fasta"
    )[0]

    POS_CHARGE = {"K", "R", "H"}
    NEG_CHARGE = {"D", "E"}

    sites = np.array([9, 10, 11, 12, 13, 16, 17, 25, 26]) - 1
    print("sites upper face", sites)
    w_bias = np.zeros((PROTEIN_INIT.shape[0], 20, 1))
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

            if i != wt_idx and is_charge_flip:
                w_bias[site, i, 0] = 1

    L = PROTEIN_INIT.shape[0]
    Q = 20
    T = args.T + 1

    Q_C = args.D
    beta_rbm = args.beta_rbm

    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))
    g = np.array(np.expand_dims(RBM.vlayer.fields[:, :], axis=-1))
    print("g", g.shape)
    gamma_f_list = create_gamma_functions(
        torch.tensor(RBM.hlayer.gamma_plus),
        torch.tensor(RBM.hlayer.gamma_minus),
        torch.tensor(RBM.hlayer.theta_plus),
        torch.tensor(RBM.hlayer.theta_minus),
        L,
        beta_rbm=beta_rbm,
    )

    all_s_functions = []
    w_components = []
    name_array = []

    w_components.append(wgamma)
    all_s_functions.extend(gamma_f_list)
    name_array.extend(["gamma " + str(i) for i in range(len(gamma_f_list))])

    w = torch.tensor(np.concatenate(w_components, axis=-1))

    def escape_function(G_t):
        epsilon = 0.05
        return -torch.log(1 - torch.exp(-epsilon - G_t)) * args.beta_ab

    all_s_functions.append(escape_function)
    name_array.append("escape")
    w = torch.cat((w, torch.tensor(w_bias)), dim=-1)

    K = len(name_array)

    print(f"Number of functions: {K}")

    q = torch.zeros(T - 1, requires_grad=True)

    m_0 = torch.zeros(K)

    for k in range(K):
        m_0[k] = sum(w[i, PROTEIN_INIT[i], k] for i in range(len(PROTEIN_INIT)))  # / L

    mint = torch.zeros(T, K)
    for t in range(1, T):
        mint[t] = m_0
    m = torch.zeros(T, K, dtype=torch.float64, requires_grad=True)
    m_updated = m.clone()

    m_updated[0] = m_0

    for t in range(1, T):
        m_updated[t] = m_0

    m = m_updated.clone().detach().requires_grad_(True)

    if torch.isnan(m).any():
        print("m has nan at init")
        raise ValueError("m has nan at init")

    if torch.isnan(w).any():
        print("w has nan at init")
        raise ValueError("w has nan at init")
    mopt, qopt, m_array, q_array, dm_array, dq_array, qhat_array, mhat_array = (
        GradDescent_free(
            mstart=m,
            qstart=q,
            w=w,
            g=g,
            gamma_f_list=all_s_functions,
            beta_rbm=args.beta_rbm,
            beta_phi=args.beta_phi,
            niter=args.N_ITER,
            eps=args.EPS,
            L=L,
            Q=Q,
            Q_C=Q_C,
            protein_init=PROTEIN_INIT,
            MU=args.MU,
            phi=args.phi,
        )
    )

    os.chdir("../mean_field_theory")
    folder = args.folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(f"{folder}/mopt.npy", mopt.detach().numpy())
    np.save(f"{folder}/qopt.npy", qopt.detach().numpy())
    print("mopt", mopt)
    print("mopt", mopt.shape)
    m_array = torch.stack(m_array)
    m_array = m_array.detach().numpy()
    print("m_array", m_array.shape)
    np.save(f"{folder}/m_array.npy", m_array)

    q_array = torch.stack(q_array)
    q_array = q_array.detach().numpy()
    print("q_array", q_array.shape)
    np.save(f"{folder}/q_array.npy", q_array)

    dm_array = torch.stack(dm_array)
    dm_array = dm_array.detach().numpy()
    print("dm_array", dm_array.shape)
    np.save(f"{folder}/dm_array.npy", dm_array)

    dq_array = torch.stack(dq_array)
    dq_array = dq_array.detach().numpy()
    print("dq_array", dq_array.shape)
    np.save(f"{folder}/dq_array.npy", dq_array)

    qhat_array = torch.stack(qhat_array)
    qhat_array = qhat_array.detach().numpy()
    print("qhat_array", qhat_array.shape)
    np.save(f"{folder}/qhat_array.npy", qhat_array)

    mhat_array = torch.stack(mhat_array)
    mhat_array = mhat_array.detach().numpy()
    print("mhat_array", mhat_array.shape)
    np.save(f"{folder}/mhat_array.npy", mhat_array)

    data_numpy = q_array

    plt.figure(figsize=(10, 6))
    for i in range(data_numpy.shape[1]):
        plt.plot(data_numpy[:, i], alpha=0.75, label=f"q_{i + 1}", linewidth=2)

    plt.axhline(
        y=Q_C,
        color="green",
        linestyle=":",
        linewidth=2,
        label="Q (hard wall)",
    )
    plt.axhline(
        y=data_numpy[-1],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="q_i",
    )

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Q", fontsize=14)
    plt.title("Q evolution Over Iterations", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(args.folder + "/Q_Evolution_lattice.png", dpi=300)

    plt.show()

    m_array_np = m_array

    iterations = len(m_array_np)

    rows = (K + 3) // 4
    cols = min(K, 4)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 6 * rows))

    axes = axes.flatten() if K > 1 else [axes]

    for k in range(K):
        for row in range(m_array_np[0].shape[0]):
            row_values = [m_array_np[i][row, k] for i in range(iterations)]
            axes[k].plot(
                range(iterations),
                row_values,
                marker="o",
                label=f"m {row}",
                alpha=0.8,
                markersize=4,
            )

        axes[k].set_title(name_array[k], fontsize=14)
        axes[k].set_xlabel("Iterations", fontsize=12)
        axes[k].set_ylabel("m", fontsize=12)
        axes[k].legend(fontsize=10)
        axes[k].grid(True, linestyle="--", alpha=0.6)

    for idx in range(K, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.folder + "/m_Evolution_lattice.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean field implementation lattice model."
    )
    parser.add_argument("--T", type=int, default=6, help="Number of time steps")
    parser.add_argument("--N_ITER", type=int, default=200, help="Number of iterations")
    parser.add_argument("--EPS", type=float, default=0.05, help="Step size")
    parser.add_argument("--D", type=float, default=6, help="Gamma coefficient")

    parser.add_argument(
        "--folder", type=str, default="results_script", help="Output folder"
    )
    parser.add_argument("--beta_rbm", type=float, default=1, help="Beta rbm")
    parser.add_argument("--beta_phi", type=float, default=1, help="Beta phi")
    parser.add_argument("--MU", type=float, default=0.01, help="Mutation rate phi evo")
    parser.add_argument(
        "--phi",
        type=str,
        choices=["cont", "evo"],
        default="cont",
        help="Phi continuity type",
    )
    parser.add_argument("--beta_ab", type=float, default=1, help="Immune pressure")

    args = parser.parse_args()
    main(args)
