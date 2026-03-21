import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from utils_mean_field import *

# Global imports
main_path = "../"
sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
sys.path.append(main_path + "covid/")
os.chdir(main_path + "covid/")

from global_variables import *
from utils_evaluate_seq import *

os.chdir("../mean_field_theory")

from utils_rbm_mean_field import *

import utilities, Proteins_utils, sequence_logo, plots_utils


def main(args):
    os.chdir(main_path + "covid/")

    PROT_INIT = Proteins_utils.load_FASTA("../covid/exp_data/wt_omicron.fasta")[0][
        BEGIN:-END
    ]
    L = PROT_INIT.shape[0]
    Q = 20
    T = args.T + 1
    Q_C = args.D

    ab_names = list(ESCAPE_VECTORS.keys())
    wab = np.zeros((L, Q, len(ab_names)))
    for idx, ab in enumerate(ab_names):
        wab[:, :, idx] = ESCAPE_VECTORS[ab].reshape(L, Q)

    for i in range(len(ab_names)):
        if np.any(wab[:, :, i] > 0):
            raise ValueError(f"Escape vector {ab_names[i]} has positive coeffs")

    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))
    g = np.expand_dims(RBM.vlayer.fields[:, :], axis=-1)
    gamma_f_list = create_gamma_functions(
        torch.tensor(RBM.hlayer.gamma_plus),
        torch.tensor(RBM.hlayer.gamma_minus),
        torch.tensor(RBM.hlayer.theta_plus),
        torch.tensor(RBM.hlayer.theta_minus),
        L,
        beta_rbm=args.beta_rbm,
    )

    all_s_functions = []
    w_components = []
    name_array = []

    ab_indices = [ab_names.index(name) for name in args.ab_list]
    betas_ab = np.zeros(len(ab_names))
    for i in ab_indices:
        betas_ab[i] = args.beta_ab

    ab_function_list = create_ab_functions(len(ab_names), betas_ab, L=L)
    all_s_functions.extend(ab_function_list)
    w_components.append(wab)
    name_array.extend(ab_names)

    w_components.append(wgamma)
    all_s_functions.extend(gamma_f_list)
    name_array.extend(["gamma " + str(i) for i in range(len(gamma_f_list))])

    w = torch.tensor(np.concatenate(w_components, axis=-1))
    K = len(name_array)

    print(f"Number of functions: {K}")

    q = torch.ones(T - 1, requires_grad=True)

    m_0 = torch.zeros(K)

    for k in range(K):
        m_0[k] = sum(w[i, PROT_INIT[i], k] for i in range(len(PROT_INIT))) / L

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
            protein_init=PROT_INIT,
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
        y=1 - np.sum(1 - data_numpy[-1]),
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Path cumulated overlap (1 - sum(1 - q_i))",
    )

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Q", fontsize=14)
    plt.title("Q (site average overlap) evolution Over Iterations", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(args.folder + "/Q_Evolution_covid.png", dpi=300)

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
    plt.savefig(args.folder + "/m_Evolution_covid.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean field implementation using real COVID model."
    )
    parser.add_argument("--T", type=int, default=5, help="Number of time steps")
    parser.add_argument("--N_ITER", type=int, default=200, help="Number of iterations")
    parser.add_argument("--EPS", type=float, default=0.05, help="Step size")
    parser.add_argument("--D", type=float, default=5, help="Gamma coefficient")
    parser.add_argument(
        "--folder", type=str, default="results_script", help="Output folder"
    )
    parser.add_argument("--beta_ab", type=float, default=1, help="Beta ab")
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
    parser.add_argument(
        "--ab_list",
        nargs="+",
        type=str,
        default=[],
        help="List of antibody names to apply beta_ab to",
    )
    args = parser.parse_args()
    main(args)
