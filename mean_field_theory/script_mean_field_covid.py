import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from utils_mean_field import *

# Global imports (must be outside the function)
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

    # Load protein sequences
    PROT_INIT = Proteins_utils.load_FASTA("../covid/exp_data/wt_omicron.fasta")[0][
        BEGIN:-END
    ]

    L = PROT_INIT.shape[0]
    Q = 20
    T = args.T + 1

    # Constants
    GAMMA = args.D / L  # D/L
    Q_C = 1 - GAMMA / args.T

    # Antibody weights
    ab_names = list(ESCAPE_VECTORS.keys())
    wab = np.zeros((L, Q, len(ab_names)))
    for idx, ab in enumerate(ab_names):
        w_ab = ESCAPE_VECTORS[ab].reshape(L, Q)  # "no bias"
        wab[:, :, idx] = w_ab

    # make sure only<=0 coeffs
    for i in range(len(ab_names)):
        if np.any(wab[:, :, i] > 0):
            raise ValueError(f"Escape vector {ab_names[i]} has positive coeffs")

    # RBM weights
    wgamma = np.transpose(RBM.weights[:, :, :], (1, 2, 0))
    g = np.expand_dims(RBM.vlayer.fields[:, :], axis=-1)
    gamma_f_list = create_gamma_functions(
        torch.tensor(RBM.hlayer.gamma_plus),
        torch.tensor(RBM.hlayer.gamma_minus),
        torch.tensor(RBM.hlayer.theta_plus),
        torch.tensor(RBM.hlayer.theta_minus),
        beta_rbm=args.beta_rbm,
    )

    all_s_functions = []
    w_components = []
    name_array = []

    ab_function_list = create_ab_functions(len(ab_names), args.beta_ab)
    all_s_functions.extend(ab_function_list)
    w_components.append(wab)
    name_array.extend(ab_names)

    w_components.append(wgamma)
    all_s_functions.extend(gamma_f_list)
    name_array.extend(["gamma " + str(i) for i in range(len(gamma_f_list))])

    w = torch.tensor(np.concatenate(w_components, axis=-1))
    K = len(name_array)

    print(f"Number of functions: {K}")

    # Initialize variables for gradient descent
    q = torch.ones(T - 1, requires_grad=True)

    m_0 = torch.zeros(K)

    for k in range(K):
        m_0[k] = sum(w[i, PROT_INIT[i], k] for i in range(len(PROT_INIT))) / L

    mint = torch.zeros(T, K)
    for t in range(1, T):
        mint[t] = m_0
    # Initialize other m[t] values between m[0] and m[T-1]
    m = torch.zeros(T, K, dtype=torch.float64, requires_grad=True)
    m_updated = m.clone()

    # Assign m[0] and m[T-1]
    m_updated[0] = m_0

    # Assign other m[t] values between m[0] and m[T-1]
    for t in range(1, T):
        m_updated[t] = m_0

    # Ensure m retains gradients
    m = m_updated.clone().detach().requires_grad_(True)

    # make sure no nan in m
    if torch.isnan(m).any():
        print("m has nan at init")
        raise ValueError("m has nan at init")

    # no nan in w
    if torch.isnan(w).any():
        print("w has nan at init")
        raise ValueError("w has nan at init")

    # Perform gradient descent
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

    # Save results
    os.chdir("../mean_field_theory")
    folder = args.folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    # # Save the final results
    np.save(f"{folder}/mopt.npy", mopt.detach().numpy())
    np.save(f"{folder}/qopt.npy", qopt.detach().numpy())
    print("mopt", mopt)
    print("mopt", mopt.shape)
    # m_array to array. It is a list of tensors
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

    plt.figure(figsize=(10, 6))  # Larger size for clarity
    for i in range(data_numpy.shape[1]):  # Plot each column (line) with a label
        plt.plot(data_numpy[:, i], alpha=0.75, label=f"q_{i + 1}", linewidth=2)

    # Add horizontal lines
    plt.axhline(
        y=Q_C,
        color="green",
        linestyle=":",
        linewidth=2,
        label="Q (hard wall)",  # Plain text label
    )
    plt.axhline(
        y=1 - np.sum(1 - data_numpy[-1]),
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Path cumulated overlap (1 - sum(1 - q_i))",  # Plain text label
    )

    # Enhancing the plot for publication
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Q", fontsize=14)
    plt.title("Q (site average overlap) evolution Over Iterations", fontsize=16)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save the figure with high resolution for publication
    plt.savefig(args.folder + "/Q_Evolution_covid.png", dpi=300)

    # Show the plot
    plt.show()

    m_array_np = m_array

    # Determine dimensions
    iterations = len(m_array_np)

    # Prepare the subplots: arrange in (K // 4) rows and 4 columns for clarity
    rows = (K + 3) // 4  # Calculate the number of rows needed (ceiling of K/4)
    cols = min(K, 4)  # Maximum of 4 columns
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 6 * rows)
    )  # Dynamically adjust figure size

    # Flatten axes array for easy indexing and handle edge cases
    axes = axes.flatten() if K > 1 else [axes]  # Flatten for single-dimension indexing

    # Plot evolution for each antibody
    for k in range(K):
        for row in range(
            m_array_np[0].shape[0]
        ):  # Number of rows (values per antibody)
            row_values = [m_array_np[i][row, k] for i in range(iterations)]
            axes[k].plot(
                range(iterations),
                row_values,
                marker="o",
                label=f"m {row}",
                alpha=0.8,
                markersize=4,
            )

        # Customize each subplot
        axes[k].set_title(name_array[k], fontsize=14)
        axes[k].set_xlabel("Iterations", fontsize=12)
        axes[k].set_ylabel("m", fontsize=12)
        axes[k].legend(fontsize=10)
        axes[k].grid(True, linestyle="--", alpha=0.6)

    # Hide any unused subplots (if K is not a multiple of 4)
    for idx in range(K, len(axes)):
        axes[idx].axis("off")  # Turn off unused axes

    # Adjust layout for clarity
    plt.tight_layout()
    plt.savefig(args.folder + "/m_Evolution_covid.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean field implementation using real COVID model."
    )
    parser.add_argument("--T", type=int, default=20, help="Number of time steps")
    parser.add_argument("--N_ITER", type=int, default=5, help="Number of iterations")
    parser.add_argument("--EPS", type=float, default=0.5, help="Step size")
    parser.add_argument("--D", type=float, default=20, help="Gamma coefficient")

    parser.add_argument(
        "--folder", type=str, default="results_script2", help="Output folder"
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

    args = parser.parse_args()
    main(args)
