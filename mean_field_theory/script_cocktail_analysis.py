import os
import sys
import torch
import numpy as np
import pandas as pd
from utils_mean_field import *
from script_mean_field_covid import *
from tqdm import tqdm

main_path = "../"

sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
import Proteins_utils

sys.path.append(main_path + "covid/")
os.chdir(main_path + "covid/")

from global_variables import *

os.chdir("../mean_field_theory")
os.listdir()


# %%
PROT_INIT = Proteins_utils.load_FASTA("../covid/exp_data/wt_omicron.fasta")[0]
PROT_INIT = PROT_INIT[BEGIN:-END]


# %%
L = PROT_INIT.shape[0]
Q = 20


gamma_f_list = create_gamma_functions(
    torch.tensor(RBM.hlayer.gamma_plus),
    torch.tensor(RBM.hlayer.gamma_minus),
    torch.tensor(RBM.hlayer.theta_plus),
    torch.tensor(RBM.hlayer.theta_minus),
)


# Combine weights and functions
selection_coeff_list = []
selection_coeff_list_gamma_only = []
selection_coeff_list_gamma_only.extend(create_ab_functions(n_ab=29, beta_ab=0))
selection_coeff_list_gamma_only.extend(gamma_f_list)

selection_coeff_list.extend(create_ab_functions(n_ab=29, beta_ab=1))
selection_coeff_list.extend(gamma_f_list)


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
g = RBM.vlayer.fields
gamma_f_list = create_gamma_functions(
    torch.tensor(RBM.hlayer.gamma_plus),
    torch.tensor(RBM.hlayer.gamma_minus),
    torch.tensor(RBM.hlayer.theta_plus),
    torch.tensor(RBM.hlayer.theta_minus),
    beta_rbm=1,
)

w_components = []
w_components.append(wab)
w_components.append(wgamma)
w = np.concatenate(w_components, axis=-1)


# %%


def get_zeta(mopt, selection_coeff_list, selection_coeff_list_gamma_only):
    """
    returns potential of gamma and ab
    """
    zeta_values = []
    zeta_values_gamma_only = []

    for j in range(mopt.shape[0]):
        zeta_values.append(
            Zeta(torch.tensor(mopt[j]), selection_coeff_list, L).detach().numpy()
        )
        zeta_values_gamma_only.append(
            Zeta(torch.tensor(mopt[j]), selection_coeff_list_gamma_only, L)
            .detach()
            .numpy()
        )
    zeta_values = np.array(zeta_values)
    zeta_values_gamma_only = np.array(zeta_values_gamma_only)
    return (
        zeta_values_gamma_only,
        zeta_values - zeta_values_gamma_only,
    )  # gamma potential, Ab potential


def get_phi(qopt, q_c):
    phi_values = Phi_cont(torch.tensor(qopt), q_c, L).detach().numpy()
    return phi_values


def get_field(path_proba):
    # bring the time axis to the front: (T, L, Q)
    path_proba = np.moveaxis(path_proba, 0, -1)  # if you really need this
    field = np.empty(path_proba.shape[0])  # one value per time-step

    for t in range(path_proba.shape[0]):  # loop over timesteps
        field[t] = np.sum(path_proba[t] * g.T)  # element-wise product

    return -field / PROT_INIT.shape[0]


def path_entropy_mean_field(mhat, qhat, mopt, qopt, w=w, g=g, PROT_INIT=PROT_INIT):
    # make sur eno inf
    if np.any(np.isinf(mhat)) or np.any(np.isinf(qhat)):
        print("mhat or qhat has inf values, returning NaN")
        return np.nan, None
    if np.any(np.isinf(mopt)) or np.any(np.isinf(qopt)):
        print("mopt or qopt has inf values, returning NaN")
        return np.nan, None
    sum_m_mhat = np.sum(np.sum(mhat * mopt, axis=0))
    sum_q_qhat = np.sum(np.sum(qhat * qopt, axis=0))
    grad_g_list = []
    sum_logZ = 0
    for i in tqdm(range(PROT_INIT.shape[0])):
        Zi, grad_q, grad_m, grad_g = compute_transfer_matrix_Z_free_with_epsilon(
            Q, qhat, mhat, w[i], g[i], v1_fixed=PROT_INIT[i], beta_rbm=1
        )
        # make all tensors numpy arrays
        grad_q = grad_q.detach().numpy()
        grad_m = grad_m.detach().numpy()
        grad_g = grad_g.detach().numpy()  # grad_g.shape == (T, Q)
        grad_g_list.append(grad_g)

        sum_logZ += Zi

    sum_logZ = np.array(sum_logZ) / PROT_INIT.shape[0]
    print("sum logZ", sum_logZ)
    s = sum_logZ - sum_m_mhat - sum_q_qhat

    # shape of grad_g: L, T, Q
    grad_g_list = np.array(grad_g_list)
    print("grad_g_list", grad_g_list.shape)
    print("g", g.shape)

    sum_field_neg_energy = np.sum(grad_g_list * g[:, None, :]) / PROT_INIT.shape[0]

    print("s=", s)
    print("sum_field_energy", sum_field_neg_energy)
    s = s - sum_field_neg_energy
    return s, grad_g_list


def main():
    D = 9
    T = 9
    GAMMA = D / L
    Q_C = 1 - GAMMA / T

    results = []

    beta_ab_list = [0, 0.1, 0.5, 1, 3, 5]
    ab_combos = [
        "REGN10933",
        "COV2-2196",
        "REGN10987",
        "REGN10933_REGN10987",
        "REGN10933_COV2-2196",
    ]
    dir = "results_scripts/cocktail"

    for ab_combo in ab_combos:
        for beta_ab in beta_ab_list:
            path = f"mean_f_free_ab_beta_ab_{beta_ab}_D_{D + 1}_ab_{ab_combo}"

            try:
                # Load data
                mopt = np.load(os.path.join(dir, path, "mopt.npy"))
                qopt = np.load(os.path.join(dir, path, "qopt.npy"))
                mhat = np.load(os.path.join(dir, path, "mhat_array.npy"))[-1]
                qhat = np.load(os.path.join(dir, path, "qhat_array.npy"))[-1]

                print(f"\n--- beta_ab = {beta_ab}, ab_combo = {ab_combo} ---")
                print("mopt shape:", mopt.shape)
                print("qopt shape:", qopt.shape)
                print("mhat shape:", mhat.shape)
                print("qhat shape:", qhat.shape)
                path_entropy, field_proba = path_entropy_mean_field(
                    mhat, qhat, mopt, qopt, w=w, g=g, PROT_INIT=PROT_INIT
                )
                print("Path entropy mean field =", path_entropy)
                ab_list = ab_combo.split("_")

                ab_indices = [ab_names.index(name) for name in ab_list]
                betas_ab = np.zeros(len(ab_names))
                for i in ab_indices:
                    betas_ab[i] = 1  # don t use beta_ab
                selection_coeff_list = []
                selection_coeff_list.extend(
                    create_ab_functions(n_ab=29, beta_ab=betas_ab)
                )
                selection_coeff_list.extend(gamma_f_list)

                # Compute metrics
                zeta_gamma, zeta_ab = get_zeta(
                    mopt, selection_coeff_list, selection_coeff_list_gamma_only
                )
                zeta_gamma = zeta_gamma.sum()
                zeta_ab = zeta_ab.sum()
                print("zeta_gamma =", zeta_gamma)
                print("zeta_ab =", zeta_ab)

                phi = get_phi(qopt, Q_C).sum()
                print("phi =", phi)

                field_energy = get_field(field_proba).sum()
                print("field energy =", field_energy)

                # Store results
                results.append(
                    {
                        "beta_ab": beta_ab,
                        "ab_combo": ab_combo,
                        "zeta_gamma": zeta_gamma,
                        "zeta_ab": zeta_ab,
                        "phi": phi,
                        "field_energy": field_energy,
                        "entropy": path_entropy,
                    }
                )

            except Exception as e:
                print(f"❌ Failed for beta_ab = {beta_ab}, ab_combo = {ab_combo}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("test.csv", index=False)
    print("\n✅ Saved results to test.csv")


if __name__ == "__main__":
    main()
