import torch
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def delta(v_t, v_t_plus_1):
    """
    Returns 1 if v_t == v_t_plus_1, otherwise 0.
    """
    return 1 if v_t == v_t_plus_1 else 0


def compute_transfer_matrix_Z_free_with_epsilon(Q, q, m, w, g, v1_fixed=0, beta_rbm=1):
    """
    Compute Z with epsilon dependency and gradients with respect to epsilon using the transfer matrix method.
    """
    q = torch.tensor(q, requires_grad=True, dtype=torch.float64)
    m = torch.tensor(m, requires_grad=True, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    g = torch.tensor(g, requires_grad=True, dtype=torch.float64)
    T = m.shape[0]

    # Initialization: Fix v1 = 0
    T_mat = torch.zeros(Q, dtype=torch.float64)
    for v2 in range(Q):
        T_mat[v2] = (
            delta(v1_fixed, v2) * q[0]
            + torch.sum(m[0] * w[v1_fixed])
            + g[v1_fixed] * beta_rbm
        )

    # Recursive updates for intermediate time steps (2 to T-1)
    for t in range(1, T - 1):
        new_T_mat = []
        for vp in range(Q):
            contributions = [
                T_mat[v]
                + delta(v, vp) * q[t]
                + torch.sum(m[t] * w[v])
                + g[v] * beta_rbm
                for v in range(Q)
            ]
            new_T_mat.append(torch.logsumexp(torch.stack(contributions), dim=0))
        T_mat = torch.stack(new_T_mat)

    # Final summation over all possible states
    contributions = [
        T_mat[v] + torch.sum(m[T - 1] * w[v]) + g[v] * beta_rbm for v in range(Q)
    ]
    log_Z = torch.logsumexp(torch.stack(contributions), dim=0)

    # Compute gradients w.r.t epsilon
    log_Z.backward()
    grad_q = q.grad.clone()
    grad_m = m.grad.clone()
    grad_g = g.grad.clone()
    # remove the torch exp(Z) to avoid numerical issues
    return log_Z.item(), grad_q, grad_m, grad_g


# def compute_enumerate_matrix_Z_free_with_gradients(
#     Q, q, m, w, g, v1_fixed=0, beta_rbm=1
# ):
#     """
#     Compute Z and gradients of log Z with respect to parameters q and m using enumeration.
#     """
#     q = torch.tensor(q, requires_grad=True, dtype=torch.float64)
#     m = torch.tensor(m, requires_grad=True, dtype=torch.float64)
#     g = torch.tensor(g, requires_grad=True, dtype=torch.float64)
#     w = torch.tensor(w, dtype=torch.float64)
#     T = m.shape[0]

#     Z_enumerated = 0
#     # Enumerate all possible paths
#     for path in product(range(Q), repeat=T - 1):  # T-1 intermediate states
#         full_path = (v1_fixed,) + path
#         energy = 0
#         for t in range(T - 1):
#             energy += (
#                 delta(full_path[t], full_path[t + 1]) * q[t]
#                 + torch.sum(m[t] * w[full_path[t]])
#                 + g[full_path[t]] * beta_rbm
#             )
#         energy += torch.sum(m[T - 1] * w[full_path[-1]]) + g[full_path[-1]] * beta_rbm
#         Z_enumerated += torch.exp(energy)

#     log_Z = torch.log(Z_enumerated)

#     # Compute gradients
#     log_Z.backward()
#     grad_q = q.grad.clone()
#     grad_m = m.grad.clone()
#     grad_g = g.grad.clone()

#     # check nan
#     if torch.isnan(grad_q).any():
#         raise ValueError("NaN value in grad_q")
#     if torch.isnan(grad_m).any():
#         raise ValueError("NaN value in grad_m")
#     if torch.isnan(grad_g).any():
#         raise ValueError("NaN value in grad_g")

#     return torch.exp(log_Z).item(), grad_q, grad_m, grad_g


def Phi_cont(q, Q_C, L):
    """
    Compute the path continuity Φ_cont(q).
    Parameters:
    - q: PyTorch tensor representing the state variable.

    Returns:
    - Φ_cont(q) as a PyTorch tensor.
    """
    T = q.shape[0] - 1
    coeff = 1 / (T**2)
    # Compute Φ_cont(q) using element-wise tensor operations
    valid = (q > Q_C) & (q <= 1)  # Boolean mask for valid range
    phi = torch.where(valid, coeff / (q - Q_C), float("inf"))  # Apply the condition
    return phi / L


def dPhi_cont(q, Q_C, L):
    """
    Compute the derivative of Φ_cont(q) with respect to q.
    Parameters:
    - q: PyTorch tensor representing the state variable.

    Returns:
    - Derivative of Φ_cont(q) as a PyTorch tensor.
    """
    T = q.shape[0] - 1

    coeff = 1 / (T**2)
    # Compute dΦ/dq for the valid range
    valid = (q > Q_C) & (q <= 1)  # Boolean mask for valid range
    dphi = torch.where(
        valid, -coeff / (q - Q_C) ** 2, 0.0
    )  # Derivative is zero outside valid range
    return dphi / L


def q_hat_cont(q, Q_C, L, beta_phi=1):
    """
    Compute q_hat based on the provided equation.
    """

    grad_q = dPhi_cont(q, Q_C, L)
    return -grad_q * beta_phi


def Phi_evo(q, MU):
    """
    Compute the evolutionary potential Φ_Evo(q).

    Parameters:
    - q: PyTorch tensor representing the state variable.
    - mu: Scalar representing the mutation rate.
    - A: Scalar parameter.

    Returns:
    - Φ_Evo(q) as a PyTorch tensor.
    """
    A = 20
    MU = torch.tensor(MU, dtype=q.dtype, device=q.device)  # Ensure MU is a tensor
    valid = (q >= 0) & (q <= 1)  # Ensure q is in valid range
    term = 1 + A / (torch.exp(MU * A / (A - 1)) - 1)
    phi = torch.where(valid, (1 - q) * torch.log(term), float("inf"))  # Apply condition
    return phi


def dPhi_evo(q, MU):
    """
    Compute the derivative of Φ_Evo(q) with respect to q.

    Parameters:
    - q: PyTorch tensor representing the state variable.
    - mu: Scalar representing the mutation rate.
    - A: Scalar parameter.

    Returns:
    - Derivative of Φ_Evo(q) as a PyTorch tensor.
    """
    A = 20
    MU = torch.tensor(MU, dtype=q.dtype, device=q.device)  # Ensure MU is a tensor
    valid = (q >= 0) & (q <= 1)  # Ensure q is in valid range
    term = 1 + A / (torch.exp(MU * A / (A - 1)) - 1)
    dphi = torch.where(valid, -torch.log(term), 0.0)  # Derivative formula
    return dphi


def q_hat_evo(q, MU, beta_phi=1):
    """
    Compute q_hat based on the provided equation.
    """
    grad_q = np.zeros(q.shape)

    grad_q = dPhi_evo(q, MU)
    return -grad_q * beta_phi


# Binding free energy (G)
def get_G(mt, L):
    return L * mt


def Zeta(mt_list, gamma_f_list, L):
    """
    Compute Zeta potential of intermediary variants

    Parameters:
    - gammat_list: Tensor of mt values (dimension K).
    - gamma_f_list: List of rbm gamma functions.

    Returns:
    - Psi value as a tensor.
    """
    Energy = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    for i in range(mt_list.shape[0]):  # Iterate over elements of the tensors
        G_t = get_G(mt_list[i], L)  # dimension 1
        E_i = gamma_f_list[i](G_t)
        Energy = Energy + E_i
        if torch.isnan(E_i):
            print("function no", i)
            print("G_t", G_t)
            raise ValueError("NaN encountered in S computation.")
    return Energy / L


def dZeta(mt_list, gamma_f_list, L):
    """
    Compute the derivative of Zeta with respect to mt_list.

    Parameters:
    - mt_list: Tensor of mt values (dimension K).
    - gamma_f_list: List of rbm gamma functions.
    - L: Parameter used in the computation (passed to get_G).

    Returns:
    - Gradients with respect to mt_list as a tensor.
    """
    # Ensure mt_list requires gradients
    mt_list = mt_list.clone().detach().requires_grad_(True)

    # Initialize proba with gradient tracking
    Energy = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    # Loop over each element in mt_list
    for i in range(mt_list.shape[0]):
        G_t = get_G(mt_list[i], L)  # Compute G value for current mt
        E_i = gamma_f_list[i](G_t)  # Compute the contribution using the gamma function
        Energy = Energy + E_i
        # get grad of proba_i vs gammat_list[i] and make sure it is not nan
        if torch.isnan(E_i):
            print(f"function no {i}")
            print(f"G_t: {G_t}")
            raise ValueError("NaN encountered in gamma computation.")

    # Compute Zeta as defined in the original function
    zeta_value = Energy / L

    # Backpropagate to compute gradients with respect to mt_list
    zeta_value.backward()

    # Return the computed gradients for mt_list
    return mt_list.grad


def m_hat_t(
    mt,
    gamma_f_list,
    L,
):
    """
    Compute m_hat based on the provided equation.
    """

    zeta_hat_t = dZeta(mt, gamma_f_list, L)

    m_hat_t = -zeta_hat_t

    return m_hat_t


def m_hat(m, gamma_f_list, L):
    """
    Compute m_hat for the entire tensor m, which contains T vectors mt.
    Parameters:
    - m: Tensor of size T*K containing the state vectors over T timesteps.

    Returns:
    - m_hat_result: Tensor of size T*K, with each row corresponding to m_hat_t at timestep t.
    """
    # Create a list to store each timestep's result
    m_hat_list = []
    T = m.shape[0]

    # Intermediate timesteps
    for t in range(T):
        m_hat_list.append(m_hat_t(m[t], gamma_f_list, L))

    # Stack the results into a single tensor
    m_hat_result = torch.stack(m_hat_list, dim=0)

    return m_hat_result


def GradDescent_free(
    mstart,
    qstart,
    w,
    g,
    gamma_f_list,
    beta_rbm,
    beta_phi,
    niter,
    eps,
    L,
    Q,
    Q_C=0.0,
    protein_init=None,
    MU=0.01,
    phi="cont",
):
    """
    Perform gradient descent to optimize m and q.
    Parameters:
    - mstart: Initial tensor for m of size T*K.
    - qstart: Initial tensor for q of size T-1.
    - w: Weights tensor of size Q*K.
    - niter: Number of iterations for gradient descent.
    - eps: Learning rate.

    Returns:
    - m, q: Optimized tensors.
    - m_array, q_array: Lists of m and q values at each step.
    """
    m = mstart.clone().detach().requires_grad_(True)  # Detach and allow gradients
    q = qstart.clone().detach().requires_grad_(True)

    # check basence of nan
    if torch.isnan(m).any() or torch.isnan(q).any():
        raise ValueError("NaN value in mstart or qstart")

    T = m.shape[0]

    m_array = []
    dm_array = []
    q_array = []
    dq_array = []
    qhat_array = []
    mhat_array = []

    m_array.append(m.clone().detach())
    q_array.append(q.clone().detach())

    for i in tqdm(range(niter)):
        _, dq, dm = 0, 0, 0
        if phi == "cont":
            qhat = q_hat_cont(q, Q_C, L, beta_phi)
        elif phi == "evo":
            qhat = q_hat_evo(q, MU, beta_phi)
            Q_C = 0.0

        mhat = m_hat(
            m=m,
            gamma_f_list=gamma_f_list,
            L=L,
        )
        qhat_array.append(qhat.clone().detach())
        mhat_array.append(mhat.clone().detach())

        # check for nan
        if torch.isnan(mhat).any():
            raise ValueError("NaN value in mhat")
        # Compute the transfer matrix Z and its gradients

        for i in range(L):
            _i, dqi, dmi, dgi = compute_transfer_matrix_Z_free_with_epsilon(
                Q, qhat, mhat, w[i], g[i], protein_init[i], beta_rbm
            )
            dq += dqi
            dm += dmi

        dm = dm / L
        # print('dm:',dm)
        # print('m:',m)
        dq = dq / L

        # chzck for nan
        if torch.isnan(dm).any() or torch.isnan(dq).any():
            raise ValueError("NaN value in dm or dq")

        dm_array.append(dm.clone().detach())
        dq_array.append(dq.clone().detach())

        # Fix gradients for boundary conditions
        dm[0] = m[0]

        eps = eps * 0.98
        # Update m and q without in-place modification
        m = m + eps * (dm - m)
        q = q + eps * (dq - q)

        for i in range(len(q)):
            q = torch.where(q < Q_C, torch.tensor(Q_C + 0.001, dtype=q.dtype), q)

        # Detach and store current values for tracking
        m_array.append(m.clone().detach())
        q_array.append(q.clone().detach())

        # Retain graph for subsequent backward calls
        m = m.detach().requires_grad_(True)
        q = q.detach().requires_grad_(True)

    return m, q, m_array, q_array, dm_array, dq_array, qhat_array, mhat_array


def compute_transfer_matrix_Z_free_with_epsilon(Q, q, m, w, g, v1_fixed=0, beta_rbm=1):
    """
    Compute Z with epsilon dependency and gradients with respect to epsilon using the transfer matrix method.
    """
    q = torch.tensor(q, requires_grad=True, dtype=torch.float64)
    m = torch.tensor(m, requires_grad=True, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    T = m.shape[0]

    # repeat g T times to match the shape of m. use copies of g
    g = [g.copy() for i in range(T)]
    g = np.stack(g)
    g = torch.tensor(g, requires_grad=True, dtype=torch.float64)

    # Initialization: Fix v1 = 0
    T_mat = torch.zeros(Q, dtype=torch.float64)
    for v2 in range(Q):
        T_mat[v2] = (
            delta(v1_fixed, v2) * q[0]
            + torch.sum(m[0] * w[v1_fixed])
            + g[0, v1_fixed] * beta_rbm
        )

    # Recursive updates for intermediate time steps (2 to T-1)
    for t in range(1, T - 1):
        new_T_mat = []
        for vp in range(Q):
            contributions = [
                T_mat[v]
                + delta(v, vp) * q[t]
                + torch.sum(m[t] * w[v])
                + g[t, v] * beta_rbm
                for v in range(Q)
            ]
            new_T_mat.append(torch.logsumexp(torch.stack(contributions), dim=0))
        T_mat = torch.stack(new_T_mat)

    # Final summation over all possible states
    contributions = [
        T_mat[v] + torch.sum(m[T - 1] * w[v]) + g[T - 1, v] * beta_rbm for v in range(Q)
    ]
    log_Z = torch.logsumexp(torch.stack(contributions), dim=0)

    # Compute gradients w.r.t epsilon
    log_Z.backward()
    grad_q = q.grad.clone()
    grad_m = m.grad.clone()
    grad_g = g.grad.clone()
    # modified: was exp(log_Z) before
    return log_Z.item(), grad_q, grad_m, grad_g


def delta(v1, v2):
    """Kronecker delta function."""
    return 1.0 if v1 == v2 else 0.0


def compute_frequency(Q, q, m, w, g, v1_fixed=0, vT_fixed=1, beta_rbm=1, free=True):
    """
    Compute amino acid frequencies at each time step by evaluating
    the derivative of log Z w.r.t epsilon at epsilon = 0.
    """
    if beta_rbm == 0:
        beta_rbm = 1
        g = g * 0
    T = m.shape[0]
    # very small epsilon
    # epsilon_zero = torch.tensor(1e-6 * np.random.randn(T, Q), dtype=torch.float64)
    if free:
        _, _, _, grad_g = compute_transfer_matrix_Z_free_with_epsilon(
            Q, q, m, w, g, v1_fixed, beta_rbm=beta_rbm
        )

    return grad_g / beta_rbm
