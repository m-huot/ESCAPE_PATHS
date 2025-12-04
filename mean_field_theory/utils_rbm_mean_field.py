import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from utils_mean_field import *
import torch
from collections.abc import Iterable

# Global imports (must be outside the function)
main_path = "../"
sys.path.append(main_path + "PGM/source/")
sys.path.append(main_path + "PGM/utilities/")
sys.path.append(main_path + "covid/")
os.chdir(main_path + "covid/")

from global_variables import *
from utils_evaluate_seq import *

os.chdir("../mean_field_theory")

import utilities, Proteins_utils, sequence_logo, plots_utils


def cgf_from_inputs_dReLU(
    I,
    gamma_plus,
    gamma_minus,
    theta_plus,
    theta_minus,
):
    # Define a small constant to avoid log(0)
    eps = 1e-12

    # Ensure all constants are computed in the same precision
    sqrt_gamma_plus = torch.sqrt(gamma_plus)
    sqrt_gamma_minus = torch.sqrt(gamma_minus)
    log_gamma_plus = torch.log(gamma_plus + eps)
    log_gamma_minus = torch.log(gamma_minus + eps)

    # Clamp the activations to a safe range

    def log_erf_times_gauss_torch(x):
        # Use the same device and dtype as x
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))
        logsqrtpiover2 = 0.5 * torch.log(
            torch.tensor(torch.pi / 2, dtype=x.dtype, device=x.device) + eps
        )

        # For x < 4, compute the exact form adding eps to avoid log(0)
        branch1 = 0.5 * x**2 + torch.log(torch.erfc(x / sqrt2) + eps) + logsqrtpiover2

        # For x >= 4, use an asymptotic expansion, again adding eps where needed
        branch2 = -torch.log(x + eps) + torch.log(1 - 1 / (x**2) + 3 / (x**4) + eps)

        return torch.where(x < 4, branch1, branch2)

    Z_plus = (
        log_erf_times_gauss_torch((-I + theta_plus) / sqrt_gamma_plus)
        - 0.5 * log_gamma_plus
    )
    Z_minus = (
        log_erf_times_gauss_torch((I + theta_minus) / sqrt_gamma_minus)
        - 0.5 * log_gamma_minus
    )

    # Use torch.logaddexp to stably combine the two branches
    return torch.logaddexp(Z_plus, Z_minus)


import torch


def create_gamma_functions(
    gamma_plus_array,
    gamma_minus_array,
    theta_plus_array,
    theta_minus_array,
    L,
    beta_rbm=1,
):
    """
    Create a list of -gamma functions using the given hyperparameter arrays.

    Parameters:
        theta_plus_array (torch.Tensor): Array of theta_plus values.
        theta_minus_array (torch.Tensor): Array of theta_minus values.
        gamma_plus_array (torch.Tensor): Array of gamma_plus values.
        gamma_minus_array (torch.Tensor): Array of gamma_minus values.

    Returns:
        List[callable]: A list of -gamma functions, one for each set of hyperparameters.
    """
    if not (
        len(theta_plus_array)
        == len(theta_minus_array)
        == len(gamma_plus_array)
        == len(gamma_minus_array)
    ):
        raise ValueError("All input arrays must have the same length.")

    gamma_functions = []

    for i in range(len(theta_plus_array)):
        # Capture the current set of hyperparameters
        theta_plus = theta_plus_array[i]
        theta_minus = theta_minus_array[i]
        gamma_plus = gamma_plus_array[i]
        gamma_minus = gamma_minus_array[i]

        # Define a gamma function for the current set of hyperparameters
        def cgf_from_inputs_dReLU_func(
            I,
            gamma_plus=gamma_plus,
            gamma_minus=gamma_minus,
            theta_plus=theta_plus,
            theta_minus=theta_minus,
        ):
            return (
                -cgf_from_inputs_dReLU(
                    I, gamma_plus, gamma_minus, theta_plus, theta_minus
                )
                * beta_rbm
                # E=minus gamma
            )

        # Append the function to the list
        gamma_functions.append(cgf_from_inputs_dReLU_func)

    return gamma_functions


def create_ab_functions(n_ab, beta_ab, L):
    """
    Create a list of n_ab escape functions.
    If beta_ab is a float, it is broadcast to all functions.
    If beta_ab is a list or array, beta_ab[i] is used for function i.
    """
    if isinstance(beta_ab, Iterable) and not isinstance(beta_ab, (str, bytes)):
        beta_ab = list(beta_ab)
        if len(beta_ab) != n_ab:
            raise ValueError("Length of beta_ab must match n_ab.")
    else:
        beta_ab = [beta_ab] * n_ab  # broadcast scalar to list

    functions = []
    for i in range(n_ab):
        beta = beta_ab[i]  # capture by default argument

        def selection_coeff(G_t, beta=beta):
            if G_t >= 0:
                raise ValueError("G_t must be less than 0 for antibodies")
            return -torch.log(1 - torch.exp(G_t)) * beta

        functions.append(selection_coeff)

    return functions
