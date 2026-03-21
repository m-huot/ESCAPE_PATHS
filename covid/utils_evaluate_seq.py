from global_variables import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

import sys

pgm_path = "../PGM/"
sys.path.append(pgm_path + "source/")
sys.path.append(pgm_path + "utilities/")

import utilities, Proteins_utils, sequence_logo, plots_utils
import rbm, RBM_utils
import numpy as np
import os
import random
from tqdm import tqdm


def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def distances_wt(generated_sequences):
    """
    Output: List of hamming distances between WT and generated sequences
    """
    return [hamming(WT_SEQ, s) for s in generated_sequences]


def av_distances_wt(generated_sequences):
    """
    Output: Average /std of hamming distance between WT and generated sequences
    """
    return np.mean([hamming(WT_SEQ, s) for s in generated_sequences]), np.std(
        [hamming(WT_SEQ, s) for s in generated_sequences]
    )


def one_hot_encode_concat(s):
    num_categories = 20
    one_hot_matrix = np.zeros((len(s), num_categories))
    for i, char in enumerate(s):
        one_hot_matrix[i, char] = 1
    one_hot_encoded = one_hot_matrix.flatten()

    return one_hot_encoded


def get_log_bindings(s, escape_vectors=ESCAPE_VECTORS):
    s_hot = one_hot_encode_concat(s)
    return np.array([np.dot(s_hot, escape_vectors[aa]) for aa in escape_vectors.keys()])


def get_expected_log_bindings(proba, escape_vectors=ESCAPE_VECTORS):
    proba = proba.flatten()

    return np.array([np.dot(proba, escape_vectors[aa]) for aa in escape_vectors.keys()])


def get_ab_energy(s, escape_vectors=ESCAPE_VECTORS):
    """
    Get the energy of the antibody given the sequence s.
    """
    log_bindings = get_log_bindings(s, escape_vectors)
    if np.any(log_bindings > 0):
        raise ValueError("Log bindings must be negative for antibodies")
    ab_energies = -np.log(1 - np.exp(log_bindings))
    return ab_energies.sum()
