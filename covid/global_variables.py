import sys
import numpy as np
import os


pgm_path = "../PGM/"

sys.path.append(pgm_path + "source/")
sys.path.append(pgm_path + "utilities/")

import utilities, Proteins_utils, sequence_logo, plots_utils
import rbm, RBM_utils

BEGIN = 18
END = 5
# RBM = RBM_utils.loadRBM("../new_RBM_Covid_best2.data")
# print all files in the directory
# RBM = RBM_utils.loadRBM("wt_RBM_Covid.data")
RBM = RBM_utils.loadRBM("rbms/ESMIF_RBM_wt_Covid_temp_1_nH_50_l1B_0.2.data")


def load_kd_vectors(directory):
    """
    Load all q vectors from the specified directory.
    """
    kd_vectors = {}  # Dictionary to store q vectors, keyed by the antibody name
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            # Extract the antibody name from the filename
            antibody_name = filename.replace("delta_G.npy", "")
            # Load the q vector
            kd_vector = np.load(os.path.join(directory, filename))
            # print type
            kd_vectors[antibody_name] = kd_vector
    # asser no inf, nan or raise error
    for antibody_name, kd_vector in kd_vectors.items():
        if np.any(np.isinf(kd_vector)):
            raise ValueError(f"Inf value in {antibody_name}")
        elif np.any(np.isnan(kd_vector)):
            raise ValueError(f"NaN value in {antibody_name}")

    print(f"Loaded {len(kd_vectors)} KD vectors")
    return kd_vectors


escape_vectors_directory = "exp_data/escape_vectors"
ESCAPE_VECTORS = load_kd_vectors(escape_vectors_directory)


with open("exp_data/wt_omicron.fasta") as f:
    if f.readline().strip() != ">wt":
        print("Error: expected >wt")
    else:
        WT = f.readline().strip()

WT_SEQ = Proteins_utils.load_FASTA("exp_data/wt_omicron.fasta")[0]
WT_SEQ = WT_SEQ[BEGIN:-END]

CODE_RBM = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
