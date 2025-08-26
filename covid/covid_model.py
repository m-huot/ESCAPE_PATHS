import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import typing


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

from global_variables import *
from AB_RBM_EnergyModel import *

import math


class Covid_Model:
    # init with a list of tagets antibodies and array o
    def __init__(self, c_schedule=[-7], target_abs=None, beta=1):
        """
        Initialize the model with target antibodies and target c values.
        target_abs: list of strings
        c_schedule: list of floats
        """

        self.c_schedule = c_schedule

        self.energy_model = Ab_EnergyModel()
        if target_abs is None:
            self.target_abs = list(self.energy_model.kd_vectors.keys())
        else:
            self.target_abs = target_abs
        self.beta=beta

    def __call__(self, seq, t=0):
        """
        Calculate the energy of the sequence
        """
        ab_names = list(self.energy_model.kd_vectors.keys())

        concentrations = np.ones(len(ab_names)) * (-12)
        for i in range(len(ab_names)):
            if ab_names[i] in self.target_abs:
                concentrations[i] = self.c_schedule[t]

        self.energy_model.raw_concentrations = concentrations

        return -self.energy_model(seq)*self.beta


class RBM_score_model:
    def __init__(
        self,
        rbm=RBM,
    ):
        self.rbm = rbm

    def forward(self, s):
        return -self.rbm.free_energy(s)[0]

    def __call__(self, s):
        return self.forward(s)
