import argparse
import numpy as np
import random
import math
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

CODE_LATTICE = [
    "C",
    "M",
    "F",
    "I",
    "L",
    "V",
    "W",
    "Y",
    "A",
    "G",
    "T",
    "S",
    "N",
    "Q",
    "D",
    "E",
    "H",
    "R",
    "K",
    "P",
]


def string2seq(s):
    """
    Convert a string to a sequence of integers.
    """
    return np.array([CODE_LATTICE.index(aa) + 1 for aa in s]).flatten()


class ProteinLattice:
    def __init__(
        self,
        energy_file="energies.dat",
        contact_map_file="contact_maps_10000.dat",
        structure_idx=0,
        num_structures=10000,
        protein_size=27,
        alphabet_size=20,
    ):
        energy_file = os.path.join(os.path.dirname(__file__), energy_file)
        contact_map_file = os.path.join(os.path.dirname(__file__), contact_map_file)
        """
        Initialize the class with the required files and parameters.

        :param energy_file: Path to the Miyazawa-Jernigan interaction energy file (energies.dat)
        :param contact_map_file: Path to the contact map file (contact_maps_10000.dat)
        :param num_structures: Number of possible protein structures
        :param protein_size: Length of the protein sequence (default 27)
        :param alphabet_size: Number of amino acid types (default 20)
        """
        self.num_structures = num_structures
        self.protein_size = protein_size
        self.alphabet_size = alphabet_size
        self.structure_idx = structure_idx
        self.energies = self.load_energies(energy_file)
        self.contact1, self.contact2 = self.load_contact_maps(contact_map_file)
        self.code = CODE_LATTICE

    def load_energies(self, energy_file):
        """
        Load the Miyazawa-Jernigan interaction energies from the file.
        """
        energy_file = os.path.join(os.path.dirname(__file__), energy_file)
        energies = np.zeros((self.alphabet_size, self.alphabet_size))
        with open(energy_file, "r") as f:
            for i in range(self.alphabet_size * self.alphabet_size):
                inter1, inter2, energy = map(float, f.readline().split())
                energies[int(inter1) - 1][int(inter2) - 1] = energy
        return energies

    def load_contact_maps(self, contact_map_file):
        """
        Load the contact maps for the structures.
        """
        contact_map_file = os.path.join(os.path.dirname(__file__), contact_map_file)
        contact1 = np.zeros((self.num_structures, 28), dtype=int)
        contact2 = np.zeros((self.num_structures, 28), dtype=int)
        with open(contact_map_file, "r") as f:
            for i in range(self.num_structures):
                for j in range(28):
                    inter1, c1, c2 = map(int, f.readline().split())
                    contact1[i][j] = c1
                    contact2[i][j] = c2
        return contact1, contact2

    def calculate_energy(self, seq, structure_idx):
        """
        Calculate the energy of the sequence for the given structure.
        """
        energy = 0.0
        for k in range(28):
            aa1 = seq[self.contact1[structure_idx][k] - 1]
            aa2 = seq[self.contact2[structure_idx][k] - 1]
            energy += self.energies[aa1 - 1][aa2 - 1]
        return energy

    def calculate_pnat(self, seq):
        """
        Calculate the Boltzmann probability (Pnat) for a sequence folding into a structure.
        """
        pnat = 0.0
        sum_exp = 0.0
        for j in range(self.num_structures):
            energy = self.calculate_energy(seq, j)
            exp_energy = math.exp(-energy)
            sum_exp += exp_energy
            if j == self.structure_idx:
                pnat = exp_energy
        return pnat / sum_exp

    def mutate_sequence(self, seq):
        """
        Mutate a sequence by selecting random sites and changing to random amino acids.
        """
        new_seq = seq[:]
        num_mutations = np.random.poisson(1.0)
        for _ in range(num_mutations):
            index = random.randint(0, self.protein_size - 1)
            new_seq[index] = random.randint(1, self.alphabet_size)
        return new_seq


class LatticeModel:
    def __init__(self, lattice_model):
        self.lattice_model = lattice_model

        self.cache = OrderedDict()

    def __call__(self, x):

        x_tuple = tuple(x)

        if x_tuple in self.cache:
            self.cache.move_to_end(x_tuple)
            return self.cache[x_tuple]

        pnat_value = self.lattice_model.calculate_pnat(x)

        self.cache[x_tuple] = pnat_value

        if len(self.cache) > 100:
            self.cache.popitem(last=False)

        return pnat_value


def is_adjacent(posA, posB):
    """Return True if posA and posB differ by 1 in exactly one coordinate."""
    return sum(abs(a - b) for a, b in zip(posA, posB)) == 1


def backtrack_lattice_protein(contact_pairs):
    """
    Backtracking search for a valid cubic lattice placement of N residues,
    satisfying that consecutive residues (i, i+1) and specified contact pairs
    are adjacent (only axis-aligned moves, so no diagonal moves are allowed).

    Parameters:
      - N: total number of residues.
      - contact_pairs: list of tuples (r1, r2) representing extra adjacency constraints.

    Returns:
      A dictionary {residue: (x, y, z)} if a valid configuration is found,
      otherwise None.
    """
    N = 27
    adjacency = {r: set() for r in range(1, N + 1)}
    for i in range(1, N):
        adjacency[i].add(i + 1)
        adjacency[i + 1].add(i)
    for r1, r2 in contact_pairs:
        adjacency[r1].add(r2)
        adjacency[r2].add(r1)

    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    positions = [None] * (N + 1)
    used = set()

    def place_residue(r):
        if r > N:
            return True

        if r == 1:
            positions[r] = (0, 0, 0)
            used.add((0, 0, 0))
            return place_residue(r + 1)
        else:
            x_prev, y_prev, z_prev = positions[r - 1]
            for dx, dy, dz in directions:
                candidate = (x_prev + dx, y_prev + dy, z_prev + dz)
                if candidate in used:
                    continue

                positions[r] = candidate
                used.add(candidate)

                valid = True
                for neighbor in adjacency[r]:
                    if neighbor < r:
                        if not is_adjacent(positions[r], positions[neighbor]):
                            valid = False
                            break

                if valid:
                    if place_residue(r + 1):
                        return True

                used.remove(candidate)
                positions[r] = None

            return False

    if place_residue(1):
        return {r: positions[r] for r in range(1, N + 1)}
    else:
        return None


def _plot_lattice_protein(positions, node_values=None):
    """
    Plot the lattice protein configuration with optional node coloring.

    Parameters:
    - positions (dict): residue index -> (x, y, z)
    - node_values (array-like or None): 27 values in [0, 1] for coloring each node from blue to red

    Notes:
    - Nodes are colored based on node_values if provided (colormap: blue to red).
    - Otherwise, all nodes are cyan with black edges and blue text.
    - Residue labels, connections, and 3D plotting are maintained.
    """
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.view_init(elev=10, azim=10)

    xs, ys, zs = [], [], []

    use_colormap = node_values is not None and len(node_values) == 27
    if use_colormap:
        cmap = cm.get_cmap("coolwarm")  # blue to red

    for r, (x, y, z) in positions.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)

        if use_colormap:
            color_val = node_values[r - 1]
            node_color = cmap(color_val)
        else:
            node_color = "cyan"

        ax.scatter(x, y, z, color=node_color, s=200, edgecolors="black", zorder=5)
        ax.text(
            x,
            y,
            z,
            str(r),
            fontsize=9,
            ha="center",
            va="center",
            zorder=30,
            color="blue" if not use_colormap else "black",
            weight="bold",
        )

    sorted_keys = sorted(positions.keys())
    for i in range(len(sorted_keys) - 1):
        r1 = sorted_keys[i]
        r2 = sorted_keys[i + 1]
        x1, y1, z1 = positions[r1]
        x2, y2, z2 = positions[r2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", linewidth=2, zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_box_aspect([1, 1, 1])
    margin = 1
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_zlim(min(zs) - margin, max(zs) + margin)

    plt.show()


def get_contact_list(protein_idx, filename="contact_maps_10000.dat"):
    """
    Reads a .dat file containing structure contacts and returns the contact list
    for the given protein index.

    The file is assumed to have three space-separated columns:
      [structure_idx] [residue1] [residue2]

    Parameters:
      protein_idx (int): The protein index for which to get the contacts.
      filename (str): The path to the .dat file (default: 'contacts.dat').

    Returns:
      list of tuples: A list where each element is a tuple (residue1, residue2)
                      corresponding to a contact for the specified protein index.
    """
    filename = os.path.join(os.path.dirname(__file__), filename)
    contact_list = []
    protein_idx += 1  # start at 1

    with open(filename, "r") as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # or raise an error if the file format is unexpected

            idx, r1, r2 = parts
            try:
                idx = int(idx)
                r1 = int(r1)
                r2 = int(r2)
            except ValueError:
                continue  # or handle error

            if idx == protein_idx:
                contact_list.append((r1, r2))

    return contact_list


def plot_lattice_protein(protein_idx, node_values=None):
    contact_pairs = get_contact_list(protein_idx, filename="contact_maps_10000.dat")
    positions = backtrack_lattice_protein(contact_pairs)
    if positions:
        _plot_lattice_protein(positions, node_values=node_values)
    else:
        print(f"No valid configuration found for protein index {protein_idx}.")
