import argparse
import numpy as np
import random
import math


class ProteinLatticeMSAGenerator:
    def __init__(
        self,
        energy_file,
        contact_map_file,
        num_structures=10000,
        protein_size=27,
        alphabet_size=20,
    ):
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
        self.energies = self.load_energies(energy_file)
        self.contact1, self.contact2 = self.load_contact_maps(contact_map_file)
        self.code = [
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

    def load_energies(self, energy_file):
        """
        Load the Miyazawa-Jernigan interaction energies from the file.
        """
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
        contact1 = np.zeros((self.num_structures, 28), dtype=int)
        contact2 = np.zeros((self.num_structures, 28), dtype=int)
        with open(contact_map_file, "r") as f:
            for i in range(self.num_structures):
                for j in range(28):
                    inter1, c1, c2 = map(int, f.readline().split())
                    contact1[i][j] = c1
                    contact2[i][j] = c2
        return contact1, contact2

    def random_sequence(self):
        """
        Generate a random sequence of amino acids.
        """
        return [random.randint(1, self.alphabet_size) for _ in range(self.protein_size)]

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

    def calculate_pnat(self, seq, structure_idx):
        """
        Calculate the Boltzmann probability (Pnat) for a sequence folding into a structure.
        """
        pnat = 0.0
        sum_exp = 0.0
        for j in range(self.num_structures):
            energy = self.calculate_energy(seq, j)
            exp_energy = math.exp(-energy)
            sum_exp += exp_energy
            if j == structure_idx:
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

    def generate_msa(
        self, structure_idx, beta, mc_steps, mc_steps_warming, msa_size, output_file
    ):
        """
        Generate an MSA of sequences that fold into the given structure.

        :param structure_idx: Index of the structure for which sequences will fold
        :param beta: Inverse temperature for the Monte Carlo acceptance criterion
        :param mc_steps: Number of Monte Carlo steps between each sequence sampling
        :param msa_size: Number of sequences in the final MSA
        :param output_file: Path to the output file to store the MSA
        """
        # Initialize the first sequence randomly
        seq = self.random_sequence()

        # Calculate the initial Pnat
        pnat = self.calculate_pnat(seq, structure_idx)

        count = 0

        with open(output_file, "w") as f:
            while count < msa_size:
                if count == 0:
                    mc_steps_current = mc_steps_warming
                else:
                    mc_steps_current = mc_steps
                for _ in range(mc_steps_current):
                    # Mutate the sequence
                    new_seq = self.mutate_sequence(seq)

                    # Calculate new Pnat for the mutated sequence
                    new_pnat = self.calculate_pnat(new_seq, structure_idx)

                    # Metropolis criterion
                    if new_pnat >= pnat or random.random() < (new_pnat / pnat) ** beta:
                        seq = new_seq
                        pnat = new_pnat

                # Add the sequence to the MSA
                msa_seq = "".join([self.code[aa - 1] for aa in seq])
                f.write(f"{beta} {pnat} {msa_seq}\n")
                print(f"Sequence {count + 1} added to MSA: {msa_seq}")
                count += 1


def main():
    # Setup argparse for command line argument parsing
    parser = argparse.ArgumentParser(description="Generate lattice protein MSAs.")
    parser.add_argument(
        "--energy_file",
        type=str,
        default="energies.dat",
        help="Path to the energy file.",
    )
    parser.add_argument(
        "--contact_map_file",
        type=str,
        default="contact_maps_10000.dat",
        help="Path to the contact map file.",
    )
    parser.add_argument(
        "--structure_idx", type=int, default=0, help="Structure index for the MSA."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1000.0,
        help="Inverse temperature for the Monte Carlo simulation.",
    )
    parser.add_argument(
        "--mc_steps",
        type=int,
        default=100,
        help="Number of Monte Carlo steps between sequence samples.",
    )
    parser.add_argument(
        "--mc_steps_warming",
        type=int,
        default=1000,
        help="Monte Carlo steps for warming up.",
    )
    parser.add_argument(
        "--msa_size",
        type=int,
        default=100,
        help="Number of sequences in the final MSA.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_msa.txt",
        help="Path to the output file for the MSA.",
    )

    args = parser.parse_args()

    # Initialize the generator with the provided arguments
    generator = ProteinLatticeMSAGenerator(args.energy_file, args.contact_map_file)

    # Generate MSA based on the arguments
    generator.generate_msa(
        args.structure_idx,
        args.beta,
        args.mc_steps,
        args.mc_steps_warming,
        args.msa_size,
        args.output_file,
    )


if __name__ == "__main__":
    main()
