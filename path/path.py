import random
import numpy as np
from tqdm import tqdm
import os


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


def find_mutated_positions(v_start, v_end):
    positions = [
        i for i, (start, end) in enumerate(zip(v_start, v_end)) if start != end
    ]
    return positions


def M(v, i, z):
    """
    mutate the list v at site i with mutation z
    """
    vbis = v.copy()

    vbis[i] = z
    return vbis


def Pi_fixed(v, vbis):
    """
    return the probability of mutating from v to vbis
    """
    if (v == vbis).all():
        return 1
    else:
        return np.exp(-2 * hamming(v, vbis))


def hamming(v, vbis):
    """
    return the hamming distance between v and vbis
    """
    return sum([1 for (start, end) in zip(v, vbis) if start != end])


class Path:
    """
    Args
    T: int
        number of mutations
    Pi: function
        probability of mutating from v to vbis
    Rbm: function
        probability of v to be a native structure
    beta_pi: float
        inverse temperature for Pi
    beta_rbm: float
        inverse temperature for Rbm
    v_start: list
        initial sequence

    alphabet: list
        list of amino acids
    """

    def __init__(
        self,
        T,
        Pi,
        Rbm,
        beta_pi,
        beta_rbm,
        v_start,
        v_end=None,
        extrem="free",
        alphabet=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        code=CODE_RBM,
    ):
        self.v_start = v_start
        self.v_end = v_end
        self.T = T
        self.alphabet = alphabet
        self.Pi = Pi
        self.extrem = extrem
        self.Rbm = Rbm
        self.beta_rbm = beta_rbm
        self.beta_pi = beta_pi

        # check T>=len(positions)
        if self.extrem == "fixed" and self.T + 1 < len(
            find_mutated_positions(self.v_start, self.v_end)
        ):
            raise ValueError(
                "T must be greater than or equal to the number of positions to mutate"
            )

        self.intermediary_strings = self.generate_intermediary_strings()
        self.code = code

    def likelihood(self):
        lll_pi = 0
        for i in range(len(self.intermediary_strings) - 1):
            lll_pi += self.Pi(
                self.intermediary_strings[i], self.intermediary_strings[i + 1]
            )
        lll_rbm = 0
        for i in range(len(self.intermediary_strings)):
            lll_rbm += self.Rbm(self.intermediary_strings[i])

        return lll_pi, lll_rbm

    def generate_intermediary_strings(self):
        if self.extrem == "fixed":
            positions = find_mutated_positions(self.v_start, self.v_end)
            stay_step_number = self.T + 1 - len(positions)
            mutations_path = positions.copy()
            for i in range(stay_step_number):
                mutations_path.append("x")
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(mutations_path)

            intermediary_strings = [self.v_start]

            for i in range(len(mutations_path)):
                mutated_position = mutations_path[i]
                if mutated_position == "x":
                    new_mutant = intermediary_strings[-1]
                else:
                    new_mutant = M(
                        intermediary_strings[-1],
                        mutated_position,
                        self.v_end[mutated_position],
                    )
                intermediary_strings.append(new_mutant)

        elif self.extrem == "free":
            intermediary_strings = [self.v_start]
            for i in range(self.T):
                new_mutant = M(
                    intermediary_strings[-1],
                    np.random.randint(0, len(intermediary_strings[-1])),
                    np.random.choice(self.alphabet),
                )
                intermediary_strings.append(new_mutant)
        intermediary_strings = np.array(intermediary_strings)
        return intermediary_strings

    def get_mutation_path(self):
        intermediary_strings = self.intermediary_strings
        mutation_positions = []
        for i in range(len(intermediary_strings) - 1):
            if intermediary_strings[i] != intermediary_strings[i + 1]:
                mutation_positions.append(
                    find_mutated_positions(
                        intermediary_strings[i], intermediary_strings[i + 1]
                    )[0]
                )
            else:
                mutation_positions.append("x")
        return mutation_positions

    def generate_paths(
        self,
        output_directory,
        warming_steps=10000,
        verbose=True,
        sampling_steps=500,
        paths_nb=10,
    ):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory {output_directory}")

        betas_discount = np.append(
            np.linspace(0.1, 1, warming_steps - warming_steps // 3),
            1.0 ** np.ones(warming_steps // 3),
        )

        acc_count = 0
        for i in tqdm(range(warming_steps)):
            self, acc = MC_mutate_path(
                self,
                Pi=self.Pi,
                Rbm=self.Rbm,
                beta_pi=self.beta_pi * betas_discount[i],
                beta_rbm=self.beta_rbm * betas_discount[i],
            )
            acc_count += acc

            if i % (warming_steps // 10) == 0 and verbose:
                print("accepted ratio", acc_count / (warming_steps // 10))
                acc_count = 0
                path_lll_pi, path_lll_rbm = self.likelihood()
                print("lll_pi", path_lll_pi)
                print("path_lll_rbm", path_lll_rbm)
                pnat_path = [
                    self.Rbm(self.intermediary_strings[i])
                    for i in range(len(self.intermediary_strings))
                ]
                print(pnat_path)

        for path_no in tqdm(range(paths_nb)):
            betas_discount = np.linspace(0.1, 1, sampling_steps)
            for sample_idx in range(sampling_steps):
                self, acc = MC_mutate_path(
                    self,
                    Pi=self.Pi,
                    Rbm=self.Rbm,
                    beta_pi=self.beta_pi * betas_discount[sample_idx],
                    beta_rbm=self.beta_rbm * betas_discount[sample_idx],
                )

            path_array = self.intermediary_strings
            results = []
            fasta_records = []

            for idx, sequence in enumerate(path_array):
                pnat = self.Rbm(sequence)
                seq_string = "".join([self.code[num] for num in sequence])
                results.append(f"1 {pnat} {seq_string}\n")
                fasta_records.append(
                    f">seq_{path_no}_{idx} pnat={pnat:.4f}\n{seq_string}\n"
                )

            # Save the results text file
            output_txt = os.path.join(output_directory, f"output_{path_no}.txt")
            with open(output_txt, "w") as f:
                f.writelines(results)
            if verbose:
                print(f"Saved results to {output_txt}")

            # Save the FASTA file
            output_fasta = os.path.join(output_directory, f"output_{path_no}.fasta")
            with open(output_fasta, "w") as f:
                f.writelines(fasta_records)
            if verbose:
                print(f"Saved FASTA to {output_fasta}")


def MC_mutate_path(path, Pi, Rbm, beta_pi, beta_rbm):
    """
    mutate a path according to the Metropolis-Hastings algorithm
    """

    def compute_delta(v_prev, u):
        # δ = L if u == v_prev, else 1
        return len(v_prev) if np.array_equal(u, v_prev) else 1

    t = np.random.randint(0, path.T) + 1  # position to mutate: cannot be first variant

    # intermediary strings to int16
    path.intermediary_strings = np.array(path.intermediary_strings, dtype=np.int16)
    v_prev = path.intermediary_strings[t - 1]
    v = path.intermediary_strings[t]

    # choose proposal v_new
    if t < path.T:
        v_next = path.intermediary_strings[t + 1]
        ham = hamming(v_prev, v_next)

        if ham == 0:
            random_i = np.random.randint(0, len(v))
            random_letter = np.random.choice(path.alphabet)
            v_new = M(v_prev, random_i, random_letter)

        elif ham == 1:
            i = [l for l, (a, b) in enumerate(zip(v_prev, v_next)) if a != b][0]
            random_letter = np.random.choice(path.alphabet)
            v_new = M(v_prev, i, random_letter)

        elif ham == 2:
            diffs = [l for l, (a, b) in enumerate(zip(v_prev, v_next)) if a != b]
            i, j = diffs
            c1 = M(v_prev, i, v_next[i])
            c2 = M(v_prev, j, v_next[j])
            v_new = c2 if (v == c1).all() else c1
        else:
            raise ValueError("v_prev and v_next must differ by at most 2 mutations")

        ΔlogP = (
            Pi(v_prev, v_new) + Pi(v_new, v_next) - (Pi(v_prev, v) + Pi(v, v_next))
        ) * beta_pi + (Rbm(v_new) - Rbm(v)) * beta_rbm

    else:
        random_i = np.random.randint(0, len(v))
        random_letter = np.random.choice(path.alphabet)
        v_new = M(v_prev, random_i, random_letter)

        ΔlogP = (Pi(v_prev, v_new) - Pi(v_prev, v)) * beta_pi + (
            Rbm(v_new) - Rbm(v)
        ) * beta_rbm

    # compute MH acceptance with proposal asymmetry
    δ_v = compute_delta(v_prev, v)
    δ_vnew = compute_delta(v_prev, v_new)
    p_acc = min(np.exp(ΔlogP) * (δ_v / δ_vnew), 1.0)

    # accept or reject
    if np.random.rand() < p_acc:
        path.intermediary_strings[t] = v_new
        return path, True
    else:
        return path, False
