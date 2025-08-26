import numpy as np
import random
import os
from typing import Callable, List
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Alphabet definitions
# ---------------------------------------------------------------------------

# Nucleotides encoded as integers ------------------------------------------------
CODE_NT: List[str] = ["A", "C", "G", "T"]  # index → base letter
NT_ALPHABET: List[int] = list(range(4))  # 0=A, 1=C, 2=G, 3=T
NT_TO_CODE = {nt: i for i, nt in enumerate(CODE_NT)}

# Amino‑acid alphabet expected by the RBM ---------------------------------------
CODE_RBM: List[str] = [
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
AA_TO_IDX = {aa: idx for idx, aa in enumerate(CODE_RBM)}  # letter → integer

# Standard genetic code (DNA codons → amino‑acid letter) -------------------------
STD_GENETIC_CODE = {
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TGT": "C",
    "TGC": "C",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TTT": "F",
    "TTC": "F",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "CAT": "H",
    "CAC": "H",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "AAA": "K",
    "AAG": "K",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATG": "M",
    "AAT": "N",
    "AAC": "N",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TGG": "W",
    "TAT": "Y",
    "TAC": "Y",
    # Stop codons
}
STOP_CODONS = {"TAA", "TAG", "TGA"}

# DNA‑codon → integer amino‑acid mapping (stop codons excluded!) -----------------
CODON_TO_IDX = {codon: AA_TO_IDX[aa] for codon, aa in STD_GENETIC_CODE.items()}

# ---------------------------------------------------------------------------
#  Codon-bias term  log f(c | a)  under uniform bias -----------------------------
# ---------------------------------------------------------------------------

# Count synonymous codons for each amino acid
CODONS_BY_AA = {}
for codon, aa in STD_GENETIC_CODE.items():
    CODONS_BY_AA.setdefault(aa, []).append(codon)

# Pre-compute log f(c|a) = log(1 / N(a)) for every sense codon
LOG_F_CODON = {
    codon: np.log(1.0 / len(CODONS_BY_AA[aa])) for codon, aa in STD_GENETIC_CODE.items()
}


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


def nts_to_str(v: np.ndarray) -> str:
    """Convert integer‑encoded nt array to string like 'ACGT…'."""
    return "".join(CODE_NT[x] for x in v)


def str_to_nts(s: str) -> np.ndarray:
    """Convert nucleotide string like 'ACGT…' to integer‑encoded nt array."""
    return np.array([NT_TO_CODE[char] for char in s], dtype=np.int8)


def has_stop(nt_seq: np.ndarray) -> bool:
    """Return True if *nt_seq* (int‑encoded) contains any stop codon."""
    s = nts_to_str(nt_seq)
    return any(s[i : i + 3] in STOP_CODONS for i in range(0, len(s), 3))


def translate_idx(nt_seq: np.ndarray) -> np.ndarray:
    """Translate integer‑encoded nucleotide sequence to integer‑encoded AA seq.

    Raises
    ------
    ValueError
        If sequence length is not a multiple of 3 or contains stop/unknown codon.
    """
    if has_stop(nt_seq):
        raise ValueError("Sequence contains stop codon – cannot translate.")

    s = nts_to_str(nt_seq)
    if len(s) % 3 != 0:
        raise ValueError("Nucleotide sequence length must be a multiple of 3.")

    aa_idx: List[int] = []
    for i in range(0, len(s), 3):
        codon = s[i : i + 3]
        try:
            aa_idx.append(CODON_TO_IDX[codon])
        except KeyError:
            raise ValueError(f"Unknown codon {codon}.")
    return np.asarray(aa_idx, dtype=np.int16)


def hamming(v1: np.ndarray, v2: np.ndarray) -> int:
    return int(np.sum(v1 != v2))


def find_mutated_positions(v_start: np.ndarray, v_end: np.ndarray):
    return [i for i, (a, b) in enumerate(zip(v_start, v_end)) if a != b]


def mutate(v: np.ndarray, i: int, new_nt: int) -> np.ndarray:
    out = v.copy()
    out[i] = new_nt
    return out


# ---------------------------------------------------------------------------
#  Path sampler in nucleotide space (no Pi term)
# ---------------------------------------------------------------------------


class PathNT:
    """MCMC sampler generating evolutionary paths in nucleotide space.

    * Each edge (v_t−1 → v_t) differs by ≤ 1 nucleotide (0 allowed).
    * Candidates containing a stop codon are **automatically rejected**.
    """

    def __init__(
        self,
        T: int,
        rbm_energy: Callable[[np.ndarray], float],
        beta: float,
        v_start: np.ndarray,
        v_end: np.ndarray | None = None,
        extrem: str = "free",
        alphabet: List[int] = NT_ALPHABET,
        seed: int | None = None,
    ):
        self.T, self.beta = int(T), float(beta)
        self.rbm_energy = rbm_energy
        self.v_start = np.asarray(v_start, dtype=np.int16)
        self.v_end = None if v_end is None else np.asarray(v_end, dtype=np.int16)
        self.extrem, self.alphabet = extrem, alphabet
        self.random_state(seed)

        if extrem == "fixed" and self.v_end is None:
            raise ValueError("v_end required for extrem='fixed'.")
        if extrem == "fixed" and self.T + 1 < len(
            find_mutated_positions(self.v_start, self.v_end)
        ):
            raise ValueError("T too small for fixed endpoints.")
        if has_stop(self.v_start) or (self.v_end is not None and has_stop(self.v_end)):
            raise ValueError("Start/target sequence contains stop codon.")

        self.intermediary_strings = self._initial_path()

    # ------------------------------------------------------------------
    def random_state(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def nt_energy(self, nt_seq):
        """Return minus energy"""
        aa_idx_seq = translate_idx(nt_seq)  # integer-encoded AA seq
        E_aa = -self.rbm_energy(aa_idx_seq)  # RBM term

        codon_str = nts_to_str(nt_seq)
        logf_sum = sum(
            LOG_F_CODON[codon_str[i : i + 3]] for i in range(0, len(codon_str), 3)
        )
        # logf_sum = 0

        return -E_aa + logf_sum

    # ------------------------------------------------------------------
    def _initial_path(self):
        if self.extrem == "fixed":
            positions = find_mutated_positions(self.v_start, self.v_end)
            stays = self.T + 1 - len(positions)
            moves = positions + ["x"] * stays
            random.shuffle(moves)
            nodes = [self.v_start]
            for m in moves:
                nodes.append(
                    nodes[-1] if m == "x" else mutate(nodes[-1], m, self.v_end[m])
                )
            return np.asarray(nodes, dtype=np.int16)
        nodes = [self.v_start]
        for _ in range(self.T):
            while True:
                cur = nodes[-1]
                pos = np.random.randint(len(cur))
                nt = random.choice(self.alphabet)
                if nt == cur[pos]:
                    continue
                cand = mutate(cur, pos, nt)
                if not has_stop(cand):
                    nodes.append(cand)
                    break
        return np.asarray(nodes, dtype=np.int16)

    # ------------------------------------------------------------------
    @staticmethod
    def _delta(v_prev, u):
        return len(v_prev) if np.array_equal(v_prev, u) else 1

    def _valid_candidate(self, seq):
        return not has_stop(seq)

    def mh_step(self):
        t = np.random.randint(1, self.T + 1)
        v_prev, v = self.intermediary_strings[t - 1], self.intermediary_strings[t]

        # ------------------ proposal generation (≤1 nt diff) -------------
        max_retry = 100
        for _ in range(max_retry):
            if t < self.T:
                v_next = self.intermediary_strings[t + 1]
                ham = hamming(v_prev, v_next)
                if ham == 0:
                    pos = np.random.randint(len(v_prev))
                    nt = random.choice(self.alphabet)
                    v_new = mutate(v_prev, pos, nt)

                elif ham == 1:
                    i = next(
                        idx for idx, (a, b) in enumerate(zip(v_prev, v_next)) if a != b
                    )
                    nt = random.choice(self.alphabet)
                    v_new = mutate(v_prev, i, nt)

                elif ham == 2:
                    diffs = [
                        idx for idx, (a, b) in enumerate(zip(v_prev, v_next)) if a != b
                    ]
                    i, j = diffs
                    cand1 = mutate(v_prev, i, v_next[i])
                    cand2 = mutate(v_prev, j, v_next[j])
                    v_new = cand2 if np.array_equal(v, cand1) else cand1
                else:
                    raise ValueError("Neighbours differ by >2 nts.")
            else:
                pos = np.random.randint(len(v_prev))
                nt = random.choice(self.alphabet)
                v_new = mutate(v_prev, pos, nt)

            if self._valid_candidate(v_new):
                break
        else:
            return False  # failed to find stop‑free candidate

        # ------------------ MH acceptance -------------------------------
        delta_logP = self.beta * (self.nt_energy(v_new) - self.nt_energy(v))

        δ_v = self._delta(v_prev, v)
        δ_vnew = self._delta(v_prev, v_new)

        p_acc = min(1.0, np.exp(delta_logP) * (δ_v / δ_vnew))

        if np.random.rand() < p_acc:
            self.intermediary_strings[t] = v_new
            return True
        return False

    # ------------------------------------------------------------------
    def sample_paths(
        self,
        out_dir,
        warming_steps=10000,
        sampling_steps=1000,
        paths_nb=10,
        verbose=True,
    ):
        os.makedirs(out_dir, exist_ok=True)
        for _ in tqdm(range(warming_steps), disable=not verbose, desc="burn‑in"):
            self.mh_step()
        for p in range(paths_nb):
            for _ in range(sampling_steps):
                self.mh_step()
            txt, fasta = [], []
            for idx, nt_seq in enumerate(self.intermediary_strings):
                logp = self.nt_energy(nt_seq)
                s = nts_to_str(nt_seq)
                txt.append(f"1 {logp} {s}\n")
                fasta.append(f">path{p}_step{idx} logp={logp:.4f}\n{s}\n")
            with open(os.path.join(out_dir, f"output_{p}.txt"), "w") as fh:
                fh.writelines(txt)
            with open(os.path.join(out_dir, f"output_{p}.fasta"), "w") as fh:
                fh.writelines(fasta)
            if verbose:
                print(f"Saved path {p} to {out_dir}")


# ---------------------------------------------------------------------------
#  Wrapper & dummy RBM
# ---------------------------------------------------------------------------


def make_nt_energy_from_rbm(rbm_fn):
    return lambda nt_seq: rbm_fn(translate_idx(nt_seq))


def dummy_rbm(aa_idx_seq):
    return -float(np.sum(aa_idx_seq != AA_TO_IDX["A"]))


# ---------------------------------------------------------------------------
