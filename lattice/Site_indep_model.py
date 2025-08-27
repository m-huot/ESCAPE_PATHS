import numpy as np


class SiteIndependentModel:
    """
    A site-independent (position-specific) generative model for discrete sequences.

    Parameters
    ----------
    beta : float, optional
        Inverse-temperature scaling factor applied to the log-likelihood.
    num_values : int, optional
        Alphabet size (default 21 ⇒ symbols 0 … 20).
    """

    def __init__(self, beta: float = 1.0, num_values: int = 20) -> None:
        self.beta = float(beta)
        self.num_values = int(num_values)
        self._log_probs: np.ndarray | None = None  # shape (L, num_values)

    # --------------------------------------------------------------------- #
    # Fitting                                                               #
    # --------------------------------------------------------------------- #
    def fit(
        self,
        sequences: np.ndarray,  # shape (N, L)  | dtype int
        frequencies: np.ndarray | None = None,  # shape (N,)   | dtype float
    ) -> None:
        """Estimate per-site symbol probabilities with optional sequence weights."""
        if sequences.ndim != 2:
            raise ValueError("`sequences` must be a 2-D array of shape (N, L).")
        N, L = sequences.shape
        if frequencies is None:
            frequencies = np.ones(N, dtype=float)
        frequencies = frequencies.astype(float)

        # One-hot encode: shape (N, L, num_values)
        one_hot = np.zeros((N, L, self.num_values), dtype=float)
        one_hot[
            np.arange(N)[:, None],  # broadcast over positions
            np.arange(L)[None, :],
            sequences,
        ] = 1.0

        # Weighted counts per position / symbol
        weighted_counts = (frequencies[:, None, None] * one_hot).sum(axis=0)  # (L, V)

        # Convert to probabilities then log-probabilities
        totals = weighted_counts.sum(axis=1, keepdims=True)  # (L, 1)
        probs = weighted_counts / np.clip(totals, 1e-12, None)  # avoid 0-div
        self._log_probs = np.log(np.clip(probs, 1e-12, None))  # store log(P)

    # --------------------------------------------------------------------- #
    # Scoring                                                               #
    # --------------------------------------------------------------------- #
    def log_prob(self, sequence: np.ndarray) -> float:
        """
        Log-likelihood of a single sequence (length L) under the model.
        Returns **natural-log space** value (not multiplied by −1).
        """
        if self._log_probs is None:
            raise RuntimeError("Call `.fit()` before scoring.")
        if sequence.ndim != 1:
            raise ValueError("`sequence` must be a 1-D array of length L.")
        return self._log_probs[np.arange(sequence.size), sequence].sum() * self.beta

    def neg_log_prob(self, sequence: np.ndarray) -> float:
        """Convenience wrapper returning the negative log-likelihood."""
        return -self.log_prob(sequence)

    # --------------------------------------------------------------------- #
    # Sampling                                                               #
    # --------------------------------------------------------------------- #
    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Draw `n` independent samples from the fitted model.

        Returns
        -------
        sequences : ndarray
            Shape (n, L) if n>1 else (L,)  – dtype int64.
        """
        if self._log_probs is None:
            raise RuntimeError("Call `.fit()` before sampling.")
        rng = np.random.default_rng() if rng is None else rng
        probs = np.exp(self._log_probs)  # (L, V)
        L, V = probs.shape

        out = np.empty((n, L), dtype=np.int64)
        for pos in range(L):
            out[:, pos] = rng.choice(V, p=probs[pos], size=n)

        return out.squeeze(0) if n == 1 else out


# ===================================================================== #
# A baseline model with *uniform* per-site probabilities                #
# ===================================================================== #
class RandomSiteIndependentModel:
    """
    Site-independent model in which every symbol at every position
    has equal probability (1 / num_values).
    """

    def __init__(self, sequence_length: int, beta: float = 1.0, num_values: int = 21):
        self.L = int(sequence_length)
        self.beta = float(beta)
        self.num_values = int(num_values)
        # Store log-probabilities once; they never change
        self._log_prob_row = np.log(np.full(self.num_values, 1.0 / self.num_values))
        self._log_probs = np.tile(self._log_prob_row, (self.L, 1))  # shape (L, V)

    # No-op `fit` to match the API
    def fit(self, *args, **kwargs) -> None:
        pass

    # Same interface as the first class
    def log_prob(self, sequence: np.ndarray) -> float:
        if sequence.ndim != 1 or sequence.size != self.L:
            raise ValueError(f"`sequence` must be a 1-D array of length {self.L}.")
        return self._log_probs[np.arange(self.L), sequence].sum() * self.beta

    def neg_log_prob(self, sequence: np.ndarray) -> float:
        return -self.log_prob(sequence)

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        out = rng.integers(0, self.num_values, size=(n, self.L), dtype=np.int64)
        return out.squeeze(0) if n == 1 else out
