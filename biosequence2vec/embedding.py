"""Core implementation of the BioSequence2Vec embedding.

Each of the t output coordinates is the scaled sum of +1/-1 values that a
4-wise independent hash function assigns to the k-mers of a sequence:

    x_hat[i] = (1 / sqrt(t)) * sum_j h_i(alpha_j),

where alpha_j runs over the k-mers of the sequence. The hash functions follow
the Carter-Wegman polynomial construction

    g(a) = (a0 + a1*a + a2*a^2 + a3*a^3) mod p,   h(a) = -1 if g(a) is even else +1,

with the k-mer encoded as an integer in base |alphabet|. The same t hash
functions are used for every sequence, so inner products between embeddings
estimate the k-mer spectrum kernel.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["BioSequence2Vec", "kmer_spectrum", "spectrum_kernel"]


def _kmer_codes(seq: str, k: int, char_to_code: Dict[str, int]) -> np.ndarray:
    """Return the base-|alphabet| integer code of every valid k-mer in seq.

    k-mers that contain a character outside the alphabet are skipped.
    """
    base = len(char_to_code)
    codes = np.fromiter((char_to_code.get(c, -1) for c in seq), dtype=np.int64, count=len(seq))
    n_kmers = len(seq) - k + 1
    if n_kmers <= 0:
        return np.empty(0, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (windows >= 0).all(axis=1)
    windows = windows[valid]
    powers = base ** np.arange(k - 1, -1, -1, dtype=np.int64)
    return windows @ powers


class BioSequence2Vec:
    """Hash-based embedding of biological sequences.

    Parameters
    ----------
    k : int
        Length of the k-mers.
    t : int
        Number of hash functions, which is also the embedding dimension.
    p : int
        Prime modulus of the hash functions. A value of at least |alphabet|**k
        avoids k-mers that share a residue modulo p.
    alphabet : str or sequence of str, optional
        Symbols of the alphabet. If omitted, the alphabet is inferred from the
        sequences passed to ``fit``.
    seed : int, optional
        Seed of the random generator that draws the hash coefficients.
    """

    def __init__(self, k: int = 3, t: int = 1000, p: int = 4999,
                 alphabet: Optional[Sequence[str]] = None, seed: Optional[int] = 0):
        if k < 1 or t < 1:
            raise ValueError("k and t must be positive integers")
        if p < 3 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            raise ValueError("p must be an odd prime")
        self.k = k
        self.t = t
        self.p = p
        self.seed = seed
        self.alphabet = None if alphabet is None else list(alphabet)
        self.coefficients_ = None
        self.char_to_code_ = None
        if self.alphabet is not None:
            self._setup()

    def _setup(self) -> None:
        self.char_to_code_ = {c: i for i, c in enumerate(self.alphabet)}
        rng = np.random.default_rng(self.seed)
        # Row i holds (a0, a1, a2, a3) of hash function i. Drawn once and reused
        # for every sequence.
        self.coefficients_ = rng.integers(0, self.p, size=(self.t, 4), dtype=np.int64)

    def fit(self, sequences: Iterable[str]) -> "BioSequence2Vec":
        """Infer the alphabet (if not given) and draw the hash functions."""
        if self.alphabet is None:
            symbols = set()
            for s in sequences:
                symbols.update(s)
            self.alphabet = sorted(symbols)
        self._setup()
        return self

    def _signs(self, keys: np.ndarray) -> np.ndarray:
        """Return the t x len(keys) matrix of hash values in {-1, +1}."""
        a = self.coefficients_
        x = keys % self.p
        # Horner's rule modulo p keeps every intermediate value below p**2.
        g = a[:, 3:4] * x % self.p
        g = (g + a[:, 2:3]) * x % self.p
        g = (g + a[:, 1:2]) * x % self.p
        g = (g + a[:, 0:1]) % self.p
        return (2 * (g & 1) - 1).astype(np.float64)

    def transform_one(self, seq: str) -> np.ndarray:
        """Embed a single sequence."""
        if self.coefficients_ is None:
            raise RuntimeError("call fit() first or pass an alphabet")
        codes = _kmer_codes(seq, self.k, self.char_to_code_)
        if codes.size == 0:
            return np.zeros(self.t)
        # Identical k-mers receive identical signs, so summing over distinct
        # k-mers weighted by their counts equals summing over all occurrences.
        keys, counts = np.unique(codes, return_counts=True)
        return self._signs(keys) @ counts.astype(np.float64) / np.sqrt(self.t)

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        """Embed a collection of sequences into an (N, t) array."""
        return np.vstack([self.transform_one(s) for s in sequences])

    def fit_transform(self, sequences: Sequence[str]) -> np.ndarray:
        sequences = list(sequences)
        return self.fit(sequences).transform(sequences)


def kmer_spectrum(sequences: Iterable[str], k: int) -> List[Counter]:
    """Exact k-mer counts of each sequence, stored sparsely as Counters."""
    out = []
    for s in sequences:
        out.append(Counter(s[j:j + k] for j in range(len(s) - k + 1)))
    return out


def spectrum_kernel(a: Counter, b: Counter) -> float:
    """Spectrum kernel (inner product of k-mer counts) of two sequences."""
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(key, 0) for key, v in a.items()))
