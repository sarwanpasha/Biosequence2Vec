"""Compare BioSequence2Vec inner products with the exact k-mer spectrum kernel.

For random pairs of sequences, the script reports the mean absolute error of the
estimated cosine similarity for several embedding dimensions t. By the variance
bound of the method, the error should shrink roughly as 1/sqrt(t).

Example:
    python scripts/check_kernel_approximation.py --data examples/toy_sequences.csv --k 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from biosequence2vec import BioSequence2Vec, kmer_spectrum, spectrum_kernel  # noqa: E402
from biosequence2vec.io import read_labeled_sequences  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--p", type=int, default=None,
                        help="prime modulus; defaults to the smallest prime that is at least |alphabet|**k")
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 100, 1000, 5000])
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seqs, _ = read_labeled_sequences(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(seqs), size=(args.pairs, 2))
    spectra = kmer_spectrum(seqs, args.k)
    norms = np.sqrt([spectrum_kernel(s, s) for s in spectra])
    exact = np.array([spectrum_kernel(spectra[a], spectra[b]) / (norms[a] * norms[b]) for a, b in idx])

    alphabet = sorted(set("".join(seqs)))
    p = args.p
    if p is None:
        p = len(alphabet) ** args.k
        while p < 3 or any(p % d == 0 for d in range(2, int(p ** 0.5) + 1)):
            p += 1
    used = sorted(set(idx.ravel()))
    print(f"k = {args.k}, alphabet size = {len(alphabet)}, p = {p}")
    for t in args.dims:
        model = BioSequence2Vec(k=args.k, t=t, p=p, alphabet=alphabet, seed=args.seed)
        emb = dict(zip(used, model.transform([seqs[i] for i in used])))
        est = np.array([emb[a] @ emb[b] / (norms[a] * norms[b]) for a, b in idx])
        err = np.abs(est - exact)
        print(f"t = {t:6d}  mean abs error = {err.mean():.4f}  95th percentile = {np.quantile(err, 0.95):.4f}")


if __name__ == "__main__":
    main()
