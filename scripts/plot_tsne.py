"""Plot a two-dimensional t-SNE projection of BioSequence2Vec embeddings.

The figure has no title and uses large fonts so that it remains readable
when scaled down in a document.

Example:
    python scripts/plot_tsne.py --data examples/toy_sequences.csv --t 1000 --out tsne.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from biosequence2vec import BioSequence2Vec  # noqa: E402
from biosequence2vec.io import read_labeled_sequences  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--t", type=int, default=1000)
    parser.add_argument("--p", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tsne.png")
    args = parser.parse_args()

    seqs, labels = read_labeled_sequences(args.data)
    X = BioSequence2Vec(k=args.k, t=args.t, p=args.p, seed=args.seed).fit_transform(seqs)
    perplexity = min(30, max(5, len(seqs) // 4))
    Y = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init="pca").fit_transform(X)

    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(set(labels)):
        mask = np.array(labels) == lab
        ax.scatter(Y[mask, 0], Y[mask, 1], s=12, label=lab)
    ax.legend(markerscale=2, fontsize=14, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
