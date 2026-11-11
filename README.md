# BioSequence2Vec

BioSequence2Vec is an alignment-free embedding for biological sequences. It maps a protein or nucleotide sequence of any length to a fixed-length numerical vector that can be used with any machine learning model.

Each of the `t` coordinates of the embedding is the scaled sum of `+1`/`-1` values that a 4-wise independent hash function assigns to the k-mers of the sequence. The inner product of two embeddings is an unbiased estimate of the k-mer spectrum kernel of the two sequences, and its variance decreases as `1/t`. The method never builds the `|alphabet|^k` k-mer count vector or an `N x N` kernel matrix.

## Installation

```bash
git clone https://github.com/sarwanpasha/Biosequence2Vec.git
cd Biosequence2Vec
pip install -r requirements.txt
```

Python 3.9 or higher is required.

## Quick start

```python
from biosequence2vec import BioSequence2Vec

sequences = ["MFVFLVLLPLVSSQCVNLT", "MFVFLVLLPLVSSQCVNLI", "ACGTACGTTAGC"]
model = BioSequence2Vec(k=3, t=1000, p=4999, seed=0)
X = model.fit_transform(sequences)   # array of shape (3, 1000)
```

Use the same fitted model (same hash functions) for training and test sequences:

```python
model = BioSequence2Vec(k=3, t=1000).fit(train_sequences)
X_train = model.transform(train_sequences)
X_test = model.transform(test_sequences)
```

### Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `k` | k-mer length | 3 |
| `t` | number of hash functions (embedding dimension) | 1000 |
| `p` | prime modulus of the hash functions | 4999 |
| `alphabet` | sequence alphabet; inferred by `fit` if omitted | `None` |
| `seed` | seed for drawing the hash functions | 0 |

Choosing `p` at least as large as `|alphabet|^k` gives every k-mer its own hash input. With a smaller `p`, k-mers whose integer codes are equal modulo `p` share the same signs.

## Scripts

All scripts accept a CSV/TSV file with `sequence` and `label` columns, or a FASTA file where the label is the text after the last `|` of each header (for example `>seq1|B.1.1.7`). To create a small synthetic dataset for testing, run:

```bash
python examples/make_toy_data.py
```

This writes `examples/toy_sequences.csv`.

Evaluate seven classifiers (SVM, naive Bayes, MLP, k-NN, random forest, logistic regression, decision tree) over repeated 70/30 splits:

```bash
python scripts/run_classification.py --data examples/toy_sequences.csv --k 3 --t 1000 --runs 5 --out results.csv
```

Add `--standardize` to standardize the embedding coordinates, which is recommended for linear models.

Check how closely embedding inner products approximate the exact spectrum kernel for different `t`:

```bash
python scripts/check_kernel_approximation.py --data examples/toy_sequences.csv --k 3 --dims 10 100 1000 5000
```

Plot a t-SNE projection of the embeddings:

```bash
python scripts/plot_tsne.py --data examples/toy_sequences.csv --t 1000 --out tsne.png
```

## Repository layout

```
biosequence2vec/
    embedding.py                  BioSequence2Vec class and exact k-mer spectrum helpers
    io.py                         readers for CSV/TSV and FASTA files
scripts/
    run_classification.py         classification benchmark with seven classifiers
    check_kernel_approximation.py comparison with the exact spectrum kernel
    plot_tsne.py                  t-SNE visualization
examples/
    make_toy_data.py              generator of a small synthetic dataset
```

## Citation

If you use this code, please cite:

Ali S, Sardar U, Patterson M, Khan I. BioSequence2Vec: Efficient embedding generation for biological sequences. In: Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD). Springer; 2023. p. 173-185. https://doi.org/10.1007/978-3-031-33377-4_14
