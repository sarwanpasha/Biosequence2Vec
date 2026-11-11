"""Embed labeled sequences with BioSequence2Vec and evaluate seven classifiers.

Each run draws a new stratified 70/30 train-test split. The script reports the
mean and standard deviation of every metric over the runs.

Example:
    python scripts/run_classification.py --data examples/toy_sequences.csv --k 3 --t 1000 --runs 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from biosequence2vec import BioSequence2Vec  # noqa: E402
from biosequence2vec.io import read_labeled_sequences  # noqa: E402


def classifiers(seed: int, standardize: bool):
    models = {
        "SVM": SVC(kernel="linear", random_state=seed),
        "NB": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "LR": LogisticRegression(max_iter=1000),
        "DT": DecisionTreeClassifier(random_state=seed),
    }
    if standardize:
        models = {name: make_pipeline(StandardScaler(), m) for name, m in models.items()}
    return models


def evaluate(model, X_train, y_train, X_test, y_test, n_classes):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        try:
            proba = model.predict_proba(X_test)
        except AttributeError:
            dec = model.decision_function(X_test)
            if dec.ndim == 1:
                dec = np.column_stack([-dec, dec])
            dec = np.exp(dec - dec.max(axis=1, keepdims=True))
            proba = dec / dec.sum(axis=1, keepdims=True)
        full = np.zeros((len(y_test), n_classes))
        full[:, model.classes_] = proba
        present = np.unique(y_test)
        if len(present) > 1:
            full = full[:, present]
            full = full / np.clip(full.sum(axis=1, keepdims=True), 1e-12, None)
            y_idx = np.searchsorted(present, y_test)
            scores["roc_auc_ovr"] = roc_auc_score(y_idx, full, multi_class="ovr",
                                                  labels=np.arange(len(present)))
        else:
            scores["roc_auc_ovr"] = float("nan")
    except (AttributeError, ValueError):
        scores["roc_auc_ovr"] = float("nan")
    scores["train_time_s"] = train_time
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV/TSV with 'sequence' and 'label' columns, or FASTA")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--t", type=int, default=1000)
    parser.add_argument("--p", type=int, default=4999)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--standardize", action="store_true",
                        help="standardize embedding coordinates before training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    seqs, labels = read_labeled_sequences(args.data)
    y = LabelEncoder().fit_transform(labels)
    n_classes = int(y.max()) + 1

    start = time.perf_counter()
    X = BioSequence2Vec(k=args.k, t=args.t, p=args.p, seed=args.seed).fit_transform(seqs)
    print(f"Embedded {len(seqs)} sequences into {X.shape[1]} dimensions "
          f"in {time.perf_counter() - start:.2f} s")

    results = {}
    for run in range(args.runs):
        stratify = y if np.bincount(y).min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed + run, stratify=stratify)
        for name, model in classifiers(args.seed + run, args.standardize).items():
            results.setdefault(name, []).append(evaluate(model, X_tr, y_tr, X_te, y_te, n_classes))

    metrics = list(next(iter(results.values()))[0].keys())
    with open(args.out, "w") as fh:
        fh.write("classifier," + ",".join(f"{m}_mean,{m}_std" for m in metrics) + "\n")
        for name, runs in results.items():
            row = [name]
            for m in metrics:
                vals = np.array([r[m] for r in runs], dtype=float)
                row += [f"{np.nanmean(vals):.4f}", f"{np.nanstd(vals):.4f}"]
            fh.write(",".join(row) + "\n")
            print(f"{name:4s} accuracy {np.mean([r['accuracy'] for r in runs]):.3f}  "
                  f"macro F1 {np.mean([r['f1_macro'] for r in runs]):.3f}")
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
