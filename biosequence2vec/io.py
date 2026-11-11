"""Helpers for reading labeled sequences."""

from __future__ import annotations

import csv
from typing import List, Tuple


def read_labeled_sequences(path: str, seq_col: str = "sequence",
                           label_col: str = "label") -> Tuple[List[str], List[str]]:
    """Read sequences and labels from a CSV/TSV file or a FASTA file.

    CSV/TSV files need a header with the sequence and label columns.
    In FASTA files the label is the text after the last '|' in each header,
    for example ``>seq1|B.1.1.7``.
    """
    if path.lower().endswith((".fa", ".fasta", ".fas")):
        return _read_fasta(path)
    delimiter = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    seqs, labels = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            seqs.append(row[seq_col].strip().upper())
            labels.append(row[label_col].strip())
    return seqs, labels


def _read_fasta(path: str) -> Tuple[List[str], List[str]]:
    seqs, labels, chunks, label = [], [], [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    seqs.append("".join(chunks).upper())
                    labels.append(label)
                label = line[1:].split("|")[-1].strip()
                chunks = []
            else:
                chunks.append(line)
    if label is not None:
        seqs.append("".join(chunks).upper())
        labels.append(label)
    return seqs, labels
