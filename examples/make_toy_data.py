"""Create a small synthetic dataset for testing the scripts.

Three classes of random protein-like sequences, each with a class-specific
motif inserted at random positions. The output is examples/toy_sequences.csv.
"""

import os
import random

random.seed(7)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MOTIFS = {"class_A": "MFVFLVLLP", "class_B": "KRRGQWEHT", "class_C": "YYNDPCSAV"}

rows = ["sequence,label"]
for label, motif in MOTIFS.items():
    for _ in range(40):
        length = random.randint(150, 250)
        seq = [random.choice(AMINO_ACIDS) for _ in range(length)]
        for _ in range(6):
            pos = random.randint(0, length - len(motif))
            seq[pos:pos + len(motif)] = list(motif)
        rows.append("".join(seq) + "," + label)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_sequences.csv")
with open(out, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"Wrote {len(rows) - 1} sequences to {out}")
