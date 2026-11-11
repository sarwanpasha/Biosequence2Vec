"""BioSequence2Vec: hash-based embeddings for biological sequences."""

from .embedding import BioSequence2Vec, kmer_spectrum, spectrum_kernel

__all__ = ["BioSequence2Vec", "kmer_spectrum", "spectrum_kernel"]
__version__ = "1.0.0"
