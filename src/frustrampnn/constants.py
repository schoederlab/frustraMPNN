"""
Package-wide constants for FrustraMPNN.

This module defines constants used throughout the package including
amino acid alphabets, model dimensions, and frustration thresholds.
"""

__all__ = [
    "ALPHABET",
    "VOCAB_DIM",
    "HIDDEN_DIM",
    "EMBED_DIM",
    "AA_3_TO_1",
    "AA_1_TO_3",
    "FRUSTRATION_THRESHOLDS",
    "FRUSTRATION_COLORS",
]

# Standard 21-character amino acid alphabet (with X for unknown)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
VOCAB_DIM = len(ALPHABET)

# Model dimensions (from ProteinMPNN)
HIDDEN_DIM = 128
EMBED_DIM = 128

# Three-letter to one-letter amino acid conversion
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',  # Selenomethionine treated as methionine
}

# One-letter to three-letter amino acid conversion
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items() if k != 'MSE'}

# Frustration classification thresholds
FRUSTRATION_THRESHOLDS = {
    'highly': -1.0,      # FrstIndex <= -1.0
    'minimally': 0.58,   # FrstIndex >= 0.58
    # neutral: -1.0 < FrstIndex < 0.58
}

# Frustration visualization colors (matching frustrapy)
FRUSTRATION_COLORS = {
    'highly': 'red',
    'neutral': 'gray',
    'minimally': 'green',
    'native': 'blue',
}


