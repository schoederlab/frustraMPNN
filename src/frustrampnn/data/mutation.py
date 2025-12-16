"""
Mutation data structures for FrustraMPNN.

This module provides the Mutation dataclass and related utilities
for representing amino acid mutations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from frustrampnn.constants import ALPHABET

__all__ = [
    "Mutation",
    "parse_mutation_string",
    "generate_ssm_mutations",
]


@dataclass
class Mutation:
    """
    Represents a single amino acid mutation.

    Attributes:
        position: 0-indexed position in the sequence
        wildtype: Single-letter wild-type amino acid
        mutation: Single-letter mutant amino acid
        frustration: Target frustration value (for training)
        pdb: PDB identifier
        chain: Chain identifier
        weight: Sample weight (for loss reweighting)

    Example:
        >>> mut = Mutation(position=72, wildtype='A', mutation='G')
        >>> print(mut)
        Mutation(A73G)  # Note: displayed as 1-indexed
    """

    position: int
    wildtype: str
    mutation: str
    frustration: torch.Tensor | None = None
    pdb: str | None = None
    chain: str | None = None
    weight: float | None = None

    def __post_init__(self) -> None:
        """Validate mutation data."""
        if self.wildtype not in ALPHABET:
            raise ValueError(f"Invalid wildtype amino acid: {self.wildtype}")
        if self.mutation not in ALPHABET:
            raise ValueError(f"Invalid mutation amino acid: {self.mutation}")
        if self.position < 0:
            raise ValueError(f"Position must be non-negative: {self.position}")

    def __str__(self) -> str:
        """Return human-readable mutation string (1-indexed)."""
        return f"{self.wildtype}{self.position + 1}{self.mutation}"

    def __repr__(self) -> str:
        return f"Mutation({self})"

    @property
    def is_synonymous(self) -> bool:
        """Check if mutation is synonymous (same amino acid)."""
        return self.wildtype == self.mutation

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "position": self.position,
            "wildtype": self.wildtype,
            "mutation": self.mutation,
            "pdb": self.pdb,
            "chain": self.chain,
        }


def parse_mutation_string(mutation_str: str) -> Mutation:
    """
    Parse a mutation string into a Mutation object.

    Supports formats:
        - "A73G" (1-indexed)
        - "A72G" with zero_indexed=True

    Args:
        mutation_str: Mutation string like "A73G"

    Returns:
        Mutation: Parsed mutation object (0-indexed internally)

    Example:
        >>> mut = parse_mutation_string("A73G")
        >>> mut.position  # 0-indexed
        72
    """
    # Pattern: single letter, number, single letter
    pattern = r"^([A-Z])(\d+)([A-Z])$"
    match = re.match(pattern, mutation_str.upper())

    if not match:
        raise ValueError(
            f"Invalid mutation format: {mutation_str}. Expected format: A73G"
        )

    wildtype, position_str, mutation = match.groups()
    position = int(position_str) - 1  # Convert to 0-indexed

    return Mutation(
        position=position,
        wildtype=wildtype,
        mutation=mutation,
    )


def generate_ssm_mutations(
    sequence: str,
    positions: list[int] | None = None,
    include_synonymous: bool = True,
) -> list[Mutation]:
    """
    Generate site-saturation mutagenesis mutations.

    Creates all possible single amino acid mutations for each position
    in the sequence.

    Args:
        sequence: Amino acid sequence
        positions: Specific positions to mutate (0-indexed). None = all positions.
        include_synonymous: Include mutations to the same amino acid

    Returns:
        List[Mutation]: List of all possible mutations

    Example:
        >>> mutations = generate_ssm_mutations("MAG", positions=[0])
        >>> len(mutations)  # 20 mutations at position 0
        20
    """
    mutations = []

    for pos, wt_aa in enumerate(sequence):
        # Skip if position not in requested list
        if positions is not None and pos not in positions:
            continue

        # Skip gaps and unknown residues
        if wt_aa == "-" or wt_aa not in ALPHABET:
            continue

        # Generate all mutations at this position
        for mut_aa in ALPHABET[:-1]:  # Exclude 'X'
            if not include_synonymous and wt_aa == mut_aa:
                continue

            mutations.append(
                Mutation(
                    position=pos,
                    wildtype=wt_aa,
                    mutation=mut_aa,
                )
            )

    return mutations
