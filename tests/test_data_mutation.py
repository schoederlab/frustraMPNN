"""Tests for Mutation dataclass and utilities."""

import pytest


def test_mutation_creation():
    """Test basic Mutation creation."""
    from frustrampnn.data import Mutation

    mut = Mutation(position=72, wildtype="A", mutation="G")

    assert mut.position == 72
    assert mut.wildtype == "A"
    assert mut.mutation == "G"


def test_mutation_str():
    """Test Mutation string representation (1-indexed)."""
    from frustrampnn.data import Mutation

    mut = Mutation(position=72, wildtype="A", mutation="G")
    assert str(mut) == "A73G"  # 1-indexed for display


def test_mutation_repr():
    """Test Mutation repr."""
    from frustrampnn.data import Mutation

    mut = Mutation(position=72, wildtype="A", mutation="G")
    assert repr(mut) == "Mutation(A73G)"


def test_mutation_validation_invalid_wildtype():
    """Test Mutation validation for invalid wildtype."""
    from frustrampnn.data import Mutation

    with pytest.raises(ValueError, match="Invalid wildtype amino acid"):
        Mutation(position=0, wildtype="Z", mutation="A")


def test_mutation_validation_invalid_mutation():
    """Test Mutation validation for invalid mutation."""
    from frustrampnn.data import Mutation

    with pytest.raises(ValueError, match="Invalid mutation amino acid"):
        Mutation(position=0, wildtype="A", mutation="Z")


def test_mutation_validation_negative_position():
    """Test Mutation validation for negative position."""
    from frustrampnn.data import Mutation

    with pytest.raises(ValueError, match="Position must be non-negative"):
        Mutation(position=-1, wildtype="A", mutation="G")


def test_mutation_is_synonymous():
    """Test synonymous mutation detection."""
    from frustrampnn.data import Mutation

    syn = Mutation(position=0, wildtype="A", mutation="A")
    assert syn.is_synonymous

    nonsyn = Mutation(position=0, wildtype="A", mutation="G")
    assert not nonsyn.is_synonymous


def test_mutation_to_dict():
    """Test Mutation to_dict method."""
    from frustrampnn.data import Mutation

    mut = Mutation(position=72, wildtype="A", mutation="G", pdb="1UBQ", chain="A")
    d = mut.to_dict()

    assert d["position"] == 72
    assert d["wildtype"] == "A"
    assert d["mutation"] == "G"
    assert d["pdb"] == "1UBQ"
    assert d["chain"] == "A"


def test_parse_mutation_string():
    """Test mutation string parsing."""
    from frustrampnn.data import parse_mutation_string

    mut = parse_mutation_string("A73G")

    assert mut.position == 72  # 0-indexed
    assert mut.wildtype == "A"
    assert mut.mutation == "G"


def test_parse_mutation_string_lowercase():
    """Test mutation string parsing with lowercase."""
    from frustrampnn.data import parse_mutation_string

    mut = parse_mutation_string("a73g")

    assert mut.position == 72
    assert mut.wildtype == "A"
    assert mut.mutation == "G"


def test_parse_mutation_string_invalid_format():
    """Test invalid mutation string handling."""
    from frustrampnn.data import parse_mutation_string

    with pytest.raises(ValueError, match="Invalid mutation format"):
        parse_mutation_string("invalid")


def test_parse_mutation_string_missing_mutation():
    """Test mutation string with missing mutation."""
    from frustrampnn.data import parse_mutation_string

    with pytest.raises(ValueError, match="Invalid mutation format"):
        parse_mutation_string("A73")


def test_generate_ssm_mutations():
    """Test site-saturation mutagenesis generation."""
    from frustrampnn.data import generate_ssm_mutations

    mutations = generate_ssm_mutations("MAG")

    # 3 positions × 20 amino acids = 60 mutations
    assert len(mutations) == 60

    # Check first position mutations
    pos0_muts = [m for m in mutations if m.position == 0]
    assert len(pos0_muts) == 20


def test_generate_ssm_mutations_specific_positions():
    """Test SSM with specific positions."""
    from frustrampnn.data import generate_ssm_mutations

    mutations = generate_ssm_mutations("MAG", positions=[0, 2])

    # 2 positions × 20 amino acids = 40 mutations
    assert len(mutations) == 40

    # No mutations at position 1
    pos1_muts = [m for m in mutations if m.position == 1]
    assert len(pos1_muts) == 0


def test_generate_ssm_mutations_no_synonymous():
    """Test SSM without synonymous mutations."""
    from frustrampnn.data import generate_ssm_mutations

    mutations = generate_ssm_mutations("MAG", include_synonymous=False)

    # 3 positions × 19 amino acids = 57 mutations
    assert len(mutations) == 57

    # No synonymous mutations
    for mut in mutations:
        assert not mut.is_synonymous


def test_generate_ssm_mutations_with_gap():
    """Test SSM with gap in sequence."""
    from frustrampnn.data import generate_ssm_mutations

    mutations = generate_ssm_mutations("M-G")

    # 2 valid positions × 20 amino acids = 40 mutations
    assert len(mutations) == 40


def test_generate_ssm_mutations_with_unknown():
    """Test SSM with unknown residue in sequence."""
    from frustrampnn.data import generate_ssm_mutations

    # 'B' is not in ALPHABET, should be skipped
    mutations = generate_ssm_mutations("MBG")

    # Only M and G are valid, so 2 × 20 = 40
    # But B is not in ALPHABET, so it's skipped
    # Actually M and G are valid = 2 positions × 20 = 40
    assert len(mutations) == 40


def test_mutation_with_frustration():
    """Test Mutation with frustration value."""
    import torch

    from frustrampnn.data import Mutation

    frustration = torch.tensor([0.5])
    mut = Mutation(position=0, wildtype="A", mutation="G", frustration=frustration)

    assert mut.frustration is not None
    assert mut.frustration.item() == 0.5


def test_mutation_with_weight():
    """Test Mutation with weight."""
    from frustrampnn.data import Mutation

    mut = Mutation(position=0, wildtype="A", mutation="G", weight=1.5)

    assert mut.weight == 1.5
