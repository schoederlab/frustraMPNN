"""Tests for constants module."""



def test_alphabet():
    """Test ALPHABET constant."""
    from frustrampnn import ALPHABET

    assert len(ALPHABET) == 21
    assert 'A' in ALPHABET
    assert 'X' in ALPHABET


def test_vocab_dim():
    """Test VOCAB_DIM matches ALPHABET length."""
    from frustrampnn import ALPHABET, VOCAB_DIM

    assert VOCAB_DIM == len(ALPHABET)
    assert VOCAB_DIM == 21


def test_aa_conversion():
    """Test amino acid conversion dictionaries."""
    from frustrampnn import AA_1_TO_3, AA_3_TO_1

    assert AA_3_TO_1['ALA'] == 'A'
    assert AA_3_TO_1['TRP'] == 'W'
    assert AA_3_TO_1['MSE'] == 'M'  # Selenomethionine
    assert AA_1_TO_3['A'] == 'ALA'
    assert AA_1_TO_3['W'] == 'TRP'


def test_frustration_thresholds():
    """Test frustration threshold values."""
    from frustrampnn import FRUSTRATION_THRESHOLDS

    assert FRUSTRATION_THRESHOLDS['highly'] == -1.0
    assert FRUSTRATION_THRESHOLDS['minimally'] == 0.58


def test_frustration_colors():
    """Test frustration color scheme."""
    from frustrampnn import FRUSTRATION_COLORS

    assert FRUSTRATION_COLORS['highly'] == 'red'
    assert FRUSTRATION_COLORS['minimally'] == 'green'
    assert FRUSTRATION_COLORS['neutral'] == 'gray'
    assert FRUSTRATION_COLORS['native'] == 'blue'


def test_constants_from_model():
    """Test that constants can be imported from model module."""
    from frustrampnn.model import ALPHABET, VOCAB_DIM

    assert ALPHABET == 'ACDEFGHIKLMNPQRSTVWYX'
    assert VOCAB_DIM == 21


