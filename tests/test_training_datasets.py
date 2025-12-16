"""Tests for training dataset classes."""

import pytest
import torch

from frustrampnn.training.config import TrainingConfig
from frustrampnn.training.datasets.base import (
    ALPHABET,
    BaseDataset,
    TrainingMutation,
    get_dataset,
    seq1_index_to_seq2_index,
)


class TestTrainingMutation:
    """Tests for TrainingMutation dataclass."""

    def test_create_mutation(self):
        """Test creating a basic mutation."""
        mut = TrainingMutation(
            position=42,
            wildtype="A",
            mutation="G",
        )
        assert mut.position == 42
        assert mut.wildtype == "A"
        assert mut.mutation == "G"
        assert mut.frustration is None
        assert mut.pdb == ""
        assert mut.weight is None

    def test_mutation_with_frustration(self):
        """Test mutation with frustration value."""
        frustration = torch.tensor([0.5])
        mut = TrainingMutation(
            position=10,
            wildtype="M",
            mutation="L",
            frustration=frustration,
            pdb="1UBQ",
            weight=1.5,
        )
        assert torch.equal(mut.frustration, frustration)
        assert mut.pdb == "1UBQ"
        assert mut.weight == 1.5

    def test_mutation_all_fields(self):
        """Test mutation with all fields populated."""
        frustration = torch.tensor([-0.8])
        mut = TrainingMutation(
            position=0,
            wildtype="W",
            mutation="F",
            frustration=frustration,
            pdb="4BLM",
            weight=2.0,
        )
        assert mut.position == 0
        assert mut.wildtype == "W"
        assert mut.mutation == "F"
        assert mut.frustration.item() == pytest.approx(-0.8)
        assert mut.pdb == "4BLM"
        assert mut.weight == 2.0


class TestAlphabet:
    """Tests for amino acid alphabet."""

    def test_alphabet_length(self):
        """Test alphabet has correct length."""
        assert len(ALPHABET) == 21  # 20 amino acids + gap

    def test_alphabet_contains_standard_aa(self):
        """Test alphabet contains standard amino acids."""
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for aa in standard_aa:
            assert aa in ALPHABET

    def test_alphabet_contains_gap(self):
        """Test alphabet contains gap character."""
        assert "-" in ALPHABET


class TestSequenceAlignment:
    """Tests for sequence alignment utilities."""

    @pytest.fixture
    def mock_alignment_no_gaps(self):
        """Create mock alignment without gaps."""

        class MockAlign:
            seqA = "ACDEF"
            seqB = "ACDEF"

        return MockAlign()

    @pytest.fixture
    def mock_alignment_with_gap(self):
        """Create mock alignment with gap in seqB."""

        class MockAlign:
            seqA = "ACDEF"
            seqB = "AC-EF"

        return MockAlign()

    def test_seq1_index_to_seq2_index_no_gaps(self, mock_alignment_no_gaps):
        """Test index conversion without gaps."""
        align = mock_alignment_no_gaps
        assert seq1_index_to_seq2_index(align, 0) == 0
        assert seq1_index_to_seq2_index(align, 2) == 2
        assert seq1_index_to_seq2_index(align, 4) == 4

    def test_seq1_index_to_seq2_index_with_gap(self, mock_alignment_with_gap):
        """Test index conversion with gap in seq2."""
        align = mock_alignment_with_gap
        assert seq1_index_to_seq2_index(align, 0) == 0  # A
        assert seq1_index_to_seq2_index(align, 1) == 1  # C
        # Position 2 (D) maps to gap, should return None
        assert seq1_index_to_seq2_index(align, 2) is None
        assert seq1_index_to_seq2_index(align, 3) == 2  # E
        assert seq1_index_to_seq2_index(align, 4) == 3  # F


class TestGetDataset:
    """Tests for get_dataset factory function."""

    def test_get_dataset_fireprot(self):
        """Test getting FireProt dataset."""
        config = TrainingConfig(datasets="fireprot")
        # Should raise FileNotFoundError without actual data
        with pytest.raises(FileNotFoundError):
            get_dataset(config, "train")

    def test_get_dataset_megascale(self):
        """Test getting MegaScale dataset."""
        config = TrainingConfig(datasets="megascale")
        # Should raise FileNotFoundError without actual data
        with pytest.raises(FileNotFoundError):
            get_dataset(config, "train")

    def test_get_dataset_combo(self):
        """Test getting Combo dataset."""
        config = TrainingConfig(datasets="combo")
        # Should raise error without actual data (FileNotFoundError or AssertionError)
        with pytest.raises((FileNotFoundError, AssertionError)):
            get_dataset(config, "train")

    def test_get_dataset_invalid(self):
        """Test getting invalid dataset raises error."""
        config = TrainingConfig(datasets="invalid")
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset(config, "train")


class TestBaseDataset:
    """Tests for BaseDataset abstract class."""

    def test_base_dataset_is_abstract(self):
        """Test that BaseDataset cannot be instantiated directly."""
        config = TrainingConfig()
        with pytest.raises(TypeError):
            BaseDataset(config, "train")


class TestMutationFiltering:
    """Tests for mutation filtering logic."""

    def test_filter_multi_mutations(self):
        """Test filtering multi-point mutations."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "mut_type": ["A1G", "A2G:B3C", "A3G"],
            }
        )
        df = df[~df["mut_type"].str.contains(":", na=False)]
        assert len(df) == 2
        assert "A1G" in df["mut_type"].values
        assert "A3G" in df["mut_type"].values

    def test_filter_insertions(self):
        """Test filtering insertion mutations."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "mut_type": ["A1G", "ins_A1", "A3G"],
            }
        )
        df = df[~df["mut_type"].str.contains("ins", case=False, na=False)]
        assert len(df) == 2

    def test_filter_deletions(self):
        """Test filtering deletion mutations."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "mut_type": ["A1G", "del_A1", "A3G"],
            }
        )
        df = df[~df["mut_type"].str.contains("del", case=False, na=False)]
        assert len(df) == 2

    def test_filter_missing_frustration(self):
        """Test filtering missing frustration values."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "mut_type": ["A1G", "A2G", "A3G"],
                "frustration": [0.5, "-", 0.7],
            }
        )
        df = df[df["frustration"] != "-"]
        assert len(df) == 2

    def test_combined_filtering(self):
        """Test combined filtering of all invalid mutations."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "mut_type": ["A1G", "A2G:B3C", "ins_A1", "del_A1", "A5G"],
                "frustration": [0.5, 0.6, 0.7, 0.8, "-"],
            }
        )

        # Apply all filters
        df = df[~df["mut_type"].str.contains(":", na=False)]
        df = df[~df["mut_type"].str.contains("ins", case=False, na=False)]
        df = df[~df["mut_type"].str.contains("del", case=False, na=False)]
        df = df[df["frustration"] != "-"]

        assert len(df) == 1
        assert df.iloc[0]["mut_type"] == "A1G"
