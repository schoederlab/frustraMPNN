"""Integration tests for FrustraMPNN.

This module contains integration tests that verify complete workflows
work correctly end-to-end. These tests exercise multiple components
working together.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# =============================================================================
# Markers
# =============================================================================

pytestmark = pytest.mark.integration


# =============================================================================
# PDB Parsing Integration Tests
# =============================================================================


class TestPDBParsingIntegration:
    """Integration tests for PDB parsing workflow."""

    def test_parse_and_extract_sequence(self, test_pdb_path: Path) -> None:
        """Test parsing PDB and extracting sequence."""
        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        from frustrampnn.model import alt_parse_PDB

        result = alt_parse_PDB(str(test_pdb_path), input_chain_list=["A"])

        assert len(result) == 1
        pdb_dict = result[0]

        # Verify sequence
        assert "seq_chain_A" in pdb_dict
        seq = pdb_dict["seq_chain_A"]
        assert seq.startswith("MQIFVKTLTGK")
        assert len(seq) == 76  # 1UBQ has 76 residues

    def test_parse_and_extract_coordinates(self, test_pdb_path: Path) -> None:
        """Test parsing PDB and extracting coordinates."""
        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        from frustrampnn.model import alt_parse_PDB

        result = alt_parse_PDB(str(test_pdb_path), input_chain_list=["A"])
        pdb_dict = result[0]

        # Verify coordinates
        assert "coords_chain_A" in pdb_dict
        coords = pdb_dict["coords_chain_A"]

        assert "N_chain_A" in coords
        assert "CA_chain_A" in coords
        assert "C_chain_A" in coords
        assert "O_chain_A" in coords

        # Verify coordinate dimensions
        ca_coords = coords["CA_chain_A"]
        assert len(ca_coords) == 76  # 76 residues
        assert len(ca_coords[0]) == 3  # 3D coordinates

    def test_parse_and_extract_residue_numbers(self, test_pdb_path: Path) -> None:
        """Test parsing PDB and extracting residue numbers."""
        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        from frustrampnn.model import alt_parse_PDB

        result = alt_parse_PDB(str(test_pdb_path), input_chain_list=["A"])
        pdb_dict = result[0]

        # Verify residue numbers
        assert "resn_list" in pdb_dict
        resn_list = pdb_dict["resn_list"]
        assert len(resn_list) == 76


# =============================================================================
# Mutation Generation Integration Tests
# =============================================================================


class TestMutationGenerationIntegration:
    """Integration tests for mutation generation workflow."""

    def test_generate_mutations_from_sequence(self) -> None:
        """Test generating mutations from a sequence."""
        from frustrampnn.data import generate_ssm_mutations

        sequence = "MQIFVKTLTGK"  # First 11 residues of 1UBQ
        mutations = generate_ssm_mutations(sequence)

        # 11 positions × 20 amino acids = 220 mutations
        assert len(mutations) == 220

        # Verify first position mutations
        pos0_muts = [m for m in mutations if m.position == 0]
        assert len(pos0_muts) == 20
        assert all(m.wildtype == "M" for m in pos0_muts)

    def test_generate_mutations_specific_positions(self) -> None:
        """Test generating mutations for specific positions."""
        from frustrampnn.data import generate_ssm_mutations

        sequence = "MQIFVKTLTGK"
        mutations = generate_ssm_mutations(sequence, positions=[0, 5, 10])

        # 3 positions × 20 amino acids = 60 mutations
        assert len(mutations) == 60

        # Verify only specified positions
        positions = set(m.position for m in mutations)
        assert positions == {0, 5, 10}

    def test_mutation_to_dict_integration(self) -> None:
        """Test mutation to dictionary conversion."""
        from frustrampnn.data import Mutation

        mut = Mutation(
            position=72,
            wildtype="A",
            mutation="G",
            pdb="1UBQ",
            chain="A",
        )

        d = mut.to_dict()

        assert d["position"] == 72
        assert d["wildtype"] == "A"
        assert d["mutation"] == "G"
        assert d["pdb"] == "1UBQ"
        assert d["chain"] == "A"


# =============================================================================
# Visualization Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization workflow."""

    def test_plot_single_residue_with_data(
        self, sample_frustration_df: pd.DataFrame
    ) -> None:
        """Test single residue plot with sample data."""
        from frustrampnn.visualization import plot_single_residue

        fig = plot_single_residue(sample_frustration_df, position=0, chain="A")

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1

    def test_plot_heatmap_with_data(
        self, sample_frustration_df: pd.DataFrame
    ) -> None:
        """Test heatmap plot with sample data."""
        from frustrampnn.visualization import plot_frustration_heatmap

        fig = plot_frustration_heatmap(sample_frustration_df, chain="A")

        assert fig is not None

    def test_plot_single_residue_plotly(
        self, sample_frustration_df: pd.DataFrame
    ) -> None:
        """Test plotly single residue plot."""
        from frustrampnn.visualization import plot_single_residue_plotly

        fig = plot_single_residue_plotly(sample_frustration_df, position=0, chain="A")

        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_heatmap_plotly(
        self, sample_frustration_df: pd.DataFrame
    ) -> None:
        """Test plotly heatmap plot."""
        from frustrampnn.visualization import plot_frustration_heatmap_plotly

        fig = plot_frustration_heatmap_plotly(sample_frustration_df, chain="A")

        assert fig is not None
        assert fig.data[0].type == "heatmap"

    def test_save_plot_to_file(
        self, sample_frustration_df: pd.DataFrame, tmp_output_dir: Path
    ) -> None:
        """Test saving plot to file."""
        from frustrampnn.visualization import plot_single_residue

        fig = plot_single_residue(sample_frustration_df, position=0, chain="A")

        output_path = tmp_output_dir / "test_plot.png"
        fig.savefig(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


# =============================================================================
# Validation Integration Tests
# =============================================================================


class TestValidationIntegration:
    """Integration tests for validation workflow."""

    def test_compute_metrics_integration(self) -> None:
        """Test computing all metrics."""
        from frustrampnn.validation import compute_all_metrics

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.1, 2.2, 2.9, 4.1, 5.0]

        metrics = compute_all_metrics(x, y)

        assert "spearman" in metrics
        assert "pearson" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "n_valid" in metrics

        assert metrics["n_valid"] == 5
        assert metrics["spearman"] > 0.9

    def test_comparison_result_creation(self) -> None:
        """Test creating comparison result."""
        from frustrampnn.validation import ComparisonResult

        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1, 2],
            spearman=0.85,
            pearson=0.82,
            rmse=0.5,
            mae=0.4,
            n_comparisons=60,
        )

        assert result.n_positions == 3
        assert result.spearman == 0.85

        # Test summary
        summary = result.summary()
        assert "test.pdb" in summary
        assert "0.85" in summary

    def test_comparison_result_to_dict(self) -> None:
        """Test comparison result to dictionary."""
        from frustrampnn.validation import ComparisonResult

        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1, 2],
            spearman=0.85,
        )

        d = result.to_dict()

        assert d["pdb_path"] == "test.pdb"
        assert d["chain"] == "A"
        assert d["n_positions"] == 3
        assert d["spearman"] == 0.85


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_info_command(self) -> None:
        """Test CLI info command."""
        from click.testing import CliRunner

        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "FrustraMPNN" in result.output
        assert "PyTorch" in result.output

    def test_cli_version(self) -> None:
        """Test CLI version option."""
        from click.testing import CliRunner

        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1." in result.output or "0." in result.output

    def test_cli_predict_help(self) -> None:
        """Test CLI predict help."""
        from click.testing import CliRunner

        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])

        assert result.exit_code == 0
        assert "--pdb" in result.output
        assert "--checkpoint" in result.output
        assert "--output" in result.output


# =============================================================================
# Data Flow Integration Tests
# =============================================================================


class TestDataFlowIntegration:
    """Integration tests for data flow through the system."""

    def test_mutation_dataclass_workflow(self) -> None:
        """Test complete mutation dataclass workflow."""
        from frustrampnn.data import parse_mutation_string

        # Parse mutation string
        mut = parse_mutation_string("A73G")

        assert mut.position == 72  # 0-indexed
        assert mut.wildtype == "A"
        assert mut.mutation == "G"

        # Convert to string (1-indexed)
        assert str(mut) == "A73G"

        # Convert to dict
        d = mut.to_dict()
        assert d["position"] == 72

    def test_frustration_classification_workflow(self) -> None:
        """Test frustration classification workflow."""
        from frustrampnn.visualization import classify_frustration

        # Test classification
        assert classify_frustration(-2.0) == "highly"
        assert classify_frustration(-1.0) == "highly"
        assert classify_frustration(-0.5) == "neutral"
        assert classify_frustration(0.0) == "neutral"
        assert classify_frustration(0.58) == "minimally"
        assert classify_frustration(1.0) == "minimally"

    def test_constants_consistency(self) -> None:
        """Test that constants are consistent across modules."""
        from frustrampnn import ALPHABET, VOCAB_DIM
        from frustrampnn.model import ALPHABET as MODEL_ALPHABET
        from frustrampnn.model import VOCAB_DIM as MODEL_VOCAB_DIM

        assert ALPHABET == MODEL_ALPHABET
        assert VOCAB_DIM == MODEL_VOCAB_DIM
        assert len(ALPHABET) == VOCAB_DIM


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_invalid_pdb_path_error(self) -> None:
        """Test error handling for invalid PDB path."""
        from frustrampnn import FrustraMPNN

        model = FrustraMPNN.__new__(FrustraMPNN)
        model.model = MagicMock()
        model.cfg = MagicMock()
        model.device = "cpu"

        with pytest.raises(FileNotFoundError, match="PDB file not found"):
            model.predict("nonexistent.pdb")

    def test_invalid_checkpoint_error(self) -> None:
        """Test error handling for invalid checkpoint."""
        from frustrampnn import FrustraMPNN

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            FrustraMPNN.from_pretrained("nonexistent.ckpt")

    def test_invalid_mutation_string_error(self) -> None:
        """Test error handling for invalid mutation string."""
        from frustrampnn.data import parse_mutation_string

        with pytest.raises(ValueError, match="Invalid mutation format"):
            parse_mutation_string("invalid")

    def test_invalid_position_in_plot_error(
        self, sample_frustration_df: pd.DataFrame
    ) -> None:
        """Test error handling for invalid position in plot."""
        from frustrampnn.visualization import plot_single_residue

        with pytest.raises(ValueError, match="No data found"):
            plot_single_residue(sample_frustration_df, position=999, chain="A")

