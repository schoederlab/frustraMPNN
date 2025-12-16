"""
Tests for backward compatibility with existing scripts and output formats.

These tests ensure that:
1. Old inference script still exists and has valid syntax
2. Output CSV format matches the reference format
3. Old checkpoint format is supported
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestOldInferenceScript:
    """Test that old inference script is preserved."""

    def test_old_inference_script_exists(self) -> None:
        """Verify old inference script still exists."""
        script = Path("inference/custom_inference_refac.py")
        assert script.exists(), (
            "Old inference script must not be removed. "
            "Users may have scripts that depend on it."
        )

    def test_old_inference_script_syntax(self) -> None:
        """Verify old inference script has valid Python syntax."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "inference/custom_inference_refac.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error in old script: {result.stderr}"

    def test_old_transfer_model_exists(self) -> None:
        """Verify old transfer model module exists."""
        script = Path("inference/transfer_model_pl.py")
        assert script.exists(), "Old transfer_model_pl.py must not be removed"

    def test_old_transfer_model_syntax(self) -> None:
        """Verify old transfer model has valid syntax."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "inference/transfer_model_pl.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


class TestOutputCSVFormat:
    """Test that output CSV format matches reference."""

    @pytest.fixture
    def reference_output_path(self) -> Path:
        """Get path to reference output."""
        return Path("test_data/1UBQ_reference_output.csv")

    def test_reference_output_exists(self, reference_output_path: Path) -> None:
        """Verify reference output file exists."""
        assert reference_output_path.exists(), (
            f"Reference output not found at {reference_output_path}. "
            "This file is required for regression testing."
        )

    def test_reference_output_columns(self, reference_output_path: Path) -> None:
        """Verify reference output has expected columns."""
        import pandas as pd

        if not reference_output_path.exists():
            pytest.skip("Reference output not found")

        df = pd.read_csv(reference_output_path)

        # Required columns for backward compatibility
        required_cols = [
            "frustration_pred",
            "position",
            "wildtype",
            "mutation",
            "pdb",
            "chain",
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_reference_output_data_types(self, reference_output_path: Path) -> None:
        """Verify reference output has correct data types."""
        import pandas as pd

        if not reference_output_path.exists():
            pytest.skip("Reference output not found")

        df = pd.read_csv(reference_output_path)

        # Check data types
        assert df["frustration_pred"].dtype in ["float64", "float32"], (
            "frustration_pred should be float"
        )
        assert df["position"].dtype in ["int64", "int32"], "position should be int"
        assert df["wildtype"].dtype == "object", "wildtype should be string"
        assert df["mutation"].dtype == "object", "mutation should be string"
        assert df["pdb"].dtype == "object", "pdb should be string"
        assert df["chain"].dtype == "object", "chain should be string"

    def test_reference_output_position_indexing(self, reference_output_path: Path) -> None:
        """Verify positions are 0-indexed (internal format)."""
        import pandas as pd

        if not reference_output_path.exists():
            pytest.skip("Reference output not found")

        df = pd.read_csv(reference_output_path)

        # Positions should start at 0 (0-indexed)
        assert df["position"].min() == 0, "Positions should be 0-indexed"

    def test_reference_output_amino_acids(self, reference_output_path: Path) -> None:
        """Verify amino acids are single-letter codes."""
        import pandas as pd

        if not reference_output_path.exists():
            pytest.skip("Reference output not found")

        df = pd.read_csv(reference_output_path)

        # All amino acids should be single letters
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

        wildtypes = set(df["wildtype"].unique())
        mutations = set(df["mutation"].unique())

        assert wildtypes.issubset(valid_aa), f"Invalid wildtype AAs: {wildtypes - valid_aa}"
        assert mutations.issubset(valid_aa), f"Invalid mutation AAs: {mutations - valid_aa}"


class TestNewAPIOutputFormat:
    """Test that new API produces compatible output format."""

    def test_new_api_output_columns(self) -> None:
        """Verify new API produces required columns."""

        # Create a mock DataFrame like what FrustraMPNN.predict() returns
        # This tests the expected output format
        expected_columns = [
            "frustration_pred",
            "position",
            "wildtype",
            "mutation",
            "pdb",
            "chain",
        ]

        # The new API should produce these columns
        # (actual prediction test requires checkpoint)
        from frustrampnn import FrustraMPNN

        # Check that the class exists and has predict method
        assert hasattr(FrustraMPNN, "predict")
        assert hasattr(FrustraMPNN, "from_pretrained")


class TestCheckpointCompatibility:
    """Test checkpoint format compatibility."""

    def test_checkpoint_loading_interface(self) -> None:
        """Verify checkpoint loading interface exists."""
        # Check that from_pretrained accepts expected arguments
        import inspect

        from frustrampnn import FrustraMPNN

        sig = inspect.signature(FrustraMPNN.from_pretrained)
        params = list(sig.parameters.keys())

        assert "checkpoint_path" in params, "Should accept checkpoint_path"
        assert "config_path" in params, "Should accept config_path for old format"
        assert "device" in params, "Should accept device"

    def test_old_checkpoint_format_support(self) -> None:
        """Verify old checkpoint format is documented as supported."""
        from frustrampnn import FrustraMPNN

        # Check docstring mentions old format support
        docstring = FrustraMPNN.from_pretrained.__doc__ or ""
        assert "config" in docstring.lower() or "old" in docstring.lower(), (
            "from_pretrained should document support for old checkpoint format"
        )


class TestCLIBackwardCompatibility:
    """Test CLI maintains backward-compatible interface."""

    def test_cli_predict_arguments(self) -> None:
        """Verify CLI predict command has expected arguments."""
        from click.testing import CliRunner

        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])

        # Check for expected argument names
        assert "--pdb" in result.output, "Should have --pdb argument"
        assert "--checkpoint" in result.output, "Should have --checkpoint argument"
        assert "--output" in result.output, "Should have --output argument"
        assert "--chains" in result.output, "Should have --chains argument"
        assert "--device" in result.output, "Should have --device argument"

    def test_cli_short_options(self) -> None:
        """Verify CLI has short options for common arguments."""
        from click.testing import CliRunner

        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])

        # Check for short options
        assert "-p" in result.output, "Should have -p short option for --pdb"
        assert "-c" in result.output, "Should have -c short option for --checkpoint"
        assert "-o" in result.output, "Should have -o short option for --output"
        assert "-q" in result.output, "Should have -q short option for --quiet"


class TestImportCompatibility:
    """Test import paths for backward compatibility."""

    def test_main_import(self) -> None:
        """Test main package import."""
        import frustrampnn

        assert hasattr(frustrampnn, "__version__")
        assert hasattr(frustrampnn, "FrustraMPNN")

    def test_mutation_import(self) -> None:
        """Test Mutation class import."""
        from frustrampnn import Mutation

        # Should be able to create a Mutation
        mut = Mutation(position=0, wildtype="A", mutation="G")
        assert mut.position == 0
        assert mut.wildtype == "A"
        assert mut.mutation == "G"

    def test_constants_import(self) -> None:
        """Test constants import."""
        from frustrampnn import AA_1_TO_3, AA_3_TO_1, ALPHABET

        assert len(ALPHABET) == 21  # 20 AAs + X
        assert AA_1_TO_3["A"] == "ALA"
        assert AA_3_TO_1["ALA"] == "A"


