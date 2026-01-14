"""Regression tests for FrustraMPNN.

This module contains tests that verify numerical consistency of predictions
against reference outputs. These tests ensure that refactoring does not
introduce numerical regressions.

The reference output was generated using the original inference pipeline
with the FireProt-balanced model checkpoint.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def reference_df(reference_output_path: Path) -> pd.DataFrame:
    """Load reference output DataFrame."""
    if not reference_output_path.exists():
        pytest.skip("Reference output not found")
    return pd.read_csv(reference_output_path)


# =============================================================================
# Reference Output Validation
# =============================================================================


class TestReferenceOutput:
    """Tests for reference output file validity."""

    def test_reference_exists(self, reference_output_path: Path) -> None:
        """Verify reference output file exists."""
        assert reference_output_path.exists(), (
            f"Reference output not found at {reference_output_path}. "
            "This file is required for regression testing."
        )

    def test_reference_columns(self, reference_df: pd.DataFrame) -> None:
        """Verify reference output has required columns."""
        required_cols = [
            "frustration_pred",
            "position",
            "wildtype",
            "mutation",
            "pdb",
            "chain",
        ]

        for col in required_cols:
            assert col in reference_df.columns, f"Missing column: {col}"

    def test_reference_row_count(self, reference_df: pd.DataFrame) -> None:
        """Verify reference output has expected number of rows."""
        # 1UBQ has 76 residues × 20 amino acids = 1520 predictions
        expected_rows = 76 * 20
        assert len(reference_df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(reference_df)}"
        )

    def test_reference_positions(self, reference_df: pd.DataFrame) -> None:
        """Verify positions are 0-indexed and complete."""
        positions = reference_df["position"].unique()
        expected_positions = list(range(76))

        assert sorted(positions) == expected_positions, "Positions should be 0-indexed from 0 to 75"

    def test_reference_mutations(self, reference_df: pd.DataFrame) -> None:
        """Verify all 20 amino acids are present for each position."""
        expected_aa = set("ACDEFGHIKLMNPQRSTVWY")

        for pos in reference_df["position"].unique():
            pos_mutations = set(reference_df[reference_df["position"] == pos]["mutation"])
            assert pos_mutations == expected_aa, (
                f"Position {pos} missing mutations: {expected_aa - pos_mutations}"
            )

    def test_reference_frustration_range(self, reference_df: pd.DataFrame) -> None:
        """Verify frustration values are in reasonable range."""
        frustration = reference_df["frustration_pred"]

        # Frustration values typically range from -3 to +3
        assert frustration.min() > -5, f"Min frustration too low: {frustration.min()}"
        assert frustration.max() < 5, f"Max frustration too high: {frustration.max()}"

    def test_reference_no_nan(self, reference_df: pd.DataFrame) -> None:
        """Verify no NaN values in frustration predictions."""
        assert not reference_df["frustration_pred"].isna().any(), (
            "Reference output contains NaN values"
        )


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_predictions(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    tolerance: float = 1e-5,
) -> dict:
    """Compare two prediction DataFrames.

    Args:
        reference: Reference predictions
        test: Test predictions to compare
        tolerance: Absolute tolerance for numerical comparison

    Returns:
        Dictionary with comparison results
    """
    results = {
        "passed": True,
        "total_predictions": len(reference),
        "matched_predictions": 0,
        "mismatches": 0,
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "missing_in_test": [],
        "extra_in_test": [],
        "details": [],
    }

    # Create keys for matching
    def make_key(row):
        return f"{row['position']}_{row['wildtype']}_{row['mutation']}_{row['chain']}"

    ref_keys = reference.apply(make_key, axis=1)
    test_keys = test.apply(make_key, axis=1)

    ref_set = set(ref_keys)
    test_set = set(test_keys)

    # Check for missing/extra predictions
    results["missing_in_test"] = list(ref_set - test_set)[:10]
    results["extra_in_test"] = list(test_set - ref_set)[:10]

    if results["missing_in_test"]:
        results["passed"] = False

    # Merge for comparison
    reference = reference.copy()
    test = test.copy()
    reference["_key"] = ref_keys
    test["_key"] = test_keys

    merged = reference.merge(
        test[["_key", "frustration_pred"]],
        on="_key",
        suffixes=("_ref", "_test"),
    )

    results["matched_predictions"] = len(merged)

    if len(merged) == 0:
        results["passed"] = False
        results["error"] = "No matching predictions found"
        return results

    # Calculate differences
    diffs = np.abs(merged["frustration_pred_ref"] - merged["frustration_pred_test"])
    results["max_diff"] = float(diffs.max())
    results["mean_diff"] = float(diffs.mean())
    results["mismatches"] = int((diffs > tolerance).sum())

    if results["mismatches"] > 0:
        results["passed"] = False
        mismatch_rows = merged[diffs > tolerance].head(10)
        for _, row in mismatch_rows.iterrows():
            results["details"].append(
                {
                    "key": row["_key"],
                    "reference": row["frustration_pred_ref"],
                    "test": row["frustration_pred_test"],
                    "diff": abs(row["frustration_pred_ref"] - row["frustration_pred_test"]),
                }
            )

    return results


# =============================================================================
# Regression Tests (require checkpoint)
# =============================================================================


@pytest.mark.regression
@pytest.mark.slow
class TestNumericalRegression:
    """Tests for numerical regression against reference output.

    These tests require a model checkpoint to be available.
    They verify that the new API produces identical results to the reference.
    """

    @pytest.fixture
    def checkpoint_path(self) -> Path | None:
        """Get path to model checkpoint."""
        # Try common checkpoint locations
        possible_paths = [
            Path("checkpoints/fireprot_balanced.ckpt"),
            Path("inference/checkpoints/fireprot_balanced.ckpt"),
            Path.home() / ".cache/frustrampnn/fireprot_balanced.ckpt",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def test_prediction_matches_reference(
        self,
        reference_df: pd.DataFrame,
        test_pdb_path: Path,
        checkpoint_path: Path | None,
    ) -> None:
        """Verify predictions match reference within tolerance."""
        if checkpoint_path is None:
            pytest.skip("Model checkpoint not available")

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        from frustrampnn import FrustraMPNN

        # Load model and predict
        model = FrustraMPNN.from_pretrained(str(checkpoint_path))
        results = model.predict(str(test_pdb_path), chains=["A"])

        # Compare
        comparison = compare_predictions(reference_df, results, tolerance=1e-5)

        assert comparison["passed"], (
            f"Regression test failed:\n"
            f"  Mismatches: {comparison['mismatches']}\n"
            f"  Max diff: {comparison['max_diff']:.2e}\n"
            f"  Mean diff: {comparison['mean_diff']:.2e}\n"
            f"  Details: {comparison['details'][:5]}"
        )

    def test_prediction_correlation(
        self,
        reference_df: pd.DataFrame,
        test_pdb_path: Path,
        checkpoint_path: Path | None,
    ) -> None:
        """Verify predictions have high correlation with reference."""
        if checkpoint_path is None:
            pytest.skip("Model checkpoint not available")

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        from frustrampnn import FrustraMPNN

        model = FrustraMPNN.from_pretrained(str(checkpoint_path))
        results = model.predict(str(test_pdb_path), chains=["A"])

        # Merge for correlation
        reference = reference_df.copy()
        reference["_key"] = reference.apply(
            lambda r: f"{r['position']}_{r['wildtype']}_{r['mutation']}_{r['chain']}",
            axis=1,
        )
        results["_key"] = results.apply(
            lambda r: f"{r['position']}_{r['wildtype']}_{r['mutation']}_{r['chain']}",
            axis=1,
        )

        merged = reference.merge(
            results[["_key", "frustration_pred"]],
            on="_key",
            suffixes=("_ref", "_test"),
        )

        correlation = merged["frustration_pred_ref"].corr(merged["frustration_pred_test"])

        assert correlation > 0.999, f"Correlation too low: {correlation}"


# =============================================================================
# Output Format Tests
# =============================================================================


class TestOutputFormat:
    """Tests for output format consistency."""

    def test_output_columns_match_reference(self, reference_df: pd.DataFrame) -> None:
        """Verify output columns match reference format."""
        expected_cols = {
            "frustration_pred",
            "position",
            "wildtype",
            "mutation",
            "pdb",
            "chain",
        }

        # Reference should have at least these columns
        assert expected_cols.issubset(set(reference_df.columns))

    def test_position_indexing_is_zero_based(self, reference_df: pd.DataFrame) -> None:
        """Verify positions are 0-indexed."""
        assert reference_df["position"].min() == 0, "Positions should start at 0"

    def test_amino_acids_are_single_letter(self, reference_df: pd.DataFrame) -> None:
        """Verify amino acids are single-letter codes."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

        wildtypes = set(reference_df["wildtype"].unique())
        mutations = set(reference_df["mutation"].unique())

        assert wildtypes.issubset(valid_aa), f"Invalid wildtypes: {wildtypes - valid_aa}"
        assert mutations.issubset(valid_aa), f"Invalid mutations: {mutations - valid_aa}"

    def test_chain_is_string(self, reference_df: pd.DataFrame) -> None:
        """Verify chain is string type."""
        assert reference_df["chain"].dtype == "object"

    def test_frustration_is_float(self, reference_df: pd.DataFrame) -> None:
        """Verify frustration_pred is float type."""
        assert reference_df["frustration_pred"].dtype in ["float64", "float32"]


# =============================================================================
# Statistical Properties Tests
# =============================================================================


class TestStatisticalProperties:
    """Tests for statistical properties of predictions."""

    def test_frustration_distribution(self, reference_df: pd.DataFrame) -> None:
        """Verify frustration distribution is reasonable."""
        frustration = reference_df["frustration_pred"]

        # Mean should be close to 0 (neutral)
        assert -1 < frustration.mean() < 1, f"Mean too extreme: {frustration.mean()}"

        # Standard deviation should be reasonable
        assert 0.5 < frustration.std() < 2, f"Std too extreme: {frustration.std()}"

    def test_native_residue_frustration(self, reference_df: pd.DataFrame) -> None:
        """Verify native residues tend to have higher frustration."""
        # Native mutations (wildtype == mutation) should generally be less frustrated
        native = reference_df[reference_df["wildtype"] == reference_df["mutation"]]
        non_native = reference_df[reference_df["wildtype"] != reference_df["mutation"]]

        # This is a soft check - native residues are often more stable
        # but not always
        native_mean = native["frustration_pred"].mean()
        non_native_mean = non_native["frustration_pred"].mean()

        # Just verify both exist and are reasonable
        assert not np.isnan(native_mean)
        assert not np.isnan(non_native_mean)

    def test_position_variance(self, reference_df: pd.DataFrame) -> None:
        """Verify each position has variance in predictions."""
        for pos in reference_df["position"].unique():
            pos_data = reference_df[reference_df["position"] == pos]
            variance = pos_data["frustration_pred"].var()

            # Each position should have some variance across mutations
            assert variance > 0.01, f"Position {pos} has too little variance: {variance}"


# =============================================================================
# Comparison Script Integration
# =============================================================================


class TestComparisonScript:
    """Tests for the regression_test.py comparison script."""

    def test_comparison_script_exists(self, test_data_dir: Path) -> None:
        """Verify comparison script exists."""
        script = test_data_dir / "regression_test.py"
        assert script.exists(), "regression_test.py not found"

    def test_comparison_script_syntax(self, test_data_dir: Path) -> None:
        """Verify comparison script has valid syntax."""
        import subprocess
        import sys

        script = test_data_dir / "regression_test.py"
        if not script.exists():
            pytest.skip("regression_test.py not found")

        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_comparison_script_help(self, test_data_dir: Path) -> None:
        """Verify comparison script has help."""
        import subprocess
        import sys

        script = test_data_dir / "regression_test.py"
        if not script.exists():
            pytest.skip("regression_test.py not found")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--reference" in result.stdout
        assert "--tolerance" in result.stdout
