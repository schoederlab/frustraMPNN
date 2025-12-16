"""
Tests for the validation module.

These tests verify the validation tools for comparing FrustraMPNN
predictions with frustrapy calculations.
"""

import numpy as np
import pandas as pd
import pytest

from frustrampnn.validation import (
    ComparisonResult,
    PositionComparison,
    compute_all_metrics,
    compute_mae,
    compute_pearson,
    compute_rmse,
    compute_spearman,
)


class TestMetrics:
    """Tests for statistical metrics."""

    def test_compute_spearman_perfect_correlation(self):
        """Test Spearman with perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        result = compute_spearman(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_compute_spearman_perfect_negative(self):
        """Test Spearman with perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        result = compute_spearman(x, y)
        assert abs(result - (-1.0)) < 1e-10

    def test_compute_spearman_no_correlation(self):
        """Test Spearman with low correlation."""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        result = compute_spearman(x, y)
        # Should be relatively low (not perfect correlation)
        assert -0.6 <= result <= 0.6

    def test_compute_spearman_with_nan(self):
        """Test Spearman handles NaN values."""
        x = [1, 2, np.nan, 4, 5]
        y = [1, 2, 3, 4, 5]
        result = compute_spearman(x, y)
        # Should compute on non-NaN values
        assert not np.isnan(result)

    def test_compute_spearman_different_lengths(self):
        """Test Spearman raises error for different lengths."""
        x = [1, 2, 3]
        y = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="same length"):
            compute_spearman(x, y)

    def test_compute_spearman_too_short(self):
        """Test Spearman raises error for too short arrays."""
        x = [1]
        y = [1]
        with pytest.raises(ValueError, match="at least 2"):
            compute_spearman(x, y)

    def test_compute_pearson_perfect_correlation(self):
        """Test Pearson with perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # y = 2x
        result = compute_pearson(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_compute_pearson_perfect_negative(self):
        """Test Pearson with perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]  # y = -2x + 12
        result = compute_pearson(x, y)
        assert abs(result - (-1.0)) < 1e-10

    def test_compute_rmse_zero_error(self):
        """Test RMSE with identical arrays."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        result = compute_rmse(x, y)
        assert abs(result) < 1e-10

    def test_compute_rmse_known_value(self):
        """Test RMSE with known value."""
        x = [1, 2, 3]
        y = [2, 3, 4]  # All differ by 1
        result = compute_rmse(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_compute_rmse_with_nan(self):
        """Test RMSE handles NaN values."""
        x = [1, 2, np.nan, 4]
        y = [1, 2, 3, 4]
        result = compute_rmse(x, y)
        assert abs(result) < 1e-10  # Non-NaN values are identical

    def test_compute_mae_zero_error(self):
        """Test MAE with identical arrays."""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        result = compute_mae(x, y)
        assert abs(result) < 1e-10

    def test_compute_mae_known_value(self):
        """Test MAE with known value."""
        x = [1, 2, 3]
        y = [2, 3, 4]  # All differ by 1
        result = compute_mae(x, y)
        assert abs(result - 1.0) < 1e-10

    def test_compute_all_metrics(self):
        """Test compute_all_metrics returns all expected keys."""
        x = [1, 2, 3, 4, 5]
        y = [1.1, 2.2, 2.9, 4.1, 5.0]
        result = compute_all_metrics(x, y)

        assert "spearman" in result
        assert "pearson" in result
        assert "rmse" in result
        assert "mae" in result
        assert "n_valid" in result

        assert result["n_valid"] == 5
        assert result["spearman"] > 0.9  # High correlation
        assert result["pearson"] > 0.9
        assert result["rmse"] < 0.5
        assert result["mae"] < 0.5


class TestPositionComparison:
    """Tests for PositionComparison dataclass."""

    def test_position_comparison_creation(self):
        """Test creating a PositionComparison."""
        pc = PositionComparison(
            position=72,
            pdb_residue_num=73,
            chain="A",
            wildtype="A",
            frustrampnn_values={"A": 0.5, "G": -0.3},
            frustrapy_values={"A": 0.4, "G": -0.2},
            spearman=0.95,
            pearson=0.93,
            rmse=0.1,
            mae=0.08,
        )

        assert pc.position == 72
        assert pc.pdb_residue_num == 73
        assert pc.chain == "A"
        assert pc.wildtype == "A"
        assert pc.n_mutations == 2

    def test_position_comparison_n_mutations(self):
        """Test n_mutations property counts common keys."""
        pc = PositionComparison(
            position=0,
            pdb_residue_num=1,
            chain="A",
            wildtype="M",
            frustrampnn_values={"A": 0.5, "G": -0.3, "C": 0.1},
            frustrapy_values={"A": 0.4, "G": -0.2},  # Missing C
        )

        assert pc.n_mutations == 2  # Only A and G are common

    def test_position_comparison_to_dict(self):
        """Test to_dict method."""
        pc = PositionComparison(
            position=72,
            pdb_residue_num=73,
            chain="A",
            wildtype="A",
            spearman=0.95,
        )

        d = pc.to_dict()
        assert d["position"] == 72
        assert d["pdb_residue_num"] == 73
        assert d["chain"] == "A"
        assert d["spearman"] == 0.95


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_creation(self):
        """Test creating a ComparisonResult."""
        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1, 2],
            spearman=0.85,
            pearson=0.82,
            rmse=0.5,
            mae=0.4,
            n_comparisons=60,
            frustrapy_time_seconds=120.5,
        )

        assert result.pdb_path == "test.pdb"
        assert result.chain == "A"
        assert result.n_positions == 3
        assert result.spearman == 0.85
        assert result.n_comparisons == 60

    def test_comparison_result_summary(self):
        """Test summary method."""
        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1, 2],
            spearman=0.85,
            pearson=0.82,
            rmse=0.5,
            mae=0.4,
            n_comparisons=60,
            frustrapy_time_seconds=120.5,
        )

        summary = result.summary()
        assert "test.pdb" in summary
        assert "chain A" in summary
        assert "0.85" in summary  # Spearman
        assert "60" in summary  # n_comparisons

    def test_comparison_result_to_dict(self):
        """Test to_dict method."""
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

    def test_comparison_result_repr(self):
        """Test __repr__ method."""
        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1, 2],
            spearman=0.85,
        )

        repr_str = repr(result)
        assert "test.pdb" in repr_str
        assert "chain='A'" in repr_str
        assert "n_positions=3" in repr_str

    def test_comparison_result_with_merged_data(self):
        """Test ComparisonResult with merged DataFrame."""
        merged_df = pd.DataFrame(
            {
                "position": [0, 0, 1, 1],
                "mutation": ["A", "G", "A", "G"],
                "frustrampnn": [0.5, -0.3, 0.2, -0.1],
                "frustrapy": [0.4, -0.2, 0.3, -0.15],
            }
        )

        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1],
            merged_data=merged_df,
        )

        assert result.n_comparisons == 4


class TestComparisonResultPlotting:
    """Tests for ComparisonResult plotting methods."""

    def test_plot_requires_matplotlib(self):
        """Test plot method requires matplotlib."""
        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0],
            merged_data=pd.DataFrame(),
        )

        # This should raise ValueError for empty data, not ImportError
        with pytest.raises(ValueError, match="No comparison data"):
            result.plot()

    def test_plot_with_data(self):
        """Test plot method with valid data."""
        pytest.importorskip("matplotlib")

        merged_df = pd.DataFrame(
            {
                "position": [0, 0, 1, 1],
                "mutation": ["A", "G", "A", "G"],
                "frustrampnn": [0.5, -0.3, 0.2, -0.1],
                "frustrapy": [0.4, -0.2, 0.3, -0.15],
            }
        )

        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1],
            merged_data=merged_df,
            spearman=0.9,
            rmse=0.1,
        )

        fig = result.plot()
        assert fig is not None

    def test_plot_per_position(self):
        """Test plot_per_position method."""
        pytest.importorskip("matplotlib")

        position_results = [
            PositionComparison(
                position=0,
                pdb_residue_num=1,
                chain="A",
                wildtype="M",
                spearman=0.9,
                rmse=0.1,
            ),
            PositionComparison(
                position=1,
                pdb_residue_num=2,
                chain="A",
                wildtype="Q",
                spearman=0.85,
                rmse=0.15,
            ),
        ]

        result = ComparisonResult(
            pdb_path="test.pdb",
            chain="A",
            positions=[0, 1],
            position_results=position_results,
            spearman=0.875,
            rmse=0.125,
        )

        fig = result.plot_per_position()
        assert fig is not None


class TestFrustrapyWrapper:
    """Tests for frustrapy wrapper functions."""

    def test_frustrapy_not_installed_error(self):
        """Test FrustrapyNotInstalledError message."""
        from frustrampnn.validation import FrustrapyNotInstalledError

        error = FrustrapyNotInstalledError()
        assert "frustrapy is not installed" in str(error)
        assert "pip install frustrapy" in str(error)


class TestImports:
    """Tests for module imports."""

    def test_import_validation_module(self):
        """Test importing validation module."""
        from frustrampnn import validation

        assert hasattr(validation, "compare_with_frustrapy")
        assert hasattr(validation, "ComparisonResult")

    def test_import_from_main_package(self):
        """Test importing validation from main package."""
        from frustrampnn import ComparisonResult, compare_with_frustrapy

        assert ComparisonResult is not None
        assert compare_with_frustrapy is not None

    def test_import_metrics(self):
        """Test importing metrics functions."""
        from frustrampnn.validation import (
            compute_all_metrics,
            compute_mae,
            compute_pearson,
            compute_rmse,
            compute_spearman,
        )

        assert compute_spearman is not None
        assert compute_pearson is not None
        assert compute_rmse is not None
        assert compute_mae is not None
        assert compute_all_metrics is not None

    def test_import_position_mapping(self):
        """Test importing position mapping utilities."""
        from frustrampnn.validation import (
            convert_pdb_numbering_to_positions,
            convert_positions_to_pdb_numbering,
            get_pdb_residue_mapping,
            get_reverse_residue_mapping,
        )

        assert get_pdb_residue_mapping is not None
        assert get_reverse_residue_mapping is not None
        assert convert_positions_to_pdb_numbering is not None
        assert convert_pdb_numbering_to_positions is not None
