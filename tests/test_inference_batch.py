"""Tests for batch prediction."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_predict_batch_import():
    """Test that predict_batch method exists."""
    from frustrampnn import FrustraMPNN

    assert hasattr(FrustraMPNN, "predict_batch")


def test_predict_batch_empty_list():
    """Test predict_batch with empty list."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)
    model.model = MagicMock()
    model.cfg = MagicMock()
    model.device = "cpu"

    result = model.predict_batch([], show_progress=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_predict_batch_handles_errors():
    """Test that predict_batch handles errors gracefully."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)
    model.model = MagicMock()
    model.cfg = MagicMock()
    model.device = "cpu"

    # Mock predict to raise an error
    with patch.object(model, "predict", side_effect=Exception("Test error")):
        with pytest.warns(UserWarning, match="Failed to process"):
            result = model.predict_batch(
                ["nonexistent1.pdb", "nonexistent2.pdb"], show_progress=False
            )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_predict_batch_combines_results():
    """Test that predict_batch combines results from multiple PDBs."""
    from frustrampnn import FrustraMPNN

    model = FrustraMPNN.__new__(FrustraMPNN)
    model.model = MagicMock()
    model.cfg = MagicMock()
    model.device = "cpu"

    # Mock predict to return DataFrames
    df1 = pd.DataFrame(
        {
            "frustration_pred": [0.5],
            "position": [0],
            "wildtype": ["A"],
            "mutation": ["G"],
            "pdb": ["pdb1"],
            "chain": ["A"],
        }
    )
    df2 = pd.DataFrame(
        {
            "frustration_pred": [0.7],
            "position": [1],
            "wildtype": ["M"],
            "mutation": ["L"],
            "pdb": ["pdb2"],
            "chain": ["A"],
        }
    )

    with patch.object(model, "predict", side_effect=[df1, df2]):
        result = model.predict_batch(["pdb1.pdb", "pdb2.pdb"], show_progress=False)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result["pdb"]) == ["pdb1", "pdb2"]
