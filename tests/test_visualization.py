"""
Tests for visualization module.

Tests the plotting functions for single-residue frustration analysis.
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def mock_frustration_data() -> pd.DataFrame:
    """Create mock frustration data for testing."""
    data = []
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        # Create mock frustration values based on amino acid
        # This creates a spread of values for testing
        value = (ord(aa) - 77) / 10  # Range roughly -1.3 to 0.8
        data.append(
            {
                "position": 0,
                "wildtype": "M",
                "mutation": aa,
                "frustration_pred": value,
                "chain": "A",
                "pdb": "test",
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def mock_multi_position_data() -> pd.DataFrame:
    """Create mock data for multiple positions."""
    data = []
    for pos in range(10):
        wt = "ACDEFGHIKL"[pos]
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            # Create position-dependent frustration values
            value = np.sin(pos * 0.5) + (ord(aa) - 77) / 20
            data.append(
                {
                    "position": pos,
                    "wildtype": wt,
                    "mutation": aa,
                    "frustration_pred": value,
                    "chain": "A",
                    "pdb": "test",
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def mock_multi_chain_data() -> pd.DataFrame:
    """Create mock data for multiple chains."""
    data = []
    for chain in ["A", "B"]:
        for pos in range(5):
            wt = "ACDEF"[pos]
            for aa in "ACDEFGHIKLMNPQRSTVWY":
                value = (ord(aa) - 77) / 10 + (0.5 if chain == "B" else 0)
                data.append(
                    {
                        "position": pos,
                        "wildtype": wt,
                        "mutation": aa,
                        "frustration_pred": value,
                        "chain": chain,
                        "pdb": "test",
                    }
                )
    return pd.DataFrame(data)


# =============================================================================
# Test classify_frustration
# =============================================================================


def test_classify_frustration_highly():
    """Test classification of highly frustrated values."""
    from frustrampnn.visualization import classify_frustration

    assert classify_frustration(-2.0) == "highly"
    assert classify_frustration(-1.5) == "highly"
    assert classify_frustration(-1.0) == "highly"  # Boundary


def test_classify_frustration_neutral():
    """Test classification of neutral values."""
    from frustrampnn.visualization import classify_frustration

    assert classify_frustration(-0.5) == "neutral"
    assert classify_frustration(0.0) == "neutral"
    assert classify_frustration(0.5) == "neutral"
    assert classify_frustration(0.57) == "neutral"  # Just below boundary


def test_classify_frustration_minimally():
    """Test classification of minimally frustrated values."""
    from frustrampnn.visualization import classify_frustration

    assert classify_frustration(0.58) == "minimally"  # Boundary
    assert classify_frustration(1.0) == "minimally"
    assert classify_frustration(2.0) == "minimally"


def test_classify_frustration_boundaries():
    """Test exact boundary values."""
    from frustrampnn.constants import FRUSTRATION_THRESHOLDS
    from frustrampnn.visualization import classify_frustration

    # Verify thresholds are as expected
    assert FRUSTRATION_THRESHOLDS["highly"] == -1.0
    assert FRUSTRATION_THRESHOLDS["minimally"] == 0.58

    # Test boundaries
    assert classify_frustration(-1.0) == "highly"
    assert classify_frustration(-0.99) == "neutral"
    assert classify_frustration(0.58) == "minimally"
    assert classify_frustration(0.57) == "neutral"


# =============================================================================
# Test plot_single_residue (matplotlib)
# =============================================================================


def test_plot_single_residue_import():
    """Test plot function can be imported."""
    from frustrampnn.visualization import plot_single_residue

    assert callable(plot_single_residue)


def test_plot_single_residue_basic(mock_frustration_data):
    """Test basic single residue plot."""
    from frustrampnn.visualization import plot_single_residue

    fig = plot_single_residue(mock_frustration_data, position=0, chain="A")
    assert fig is not None

    # Check figure has axes
    axes = fig.get_axes()
    assert len(axes) == 1

    # Check title contains position
    title = axes[0].get_title()
    assert "1" in title  # Position 0 displayed as 1


def test_plot_single_residue_no_chain(mock_frustration_data):
    """Test plot without specifying chain."""
    from frustrampnn.visualization import plot_single_residue

    fig = plot_single_residue(mock_frustration_data, position=0)
    assert fig is not None


def test_plot_single_residue_custom_title(mock_frustration_data):
    """Test plot with custom title."""
    from frustrampnn.visualization import plot_single_residue

    custom_title = "My Custom Title"
    fig = plot_single_residue(mock_frustration_data, position=0, chain="A", title=custom_title)

    axes = fig.get_axes()
    assert axes[0].get_title() == custom_title


def test_plot_single_residue_no_thresholds(mock_frustration_data):
    """Test plot without threshold lines."""
    from frustrampnn.visualization import plot_single_residue

    fig = plot_single_residue(mock_frustration_data, position=0, chain="A", show_thresholds=False)
    assert fig is not None


def test_plot_single_residue_invalid_position(mock_frustration_data):
    """Test error on invalid position."""
    from frustrampnn.visualization import plot_single_residue

    with pytest.raises(ValueError, match="No data found"):
        plot_single_residue(mock_frustration_data, position=999, chain="A")


def test_plot_single_residue_invalid_chain(mock_frustration_data):
    """Test error on invalid chain."""
    from frustrampnn.visualization import plot_single_residue

    with pytest.raises(ValueError, match="No data found"):
        plot_single_residue(mock_frustration_data, position=0, chain="Z")


def test_plot_single_residue_colors(mock_frustration_data):
    """Test that colors are correctly assigned."""
    from frustrampnn.visualization._core import prepare_single_residue_data

    df, wildtype = prepare_single_residue_data(mock_frustration_data, 0, "A")

    # Check wildtype is M
    assert wildtype == "M"

    # Check native residue is blue
    native_row = df[df["mutation"] == "M"]
    assert native_row["color"].iloc[0] == "blue"

    # Check other colors based on frustration values
    for _, row in df.iterrows():
        if row["mutation"] == "M":
            assert row["color"] == "blue"
        elif row["frustration_pred"] <= -1.0:
            assert row["color"] == "red"
        elif row["frustration_pred"] >= 0.58:
            assert row["color"] == "green"
        else:
            assert row["color"] == "gray"


# =============================================================================
# Test plot_single_residue_plotly
# =============================================================================


def test_plot_single_residue_plotly_import():
    """Test plotly plot function can be imported."""
    from frustrampnn.visualization import plot_single_residue_plotly

    assert callable(plot_single_residue_plotly)


def test_plot_single_residue_plotly_basic(mock_frustration_data):
    """Test basic plotly single residue plot."""
    from frustrampnn.visualization import plot_single_residue_plotly

    fig = plot_single_residue_plotly(mock_frustration_data, position=0, chain="A")
    assert fig is not None

    # Check figure has data
    assert len(fig.data) > 0


def test_plot_single_residue_plotly_layout(mock_frustration_data):
    """Test plotly plot layout."""
    from frustrampnn.visualization import plot_single_residue_plotly

    fig = plot_single_residue_plotly(
        mock_frustration_data, position=0, chain="A", width=1000, height=600
    )

    assert fig.layout.width == 1000
    assert fig.layout.height == 600


# =============================================================================
# Test plot_frustration_heatmap (matplotlib)
# =============================================================================


def test_plot_frustration_heatmap_import():
    """Test heatmap function can be imported."""
    from frustrampnn.visualization import plot_frustration_heatmap

    assert callable(plot_frustration_heatmap)


def test_plot_frustration_heatmap_basic(mock_multi_position_data):
    """Test basic heatmap plot."""
    from frustrampnn.visualization import plot_frustration_heatmap

    fig = plot_frustration_heatmap(mock_multi_position_data, chain="A")
    assert fig is not None

    # Check figure has axes
    axes = fig.get_axes()
    assert len(axes) >= 1  # Main axes + colorbar


def test_plot_frustration_heatmap_custom_colormap(mock_multi_position_data):
    """Test heatmap with custom colormap."""
    from frustrampnn.visualization import plot_frustration_heatmap

    fig = plot_frustration_heatmap(
        mock_multi_position_data, chain="A", cmap="viridis", vmin=-2, vmax=2
    )
    assert fig is not None


def test_plot_frustration_heatmap_invalid_chain(mock_multi_position_data):
    """Test error on invalid chain."""
    from frustrampnn.visualization import plot_frustration_heatmap

    with pytest.raises(ValueError, match="No data found"):
        plot_frustration_heatmap(mock_multi_position_data, chain="Z")


# =============================================================================
# Test plot_frustration_heatmap_plotly
# =============================================================================


def test_plot_frustration_heatmap_plotly_import():
    """Test plotly heatmap function can be imported."""
    from frustrampnn.visualization import plot_frustration_heatmap_plotly

    assert callable(plot_frustration_heatmap_plotly)


def test_plot_frustration_heatmap_plotly_basic(mock_multi_position_data):
    """Test basic plotly heatmap."""
    from frustrampnn.visualization import plot_frustration_heatmap_plotly

    fig = plot_frustration_heatmap_plotly(mock_multi_position_data, chain="A")
    assert fig is not None

    # Check figure has heatmap data
    assert len(fig.data) > 0
    assert fig.data[0].type == "heatmap"


# =============================================================================
# Test multi-chain handling
# =============================================================================


def test_plot_single_residue_multi_chain(mock_multi_chain_data):
    """Test plotting with multi-chain data requires chain specification."""
    from frustrampnn.visualization import plot_single_residue

    # Should work with chain specified
    fig_a = plot_single_residue(mock_multi_chain_data, position=0, chain="A")
    fig_b = plot_single_residue(mock_multi_chain_data, position=0, chain="B")

    assert fig_a is not None
    assert fig_b is not None


def test_plot_heatmap_multi_chain(mock_multi_chain_data):
    """Test heatmap with multi-chain data."""
    from frustrampnn.visualization import plot_frustration_heatmap

    # Should work with chain specified
    fig_a = plot_frustration_heatmap(mock_multi_chain_data, chain="A")
    fig_b = plot_frustration_heatmap(mock_multi_chain_data, chain="B")

    assert fig_a is not None
    assert fig_b is not None


# =============================================================================
# Test exports from package
# =============================================================================


def test_exports_from_visualization_module():
    """Test all expected functions are exported from visualization module."""
    from frustrampnn import visualization

    assert hasattr(visualization, "plot_single_residue")
    assert hasattr(visualization, "plot_single_residue_plotly")
    assert hasattr(visualization, "plot_frustration_heatmap")
    assert hasattr(visualization, "plot_frustration_heatmap_plotly")
    assert hasattr(visualization, "classify_frustration")


def test_exports_from_main_package():
    """Test visualization functions are exported from main package."""
    import frustrampnn

    assert hasattr(frustrampnn, "plot_single_residue")
    assert hasattr(frustrampnn, "plot_single_residue_plotly")
    assert hasattr(frustrampnn, "plot_frustration_heatmap")
    assert hasattr(frustrampnn, "plot_frustration_heatmap_plotly")
    assert hasattr(frustrampnn, "classify_frustration")


# =============================================================================
# Test Sequence Map
# =============================================================================


@pytest.fixture
def mock_native_frustration_data() -> pd.DataFrame:
    """Create mock native frustration data for sequence map testing."""
    data = []
    for pos in range(20):
        wt = "ACDEFGHIKLMNPQRSTVWY"[pos]
        # Create varying frustration values
        if pos < 5:
            category = "highly"
            value = -1.5
        elif pos < 15:
            category = "neutral"
            value = 0.0
        else:
            category = "minimally"
            value = 1.0
        data.append(
            {
                "position": pos,
                "wildtype": wt,
                "frustration": value,
                "category": category,
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def mock_calculated_frustration_data() -> pd.DataFrame:
    """Create mock calculated frustration data (slightly different from predicted)."""
    data = []
    for pos in range(20):
        wt = "ACDEFGHIKLMNPQRSTVWY"[pos]
        # Create slightly different categories to show disagreement
        if pos < 4:  # One less highly frustrated
            category = "highly"
            value = -1.5
        elif pos < 16:  # Two more neutral
            category = "neutral"
            value = 0.0
        else:
            category = "minimally"
            value = 1.0
        data.append(
            {
                "position": pos,
                "wildtype": wt,
                "frustration": value,
                "category": category,
            }
        )
    return pd.DataFrame(data)


def test_get_native_frustration_per_position_import():
    """Test get_native_frustration_per_position can be imported."""
    from frustrampnn.visualization import get_native_frustration_per_position

    assert callable(get_native_frustration_per_position)


def test_get_native_frustration_per_position(mock_multi_position_data):
    """Test extracting native frustration values."""
    from frustrampnn.visualization import get_native_frustration_per_position

    result = get_native_frustration_per_position(mock_multi_position_data, chain="A")

    # Should have one row per position
    assert len(result) == 10

    # Should have required columns
    assert "position" in result.columns
    assert "wildtype" in result.columns
    assert "frustration" in result.columns
    assert "category" in result.columns


def test_plot_sequence_map_import():
    """Test plot_sequence_map can be imported."""
    from frustrampnn.visualization import plot_sequence_map

    assert callable(plot_sequence_map)


def test_plot_sequence_map_predicted_only(mock_native_frustration_data):
    """Test sequence map with only predicted data."""
    from frustrampnn.visualization import plot_sequence_map

    fig = plot_sequence_map(mock_native_frustration_data)
    assert fig is not None

    # Check figure has axes
    axes = fig.get_axes()
    assert len(axes) >= 1


def test_plot_sequence_map_comparison(
    mock_native_frustration_data, mock_calculated_frustration_data
):
    """Test sequence map comparing calculated vs predicted."""
    from frustrampnn.visualization import plot_sequence_map

    fig = plot_sequence_map(
        mock_native_frustration_data,
        mock_calculated_frustration_data,
        title="Test Comparison",
    )
    assert fig is not None


def test_plot_sequence_map_plotly_import():
    """Test plot_sequence_map_plotly can be imported."""
    from frustrampnn.visualization import plot_sequence_map_plotly

    assert callable(plot_sequence_map_plotly)


def test_plot_sequence_map_plotly_basic(mock_native_frustration_data):
    """Test basic plotly sequence map."""
    from frustrampnn.visualization import plot_sequence_map_plotly

    fig = plot_sequence_map_plotly(mock_native_frustration_data)
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_sequence_map_plotly_comparison(
    mock_native_frustration_data, mock_calculated_frustration_data
):
    """Test plotly sequence map with comparison."""
    from frustrampnn.visualization import plot_sequence_map_plotly

    fig = plot_sequence_map_plotly(
        mock_native_frustration_data,
        mock_calculated_frustration_data,
    )
    assert fig is not None
    # Should have multiple traces (calc, pred, disagree)
    assert len(fig.data) >= 2


# =============================================================================
# Test Sankey Diagram
# =============================================================================


def test_compute_category_flows_import():
    """Test compute_category_flows can be imported."""
    from frustrampnn.visualization import compute_category_flows

    assert callable(compute_category_flows)


def test_compute_category_flows(mock_native_frustration_data, mock_calculated_frustration_data):
    """Test computing category flows."""
    from frustrampnn.visualization import compute_category_flows

    calc_counts, pred_counts, flows = compute_category_flows(
        mock_calculated_frustration_data,
        mock_native_frustration_data,
    )

    # Check counts are returned
    assert isinstance(calc_counts, dict)
    assert isinstance(pred_counts, dict)
    assert isinstance(flows, dict)

    # Check all categories are present
    for cat in ["minimally", "neutral", "highly"]:
        assert cat in calc_counts
        assert cat in pred_counts

    # Check flows are tuples
    for key in flows:
        assert isinstance(key, tuple)
        assert len(key) == 2


def test_plot_frustration_sankey_import():
    """Test plot_frustration_sankey can be imported."""
    from frustrampnn.visualization import plot_frustration_sankey

    assert callable(plot_frustration_sankey)


def test_plot_frustration_sankey_basic(
    mock_native_frustration_data, mock_calculated_frustration_data
):
    """Test basic Sankey diagram."""
    from frustrampnn.visualization import plot_frustration_sankey

    fig = plot_frustration_sankey(
        mock_calculated_frustration_data,
        mock_native_frustration_data,
        title="Test Sankey",
    )
    assert fig is not None

    # Check figure has Sankey data
    assert len(fig.data) > 0
    assert fig.data[0].type == "sankey"


def test_plot_frustration_sankey_matplotlib_import():
    """Test plot_frustration_sankey_matplotlib can be imported."""
    from frustrampnn.visualization import plot_frustration_sankey_matplotlib

    assert callable(plot_frustration_sankey_matplotlib)


def test_plot_frustration_sankey_matplotlib_basic(
    mock_native_frustration_data, mock_calculated_frustration_data
):
    """Test matplotlib Sankey-style diagram."""
    from frustrampnn.visualization import plot_frustration_sankey_matplotlib

    fig = plot_frustration_sankey_matplotlib(
        mock_calculated_frustration_data,
        mock_native_frustration_data,
    )
    assert fig is not None


# =============================================================================
# Test new exports
# =============================================================================


def test_new_exports_from_visualization_module():
    """Test new functions are exported from visualization module."""
    from frustrampnn import visualization

    # Sequence map functions
    assert hasattr(visualization, "plot_sequence_map")
    assert hasattr(visualization, "plot_sequence_map_plotly")
    assert hasattr(visualization, "get_native_frustration_per_position")

    # Sankey functions
    assert hasattr(visualization, "plot_frustration_sankey")
    assert hasattr(visualization, "plot_frustration_sankey_matplotlib")
    assert hasattr(visualization, "compute_category_flows")


# =============================================================================
# Test figure cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
