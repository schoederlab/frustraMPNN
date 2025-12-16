"""Pytest configuration and fixtures for FrustraMPNN tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test_data directory."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_pdb_path(test_data_dir: Path) -> Path:
    """Return path to test PDB file."""
    return test_data_dir / "1UBQ.pdb"


@pytest.fixture
def reference_output_path(test_data_dir: Path) -> Path:
    """Return path to reference output CSV."""
    return test_data_dir / "1UBQ_reference_output.csv"


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_sequence() -> str:
    """Return a sample protein sequence."""
    return "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture
def short_sequence() -> str:
    """Return a short protein sequence for quick tests."""
    return "MQIFVKTL"


@pytest.fixture
def sample_frustration_df() -> pd.DataFrame:
    """Create sample frustration prediction DataFrame."""
    data = []
    for pos in range(5):
        wt = "MQIFV"[pos]
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            # Create mock frustration values
            value = np.sin(pos * 0.5) + (ord(aa) - 77) / 20
            data.append(
                {
                    "frustration_pred": value,
                    "position": pos,
                    "wildtype": wt,
                    "mutation": aa,
                    "pdb": "test",
                    "chain": "A",
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def reference_output_df(reference_output_path: Path) -> pd.DataFrame:
    """Load reference output DataFrame."""
    if not reference_output_path.exists():
        pytest.skip("Reference output not found")
    return pd.read_csv(reference_output_path)


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_length() -> int:
    """Default sequence length for tests."""
    return 10


@pytest.fixture
def num_neighbors() -> int:
    """Default number of neighbors for tests."""
    return 5


@pytest.fixture
def hidden_dim() -> int:
    """Default hidden dimension for tests."""
    return 32


@pytest.fixture
def sample_node_features(batch_size: int, seq_length: int, hidden_dim: int) -> torch.Tensor:
    """Create sample node features tensor."""
    return torch.randn(batch_size, seq_length, hidden_dim)


@pytest.fixture
def sample_edge_features(
    batch_size: int, seq_length: int, num_neighbors: int, hidden_dim: int
) -> torch.Tensor:
    """Create sample edge features tensor."""
    return torch.randn(batch_size, seq_length, num_neighbors, hidden_dim)


@pytest.fixture
def sample_edge_idx(batch_size: int, seq_length: int, num_neighbors: int) -> torch.Tensor:
    """Create sample edge indices tensor."""
    # Create valid neighbor indices (each node points to random neighbors)
    idx = torch.randint(0, seq_length, (batch_size, seq_length, num_neighbors))
    return idx


@pytest.fixture
def sample_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """Create sample mask tensor (all ones)."""
    return torch.ones(batch_size, seq_length)


@pytest.fixture
def sample_residue_idx(batch_size: int, seq_length: int) -> torch.Tensor:
    """Create sample residue index tensor."""
    return torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)


@pytest.fixture
def sample_chain_labels(batch_size: int, seq_length: int) -> torch.Tensor:
    """Create sample chain labels tensor (all same chain)."""
    return torch.zeros(batch_size, seq_length, dtype=torch.long)


@pytest.fixture
def sample_ca_coords(batch_size: int, seq_length: int) -> torch.Tensor:
    """Create sample CA coordinates tensor."""
    # Create a simple alpha helix-like structure
    coords = torch.zeros(batch_size, seq_length, 3)
    for i in range(seq_length):
        # Approximate alpha helix geometry
        coords[:, i, 0] = 1.5 * np.cos(i * 100 * np.pi / 180)  # x
        coords[:, i, 1] = 1.5 * np.sin(i * 100 * np.pi / 180)  # y
        coords[:, i, 2] = i * 1.5  # z (rise per residue)
    return coords


@pytest.fixture
def sample_backbone_coords(batch_size: int, seq_length: int) -> torch.Tensor:
    """Create sample backbone coordinates tensor (N, CA, C, O)."""
    # Create simple backbone geometry
    coords = torch.zeros(batch_size, seq_length, 4, 3)
    for i in range(seq_length):
        # Approximate backbone geometry
        ca_x = 1.5 * np.cos(i * 100 * np.pi / 180)
        ca_y = 1.5 * np.sin(i * 100 * np.pi / 180)
        ca_z = i * 1.5

        # N is ~1.46 Å from CA
        coords[:, i, 0, :] = torch.tensor([ca_x - 0.5, ca_y - 0.5, ca_z - 1.0])  # N
        coords[:, i, 1, :] = torch.tensor([ca_x, ca_y, ca_z])  # CA
        coords[:, i, 2, :] = torch.tensor([ca_x + 0.5, ca_y + 0.5, ca_z + 0.5])  # C
        coords[:, i, 3, :] = torch.tensor([ca_x + 1.0, ca_y + 1.0, ca_z + 0.5])  # O
    return coords


# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def model_config() -> dict[str, Any]:
    """Return default model configuration."""
    return {
        "hidden_dim": 128,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "num_neighbors": 30,
        "dropout": 0.1,
        "vocab_size": 21,
    }


@pytest.fixture
def small_model_config() -> dict[str, Any]:
    """Return small model configuration for fast tests."""
    return {
        "hidden_dim": 32,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_neighbors": 5,
        "dropout": 0.0,
        "vocab_size": 21,
    }


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    yield
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Skip Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "regression: marks regression tests")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_checkpoint(test_data_dir: Path):
    """Skip test if no checkpoint is available."""
    checkpoint_dir = Path("inference/vanilla_model_weights")
    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.pt")):
        pytest.skip("Model checkpoint not available")
