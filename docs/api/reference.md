# API Reference

Complete reference for the FrustraMPNN Python API.

## frustrampnn

### FrustraMPNN

Main class for frustration prediction.

```python
class FrustraMPNN:
    """FrustraMPNN predictor for single-residue frustration."""
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "FrustraMPNN":
        """Load a pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            config_path: Path to config YAML (for old checkpoints)
            device: Device to use ('cuda', 'cpu', or None for auto)
        
        Returns:
            FrustraMPNN instance
        
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If model loading fails
        """
    
    def predict(
        self,
        pdb_path: str,
        chains: Optional[List[str]] = None,
        positions: Optional[List[int]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Predict frustration for a protein structure.
        
        Args:
            pdb_path: Path to PDB file
            chains: Chains to analyze (None = all)
            positions: Positions to analyze (None = all, 0-indexed)
            show_progress: Show progress bar
        
        Returns:
            DataFrame with columns:
            - frustration_pred: Predicted frustration index
            - position: 0-indexed position
            - wildtype: Wild-type amino acid
            - mutation: Mutant amino acid
            - pdb: PDB identifier
            - chain: Chain identifier
        """
    
    def predict_batch(
        self,
        pdb_paths: List[str],
        chains: Optional[List[str]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Predict frustration for multiple structures.
        
        Args:
            pdb_paths: List of PDB file paths
            chains: Chains to analyze (None = all)
            show_progress: Show progress bar
        
        Returns:
            Combined DataFrame for all structures
        """
    
    def to(self, device: str) -> "FrustraMPNN":
        """Move model to device.
        
        Args:
            device: Target device ('cuda' or 'cpu')
        
        Returns:
            Self for chaining
        """
    
    @property
    def device(self) -> str:
        """Current device."""
    
    @property
    def model(self) -> torch.nn.Module:
        """Underlying PyTorch model."""
    
    @property
    def cfg(self) -> OmegaConf:
        """Model configuration."""
```

## frustrampnn.data

### Mutation

Data class for representing mutations.

```python
@dataclass
class Mutation:
    """Represents a single mutation.
    
    Attributes:
        position: 0-indexed residue position
        wildtype: Wild-type amino acid (1-letter code)
        mutation: Mutant amino acid (1-letter code)
        pdb: PDB identifier
        ddG: Optional stability change
        chain: Optional chain identifier
    """
    position: int
    wildtype: str
    mutation: str
    pdb: str
    ddG: Optional[float] = None
    chain: Optional[str] = None
```

## frustrampnn.visualization

### plot_single_residue

```python
def plot_single_residue(
    results: pd.DataFrame,
    position: int,
    chain: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_thresholds: bool = True,
    highlight_native: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot frustration for all amino acids at a position.
    
    Args:
        results: FrustraMPNN results DataFrame
        position: 0-indexed position
        chain: Chain identifier
        title: Plot title (default: auto-generated)
        figsize: Figure size
        show_thresholds: Show category threshold lines
        highlight_native: Highlight native residue in blue
        ax: Existing axes to plot on
    
    Returns:
        Matplotlib Figure
    """
```

### plot_single_residue_plotly

```python
def plot_single_residue_plotly(
    results: pd.DataFrame,
    position: int,
    chain: str,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 500,
) -> go.Figure:
    """Interactive single-residue plot with Plotly.
    
    Args:
        results: FrustraMPNN results DataFrame
        position: 0-indexed position
        chain: Chain identifier
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
    
    Returns:
        Plotly Figure
    """
```

### plot_frustration_heatmap

```python
def plot_frustration_heatmap(
    results: pd.DataFrame,
    chain: str,
    figsize: Tuple[int, int] = (20, 10),
    cmap: str = "RdYlGn_r",
    vmin: float = -2.0,
    vmax: float = 2.0,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot frustration heatmap (positions x mutations).
    
    Args:
        results: FrustraMPNN results DataFrame
        chain: Chain identifier
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        show_colorbar: Show colorbar
        title: Plot title
        ax: Existing axes to plot on
    
    Returns:
        Matplotlib Figure
    """
```

### plot_frustration_heatmap_plotly

```python
def plot_frustration_heatmap_plotly(
    results: pd.DataFrame,
    chain: str,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 600,
    colorscale: str = "RdYlGn_r",
) -> go.Figure:
    """Interactive frustration heatmap with Plotly.
    
    Args:
        results: FrustraMPNN results DataFrame
        chain: Chain identifier
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        colorscale: Plotly colorscale name
    
    Returns:
        Plotly Figure
    """
```

## frustrampnn.validation

### compare_with_frustrapy

```python
def compare_with_frustrapy(
    pdb_path: str,
    chain: str,
    frustrampnn_results: pd.DataFrame,
    positions: Optional[List[int]] = None,
    n_jobs: int = 1,
) -> ComparisonResult:
    """Compare FrustraMPNN predictions with frustrapy calculations.
    
    Args:
        pdb_path: Path to PDB file
        chain: Chain identifier
        frustrampnn_results: FrustraMPNN predictions DataFrame
        positions: Positions to compare (None = all)
        n_jobs: Number of parallel jobs for frustrapy
    
    Returns:
        ComparisonResult with metrics and detailed results
    """
```

### ComparisonResult

```python
@dataclass
class ComparisonResult:
    """Results from comparing FrustraMPNN with frustrapy.
    
    Attributes:
        spearman: Spearman correlation coefficient
        pearson: Pearson correlation coefficient
        rmse: Root mean squared error
        mae: Mean absolute error
        accuracy: Categorical accuracy
        f1_score: F1 score (macro average)
        confusion_matrix: Confusion matrix
        results: DataFrame with detailed results
    """
    spearman: float
    pearson: float
    rmse: float
    mae: float
    accuracy: float
    f1_score: float
    confusion_matrix: np.ndarray
    results: pd.DataFrame
    
    def plot(
        self,
        show_regression: bool = False,
        show_identity: bool = True,
        show_categories: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Plot comparison scatter plot.
        
        Args:
            show_regression: Show regression line
            show_identity: Show y=x identity line
            show_categories: Color points by category
            title: Plot title
        
        Returns:
            Plotly Figure
        """
```

## frustrampnn.constants

```python
# Standard amino acid alphabet
ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"

# List of 20 standard amino acids
AMINO_ACIDS: List[str] = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                          "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# Frustration category thresholds
FRUSTRATION_THRESHOLDS: Dict[str, float] = {
    "highly": -1.0,      # <= -1.0 is highly frustrated
    "minimally": 0.58,   # >= 0.58 is minimally frustrated
}

# Color scheme
FRUSTRATION_COLORS: Dict[str, str] = {
    "highly": "#FF0000",     # Red
    "neutral": "#808080",    # Gray
    "minimally": "#00FF00",  # Green
    "native": "#0000FF",     # Blue
}
```

## frustrampnn.training

### Trainer

```python
class Trainer:
    """High-level training interface.
    
    Args:
        config: TrainingConfig or path to config YAML
    """
    
    def __init__(self, config: Union[TrainingConfig, str]):
        ...
    
    def fit(self) -> None:
        """Run training."""
    
    def test(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """Evaluate on test set.
        
        Args:
            checkpoint_path: Path to checkpoint (default: best)
        
        Returns:
            Dictionary of metrics
        """
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """Training configuration.
    
    Attributes:
        project: Project name
        name: Run name
        epochs: Number of epochs
        learn_rate: Learning rate
        seed: Random seed
        ...
    """
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
```

## CLI Entry Points

```python
# Main entry point
def main():
    """CLI entry point (frustrampnn command)."""

# Subcommands
def predict_command(...):
    """frustrampnn predict"""

def batch_command(...):
    """frustrampnn batch"""

def train_command(...):
    """frustrampnn train"""

def evaluate_command(...):
    """frustrampnn evaluate"""

def info_command(...):
    """frustrampnn info"""
```

