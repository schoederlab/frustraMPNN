"""
Command-line interface for FrustraMPNN.

This module provides the CLI commands for running frustration predictions
from the command line.

Example:
    $ frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt
    $ frustrampnn info
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="frustrampnn")
def cli() -> None:
    """FrustraMPNN: Ultra-fast frustration prediction.

    Predict single-residue local energetic frustration profiles for proteins
    using a message-passing neural network.

    Examples:

        # Predict frustration for a PDB file
        frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt

        # Predict specific chains
        frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt --chains A,B

        # Show package info
        frustrampnn info
    """
    pass


@cli.command()
@click.option(
    "--pdb",
    "-p",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input PDB file",
)
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint file (.ckpt)",
)
@click.option(
    "--output",
    "-o",
    default="frustration_predictions.csv",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output CSV file (default: frustration_predictions.csv)",
)
@click.option(
    "--chains",
    default=None,
    help="Comma-separated chain IDs to analyze (default: all chains)",
)
@click.option(
    "--positions",
    default=None,
    help="Comma-separated positions to analyze (0-indexed, default: all)",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use (default: auto-detect)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config file for old-format checkpoints",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress bar",
)
def predict(
    pdb: Path,
    checkpoint: Path,
    output: Path,
    chains: str | None,
    positions: str | None,
    device: str | None,
    config: Path | None,
    quiet: bool,
) -> None:
    """Predict frustration values for a protein structure.

    This command runs site-saturation mutagenesis predictions for all
    positions in the specified chains, outputting frustration values
    for each possible amino acid substitution.

    Examples:

        # Basic usage
        frustrampnn predict --pdb 1UBQ.pdb --checkpoint model.ckpt

        # Specify output file and chains
        frustrampnn predict -p protein.pdb -c model.ckpt -o results.csv --chains A

        # Use CPU explicitly
        frustrampnn predict -p protein.pdb -c model.ckpt --device cpu
    """
    # Import here to avoid slow startup for --help
    from frustrampnn import FrustraMPNN

    # Parse chains
    chain_list = None
    if chains is not None:
        chain_list = [c.strip() for c in chains.split(",")]

    # Parse positions
    position_list = None
    if positions is not None:
        try:
            position_list = [int(p.strip()) for p in positions.split(",")]
        except ValueError as e:
            raise click.BadParameter(
                "Positions must be comma-separated integers", param_hint="--positions"
            ) from e

    # Load model
    if not quiet:
        click.echo(f"Loading model from {checkpoint}...")

    try:
        model = FrustraMPNN.from_pretrained(
            checkpoint_path=checkpoint,
            config_path=config,
            device=device,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except Exception as e:
        raise click.ClickException(f"Failed to load model: {e}") from e

    if not quiet:
        click.echo(f"Model loaded on {model.device}")
        click.echo(f"Predicting frustration for {pdb}...")

    # Run prediction
    try:
        results = model.predict(
            pdb_path=pdb,
            chains=chain_list,
            positions=position_list,
            show_progress=not quiet,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except Exception as e:
        raise click.ClickException(f"Prediction failed: {e}") from e

    # Save results
    results.to_csv(output, index=False)

    if not quiet:
        click.echo(f"Results saved to {output}")
        click.echo(f"Total predictions: {len(results)}")

        # Show summary statistics
        if len(results) > 0:
            click.echo(f"Frustration range: [{results['frustration_pred'].min():.3f}, "
                      f"{results['frustration_pred'].max():.3f}]")


@cli.command()
def info() -> None:
    """Show package information.

    Displays version, authors, and system information including
    available devices.
    """
    import frustrampnn

    click.echo(f"FrustraMPNN v{frustrampnn.__version__}")
    click.echo(f"Authors: {frustrampnn.__author__}")
    click.echo()

    # Show device info
    try:
        import torch

        click.echo("System Information:")
        click.echo(f"  PyTorch version: {torch.__version__}")
        click.echo(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            click.echo(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            click.echo(f"  CUDA version: {torch.version.cuda}")
    except ImportError:
        click.echo("  PyTorch: not installed")


@cli.command()
@click.argument(
    "pdb_files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint file (.ckpt)",
)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for CSV files (default: current directory)",
)
@click.option(
    "--chains",
    default=None,
    help="Comma-separated chain IDs to analyze (default: all chains)",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "cpu"]),
    help="Device to use (default: auto-detect)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress bar",
)
def batch(
    pdb_files: tuple[Path, ...],
    checkpoint: Path,
    output_dir: Path,
    chains: str | None,
    device: str | None,
    quiet: bool,
) -> None:
    """Predict frustration for multiple PDB files.

    Each PDB file will generate a separate output CSV file in the
    output directory, named {pdb_stem}_frustration.csv.

    Examples:

        # Process multiple files
        frustrampnn batch *.pdb --checkpoint model.ckpt

        # Specify output directory
        frustrampnn batch file1.pdb file2.pdb -c model.ckpt -o results/
    """
    from frustrampnn import FrustraMPNN

    # Parse chains
    chain_list = None
    if chains is not None:
        chain_list = [c.strip() for c in chains.split(",")]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if not quiet:
        click.echo(f"Loading model from {checkpoint}...")

    try:
        model = FrustraMPNN.from_pretrained(
            checkpoint_path=checkpoint,
            device=device,
        )
    except Exception as e:
        raise click.ClickException(f"Failed to load model: {e}") from e

    if not quiet:
        click.echo(f"Model loaded on {model.device}")
        click.echo(f"Processing {len(pdb_files)} PDB files...")

    # Process each file
    success_count = 0
    fail_count = 0

    for pdb_path in pdb_files:
        output_file = output_dir / f"{pdb_path.stem}_frustration.csv"

        try:
            results = model.predict(
                pdb_path=pdb_path,
                chains=chain_list,
                show_progress=not quiet,
            )
            results.to_csv(output_file, index=False)
            success_count += 1

            if not quiet:
                click.echo(f"  {pdb_path.name} -> {output_file.name} ({len(results)} predictions)")

        except Exception as e:
            fail_count += 1
            click.echo(f"  {pdb_path.name}: FAILED - {e}", err=True)

    # Summary
    if not quiet:
        click.echo()
        click.echo(f"Completed: {success_count} succeeded, {fail_count} failed")


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to training config YAML file",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=None,
    help="Override number of training epochs",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--resume",
    "-r",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Resume training from checkpoint",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress verbose output",
)
def train(
    config: Path,
    epochs: int | None,
    seed: int | None,
    resume: Path | None,
    quiet: bool,
) -> None:
    """Train a FrustraMPNN model.

    Train a new model from scratch or resume training from a checkpoint.
    The training configuration is specified in a YAML file.

    Examples:

        # Train with default config
        frustrampnn train --config config.yaml

        # Train with custom epochs and seed
        frustrampnn train -c config.yaml --epochs 50 --seed 42

        # Resume from checkpoint
        frustrampnn train -c config.yaml --resume checkpoint.ckpt
    """
    # Import here to avoid slow startup for --help
    from frustrampnn.training import TrainingConfig
    from frustrampnn.training.trainer import Trainer

    # Build overrides list
    overrides = []
    if epochs is not None:
        overrides.append(f"training.epochs={epochs}")
    if seed is not None:
        overrides.append(f"training.seed={seed}")

    # Load configuration
    if not quiet:
        click.echo(f"Loading config from {config}...")

    try:
        cfg = TrainingConfig.from_yaml(config, overrides=overrides if overrides else None)
    except Exception as e:
        raise click.ClickException(f"Failed to load config: {e}") from e

    if not quiet:
        click.echo(f"Project: {cfg.project}")
        click.echo(f"Name: {cfg.name}")
        click.echo(f"Dataset: {cfg.datasets}")
        click.echo(f"Epochs: {cfg.training.epochs}")
        if seed is not None:
            click.echo(f"Seed: {seed}")
        if resume:
            click.echo(f"Resuming from: {resume}")

    # Create trainer
    try:
        trainer = Trainer(cfg, resume_from=resume)
    except Exception as e:
        raise click.ClickException(f"Failed to create trainer: {e}") from e

    # Run training
    if not quiet:
        click.echo("Starting training...")

    try:
        trainer.fit()
    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user")
        return
    except Exception as e:
        raise click.ClickException(f"Training failed: {e}") from e

    if not quiet:
        click.echo("Training complete!")


@cli.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint file (.ckpt)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config YAML (optional, uses checkpoint config if not provided)",
)
@click.option(
    "--split",
    default="test",
    type=click.Choice(["train", "val", "test"]),
    help="Data split to evaluate (default: test)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file for results (JSON format)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress verbose output",
)
def evaluate(
    checkpoint: Path,
    config: Path | None,
    split: str,
    output: Path | None,
    quiet: bool,
) -> None:
    """Evaluate a trained FrustraMPNN model.

    Load a trained model checkpoint and evaluate it on a dataset split.
    If no config is provided, the config from the checkpoint is used.

    Examples:

        # Evaluate on test set
        frustrampnn evaluate --checkpoint model.ckpt

        # Evaluate with specific config
        frustrampnn evaluate -c model.ckpt --config config.yaml

        # Save results to file
        frustrampnn evaluate -c model.ckpt -o results.json
    """
    import json

    import torch

    from frustrampnn.training import TrainingConfig
    from frustrampnn.training.trainer import Trainer

    # Load config from checkpoint or file
    if config:
        if not quiet:
            click.echo(f"Loading config from {config}...")
        try:
            cfg = TrainingConfig.from_yaml(config)
        except Exception as e:
            raise click.ClickException(f"Failed to load config: {e}") from e
    else:
        if not quiet:
            click.echo(f"Loading config from checkpoint {checkpoint}...")
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
            cfg_data = ckpt.get("cfg")
            if cfg_data is None:
                raise click.ClickException(
                    "No config found in checkpoint. Please provide --config"
                )
            cfg = TrainingConfig.from_dictconfig(cfg_data)
        except click.ClickException:
            raise
        except Exception as e:
            raise click.ClickException(f"Failed to load checkpoint: {e}") from e

    # Create trainer and setup
    if not quiet:
        click.echo(f"Setting up evaluation on {split} split...")

    try:
        trainer = Trainer(cfg)
        trainer.setup()
    except Exception as e:
        raise click.ClickException(f"Failed to setup trainer: {e}") from e

    # Run evaluation
    if not quiet:
        click.echo(f"Evaluating checkpoint: {checkpoint}")

    try:
        results = trainer.test(checkpoint_path=checkpoint)
    except Exception as e:
        raise click.ClickException(f"Evaluation failed: {e}") from e

    # Display results
    if not quiet:
        click.echo(f"\nEvaluation Results ({split}):")
        for metric, value in results.items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.4f}")
            else:
                click.echo(f"  {metric}: {value}")

    # Save results if output specified
    if output:
        try:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            if not quiet:
                click.echo(f"\nResults saved to {output}")
        except Exception as e:
            raise click.ClickException(f"Failed to save results: {e}") from e


def main() -> None:
    """Entry point for the frustrampnn CLI."""
    cli()


if __name__ == "__main__":
    main()

