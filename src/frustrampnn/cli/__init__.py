"""
Command-line interface for FrustraMPNN.

This module provides the CLI entry points for the frustrampnn package.

Example:
    $ frustrampnn --help
    $ frustrampnn predict --pdb protein.pdb --checkpoint model.ckpt
    $ frustrampnn info
"""

from frustrampnn.cli.main import cli, main

__all__ = ["main", "cli"]
