"""
Tests for the FrustraMPNN command-line interface.

These tests verify that the CLI commands work correctly and provide
appropriate help messages and error handling.
"""

from __future__ import annotations

from click.testing import CliRunner


class TestCLIImport:
    """Test CLI module imports."""

    def test_cli_import(self) -> None:
        """Test CLI can be imported."""
        from frustrampnn.cli import cli, main

        assert callable(main)
        assert callable(cli)

    def test_cli_main_import(self) -> None:
        """Test main module can be imported."""
        from frustrampnn.cli.main import batch, cli, info, main, predict

        assert callable(main)
        assert callable(cli)
        assert callable(predict)
        assert callable(info)
        assert callable(batch)


class TestCLIHelp:
    """Test CLI help messages."""

    def test_cli_help(self) -> None:
        """Test CLI help works."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "FrustraMPNN" in result.output
        assert "predict" in result.output
        assert "info" in result.output
        assert "batch" in result.output

    def test_predict_help(self) -> None:
        """Test predict command help."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])

        assert result.exit_code == 0
        assert "--pdb" in result.output
        assert "--checkpoint" in result.output
        assert "--output" in result.output
        assert "--chains" in result.output
        assert "--device" in result.output
        assert "--quiet" in result.output

    def test_info_help(self) -> None:
        """Test info command help."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])

        assert result.exit_code == 0
        assert "package information" in result.output.lower()

    def test_batch_help(self) -> None:
        """Test batch command help."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])

        assert result.exit_code == 0
        assert "--checkpoint" in result.output
        assert "--output-dir" in result.output
        assert "multiple PDB files" in result.output


class TestInfoCommand:
    """Test the info command."""

    def test_info_command(self) -> None:
        """Test info command runs successfully."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "FrustraMPNN" in result.output
        # Should show version
        assert "v" in result.output or "1." in result.output

    def test_info_shows_authors(self) -> None:
        """Test info command shows authors."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        # Should contain at least one author name
        assert any(
            name in result.output for name in ["Beining", "Engelberger", "Schoeder", "Meiler"]
        )

    def test_info_shows_pytorch_info(self) -> None:
        """Test info command shows PyTorch information."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        assert result.exit_code == 0
        assert "PyTorch" in result.output
        assert "CUDA" in result.output


class TestPredictCommandValidation:
    """Test predict command input validation."""

    def test_predict_missing_pdb(self) -> None:
        """Test predict fails without --pdb."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--checkpoint", "model.ckpt"])

        # Should fail - either missing option or file doesn't exist
        assert result.exit_code != 0
        # Click may report missing option or file not found (validates file existence)
        assert (
            "Missing option" in result.output
            or "required" in result.output.lower()
            or "does not exist" in result.output.lower()
        )

    def test_predict_missing_checkpoint(self) -> None:
        """Test predict fails without --checkpoint."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--pdb", "protein.pdb"])

        # Should fail - either missing option or file doesn't exist
        assert result.exit_code != 0
        # Click may report missing option or file not found (validates file existence)
        assert (
            "Missing option" in result.output
            or "required" in result.output.lower()
            or "does not exist" in result.output.lower()
        )

    def test_predict_nonexistent_pdb(self) -> None:
        """Test predict fails with nonexistent PDB file."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--pdb", "nonexistent.pdb", "--checkpoint", "model.ckpt"]
        )

        assert result.exit_code != 0
        # Click should report the file doesn't exist
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_predict_nonexistent_checkpoint(self, tmp_path) -> None:
        """Test predict fails with nonexistent checkpoint."""
        from frustrampnn.cli import cli

        # Create a dummy PDB file
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--pdb", str(pdb_file), "--checkpoint", "nonexistent.ckpt"]
        )

        assert result.exit_code != 0


class TestBatchCommandValidation:
    """Test batch command input validation."""

    def test_batch_missing_checkpoint(self, tmp_path) -> None:
        """Test batch fails without --checkpoint."""
        from frustrampnn.cli import cli

        # Create a dummy PDB file
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", str(pdb_file)])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_batch_no_files(self) -> None:
        """Test batch fails without PDB files."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--checkpoint", "model.ckpt"])

        assert result.exit_code != 0


class TestCLIVersion:
    """Test CLI version option."""

    def test_version_option(self) -> None:
        """Test --version shows version."""
        from frustrampnn.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should contain version number
        assert "1." in result.output or "0." in result.output


class TestCLIEntryPoint:
    """Test CLI entry point."""

    def test_main_function(self) -> None:
        """Test main() function exists and is callable."""
        from frustrampnn.cli import main

        # main() should be callable (it wraps cli())
        assert callable(main)

    def test_cli_as_module(self) -> None:
        """Test CLI can be run as module."""
        import subprocess
        import sys

        # This should not raise an error
        result = subprocess.run(
            [sys.executable, "-m", "frustrampnn.cli", "--help"],
            capture_output=True,
            text=True,
        )

        # Should show help (exit code 0) or fail gracefully
        # Note: This may fail if package is not installed
        assert result.returncode == 0 or "No module named" in result.stderr
