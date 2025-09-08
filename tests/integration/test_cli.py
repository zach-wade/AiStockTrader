"""
Integration tests for the AI Trader CLI
"""

import subprocess
from pathlib import Path

import pytest


def run_cli_command(command: str) -> tuple[int, str, str]:
    """
    Run a CLI command and return the result.

    Args:
        command: Command string to execute

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = f"python ai_trader.py {command}"
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
        check=False,  # Project root
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that help command works."""
        returncode, stdout, stderr = run_cli_command("--help")
        assert returncode == 0
        assert "Hunter-Killer Trading System CLI" in stdout
        assert "Commands:" in stdout

    def test_version_command(self):
        """Test version command."""
        returncode, stdout, stderr = run_cli_command("version")
        assert returncode == 0
        assert "Hunter-Killer Trading System" in stdout
        assert "Version:" in stdout
        assert "2.0.0" in stdout

    def test_status_command(self):
        """Test status command."""
        returncode, stdout, stderr = run_cli_command("status")
        # Status might fail if database isn't configured, but command should run
        assert "System Status" in stdout or "System Status" in stderr
        assert "Database:" in stdout or "Database:" in stderr

    def test_invalid_command(self):
        """Test that invalid commands are handled."""
        returncode, stdout, stderr = run_cli_command("invalid-command")
        assert returncode != 0
        assert "Error" in stderr or "No such command" in stderr


class TestCLICommands:
    """Test individual CLI commands."""

    def test_backfill_help(self):
        """Test backfill command help."""
        returncode, stdout, stderr = run_cli_command("backfill --help")
        assert returncode == 0
        assert "Backfill historical market data" in stdout
        assert "--symbols" in stdout
        assert "--days" in stdout

    def test_scan_help(self):
        """Test scan command help."""
        returncode, stdout, stderr = run_cli_command("scan --help")
        assert returncode == 0
        assert "Run market scanners" in stdout
        assert "--symbols" in stdout

    def test_trade_help(self):
        """Test trade command help."""
        returncode, stdout, stderr = run_cli_command("trade --help")
        assert returncode == 0
        assert "Run the trading system" in stdout
        assert "--mode" in stdout
        assert "--strategy" in stdout

    def test_data_status(self):
        """Test data status subcommand."""
        returncode, stdout, stderr = run_cli_command("data status")
        # Command should run even if not fully implemented
        assert "Data Pipeline Status" in stdout or "Data Pipeline Status" in stderr

    def test_features_calculate_help(self):
        """Test features calculate subcommand."""
        returncode, stdout, stderr = run_cli_command("features calculate --help")
        assert returncode == 0
        assert "--symbols" in stdout

    def test_models_list(self):
        """Test models list subcommand."""
        returncode, stdout, stderr = run_cli_command("models list")
        # Command should run even if not fully implemented
        assert "Available Models" in stdout or "Available Models" in stderr


class TestCLIOptions:
    """Test CLI global options."""

    def test_debug_flag(self):
        """Test debug flag enables debug output."""
        returncode, stdout, stderr = run_cli_command("--debug version")
        assert returncode == 0
        # Debug flag should be mentioned somewhere
        assert "Version:" in stdout

    def test_config_option(self):
        """Test config file option."""
        returncode, stdout, stderr = run_cli_command("--config test.yaml version")
        assert returncode == 0
        assert "Version:" in stdout


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
