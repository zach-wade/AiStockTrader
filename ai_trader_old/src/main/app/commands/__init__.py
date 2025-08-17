"""
Command Modules Package

This package contains all CLI command groups for the AI Trader system.
Each module represents a logical group of related commands.
"""

from .data_commands import data
from .scanner_commands import scanner
from .trading_commands import trading
from .universe_commands import universe
from .utility_commands import utility

# Export all command groups
__all__ = [
    "trading",
    "data",
    "scanner",
    "universe",
    "utility",
]

# Command group descriptions for help text
COMMAND_GROUPS = {
    "trading": "Trading system operations (trade, backtest, positions)",
    "data": "Data pipeline and management (backfill, validate, archive)",
    "scanner": "Market scanning and screening (scan, alerts, status)",
    "universe": "Universe and layer management (populate, stats, promote)",
    "utility": "System utilities (train, features, events, status, shutdown)",
}
