#!/usr/bin/env python3
"""
AI Trader - Unified Command Line Interface (Refactored)

This is a thin entry point that delegates to modular command groups.
All business logic is contained in the command modules under src/main/app/commands/

Usage:
    python ai_trader.py <group> <command> [options]

Command Groups:
    trading     Trading system operations
    data        Data pipeline and management
    scanner     Market scanning and screening
    universe    Universe and layer management
    utility     System utilities and tools

For help on a specific group:
    python ai_trader.py <group> --help
"""

# Standard library imports
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set DATA_LAKE_PATH if running from parent directory
if not os.path.exists("data_lake") and os.path.exists("ai_trader/data_lake"):
    os.environ["DATA_LAKE_PATH"] = os.path.abspath("ai_trader/data_lake")

# Local imports
# Load environment variables FIRST, before any other imports
from main.config.env_loader import ensure_environment_loaded

ensure_environment_loaded()

# Standard library imports
import logging

# Third-party imports
import click

# Local imports
# Import all command groups
from main.app.commands import COMMAND_GROUPS, data, scanner, trading, universe, utility
from main.config import get_config_manager
from main.utils.core import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


class AliasedGroup(click.Group):
    """Custom Click group that supports command aliases."""

    def get_command(self, ctx, cmd_name):
        """Get command, supporting aliases."""
        # Define aliases for backward compatibility
        aliases = {
            "trade": ("trading", "trade"),
            "backfill": ("data", "backfill"),
            "backtest": ("trading", "backtest"),
            "train": ("utility", "train"),
            "features": ("utility", "features"),
            "events": ("utility", "events"),
            "validate": ("utility", "validate"),
            "status": ("utility", "status"),
            "shutdown": ("utility", "shutdown"),
            "scan": ("scanner", "scan"),
            "populate": ("universe", "populate"),
        }

        # Check if this is a direct alias
        if cmd_name in aliases:
            group_name, command_name = aliases[cmd_name]
            # Get the group and then the command
            group = super().get_command(ctx, group_name)
            if group and hasattr(group, "get_command"):
                return group.get_command(ctx, command_name)

        # Otherwise use normal resolution
        return super().get_command(ctx, cmd_name)


@click.group(cls=AliasedGroup, invoke_without_command=True)
@click.option(
    "--config", default=None, help="Path to configuration file (uses modular config by default)"
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"])
)
@click.option(
    "--environment", default=None, help="Environment override (paper, live, backtest, training)"
)
@click.option("--version", is_flag=True, help="Show version information")
@click.pass_context
def cli(ctx, config, log_level, environment, version):
    """
    AI Trader - Algorithmic Trading System

    A modular trading system with machine learning capabilities.

    Use 'ai_trader.py <group> --help' for help on specific command groups.
    """
    # Handle version flag
    if version:
        print("AI Trader v2.0.0 (Refactored)")
        print("Python", sys.version)
        ctx.exit(0)

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)

    # Set up logging level
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Load and validate configuration
    try:
        config_manager = get_config_manager()

        # Override environment if specified
        if environment:
            config_manager.set_environment(environment)
            logger.info(f"Environment set to: {environment}")

        # Store config in context for subcommands
        ctx.obj = {"config_manager": config_manager}

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        ctx.exit(1)


# Add all command groups
cli.add_command(trading)
cli.add_command(data)
cli.add_command(scanner)
cli.add_command(universe)
cli.add_command(utility)


# Add convenience shortcuts for common commands
@cli.command()
@click.pass_context
def help(ctx):
    """Show detailed help information."""
    click.echo("\nAI Trader Command Groups:\n")
    for group_name, description in COMMAND_GROUPS.items():
        click.echo(f"  {group_name:12} {description}")
    click.echo("\nCommon Commands:")
    click.echo("  ai_trader.py trading trade --mode paper    # Start paper trading")
    click.echo("  ai_trader.py data backfill --days 30       # Backfill 30 days of data")
    click.echo("  ai_trader.py scanner scan --pipeline       # Run full scanner pipeline")
    click.echo("  ai_trader.py universe populate             # Populate Layer 0 universe")
    click.echo("  ai_trader.py utility status                # Check system status")
    click.echo("\nFor detailed help on any group:")
    click.echo("  ai_trader.py <group> --help")


@cli.command()
@click.pass_context
def quickstart(ctx):
    """Interactive quickstart guide for new users."""
    click.echo("\n🚀 AI Trader Quickstart Guide\n")
    click.echo("Let's get you started with AI Trader!")
    click.echo("-" * 40)

    # Check configuration
    click.echo("\n1️⃣ Checking configuration...")
    try:
        config_manager = get_config_manager()
        config = config_manager.load_config("unified_config")
        click.echo("   ✅ Configuration loaded successfully")
    except Exception as e:
        click.echo(f"   ❌ Configuration error: {e}")
        click.echo("   Please check your configuration files in config/")
        ctx.exit(1)

    # Check database connection
    click.echo("\n2️⃣ Checking database connection...")
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory

    try:
        db_factory = DatabaseFactory()
        db = db_factory.create_async_database(config)
        # Standard library imports
        import asyncio

        asyncio.run(db.test_connection())
        click.echo("   ✅ Database connection successful")
    except Exception as e:
        click.echo(f"   ❌ Database error: {e}")
        click.echo("   Please check your database configuration")
        ctx.exit(1)

    # Suggest next steps
    click.echo("\n3️⃣ Suggested next steps:")
    click.echo("   a) Populate universe: ai_trader.py universe populate")
    click.echo("   b) Backfill data: ai_trader.py data backfill --days 30")
    click.echo("   c) Run scanner: ai_trader.py scanner scan --layer 0")
    click.echo("   d) Start paper trading: ai_trader.py trading trade --mode paper")

    click.echo("\n📚 For more help: ai_trader.py help")
    click.echo("💡 Tip: Use --help with any command for detailed options")


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
