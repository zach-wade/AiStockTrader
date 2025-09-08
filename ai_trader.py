#!/usr/bin/env python3
"""
AI Trader - Unified Command Line Interface for Hunter-Killer Trading System

This is the main entry point for the AI Trading System, providing a comprehensive
CLI for all trading operations including scanning, backtesting, training, and live trading.

Usage:
    python ai_trader.py <command> [options]

Commands:
    status      Check system health and configuration
    version     Show version information
    backfill    Backfill historical market data
    scan        Run market scanners for opportunities
    features    Calculate technical features
    train       Train ML models
    predict     Generate predictions
    trade       Run trading system (paper/live)
    positions   View current positions
    monitor     Real-time system monitoring
    risk        Risk management status

Examples:
    python ai_trader.py status
    python ai_trader.py trade --mode paper --strategy momentum
    python ai_trader.py scan --symbols AAPL,MSFT,NVDA
"""

import asyncio
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/ai_trader.log", mode="a")],
)
logger = logging.getLogger(__name__)

# Version information
VERSION = "2.0.0"
BUILD_DATE = "2024-12-09"
SYSTEM_NAME = "Hunter-Killer Trading System"


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(), help="Path to configuration file")
@click.pass_context
def cli(ctx, debug, config):
    """
    AI Trader - Hunter-Killer Trading System CLI

    A comprehensive trading system with ML-powered strategies,
    real-time scanning, and automated execution.
    """
    # Set debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Store context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["config_path"] = config

    # Log startup
    logger.info(f"{SYSTEM_NAME} CLI started - Version {VERSION}")


@cli.command()
@click.pass_context
def status(ctx):
    """Check system health and configuration."""
    click.echo(f"\n{'='*60}")
    click.echo(f"  {SYSTEM_NAME} - System Status")
    click.echo(f"{'='*60}\n")

    from src.application.cli.commands import check_system_status

    try:
        # Run async status check
        status_info = asyncio.run(check_system_status())

        # Display status information
        click.echo("📊 System Components:")
        click.echo(f"  • Database: {status_info['database']['status']}")
        click.echo(f"    - Connection: {status_info['database']['connection']}")
        click.echo(f"    - Tables: {status_info['database']['tables']}")

        click.echo(f"\n  • Broker: {status_info['broker']['status']}")
        click.echo(f"    - Type: {status_info['broker']['type']}")
        click.echo(f"    - Connection: {status_info['broker']['connected']}")

        click.echo(f"\n  • Market Data: {status_info['market_data']['status']}")
        click.echo(f"    - Provider: {status_info['market_data']['provider']}")

        click.echo("\n📈 Trading Status:")
        click.echo(f"  • Positions: {status_info['trading']['positions']}")
        click.echo(f"  • Open Orders: {status_info['trading']['open_orders']}")
        click.echo(f"  • Account Value: ${status_info['trading']['account_value']:,.2f}")

        click.echo("\n🔧 Configuration:")
        click.echo(f"  • Environment: {status_info['config']['environment']}")
        click.echo(f"  • Debug Mode: {ctx.obj.get('debug', False)}")
        click.echo(f"  • Config Path: {ctx.obj.get('config_path', 'default')}")

        click.echo(f"\n✅ System Status: {status_info['overall_status']}")

    except Exception as e:
        click.echo(f"❌ Error checking system status: {e}", err=True)
        logger.exception("Failed to check system status")
        sys.exit(1)

    click.echo(f"\n{'='*60}\n")


@cli.command()
def version():
    """Show version information."""
    click.echo(f"\n{'='*60}")
    click.echo(f"  {SYSTEM_NAME}")
    click.echo(f"{'='*60}\n")
    click.echo(f"  Version:     {VERSION}")
    click.echo(f"  Build Date:  {BUILD_DATE}")
    click.echo(f"  Python:      {sys.version.split()[0]}")
    click.echo(f"  Platform:    {sys.platform}")
    click.echo("\n  Components:")
    click.echo("    • Core Trading Engine:  v2.0")
    click.echo("    • Scanner System:       v2.0")
    click.echo("    • ML Models:           v2.0")
    click.echo("    • Risk Management:     v2.0")
    click.echo("\n  Repository:")
    click.echo("    https://github.com/zach-wade/AiStockTrader")
    click.echo(f"\n{'='*60}\n")


@cli.command()
@click.option("--symbols", help="Comma-separated list of symbols")
@click.option("--days", type=int, default=30, help="Number of days to backfill")
@click.option(
    "--data-type",
    default="market_data",
    type=click.Choice(["market_data", "news", "fundamentals", "all"]),
    help="Type of data to backfill",
)
@click.option(
    "--layer",
    type=click.Choice(["layer0", "layer1", "layer2", "layer3"]),
    help="Backfill all symbols in a layer",
)
@click.option(
    "--provider",
    type=click.Choice(["polygon", "alpaca", "auto"]),
    default="auto",
    help="Data provider to use",
)
@click.option(
    "--timeframe",
    default="1Day",
    type=click.Choice(["1Min", "5Min", "15Min", "1Hour", "1Day"]),
    help="Timeframe for market data",
)
@click.pass_context
def backfill(ctx, symbols, days, data_type, layer, provider, timeframe):  # noqa: PLR0913, ARG001
    """Backfill historical market data."""
    click.echo(f"📥 Backfilling {data_type} data for {days} days...")

    if layer:
        click.echo(f"   Layer: {layer}")
    elif symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        click.echo(f"   Symbols: {', '.join(symbol_list)}")
    else:
        click.echo("❌ Error: Either --symbols or --layer must be specified")
        return

    click.echo(f"   Provider: {provider}")
    click.echo(f"   Timeframe: {timeframe}")

    from src.application.cli.backfill_command import execute_backfill

    try:
        # Run the backfill
        results = asyncio.run(
            execute_backfill(
                symbols=symbols, days=days, data_type=data_type, layer=layer, provider=provider
            )
        )

        # Display results
        if "error" in results:
            click.echo(f"❌ Error: {results['error']}")
        else:
            if "market_data" in results:
                stats = results["market_data"]
                click.echo("\n✅ Market Data Backfill Complete:")
                click.echo(f"   • Records processed: {stats['records_processed']:,}")
                click.echo(f"   • Symbols processed: {stats['symbols_processed']}")
                click.echo(f"   • Duration: {stats.get('duration_seconds', 0):.2f} seconds")
                click.echo(f"   • Rate: {stats.get('records_per_second', 0):.0f} records/sec")

            if "news" in results:
                click.echo(
                    f"\n📰 News: {results['news'].get('news_items_fetched', 0)} items fetched"
                )

            if "fundamentals" in results:
                click.echo(
                    f"\n📊 Fundamentals: {results['fundamentals'].get('fundamentals_fetched', 0)} symbols fetched"
                )

    except Exception as e:
        click.echo(f"❌ Backfill failed: {e}", err=True)
        logger.exception("Backfill command failed")


@cli.command()
@click.option("--symbols", help="Comma-separated list of symbols to scan")
@click.option("--scanners", help="Specific scanners to run")
@click.option("--all", "scan_all", is_flag=True, help="Run all scanners")
@click.pass_context
def scan(ctx, symbols, scanners, scan_all):  # noqa: ARG001
    """Run market scanners for opportunities."""
    click.echo("🔍 Running market scanners...")

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        click.echo(f"   Symbols: {', '.join(symbol_list)}")

    # TODO: Implement in Phase 2
    click.echo("⚠️  Scan command will be implemented in Phase 2")


@cli.command()
@click.option("--mode", type=click.Choice(["paper", "live"]), default="paper", help="Trading mode")
@click.option("--strategy", help="Trading strategy to use")
@click.option("--symbols", help="Comma-separated list of symbols")
@click.option("--dry-run", is_flag=True, help="Simulate without executing trades")
@click.pass_context
def trade(ctx, mode, strategy, symbols, dry_run):  # noqa: ARG001
    """Run the trading system."""
    click.echo(f"\n🚀 Starting {mode.upper()} Trading...")

    if dry_run:
        click.echo("   Mode: DRY RUN (no actual trades)")

    if strategy:
        click.echo(f"   Strategy: {strategy}")

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
        click.echo(f"   Symbols: {', '.join(symbol_list)}")

    # TODO: Implement in Phase 5
    click.echo("⚠️  Trade command will be implemented in Phase 5")


@cli.command()
@click.pass_context
def positions(ctx):  # noqa: ARG001
    """View current positions and P&L."""
    click.echo("\n📊 Current Positions:")

    # TODO: Implement position viewing
    click.echo("⚠️  Positions command will be implemented in Phase 5")


@cli.command()
@click.pass_context
def monitor(ctx):  # noqa: ARG001
    """Start real-time system monitoring."""
    click.echo("\n📡 Starting real-time monitoring...")

    # TODO: Implement in Phase 6
    click.echo("⚠️  Monitor command will be implemented in Phase 6")


@cli.command()
@click.pass_context
def risk(ctx):  # noqa: ARG001
    """Check risk management status."""
    click.echo("\n⚠️  Risk Management Status:")

    # TODO: Implement in Phase 6
    click.echo("⚠️  Risk command will be implemented in Phase 6")


@cli.group()
def data():
    """Data pipeline management commands."""
    pass


@data.command("status")
def data_status():
    """Check data pipeline status."""
    click.echo("\n📊 Data Pipeline Status:")
    # TODO: Implement in Phase 1
    click.echo("⚠️  Data status will be implemented in Phase 1")


@cli.group()
def features():
    """Feature calculation commands."""
    pass


@features.command("calculate")
@click.option("--symbols", required=True, help="Comma-separated list of symbols")
def calculate_features(symbols):
    """Calculate features for symbols."""
    click.echo(f"\n🔢 Calculating features for: {symbols}")
    # TODO: Implement in Phase 3
    click.echo("⚠️  Feature calculation will be implemented in Phase 3")


@cli.group()
def models():
    """ML model management commands."""
    pass


@models.command("list")
def list_models():
    """List available models."""
    click.echo("\n🤖 Available Models:")
    # TODO: Implement in Phase 4
    click.echo("⚠️  Model listing will be implemented in Phase 4")


@cli.command()
def train():
    """Train ML models."""
    click.echo("\n🎓 Training ML models...")
    # TODO: Implement in Phase 4
    click.echo("⚠️  Training will be implemented in Phase 4")


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\n⛔ Trading system shutdown by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error in CLI")
        click.echo(f"\n❌ Fatal error: {e}", err=True)
        sys.exit(1)
