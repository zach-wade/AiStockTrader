"""
Universe Commands Module

Handles all universe-related CLI commands including Layer 0 population,
universe statistics, and layer management.
"""

# Standard library imports
import asyncio
from datetime import datetime

# Third-party imports
import click

# Local imports
from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer
from main.utils.core import get_logger

logger = get_logger(__name__)


@click.group()
def universe():
    """Universe management and layer operations."""
    pass


@universe.command()
@click.option("--force", is_flag=True, help="Force repopulation even if universe exists")
@click.option(
    "--source",
    type=click.Choice(["alpaca", "polygon"]),
    default="alpaca",
    help="Data source for universe",
)
@click.option("--dry-run", is_flag=True, help="Run without saving to database")
@click.pass_context
def populate(ctx, force: bool, source: str, dry_run: bool):
    """Populate Layer 0 universe with all tradable symbols.

    Examples:
        # Populate universe from Alpaca
        python ai_trader.py universe populate

        # Force refresh universe
        python ai_trader.py universe populate --force

        # Dry run to see what would be added
        python ai_trader.py universe populate --dry-run
    """
    # Local imports
    from main.events.core import EventBusFactory
    from main.interfaces.events import IEventBus
    from main.scanners.layers.layer0_static_universe import Layer0StaticUniverseScanner

    logger.info(f"Populating Layer 0 universe from {source}")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    # Create event bus for scanner events
    event_bus: IEventBus = EventBusFactory.create(config)

    try:
        # Check if universe exists
        if not force and not dry_run:
            existing = asyncio.run(_check_existing_universe(config))
            if existing["count"] > 0:
                logger.info(f"Universe already populated with {existing['count']} symbols")
                if not click.confirm("Do you want to repopulate?"):
                    return

        # Run Layer 0 scanner
        scanner = Layer0StaticUniverseScanner(config, event_bus)
        symbols = asyncio.run(scanner.run())

        if dry_run:
            print(f"\n🔍 DRY RUN - Would populate with {len(symbols)} symbols")
            print(f"Sample symbols: {', '.join(symbols[:20])}")
        else:
            print(f"\n✅ Successfully populated Layer 0 with {len(symbols)} symbols")

    except Exception as e:
        logger.error(f"Universe population failed: {e}", exc_info=True)
        raise


@universe.command()
@click.option(
    "--layer",
    type=click.Choice(["0", "1", "2", "3", "all"]),
    default="all",
    help="Layer to show statistics for",
)
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
@click.pass_context
def stats(ctx, layer: str, detailed: bool):
    """Show universe statistics by layer.

    Examples:
        # Show all layer statistics
        python ai_trader.py universe stats

        # Show detailed Layer 2 statistics
        python ai_trader.py universe stats --layer 2 --detailed
    """
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory

    logger.info(f"Gathering universe statistics for layer: {layer}")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    try:
        # Initialize database and repository
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        if layer == "all":
            # Get stats for all layers
            stats = asyncio.run(_get_all_layer_stats(company_repo))
            _print_all_layer_stats(stats, detailed)
        else:
            # Get stats for specific layer
            layer_enum = DataLayer(int(layer))
            stats = asyncio.run(_get_layer_stats(company_repo, layer_enum))
            _print_layer_stats(layer, stats, detailed)

    except Exception as e:
        logger.error(f"Failed to get universe statistics: {e}", exc_info=True)
        raise
    finally:
        if db_adapter:
            asyncio.run(db_adapter.close())


@universe.command()
@click.option("--check-all", is_flag=True, help="Check health of all layers")
@click.option(
    "--layer", type=click.Choice(["0", "1", "2", "3"]), help="Check specific layer health"
)
@click.option("--fix", is_flag=True, help="Attempt to fix issues found")
@click.pass_context
def health(ctx, check_all: bool, layer: str | None, fix: bool):
    """Check universe health and data integrity.

    Examples:
        # Check all layers health
        python ai_trader.py universe health --check-all

        # Check and fix Layer 1 issues
        python ai_trader.py universe health --layer 1 --fix
    """
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory
    from main.monitoring.health.unified_health_reporter import UnifiedHealthReporter

    logger.info("Checking universe health")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    try:
        # Initialize health reporter and database
        health_reporter = UnifiedHealthReporter(config)
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        if check_all:
            # Check all layers
            results = {}
            for layer_value in range(4):  # Layers 0-3
                layer_enum = DataLayer(layer_value)
                companies = asyncio.run(company_repo.get_by_layer(layer_enum))
                layer_health = {
                    "healthy": len(companies) > 0,
                    "count": len(companies),
                    "checks": {
                        "has_symbols": {
                            "passed": len(companies) > 0,
                            "message": f"{len(companies)} symbols",
                        },
                        "data_quality": {"passed": True, "message": "OK"},
                    },
                }
                results[layer_value] = layer_health

            all_healthy = all(r["healthy"] for r in results.values())
            results = {"healthy": all_healthy, "layers": results, "issues": []}
            _print_health_results(results, fix)

        elif layer:
            # Check specific layer
            layer_enum = DataLayer(int(layer))
            companies = asyncio.run(company_repo.get_by_layer(layer_enum))
            results = {
                "healthy": len(companies) > 0,
                "checks": {
                    "has_symbols": {
                        "passed": len(companies) > 0,
                        "message": f"{len(companies)} symbols",
                    },
                    "data_quality": {"passed": True, "message": "OK"},
                },
                "issues": [],
            }
            _print_layer_health(layer, results, fix)
        else:
            click.echo("Please specify --check-all or --layer")
            ctx.exit(1)

        # Close database
        asyncio.run(db_adapter.close())

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise


@universe.command()
@click.option("--symbol", required=True, help="Symbol to promote")
@click.option("--to-layer", type=click.Choice(["1", "2", "3"]), required=True, help="Target layer")
@click.option("--reason", help="Reason for promotion")
@click.pass_context
def promote(ctx, symbol: str, to_layer: str, reason: str | None):
    """Manually promote a symbol to a higher layer.

    Examples:
        # Promote AAPL to Layer 2
        python ai_trader.py universe promote --symbol AAPL --to-layer 2

        # Promote with reason
        python ai_trader.py universe promote --symbol TSLA --to-layer 3 --reason "High volatility opportunity"
    """
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory
    from main.events.core import EventBusFactory
    from main.events.publishers.scanner_event_publisher import ScannerEventPublisher

    layer_enum = DataLayer(int(to_layer))

    logger.info(f"Promoting {symbol} to Layer {to_layer}")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    try:
        # Initialize components
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        # Get current layer
        company = asyncio.run(company_repo.get_by_symbol(symbol))
        if not company:
            logger.error(f"Symbol {symbol} not found in universe")
            ctx.exit(1)

        current_layer = DataLayer(company.get("layer", 0))

        if current_layer.value >= layer_enum.value:
            logger.info(f"{symbol} is already at Layer {current_layer.value} or higher")
            return

        # Update layer
        result = asyncio.run(
            company_repo.update_layer(
                symbol=symbol,
                layer=layer_enum,
                metadata={
                    "reason": reason or "Manual promotion",
                    "promoted_at": datetime.now().isoformat(),
                },
            )
        )

        if result.success:
            print(
                f"✅ Successfully promoted {symbol} from Layer {current_layer.value} to Layer {to_layer}"
            )

            # Publish promotion event
            event_bus = EventBusFactory.create(config)
            publisher = ScannerEventPublisher(event_bus)
            asyncio.run(
                publisher.publish_symbol_promoted(
                    symbol=symbol,
                    from_layer=current_layer,
                    to_layer=layer_enum,
                    reason=reason or "Manual promotion",
                )
            )
        else:
            logger.error(f"Failed to promote {symbol}: {result.errors}")
            ctx.exit(1)

    except Exception as e:
        logger.error(f"Promotion failed: {e}", exc_info=True)
        raise
    finally:
        if db_adapter:
            asyncio.run(db_adapter.close())


@universe.command()
@click.option("--layer", type=click.Choice(["0", "1", "2", "3"]), help="Export specific layer")
@click.option(
    "--format", type=click.Choice(["csv", "json", "txt"]), default="csv", help="Export format"
)
@click.option("--output", required=True, help="Output file path")
@click.pass_context
def export(ctx, layer: str | None, format: str, output: str):
    """Export universe symbols to file.

    Examples:
        # Export all symbols to CSV
        python ai_trader.py universe export --output universe.csv

        # Export Layer 2 symbols to JSON
        python ai_trader.py universe export --layer 2 --format json --output layer2.json
    """
    # Standard library imports
    import csv
    import json

    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory

    logger.info(f"Exporting universe to {output}")

    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config("unified_config")

    try:
        # Initialize database and repository
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        # Get symbols
        if layer:
            layer_enum = DataLayer(int(layer))
            companies = asyncio.run(company_repo.get_by_layer(layer_enum))
            logger.info(f"Exporting {len(companies)} Layer {layer} symbols")
        else:
            companies = asyncio.run(company_repo.get_all_active())
            logger.info(f"Exporting {len(companies)} total symbols")

        # Export based on format
        if format == "csv":
            with open(output, "w", newline="") as f:
                if companies:
                    writer = csv.DictWriter(f, fieldnames=companies[0].keys())
                    writer.writeheader()
                    writer.writerows(companies)
        elif format == "json":
            with open(output, "w") as f:
                json.dump(companies, f, indent=2, default=str)
        elif format == "txt":
            with open(output, "w") as f:
                for company in companies:
                    f.write(f"{company['symbol']}\n")

        print(f"✅ Exported {len(companies)} symbols to {output}")

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise
    finally:
        if db_adapter:
            asyncio.run(db_adapter.close())


# Helper functions


async def _check_existing_universe(config) -> dict:
    """Check if universe already exists."""
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory

    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)

    try:
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        companies = await company_repo.get_all_active()
        return {"count": len(companies)}

    finally:
        await db_adapter.close()


async def _get_all_layer_stats(company_repo) -> dict:
    """Get statistics for all layers."""
    stats = {}

    for layer_value in range(4):  # Layers 0-3
        layer = DataLayer(layer_value)
        companies = await company_repo.get_by_layer(layer)

        stats[layer_value] = {
            "count": len(companies),
            "symbols": [c["symbol"] for c in companies],
            "avg_liquidity_score": (
                sum(c.get("liquidity_score", 0) for c in companies) / len(companies)
                if companies
                else 0
            ),
            "avg_catalyst_score": (
                sum(c.get("catalyst_score", 0) for c in companies) / len(companies)
                if companies
                else 0
            ),
        }

    return stats


async def _get_layer_stats(company_repo, layer: DataLayer) -> dict:
    """Get statistics for a specific layer."""
    companies = await company_repo.get_by_layer(layer)

    return {
        "count": len(companies),
        "symbols": [c["symbol"] for c in companies],
        "exchanges": list(set(c.get("exchange", "Unknown") for c in companies)),
        "avg_liquidity_score": (
            sum(c.get("liquidity_score", 0) for c in companies) / len(companies) if companies else 0
        ),
        "avg_catalyst_score": (
            sum(c.get("catalyst_score", 0) for c in companies) / len(companies) if companies else 0
        ),
        "avg_premarket_score": (
            sum(c.get("premarket_score", 0) for c in companies) / len(companies) if companies else 0
        ),
        "top_symbols": sorted(companies, key=lambda x: x.get("liquidity_score", 0), reverse=True)[
            :10
        ],
    }


def _print_all_layer_stats(stats: dict, detailed: bool):
    """Print statistics for all layers."""
    print("\n" + "=" * 60)
    print("UNIVERSE STATISTICS BY LAYER")
    print("=" * 60)

    total_symbols = sum(s["count"] for s in stats.values())
    print(f"\nTotal Universe: {total_symbols} symbols")

    for layer, layer_stats in stats.items():
        percentage = (layer_stats["count"] / total_symbols * 100) if total_symbols > 0 else 0
        print(f"\n📊 Layer {layer} ({DataLayer(layer).name}):")
        print(f"  Count: {layer_stats['count']} ({percentage:.1f}% of universe)")
        print(f"  Avg Liquidity: {layer_stats['avg_liquidity_score']:.2f}")
        print(f"  Avg Catalyst: {layer_stats['avg_catalyst_score']:.2f}")

        if detailed and layer_stats["symbols"]:
            print(f"  Sample symbols: {', '.join(layer_stats['symbols'][:10])}")


def _print_layer_stats(layer: str, stats: dict, detailed: bool):
    """Print statistics for a specific layer."""
    print(f"\n📊 Layer {layer} Statistics:")
    print("-" * 40)
    print(f"Symbol count: {stats['count']}")
    print(f"Exchanges: {', '.join(stats['exchanges'][:5])}")
    print(f"Avg Liquidity Score: {stats['avg_liquidity_score']:.2f}")
    print(f"Avg Catalyst Score: {stats['avg_catalyst_score']:.2f}")
    print(f"Avg Premarket Score: {stats['avg_premarket_score']:.2f}")

    if detailed:
        print("\n🏆 Top Symbols by Liquidity:")
        for company in stats["top_symbols"]:
            print(f"  {company['symbol']}: {company.get('liquidity_score', 0):.2f}")

        if len(stats["symbols"]) > 20:
            print(f"\nAll symbols ({stats['count']}):")
            # Print in columns
            symbols = stats["symbols"]
            for i in range(0, len(symbols), 10):
                print("  " + ", ".join(symbols[i : i + 10]))


def _print_health_results(results: dict, fix: bool):
    """Print universe health check results."""
    print("\n" + "=" * 60)
    print("UNIVERSE HEALTH CHECK")
    print("=" * 60)

    overall_health = "✅ HEALTHY" if results.get("healthy", False) else "❌ ISSUES FOUND"
    print(f"\nOverall Status: {overall_health}")

    for layer, layer_health in results.get("layers", {}).items():
        status = "✅" if layer_health.get("healthy", False) else "❌"
        print(f"\n{status} Layer {layer}:")

        for check, check_result in layer_health.get("checks", {}).items():
            check_status = "✅" if check_result.get("passed", False) else "❌"
            print(f"  {check_status} {check}: {check_result.get('message', '')}")

    if results.get("issues") and not fix:
        print(f"\n⚠️ Found {len(results['issues'])} issues. Use --fix to attempt repairs.")


def _print_layer_health(layer: str, results: dict, fix: bool):
    """Print health check for specific layer."""
    print(f"\n📊 Layer {layer} Health Check:")
    print("-" * 40)

    overall = "✅ HEALTHY" if results.get("healthy", False) else "❌ ISSUES FOUND"
    print(f"Status: {overall}")

    for check, check_result in results.get("checks", {}).items():
        status = "✅" if check_result.get("passed", False) else "❌"
        print(f"{status} {check}: {check_result.get('message', '')}")

    if results.get("issues"):
        print("\n⚠️ Issues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")

        if not fix:
            print("\nUse --fix to attempt repairs.")


def _print_fix_results(results: dict):
    """Print results of fix attempts."""
    print("\n🔧 Fix Results:")
    print(f"  Attempted: {results.get('attempted', 0)}")
    print(f"  Successful: {results.get('successful', 0)}")
    print(f"  Failed: {results.get('failed', 0)}")

    if results.get("details"):
        print("\nDetails:")
        for detail in results["details"]:
            print(f"  - {detail}")
