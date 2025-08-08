# File: universe/cli.py

import asyncio
import logging
import typer
from typing import Optional

from main.config.config_manager import get_config
from main.universe.universe_manager import UniverseManager

logger = logging.getLogger(__name__)

app = typer.Typer(help="AI Trader Universe Management CLI")


@app.command()
def populate(
    force: bool = typer.Option(False, "--force", help="Force re-population even if data exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing")
):
    """
    Populate the universe by running Layer 0 scanner to discover and store all tradable assets.
    """
    asyncio.run(_populate_universe(force, dry_run))


@app.command()
def stats():
    """
    Show universe statistics.
    """
    asyncio.run(_show_universe_stats())


@app.command()
def health():
    """
    Check universe health status.
    """
    asyncio.run(_check_universe_health())


@app.command()
def list_symbols(
    layer: str = typer.Option("0", "--layer", help="Layer to list symbols from (0, 1, 2, 3)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum number of symbols to show")
):
    """
    List symbols from a specific layer.
    """
    asyncio.run(_list_symbols(layer, limit))


async def _populate_universe(force: bool, dry_run: bool):
    """Populate the universe with Layer 0 scan."""
    config = get_config()
    universe_manager = UniverseManager(config)
    
    try:
        if dry_run:
            typer.echo("🔍 DRY RUN: Would populate universe with Layer 0 scan...")
            typer.echo("This would:")
            typer.echo("  1. Fetch all tradable assets from Alpaca")
            typer.echo("  2. Apply Layer 0 filters (exchange, asset class, ticker format)")
            typer.echo("  3. Store qualified companies in database")
            typer.echo("  4. Set is_active=True and layer qualifications to False")
            return
        
        typer.echo("🚀 Starting universe population...")
        
        # Check current state
        health = await universe_manager.health_check()
        if health['healthy'] and not force:
            typer.echo(f"✅ Universe already populated with {health['companies_count']} companies")
            typer.echo("Use --force to re-populate")
            return
        
        # Run population
        result = await universe_manager.populate_universe()
        
        if result['success']:
            typer.echo("✅ Universe population completed successfully!")
            typer.echo(f"⏱️  Duration: {result['duration_seconds']:.2f} seconds")
            typer.echo(f"🔍 Assets discovered: {result['assets_discovered']}")
            typer.echo(f"💾 Companies in database: {result['companies_in_db']}")
            typer.echo(f"📊 Universe stats: {result['universe_stats']}")
        else:
            typer.echo(f"❌ Universe population failed: {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Error during universe population: {e}")
        raise typer.Exit(1)
    finally:
        await universe_manager.close()


async def _show_universe_stats():
    """Show universe statistics."""
    config = get_config()
    universe_manager = UniverseManager(config)
    
    try:
        stats = await universe_manager.get_universe_stats()
        
        typer.echo("📊 Universe Statistics:")
        typer.echo(f"  Total companies: {stats['total_companies']:,}")
        typer.echo(f"  Active companies: {stats['active_companies']:,}")
        typer.echo(f"  Layer 0 (Basic): {stats.get('layer0_count', 0):,}")
        typer.echo(f"  Layer 1 (Liquid): {stats.get('layer1_count', 0):,}")
        typer.echo(f"  Layer 2 (Catalyst): {stats.get('layer2_count', 0):,}")
        typer.echo(f"  Layer 3 (Active): {stats.get('layer3_count', 0):,}")
        
        if 'active_percentage' in stats:
            typer.echo(f"  Active percentage: {stats['active_percentage']:.1f}%")
            typer.echo(f"  Layer 0 percentage: {stats.get('layer0_percentage', 0):.1f}%")
            typer.echo(f"  Layer 1 percentage: {stats.get('layer1_percentage', 0):.1f}%")
            typer.echo(f"  Layer 2 percentage: {stats.get('layer2_percentage', 0):.1f}%")
            typer.echo(f"  Layer 3 percentage: {stats.get('layer3_percentage', 0):.1f}%")
            
    except Exception as e:
        typer.echo(f"❌ Error getting universe stats: {e}")
        raise typer.Exit(1)
    finally:
        await universe_manager.close()


async def _check_universe_health():
    """Check universe health."""
    config = get_config()
    universe_manager = UniverseManager(config)
    
    try:
        health = await universe_manager.health_check()
        
        if health['healthy']:
            typer.echo("✅ Universe is healthy!")
        else:
            typer.echo("⚠️ Universe health issues detected:")
        
        typer.echo(f"  Companies count: {health['companies_count']:,}")
        typer.echo(f"  Has sufficient companies: {health['has_sufficient_companies']}")
        typer.echo(f"  Database accessible: {health['database_accessible']}")
        
        if 'universe_stats' in health:
            stats = health['universe_stats']
            typer.echo(f"  Universe stats: {stats}")
            
        if not health['healthy']:
            typer.echo("\n💡 Recommendations:")
            if not health['has_sufficient_companies']:
                typer.echo("  - Run 'ai-trader universe populate' to populate the universe")
            if not health['database_accessible']:
                typer.echo("  - Check database connection and configuration")
                
    except Exception as e:
        typer.echo(f"❌ Error checking universe health: {e}")
        raise typer.Exit(1)
    finally:
        await universe_manager.close()


async def _list_symbols(layer: str, limit: Optional[int]):
    """List symbols from a specific layer."""
    config = get_config()
    universe_manager = UniverseManager(config)
    
    try:
        symbols = await universe_manager.get_qualified_symbols(layer, limit)
        
        if not symbols:
            typer.echo(f"❌ No symbols found for layer {layer}")
            if layer != "0":
                typer.echo("💡 Layer qualification may not be implemented yet")
            return
        
        typer.echo(f"📋 Layer {layer} symbols ({len(symbols)} total):")
        
        # Show symbols in columns
        for i, symbol in enumerate(symbols):
            if i % 10 == 0:
                typer.echo("")
            typer.echo(f"{symbol:>6}", nl=False)
            if (i + 1) % 10 == 0 or i == len(symbols) - 1:
                typer.echo("")
        
        if limit and len(symbols) >= limit:
            typer.echo(f"\n(Showing first {limit} symbols)")
            
    except Exception as e:
        typer.echo(f"❌ Error listing symbols: {e}")
        raise typer.Exit(1)
    finally:
        await universe_manager.close()


if __name__ == "__main__":
    app()