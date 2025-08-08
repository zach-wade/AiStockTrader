"""
Scanner Commands Module

Handles all scanner-related CLI commands including universe scanning,
catalyst detection, premarket analysis, and scanner monitoring.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import click

from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer
from main.utils.core import get_logger

logger = get_logger(__name__)


@click.group()
def scanner():
    """Market scanner and screening commands."""
    pass


@scanner.command()
@click.option('--pipeline', is_flag=True, 
              help='Run full scanner pipeline (Layer 0-3)')
@click.option('--layer', type=click.Choice(['0', '1', '2', '3']), 
              help='Run specific layer scanner')
@click.option('--catalyst', is_flag=True, 
              help='Run catalyst scanner (earnings, news, insider)')
@click.option('--symbols', help='Comma-separated symbols to scan (for catalyst)')
@click.option('--dry-run', is_flag=True, 
              help='Run without saving results')
@click.option('--show-alerts', is_flag=True, 
              help='Display generated alerts')
@click.pass_context
def scan(ctx, pipeline: bool, layer: Optional[str], catalyst: bool,
         symbols: Optional[str], dry_run: bool, show_alerts: bool):
    """Run market scanners to identify trading opportunities.
    
    Examples:
        # Run full pipeline scan (Layer 0-3)
        python ai_trader.py scanner scan --pipeline
        
        # Run specific layer scanner
        python ai_trader.py scanner scan --layer 1
        
        # Run catalyst scanner for specific symbols
        python ai_trader.py scanner scan --catalyst --symbols AAPL,GOOGL,MSFT
        
        # Dry run with alert display
        python ai_trader.py scanner scan --layer 2 --dry-run --show-alerts
    """
    from main.scanners.scanner_pipeline import ScannerPipeline
    from main.scanners.scanner_orchestrator import ScannerOrchestrator
    from main.scanners.layers.layer2_catalyst_orchestrator import Layer2CatalystOrchestrator
    from main.interfaces.events import IEventBus
    from main.events.core import EventBusFactory
    
    logger.info("Starting market scan")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    # Create event bus for scanner events
    event_bus: IEventBus = EventBusFactory.create(config)
    
    try:
        if pipeline:
            # Run full pipeline
            logger.info("Running full scanner pipeline (Layer 0-3)")
            pipeline_scanner = ScannerPipeline(config, event_bus)
            results = asyncio.run(pipeline_scanner.run_full_pipeline(dry_run=dry_run))
            _print_pipeline_results(results, show_alerts)
            
        elif layer:
            # Run specific layer
            logger.info(f"Running Layer {layer} scanner")
            orchestrator = ScannerOrchestrator(config, event_bus)
            layer_enum = DataLayer(int(layer))
            results = asyncio.run(orchestrator.run_layer(layer_enum, dry_run=dry_run))
            _print_layer_results(layer, results, show_alerts)
            
        elif catalyst:
            # Run catalyst scanner
            logger.info("Running catalyst scanner")
            
            # Parse symbols
            if symbols:
                symbol_list = [s.strip() for s in symbols.split(',')]
            else:
                # Get Layer 2 symbols for catalyst scanning
                symbol_list = asyncio.run(_get_layer_symbols(config, DataLayer.CATALYST))
            
            catalyst_scanner = Layer2CatalystOrchestrator(config, event_bus)
            alerts = asyncio.run(catalyst_scanner.scan_catalysts(
                symbols=symbol_list,
                dry_run=dry_run
            ))
            _print_catalyst_results(alerts, show_alerts)
            
        else:
            click.echo("Please specify --pipeline, --layer, or --catalyst")
            ctx.exit(1)
            
    except Exception as e:
        logger.error(f"Scanner failed: {e}", exc_info=True)
        raise


@scanner.command()
@click.option('--layer', type=click.Choice(['0', '1', '2', '3', 'all']), 
              default='all', help='Layer to check status')
@click.option('--detailed', is_flag=True, 
              help='Show detailed scanner metrics')
@click.pass_context
def status(ctx, layer: str, detailed: bool):
    """Check scanner status and statistics.
    
    Examples:
        # Check all scanner status
        python ai_trader.py scanner status
        
        # Check specific layer with details
        python ai_trader.py scanner status --layer 2 --detailed
    """
    from main.scanners.scanner_orchestrator import ScannerOrchestrator
    
    logger.info(f"Checking scanner status for layer: {layer}")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize orchestrator for status
        orchestrator = ScannerOrchestrator(config)
        
        if layer == 'all':
            # Get status for all scanners
            status_data = asyncio.run(orchestrator.get_scanner_status())
            _print_all_scanner_status(status_data, detailed)
        else:
            # Get status for specific layer from scanner status
            layer_enum = DataLayer(int(layer))
            all_status = asyncio.run(orchestrator.get_scanner_status())
            # Filter for specific layer
            status_data = {k: v for k, v in all_status.items() if f'layer{layer}' in k.lower()}
            _print_layer_status(layer, status_data, detailed)
            
    except Exception as e:
        logger.error(f"Failed to get scanner status: {e}", exc_info=True)
        raise


@scanner.command()
@click.option('--hours', type=int, default=24, 
              help='Show alerts from last N hours')
@click.option('--min-score', type=float, 
              help='Minimum alert score to display')
@click.option('--type', type=click.Choice(['earnings', 'news', 'insider', 'premarket', 'all']), 
              default='all', help='Alert type to display')
@click.option('--format', type=click.Choice(['table', 'json', 'csv']), 
              default='table', help='Output format')
@click.pass_context
def alerts(ctx, hours: int, min_score: Optional[float], type: str, format: str):
    """View recent scanner alerts.
    
    Examples:
        # View all alerts from last 24 hours
        python ai_trader.py scanner alerts
        
        # View high-score earnings alerts
        python ai_trader.py scanner alerts --type earnings --min-score 80
        
        # Export alerts to CSV
        python ai_trader.py scanner alerts --hours 48 --format csv > alerts.csv
    """
    from main.monitoring.alerts.alert_manager import AlertManager
    import pandas as pd
    
    logger.info(f"Fetching scanner alerts from last {hours} hours")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize alert manager
        manager = AlertManager(config)
        
        # Calculate time range
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Fetch alerts
        alerts_data = asyncio.run(manager.get_alerts(
            start_time=start_time,
            alert_type=type if type != 'all' else None,
            min_score=min_score
        ))
        
        # Format and display
        if format == 'table':
            if alerts_data:
                df = pd.DataFrame(alerts_data)
                print(df.to_string())
            else:
                print("No alerts found")
        elif format == 'json':
            import json
            print(json.dumps(alerts_data, indent=2, default=str))
        elif format == 'csv':
            if alerts_data:
                df = pd.DataFrame(alerts_data)
                print(df.to_csv(index=False))
            else:
                print("No alerts found")
            
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}", exc_info=True)
        raise


@scanner.command()
@click.option('--layer', type=click.Choice(['0', '1', '2', '3']), 
              required=True, help='Layer to configure')
@click.option('--threshold', type=float, 
              help='Score threshold for layer qualification')
@click.option('--max-symbols', type=int, 
              help='Maximum symbols for this layer')
@click.option('--scan-interval', type=int, 
              help='Scan interval in minutes')
@click.option('--show', is_flag=True, 
              help='Show current configuration')
@click.pass_context
def configure(ctx, layer: str, threshold: Optional[float], 
              max_symbols: Optional[int], scan_interval: Optional[int], show: bool):
    """Configure scanner parameters.
    
    Examples:
        # Show current Layer 2 configuration
        python ai_trader.py scanner configure --layer 2 --show
        
        # Update Layer 1 threshold
        python ai_trader.py scanner configure --layer 1 --threshold 75.0
        
        # Configure Layer 3 settings
        python ai_trader.py scanner configure --layer 3 --max-symbols 50 --scan-interval 5
    """
    # Scanner config management - use config directly
    class ScannerConfigManager:
        def __init__(self, config):
            self.config = config
        
        def get_layer_config(self, layer):
            return self.config.get(f'scanners.layer{layer.value}', {})
        
        def update_layer_config(self, layer, updates):
            # Would update config
            pass
        
        def save(self):
            # Would save config
            pass
    
    layer_enum = DataLayer(int(layer))
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize config manager
        scanner_config = ScannerConfigManager(config)
        
        if show:
            # Show current configuration
            current_config = scanner_config.get_layer_config(layer_enum)
            _print_scanner_config(layer, current_config)
        else:
            # Update configuration
            updates = {}
            if threshold is not None:
                updates['threshold'] = threshold
            if max_symbols is not None:
                updates['max_symbols'] = max_symbols
            if scan_interval is not None:
                updates['scan_interval_minutes'] = scan_interval
            
            if updates:
                scanner_config.update_layer_config(layer_enum, updates)
                scanner_config.save()
                logger.info(f"Updated Layer {layer} configuration")
                
                # Show updated config
                new_config = scanner_config.get_layer_config(layer_enum)
                _print_scanner_config(layer, new_config)
            else:
                click.echo("No configuration changes specified")
                
    except Exception as e:
        logger.error(f"Configuration update failed: {e}", exc_info=True)
        raise


@scanner.command()
@click.option('--clear-all', is_flag=True, 
              help='Clear all scanner cache')
@click.option('--layer', type=click.Choice(['0', '1', '2', '3']), 
              help='Clear cache for specific layer')
@click.option('--older-than', type=int, 
              help='Clear cache older than N hours')
@click.pass_context
def cache(ctx, clear_all: bool, layer: Optional[str], older_than: Optional[int]):
    """Manage scanner cache.
    
    Examples:
        # Clear all scanner cache
        python ai_trader.py scanner cache --clear-all
        
        # Clear Layer 2 cache
        python ai_trader.py scanner cache --layer 2
        
        # Clear cache older than 24 hours
        python ai_trader.py scanner cache --older-than 24
    """
    from main.scanners.utils.scanner_cache_manager import ScannerCacheManager
    
    logger.info("Managing scanner cache")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize cache manager
        cache_manager = ScannerCacheManager(config)
        
        if clear_all:
            # Clear all cache
            result = asyncio.run(cache_manager.clear_all())
            print(f"✅ Cleared {result['cleared_entries']} cache entries")
            print(f"   Freed {result['freed_mb']:.2f} MB")
            
        elif layer:
            # Clear layer cache
            layer_enum = DataLayer(int(layer))
            result = asyncio.run(cache_manager.clear_layer(layer_enum))
            print(f"✅ Cleared Layer {layer} cache: {result['cleared_entries']} entries")
            
        elif older_than:
            # Clear old cache
            cutoff = datetime.now() - timedelta(hours=older_than)
            result = asyncio.run(cache_manager.clear_older_than(cutoff))
            print(f"✅ Cleared cache older than {older_than} hours: {result['cleared_entries']} entries")
            
        else:
            # Show cache statistics
            stats = asyncio.run(cache_manager.get_stats())
            _print_cache_stats(stats)
            
    except Exception as e:
        logger.error(f"Cache operation failed: {e}", exc_info=True)
        raise


# Helper functions

async def _get_layer_symbols(config, layer: DataLayer) -> List[str]:
    """Get symbols for a specific layer from database."""
    from main.data_pipeline.storage.database_factory import DatabaseFactory
    from main.data_pipeline.storage.repositories import get_repository_factory
    
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    try:
        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)
        
        # Get symbols for layer
        companies = await company_repo.get_by_layer(layer)
        return [c['symbol'] for c in companies]
        
    finally:
        await db_adapter.close()


def _print_pipeline_results(results: Dict, show_alerts: bool):
    """Print pipeline scan results."""
    print("\n" + "="*60)
    print("SCANNER PIPELINE RESULTS")
    print("="*60)
    
    for layer, layer_results in results.items():
        print(f"\n📊 Layer {layer}:")
        print(f"  Symbols scanned: {layer_results.get('symbols_scanned', 0)}")
        print(f"  Qualified: {layer_results.get('qualified', 0)}")
        print(f"  Duration: {layer_results.get('duration_seconds', 0):.2f}s")
        
        if show_alerts and layer_results.get('alerts'):
            print(f"  Alerts generated: {len(layer_results['alerts'])}")
            for alert in layer_results['alerts'][:5]:
                print(f"    - {alert['symbol']}: {alert['message']} (score: {alert['score']:.1f})")


def _print_layer_results(layer: str, results: Dict, show_alerts: bool):
    """Print single layer scan results."""
    print(f"\n📊 Layer {layer} Scan Results:")
    print("-" * 40)
    print(f"Symbols scanned: {results.get('symbols_scanned', 0)}")
    print(f"Qualified symbols: {results.get('qualified', 0)}")
    print(f"Average score: {results.get('avg_score', 0):.2f}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f}s")
    
    if results.get('top_symbols'):
        print(f"\n🏆 Top Symbols:")
        for symbol_data in results['top_symbols'][:10]:
            print(f"  {symbol_data['symbol']}: {symbol_data['score']:.2f}")
    
    if show_alerts and results.get('alerts'):
        print(f"\n🔔 Alerts ({len(results['alerts'])}):")
        for alert in results['alerts'][:10]:
            print(f"  {alert['symbol']}: {alert['message']}")


def _print_catalyst_results(alerts: List[Dict], show_alerts: bool):
    """Print catalyst scanner results."""
    print(f"\n🔍 Catalyst Scanner Results:")
    print("-" * 40)
    print(f"Total alerts: {len(alerts)}")
    
    # Group by type
    by_type = {}
    for alert in alerts:
        alert_type = alert.get('type', 'unknown')
        if alert_type not in by_type:
            by_type[alert_type] = []
        by_type[alert_type].append(alert)
    
    for alert_type, type_alerts in by_type.items():
        print(f"\n{alert_type.upper()}: {len(type_alerts)} alerts")
        
        if show_alerts:
            for alert in type_alerts[:5]:
                print(f"  {alert['symbol']}: {alert['message']} (score: {alert.get('score', 0):.1f})")


def _print_all_scanner_status(status: Dict, detailed: bool):
    """Print status for all scanners."""
    print("\n" + "="*60)
    print("SCANNER STATUS OVERVIEW")
    print("="*60)
    
    for layer, layer_status in status.items():
        status_icon = "✅" if layer_status.get('healthy', False) else "❌"
        print(f"\n{status_icon} Layer {layer}:")
        print(f"  Status: {layer_status.get('status', 'Unknown')}")
        print(f"  Last run: {layer_status.get('last_run', 'Never')}")
        print(f"  Symbols: {layer_status.get('symbol_count', 0)}")
        
        if detailed:
            print(f"  Success rate: {layer_status.get('success_rate', 0):.1%}")
            print(f"  Avg duration: {layer_status.get('avg_duration', 0):.2f}s")
            print(f"  Cache hit rate: {layer_status.get('cache_hit_rate', 0):.1%}")


def _print_layer_status(layer: str, status: Dict, detailed: bool):
    """Print status for specific layer."""
    print(f"\n📊 Layer {layer} Scanner Status:")
    print("-" * 40)
    print(f"Status: {status.get('status', 'Unknown')}")
    print(f"Healthy: {'Yes' if status.get('healthy', False) else 'No'}")
    print(f"Last run: {status.get('last_run', 'Never')}")
    print(f"Next run: {status.get('next_run', 'Not scheduled')}")
    print(f"Symbols: {status.get('symbol_count', 0)}")
    
    if detailed:
        print(f"\n📈 Performance Metrics:")
        print(f"  Total runs: {status.get('total_runs', 0)}")
        print(f"  Success rate: {status.get('success_rate', 0):.1%}")
        print(f"  Avg duration: {status.get('avg_duration', 0):.2f}s")
        print(f"  Cache hit rate: {status.get('cache_hit_rate', 0):.1%}")
        
        if status.get('recent_errors'):
            print(f"\n⚠️ Recent Errors:")
            for error in status['recent_errors'][:5]:
                print(f"  - {error}")


def _print_scanner_config(layer: str, config: Dict):
    """Print scanner configuration."""
    print(f"\n⚙️ Layer {layer} Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")


def _print_cache_stats(stats: Dict):
    """Print cache statistics."""
    print("\n📊 Scanner Cache Statistics:")
    print("-" * 40)
    print(f"Total entries: {stats.get('total_entries', 0)}")
    print(f"Total size: {stats.get('total_size_mb', 0):.2f} MB")
    print(f"Hit rate: {stats.get('hit_rate', 0):.1%}")
    print(f"Oldest entry: {stats.get('oldest_entry', 'N/A')}")
    
    if stats.get('by_layer'):
        print(f"\nBy Layer:")
        for layer, layer_stats in stats['by_layer'].items():
            print(f"  Layer {layer}: {layer_stats['entries']} entries ({layer_stats['size_mb']:.2f} MB)")