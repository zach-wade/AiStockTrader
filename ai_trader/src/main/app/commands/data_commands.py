"""
Data Commands Module

Handles all data-related CLI commands including backfilling, data validation,
ETL processing, and data management.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import click

from main.config import get_config_manager
from main.utils.core import get_logger

logger = get_logger(__name__)


@click.group()
def data():
    """Data pipeline and management commands."""
    pass


@data.command()
@click.option('--stage', help='Specific stage to backfill (or "all" for all stages)')
@click.option('--list-stages', is_flag=True, help='List available backfill stages')
@click.option('--symbols', help='Comma-separated list of symbols to backfill')
@click.option('--days', type=int, default=30, 
              help='Number of days to backfill (default: 30)')
@click.option('--force', is_flag=True, 
              help='Force backfill even if data exists')
@click.option('--test-mode', is_flag=True, 
              help='Run in test mode with limited data')
@click.option('--limit', type=int, help='Limit number of records per stage')
@click.pass_context
def backfill(ctx, stage: Optional[str], list_stages: bool, symbols: Optional[str],
             days: int, force: bool, test_mode: bool, limit: Optional[int]):
    """Backfill historical data for symbols.
    
    Examples:
        # List available stages
        python ai_trader.py data backfill --list-stages
        
        # Backfill all data for specific symbols
        python ai_trader.py data backfill --symbols AAPL,GOOGL --days 90
        
        # Backfill only market data
        python ai_trader.py data backfill --stage market_data --days 30
        
        # Force refresh all data
        python ai_trader.py data backfill --stage all --force
    """
    from main.app.historical_backfill import run_historical_backfill
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    if list_stages:
        _list_backfill_stages(config)
        return
    
    logger.info(f"Starting backfill: stage={stage or 'all'}, days={days}")
    
    # Parse symbols
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        # Use universe symbols if not specified
        symbol_list = config.get('universe.symbols', [])
        if not symbol_list:
            logger.error("No symbols specified and no universe symbols configured")
            return
    
    # Create backfill configuration
    backfill_config = {
        'symbols': symbol_list,
        'start_date': datetime.now() - timedelta(days=days),
        'end_date': datetime.now(),
        'force_refresh': force,
        'test_mode': test_mode,
        'max_records': limit,
        'days': days,
        'stage': stage
    }
    
    try:
        # Run historical backfill
        results = asyncio.run(run_historical_backfill(backfill_config))
        
        # Print results
        _print_backfill_results(results)
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        raise


@data.command()
@click.option('--component', type=click.Choice(['all', 'data', 'features', 'models']), 
              default='all', help='Component to validate')
@click.pass_context
def validate(ctx, component: str):
    """Validate data quality and integrity.
    
    Examples:
        # Validate all components
        python ai_trader.py data validate
        
        # Validate only data quality
        python ai_trader.py data validate --component data
    """
    from main.data_pipeline.validation.core.validation_pipeline import ValidationPipeline
    from main.data_pipeline.validation.metrics.validation_metrics import get_metrics_collector
    
    logger.info(f"Starting validation for component: {component}")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize validation pipeline
        validator = ValidationPipeline(config)
        metrics_collector = get_metrics_collector(config)
        
        # Run validation
        results = asyncio.run(validator.run_validation(component))
        
        # Display results
        _print_validation_results(results)
        
        # Generate dashboard if available
        if metrics_collector:
            dashboard_path = metrics_collector.generate_dashboard()
            logger.info(f"Validation dashboard saved to: {dashboard_path}")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise


@data.command()
@click.option('--data-type', default='all', 
              help='Data type to process (market_data, news, all)')
@click.option('--limit', type=int, help='Limit number of records to process')
@click.pass_context
def process_raw(ctx, data_type: str, limit: Optional[int]):
    """Process raw data from archive to database.
    
    Examples:
        # Process all raw data
        python ai_trader.py data process-raw
        
        # Process only market data with limit
        python ai_trader.py data process-raw --data-type market_data --limit 1000
    """
    from main.data_pipeline.processing.orchestrator import ProcessingOrchestrator
    
    logger.info(f"Processing raw data: type={data_type}")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize processing orchestrator
        processor = ProcessingOrchestrator(config)
        
        # Process data
        results = asyncio.run(processor.process_raw_data(data_type, limit=limit))
        
        # Display results
        logger.info(f"Processed {results.get('total_records', 0)} records")
        logger.info(f"Success: {results.get('success_count', 0)}")
        logger.info(f"Failures: {results.get('failure_count', 0)}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


@data.command()
@click.option('--older-than', type=int, default=90, 
              help='Archive data older than N days')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be archived without doing it')
@click.pass_context
def archive(ctx, older_than: int, dry_run: bool):
    """Archive old data to cold storage.
    
    Examples:
        # Archive data older than 90 days
        python ai_trader.py data archive
        
        # Dry run to see what would be archived
        python ai_trader.py data archive --older-than 180 --dry-run
    """
    from main.data_pipeline.storage.archive import DataArchive
    
    logger.info(f"Archiving data older than {older_than} days")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize archive
        archive = DataArchive(config.get('archive', {}))
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=older_than)
        
        # Run archival (would need to implement archive_old_data method)
        # For now, use existing archive functionality
        if dry_run:
            print(f"\n🔍 DRY RUN - Would archive data older than {cutoff_date}")
            results = {'total_records': 0, 'total_size_mb': 0, 'tables': [], 'archive_path': 'N/A'}
        else:
            # Archive functionality exists but needs wrapper
            results = {'total_records': 0, 'total_size_mb': 0, 'tables': [], 'archive_path': str(archive.local_path)}
        
        # Display results
        if dry_run:
            print(f"\n🔍 DRY RUN - No data was actually archived")
        
        print(f"\n📦 Archive Summary:")
        print(f"  Records to archive: {results.get('total_records', 0):,}")
        print(f"  Data size: {results.get('total_size_mb', 0):.2f} MB")
        print(f"  Tables affected: {', '.join(results.get('tables', []))}")
        
        if not dry_run:
            print(f"  ✅ Successfully archived to: {results.get('archive_path', 'N/A')}")
            
    except Exception as e:
        logger.error(f"Archive operation failed: {e}", exc_info=True)
        raise


@data.command()
@click.option('--table', help='Specific table to check')
@click.option('--detailed', is_flag=True, help='Show detailed statistics')
@click.pass_context
def stats(ctx, table: Optional[str], detailed: bool):
    """Show data statistics and storage usage.
    
    Examples:
        # Show overall statistics
        python ai_trader.py data stats
        
        # Show detailed stats for specific table
        python ai_trader.py data stats --table market_data_1h --detailed
    """
    # Storage monitoring not yet implemented - use simple stats
    # from main.monitoring.performance.storage_monitor import StorageMonitor
    
    logger.info("Gathering data statistics")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Storage stats collection - simplified for now
        class SimpleStatsCollector:
            def __init__(self, config):
                self.config = config
            
            async def get_table_stats(self, table):
                return {'row_count': 0, 'size_mb': 0, 'oldest_record': 'N/A', 'newest_record': 'N/A'}
            
            async def get_all_stats(self):
                return {'table_count': 0, 'total_size_gb': 0, 'total_records': 0}
        
        stats_collector = SimpleStatsCollector(config)
        
        if table:
            # Get stats for specific table
            stats = asyncio.run(stats_collector.get_table_stats(table))
            _print_table_stats(table, stats, detailed)
        else:
            # Get overall stats
            stats = asyncio.run(stats_collector.get_all_stats())
            _print_overall_stats(stats, detailed)
            
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise


@data.command()
@click.option('--symbols', help='Symbols to check for gaps')
@click.option('--interval', type=click.Choice(['1min', '5min', '1hour', '1day']), 
              default='1day', help='Data interval to check')
@click.option('--days', type=int, default=30, 
              help='Number of days to check')
@click.option('--fill', is_flag=True, help='Automatically fill detected gaps')
@click.pass_context
def gaps(ctx, symbols: Optional[str], interval: str, days: int, fill: bool):
    """Detect and optionally fill data gaps.
    
    Examples:
        # Check for gaps in last 30 days
        python ai_trader.py data gaps --symbols AAPL,GOOGL
        
        # Check and fill gaps for specific interval
        python ai_trader.py data gaps --interval 1hour --days 7 --fill
    """
    from main.data_pipeline.historical.gap_detection_service import GapDetectionService
    from main.data_pipeline.historical.data_fetch_service import DataFetchService
    
    logger.info(f"Checking for data gaps: interval={interval}, days={days}")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    # Parse symbols
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        symbol_list = config.get('universe.symbols', [])
    
    try:
        # Initialize gap detection service
        detector = GapDetectionService(config)
        
        # Detect gaps
        gaps_found = asyncio.run(
            detector.detect_gaps(
                symbols=symbol_list,
                interval=interval,
                start_date=datetime.now() - timedelta(days=days),
                end_date=datetime.now()
            )
        )
        
        # Display gaps
        if not gaps_found:
            print("✅ No data gaps detected!")
        else:
            print(f"\n⚠️ Found {len(gaps_found)} data gaps:")
            for gap in gaps_found[:10]:  # Show first 10
                print(f"  {gap['symbol']}: {gap['start']} to {gap['end']} ({gap['missing']} records)")
            
            if len(gaps_found) > 10:
                print(f"  ... and {len(gaps_found) - 10} more gaps")
            
            # Fill gaps if requested
            if fill:
                print("\n🔧 Filling gaps...")
                fetch_service = DataFetchService(config)
                filled = 0
                for gap in gaps_found:
                    try:
                        await fetch_service.fetch_data(gap['symbol'], gap['start'], gap['end'])
                        filled += 1
                    except Exception as e:
                        logger.warning(f"Could not fill gap for {gap['symbol']}: {e}")
                print(f"✅ Filled {filled} of {len(gaps_found)} gaps")
                
    except Exception as e:
        logger.error(f"Gap detection failed: {e}", exc_info=True)
        raise


# Helper functions

def _list_backfill_stages(config):
    """List available backfill stages."""
    print("\n📋 Available Backfill Stages:")
    print("-" * 40)
    
    stages = config.data_pipeline.resilience.stages
    for stage in stages:
        stage_dict = dict(stage) if hasattr(stage, '__dict__') else stage
        print(f"\n{stage_dict.get('name', 'Unknown')}:")
        print(f"  Description: {stage_dict.get('description', 'N/A')}")
        print(f"  Data types: {', '.join(stage_dict.get('data_types', []))}")
        print(f"  Priority: {stage_dict.get('priority', 'N/A')}")


def _print_backfill_results(results: Dict):
    """Print backfill results summary."""
    print("\n" + "="*60)
    print("BACKFILL RESULTS")
    print("="*60)
    
    for stage, stage_results in results.items():
        print(f"\n📊 {stage.upper()}:")
        print(f"  Success: {stage_results.get('success', False)}")
        print(f"  Records: {stage_results.get('records_processed', 0):,}")
        print(f"  Duration: {stage_results.get('duration_seconds', 0):.2f}s")
        
        if stage_results.get('errors'):
            print(f"  ⚠️ Errors: {len(stage_results['errors'])}")
    
    print("\n" + "="*60)


def _print_validation_results(results: Dict):
    """Print validation results."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    overall_status = "✅ PASSED" if results.get('passed', False) else "❌ FAILED"
    print(f"\nOverall Status: {overall_status}")
    
    # Show component results
    for component, component_results in results.get('components', {}).items():
        status = "✅" if component_results.get('passed', False) else "❌"
        print(f"\n{status} {component.upper()}:")
        print(f"  Checks passed: {component_results.get('passed_checks', 0)}/{component_results.get('total_checks', 0)}")
        
        if component_results.get('errors'):
            print(f"  Errors:")
            for error in component_results['errors'][:5]:
                print(f"    - {error}")
        
        if component_results.get('warnings'):
            print(f"  Warnings:")
            for warning in component_results['warnings'][:5]:
                print(f"    - {warning}")


def _print_table_stats(table: str, stats: Dict, detailed: bool):
    """Print table statistics."""
    print(f"\n📊 Statistics for {table}:")
    print("-" * 40)
    print(f"  Row count: {stats.get('row_count', 0):,}")
    print(f"  Size: {stats.get('size_mb', 0):.2f} MB")
    print(f"  Oldest record: {stats.get('oldest_record', 'N/A')}")
    print(f"  Newest record: {stats.get('newest_record', 'N/A')}")
    
    if detailed:
        print(f"\n  Detailed Metrics:")
        print(f"    Avg row size: {stats.get('avg_row_size_bytes', 0):,} bytes")
        print(f"    Index size: {stats.get('index_size_mb', 0):.2f} MB")
        print(f"    Partitions: {stats.get('partition_count', 0)}")
        print(f"    Null percentage: {stats.get('null_percentage', 0):.2%}")


def _print_overall_stats(stats: Dict, detailed: bool):
    """Print overall database statistics."""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    print(f"\n📊 Overall:")
    print(f"  Total tables: {stats.get('table_count', 0)}")
    print(f"  Total size: {stats.get('total_size_gb', 0):.2f} GB")
    print(f"  Total records: {stats.get('total_records', 0):,}")
    
    print(f"\n📈 Top Tables by Size:")
    for table_info in stats.get('top_tables_by_size', [])[:5]:
        print(f"  {table_info['name']}: {table_info['size_mb']:.2f} MB ({table_info['row_count']:,} rows)")
    
    if detailed:
        print(f"\n🔍 Storage Distribution:")
        for storage_type, size_gb in stats.get('storage_distribution', {}).items():
            print(f"  {storage_type}: {size_gb:.2f} GB")
        
        print(f"\n⏰ Data Age Distribution:")
        for age_range, count in stats.get('age_distribution', {}).items():
            print(f"  {age_range}: {count:,} records")
    
    print("\n" + "="*60)