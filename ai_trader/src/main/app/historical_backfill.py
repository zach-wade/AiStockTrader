# File: app/historical_backfill.py
"""
Factory and orchestrator for historical backfill operations using HistoricalManager.
This replaces the duplicated BackfillOrchestrator with the proper historical data pipeline.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

from main.config.config_manager import get_config
# HistoricalManager was refactored away - use ETLService and DataFetchService
# from main.data_pipeline.historical.manager import HistoricalManager
from main.data_pipeline.historical.etl_service import ETLService
from main.data_pipeline.historical.data_fetcher import DataFetcher
from main.data_pipeline.types import BackfillParams, DataSource, TimeInterval, DataType
from main.data_pipeline.storage.database_factory import DatabaseFactory
# from main.data_pipeline.processing.transformer import DataTransformer  # REMOVED - transformations moved to ETL
from main.data_pipeline.storage.archive_initializer import get_archive
from main.data_pipeline.storage.repositories.market_data_repository import MarketDataRepository
from main.data_pipeline.storage.repositories.company_repository import CompanyRepository
from main.data_pipeline.storage.repositories.repository_factory import get_repository_factory
from main.data_pipeline.storage.dual_storage_startup import initialize_dual_storage, start_dual_storage_consumer, stop_dual_storage
from main.data_pipeline.storage.data_lifecycle_manager import DataLifecycleManager
from main.utils.resilience import ErrorRecoveryManager, NETWORK_RETRY_CONFIG
from main.utils.app.cli import info_message, success_message, warning_message, error_message
from main.utils.core import get_logger, timer, RateLimiter
from main.utils.database import DatabasePool
from main.universe.universe_manager import UniverseManager

logger = get_logger(__name__)


def filter_valid_symbols(symbols: List[str]) -> List[str]:
    """
    Filter out known problematic symbols.
    
    Args:
        symbols: List of symbols to filter
        
    Returns:
        List of valid symbols
    """
    valid_symbols = []
    excluded_patterns = [
        # Single letters that are often invalid
        'A', 'C', 'E', 'F', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z',
        # Known problematic symbols
        'AAA', 'AAAA', 'AAAU',
        # Patterns that indicate special securities
        lambda s: s.endswith(('.PRA', '.PRB', '.PRC', '.PRD', '.PRE', '.PRF', '.PRG', '.PRH', '.PRI', '.PRJ')),
        lambda s: s.endswith(('.WS', '.WT', '.U', '.R', '.W')),  # Warrants, rights, units
        lambda s: len(s) > 5,  # Unusually long symbols
    ]
    
    for symbol in symbols:
        # Skip if matches excluded patterns
        skip = False
        for pattern in excluded_patterns:
            if callable(pattern):
                if pattern(symbol):
                    skip = True
                    break
            elif symbol == pattern:
                skip = True
                break
        
        if not skip:
            valid_symbols.append(symbol)
    
    return valid_symbols


async def create_historical_manager(config) -> HistoricalManager:
    """
    Factory function to create HistoricalManager with all required dependencies.
    
    Args:
        config: Application configuration
        
    Returns:
        Fully initialized HistoricalManager instance
    """
    info_message("Initializing historical data pipeline", "Setting up database and data sources...")
    
    # Initialize database adapter and database pool
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    # Create database pool for components that need it
    db_pool = DatabasePool()
    # Initialize the pool if needed (it may already be initialized by the context)
    if not db_pool._engine:
        db_pool.initialize()
    
    # Initialize data source clients (reuse existing context patterns)
    from main.utils.app.context import StandardAppContext
    context = StandardAppContext("historical_backfill", config)
    await context.initialize(['database', 'data_sources'])
    clients = context.data_source_manager.clients
    
    # Initialize dual storage components
    enable_dual_storage = config.get('data_pipeline', {}).get('storage', {}).get('enable_dual_storage', True)
    event_bus, cold_storage = initialize_dual_storage(
        hot_storage=db_adapter,
        enable_dual_storage=enable_dual_storage
    )
    
    # Create repository factory with dual storage support
    # Pass is_backfill=True to disable cold writes during backfill
    repo_factory = get_repository_factory(
        db_adapter=db_adapter, 
        cold_storage=cold_storage, 
        event_bus=event_bus,
        config=config,
        is_backfill=True
    )
    
    # Initialize required dependencies for DataFetcher
    resilience_manager = ErrorRecoveryManager(NETWORK_RETRY_CONFIG)
    # transformer = DataTransformer(config)  # REMOVED - transformations moved to ETL phase
    archive = get_archive()
    
    # Create market data repository with dual storage
    market_data_repo = repo_factory.create_repository('market_data')
    
    # Create BulkDataLoader if enabled for backfill
    bulk_loader = None
    bulk_config = config.get('storage', {}).get('bulk_loading', {})
    if bulk_config.get('enabled', True):
        from main.data_pipeline.storage.bulk_data_loader import BulkDataLoader
        from main.data_pipeline.storage.bulk_loaders.base import BulkLoadConfig
        
        bulk_load_config = BulkLoadConfig(
            accumulation_size=bulk_config.get('accumulation_size', 10000),
            use_copy_command=bulk_config.get('use_copy_command', True),
            disable_indexes=bulk_config.get('disable_indexes', False),
            batch_timeout_seconds=bulk_config.get('batch_timeout_seconds', 30.0),
            max_memory_mb=bulk_config.get('max_memory_mb', 500),
            parallel_archives=bulk_config.get('parallel_archives', 3)
        )
        
        bulk_loader = BulkDataLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=bulk_load_config
        )
        
        info_message("Bulk loader enabled", f"Using optimized bulk loading with accumulation size: {bulk_load_config.accumulation_size}")
    
    # Create DataFetcher (transformations now happen in ETL phase, not ingestion)
    data_fetcher = DataFetcher(
        config=config,
        resilience=resilience_manager,
        # standardizer=transformer,  # REMOVED - transformations moved to ETL phase
        archive=archive,
        market_data_repo=market_data_repo,
        bulk_loader=bulk_loader
    )
    
    # Create additional required dependencies
    calls_per_minute = config.get('data_pipeline.rate_limit.calls_per_minute', 60)
    rate_limiter = RateLimiter(
        rate=calls_per_minute,
        per=60.0  # per 60 seconds (1 minute)
    )
    from main.data_pipeline.storage.repositories import get_repository_factory
    repo_factory = get_repository_factory()
    company_repo = repo_factory.create_company_repository(db_adapter)
    
    # Create ProcessingManager for ETL operations (specifically corporate actions)
    from main.data_pipeline.processing.manager import ProcessingManager
    processing_manager = ProcessingManager(
        config=config,
        db_adapter=db_adapter
    )
    
    # Create HistoricalManager with all dependencies
    historical_manager = HistoricalManager(
        config=config,
        db_adapter=db_adapter,
        clients=clients,
        data_fetcher=data_fetcher,
        db_pool=db_pool,
        rate_limiter=rate_limiter,
        company_repo=company_repo,
        processing_manager=processing_manager
    )
    
    info_message("Historical pipeline initialized", f"Ready with {len(clients)} data sources")
    
    # Attach bulk_loader to historical_manager for later use
    historical_manager.bulk_loader = bulk_loader
    
    # Attach context to historical_manager for cleanup later
    historical_manager._app_context = context
    
    return historical_manager


def convert_cli_params_to_backfill_params(
    stages: List[str],
    symbols: Optional[List[str]] = None,
    lookback_days: int = 30,
    config: Optional[Dict] = None,
    interval_overrides: Optional[List[str]] = None
) -> BackfillParams:
    """
    Convert CLI parameters to BackfillParams structure.
    
    Args:
        stages: List of stages ('realtime', 'daily', 'news', 'yahoo_financial', 'all')
        symbols: Optional list of symbols
        lookback_days: Number of days to backfill
        config: Configuration dictionary
        interval_overrides: Optional list of specific intervals to use
        
    Returns:
        BackfillParams instance
    """
    if not config:
        config = get_config()
    
    # Get backfill stages configuration
    backfill_stages = config.get('data_pipeline', {}).get('resilience', {}).get('stages', [])
    logger.info(f"Backfill stages from config: {[s.get('name') for s in backfill_stages]}")
    logger.info(f"Requested stages: {stages}")
    
    # Initialize collections
    data_types = []
    sources = []
    intervals = []
    
    # If interval overrides provided, use those
    if interval_overrides:
        for interval_str in interval_overrides:
            try:
                # Map config interval names to enum values
                interval_mapping = {
                    '1minute': '1min',
                    '5minute': '5min',
                    '15minute': '15min',
                    '30minute': '30min',
                    '1hour': '1hour',
                    '1day': '1day',
                    '1week': '1week',
                    '1month': '1month'
                }
                mapped_interval = interval_mapping.get(interval_str, interval_str)
                interval_enum = TimeInterval(mapped_interval)
                if interval_enum not in intervals:
                    intervals.append(interval_enum)
            except ValueError:
                logger.warning(f"Invalid interval: {interval_str}")
    else:
        # Process stages to get intervals from config
        for stage_name in stages:
            if stage_name == 'all':
                # Collect all intervals from all stages
                for stage_config in backfill_stages:
                    stage_intervals = stage_config.get('intervals', [])
                    for interval_str in stage_intervals:
                        try:
                            # Map config interval names to enum values
                            interval_mapping = {
                                '1minute': '1min',
                                '5minute': '5min',
                                '15minute': '15min',
                                '30minute': '30min',
                                '1hour': '1hour',
                                '1day': '1day',
                                '1week': '1week',
                                '1month': '1month'
                            }
                            mapped_interval = interval_mapping.get(interval_str, interval_str)
                            interval_enum = TimeInterval(mapped_interval)
                            if interval_enum not in intervals:
                                intervals.append(interval_enum)
                        except ValueError:
                            logger.warning(f"Invalid interval in config: {interval_str}")
                # Also set data types for 'all'
                data_types = [
                    DataType.MARKET_DATA,
                    DataType.NEWS,
                    DataType.CORPORATE_ACTIONS,
                    DataType.FINANCIALS
                ]
                sources = [DataSource.POLYGON]  # Primary source
            else:
                # Find specific stage configuration
                stage_config = next((s for s in backfill_stages if s['name'] == stage_name), None)
                logger.info(f"Looking for stage '{stage_name}', found: {stage_config is not None}")
                if stage_config:
                    # Get intervals from stage config
                    stage_intervals = stage_config.get('intervals', [])
                    for interval_str in stage_intervals:
                        try:
                            # Map config interval names to enum values
                            interval_mapping = {
                                '1minute': '1min',
                                '5minute': '5min',
                                '15minute': '15min',
                                '30minute': '30min',
                                '1hour': '1hour',
                                '1day': '1day',
                                '1week': '1week',
                                '1month': '1month'
                            }
                            mapped_interval = interval_mapping.get(interval_str, interval_str)
                            interval_enum = TimeInterval(mapped_interval)
                            if interval_enum not in intervals:
                                intervals.append(interval_enum)
                        except ValueError:
                            logger.warning(f"Invalid interval in config: {interval_str}")
                    
                    # Get sources from stage config
                    stage_sources = stage_config.get('sources', [])
                    for source_str in stage_sources:
                        try:
                            # Map source strings to DataSource enum
                            source_mapping = {
                                'polygon': DataSource.POLYGON,
                                'alpaca': DataSource.ALPACA,
                                'yahoo': DataSource.YAHOO,
                                'benzinga': DataSource.BENZINGA,
                                'reddit': DataSource.REDDIT,
                                'polygon_market': DataSource.POLYGON,
                                'polygon_news': DataSource.POLYGON,
                                'alpaca_news': DataSource.ALPACA,
                                'alpaca_alt': DataSource.ALPACA,
                                'alpaca_corporate_actions': DataSource.ALPACA
                            }
                            source_enum = source_mapping.get(source_str, DataSource.POLYGON)
                            if source_enum not in sources:
                                sources.append(source_enum)
                        except ValueError:
                            logger.warning(f"Invalid source in config: {source_str}")
                    
                    # Get data type from stage config
                    stage_data_type = stage_config.get('data_type')
                    if stage_data_type:
                        try:
                            data_type_mapping = {
                                'market_data': DataType.MARKET_DATA,
                                'news': DataType.NEWS,
                                'corporate_actions': DataType.CORPORATE_ACTIONS,
                                'fundamentals': DataType.FINANCIALS,
                                'social_sentiment': DataType.SOCIAL_SENTIMENT,  # Use correct mapping
                                'options': DataType.OPTIONS,
                                'options_contracts': DataType.OPTIONS   # Also support this variant
                            }
                            data_type_enum = data_type_mapping.get(stage_data_type, DataType.MARKET_DATA)
                            if data_type_enum not in data_types:
                                data_types.append(data_type_enum)
                        except ValueError:
                            logger.warning(f"Invalid data type in config: {stage_data_type}")
                
                # Stage not found in config
                else:
                    logger.warning(f"Stage '{stage_name}' not found in configuration file")
                    # Skip this stage instead of using legacy mapping
    
    # Default values if nothing specified
    if not data_types:
        logger.warning("No data types found from stages, using all data types")
        data_types = [DataType.MARKET_DATA, DataType.NEWS, DataType.CORPORATE_ACTIONS, DataType.FINANCIALS]
    if not intervals:
        intervals = [TimeInterval.DAY_1]
    
    # Smart source selection based on data types
    # Instead of using ALL accumulated sources, select appropriate sources per data type
    if not sources or len(sources) > 3:  # If no sources or too many accumulated
        sources_by_type = {
            DataType.MARKET_DATA: [DataSource.POLYGON],
            DataType.NEWS: [DataSource.POLYGON],
            DataType.CORPORATE_ACTIONS: [DataSource.POLYGON],
            DataType.FINANCIALS: [DataSource.POLYGON],
            DataType.SOCIAL_SENTIMENT: [DataSource.REDDIT]
        }
        
        # Build source list based on requested data types
        smart_sources = []
        for dt in data_types:
            type_sources = sources_by_type.get(dt, [DataSource.POLYGON])
            for src in type_sources:
                if src not in smart_sources:
                    smart_sources.append(src)
        
        logger.info(f"Using smart source selection: {[s.value for s in smart_sources]} for data types: {[dt.value for dt in data_types]}")
        sources = smart_sources
    
    # Remove duplicates while preserving order
    sources = list(dict.fromkeys(sources))
    intervals = list(dict.fromkeys(intervals))
    data_types = list(dict.fromkeys(data_types))
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    
    # Log what we're using
    logger.info(f"Backfill parameters - Intervals: {[i.value for i in intervals]}, "
                f"Sources: {[s.value for s in sources]}, "
                f"Data types: {[d.value for d in data_types]}")
    
    return BackfillParams(
        symbols=symbols or [],  # Will be filled with universe symbols if empty
        sources=sources,
        intervals=intervals,
        start_date=start_date,
        end_date=end_date,
        data_types=data_types,
        max_concurrent=5,  # Conservative for stability
        retry_failed=True,
        user_requested_days=lookback_days  # Pass the user's requested days
    )


async def run_historical_backfill(backfill_config: dict) -> Dict[str, Any]:
    """
    Main entry point for historical backfill operations.
    
    Args:
        backfill_config: Dictionary with backfill parameters from CLI
        
    Returns:
        Dictionary with results summary
    """
    with timer("historical_backfill") as backfill_timer:
        try:
            info_message("Starting Historical Data Backfill")
            
            # Extract parameters
            stages = backfill_config.get('stages', ['all'])
            symbols = backfill_config.get('symbols')
            lookback_days = backfill_config.get('lookback_days', 30)
            test_mode = backfill_config.get('test_mode', False)
            limit = backfill_config.get('limit')
            
            info_message(f"Backfill configuration", f"Stages: {stages}, Days: {lookback_days}")
            
            # Load universe symbols if none provided
            if not symbols:
                warning_message(
                    "Using universe symbols for backfill",
                    "Loading symbols from universe database..."
                )
                universe_manager = UniverseManager(get_config())
                symbols = await universe_manager.get_universe_for_backfill()
                symbol_source = "universe"
                
                if not symbols:
                    warning_message("No universe symbols found", "Consider running universe population first")
                    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Fallback
                    symbol_source = "fallback"
                else:
                    # Filter out problematic symbols
                    original_count = len(symbols)
                    symbols = filter_valid_symbols(symbols)
                    if original_count != len(symbols):
                        info_message(f"Filtered symbols", f"Using {len(symbols)} valid symbols (excluded {original_count - len(symbols)} invalid)")
                    else:
                        info_message(f"Loaded {len(symbols)} symbols from universe")
            
            # Handle layer-specific symbol loading
            elif len(symbols) == 1 and symbols[0].startswith('layer'):
                layer_str = symbols[0]
                try:
                    layer_num = int(layer_str.replace('layer', ''))
                    info_message(f"Loading Layer {layer_num} symbols", "Fetching from scanner qualifications...")
                    
                    universe_manager = UniverseManager(get_config())
                    symbols = await universe_manager.get_qualified_symbols(layer_num)
                    
                    # Track that this is a layer-specific backfill for tier filtering decision
                    symbol_source = f"layer{layer_num}"
                    
                    if symbols:
                        # Filter out problematic symbols
                        original_count = len(symbols)
                        symbols = filter_valid_symbols(symbols)
                        info_message(f"Loaded Layer {layer_num} symbols", 
                                   f"Using {len(symbols)} valid symbols" + 
                                   (f" (excluded {original_count - len(symbols)} invalid)" if original_count != len(symbols) else ""))
                    else:
                        warning_message(f"No symbols found for Layer {layer_num}", 
                                      "Check if scanner pipeline has been run")
                        return {'status': 'failed', 'error': f'No symbols found for Layer {layer_num}'}
                        
                except ValueError:
                    error_message("Invalid layer specification", 
                                f"Use 'layer0', 'layer1', 'layer2', or 'layer3'")
                    return {'status': 'failed', 'error': 'Invalid layer specification'}
            else:
                # Regular symbol list, no special handling
                symbol_source = "custom"
            
            # Get structured configuration for type safety
            structured_config = get_structured_config()
            
            # Convert to BackfillParams (still use old config for compatibility)
            old_config = get_config()
            params = convert_cli_params_to_backfill_params(
                stages=stages, 
                symbols=symbols, 
                lookback_days=lookback_days,
                config=old_config
            )
            info_message(f"Data types to process: {[dt.value for dt in params.data_types]}")
            
            # Create HistoricalManager (still use old config for compatibility)
            historical_manager = await create_historical_manager(old_config)
            
            # Apply test mode or limit if specified
            if test_mode:
                test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'VTI']
                info_message("Test mode enabled", f"Using {len(test_symbols)} known good symbols")
                params.symbols = test_symbols
                symbols = test_symbols
            elif limit and len(symbols) > limit:
                info_message(f"Limiting symbols", f"Processing first {limit} of {len(symbols)} symbols")
                params.symbols = symbols[:limit]
                symbols = symbols[:limit]
            
            # Use orchestrator for full universe backfill
            if structured_config.backfill_orchestration.use_symbol_tiers and len(symbols) > 100:
                info_message("Using backfill orchestrator", f"Processing {len(symbols)} symbols with intelligent tiering")
                
                # Create orchestrator - use regular version with user days override
                from main.data_pipeline.backfill.orchestrator import (
                    BackfillOrchestrator, 
                    BackfillConfig, 
                    BackfillStage
                )
                from main.data_pipeline.storage.bulk_loaders.base import BulkLoadConfig
                
                # Create BulkLoadConfig from structured config (no manual mapping!)
                bulk_load_config = BulkLoadConfig(
                    accumulation_size=structured_config.storage.bulk_loading.accumulation_size,
                    use_copy_command=structured_config.storage.bulk_loading.use_copy_command,
                    disable_indexes=structured_config.storage.bulk_loading.disable_indexes,
                    batch_timeout_seconds=structured_config.storage.bulk_loading.batch_timeout_seconds,
                    max_memory_mb=structured_config.storage.bulk_loading.max_memory_mb,
                    parallel_archives=structured_config.storage.bulk_loading.parallel_archives
                )
                
                # Map stage names to BackfillStage enum values
                stage_mapping = {
                    'long_term': BackfillStage.MARKET_DATA,
                    'news_data': BackfillStage.NEWS,
                    'corporate_actions': BackfillStage.CORPORATE_ACTIONS,
                    'social_sentiment': BackfillStage.SOCIAL_SENTIMENT,
                    'options_data': BackfillStage.OPTIONS,
                    'scanner_daily': BackfillStage.MARKET_DATA,
                    'scanner_intraday': BackfillStage.MARKET_DATA,
                    # Also support direct enum names
                    'market_data': BackfillStage.MARKET_DATA,
                    'news': BackfillStage.NEWS,
                    'fundamentals': BackfillStage.FUNDAMENTALS,
                    'social': BackfillStage.SOCIAL_SENTIMENT,
                    'options': BackfillStage.OPTIONS
                }
                
                # Convert stage names from config to BackfillStage enums
                backfill_stages = []
                configured_stages = stages  # Use the stages from CLI/config
                logger.info(f"Configured stages: {configured_stages}")
                
                for stage_name in configured_stages:
                    if stage_name in stage_mapping:
                        stage_enum = stage_mapping[stage_name]
                        if stage_enum not in backfill_stages:
                            backfill_stages.append(stage_enum)
                            logger.info(f"Mapped stage '{stage_name}' to {stage_enum.value}")
                    else:
                        logger.warning(f"Unknown stage '{stage_name}', skipping")
                
                # If no valid stages, use defaults
                if not backfill_stages:
                    backfill_stages = [BackfillStage.MARKET_DATA, BackfillStage.NEWS, BackfillStage.CORPORATE_ACTIONS]
                    logger.info(f"No valid stages found, using defaults: {[s.value for s in backfill_stages]}")
                
                # Create BackfillConfig for regular orchestrator
                # For Layer 1 backfill, disable symbol tiers to process ALL 2,003 symbols
                use_tiers = structured_config.backfill_orchestration.use_symbol_tiers
                logger.info(f"Symbol source detected: {symbol_source if 'symbol_source' in locals() else 'unknown'}")
                if 'symbol_source' in locals() and symbol_source == 'layer1':
                    use_tiers = False
                    logger.info("🎯 Disabling symbol tier filtering for Layer 1 backfill to process all qualified symbols")
                else:
                    logger.info(f"Using default tier filtering: {use_tiers}")
                
                backfill_config = BackfillConfig(
                    use_symbol_tiers=use_tiers,
                    respect_user_days_override=True,  # Allow --days to override tier lookback
                    max_parallel_symbols_per_tier=structured_config.backfill_orchestration.max_parallel_symbols_per_tier,
                    stages=backfill_stages,  # Use our mapped stages
                    save_progress=structured_config.backfill_orchestration.save_progress,
                    progress_file=structured_config.backfill_orchestration.progress_file,
                    bulk_load_config=bulk_load_config
                )
                
                orchestrator = BackfillOrchestrator(
                    config=old_config,  # Use the original config dict
                    historical_manager=historical_manager,
                    db_adapter=historical_manager.db_adapter,
                    backfill_config=backfill_config
                )
                
                # Run orchestrated backfill
                logger.info(f"Starting orchestrator with data_types={[dt.value for dt in params.data_types] if params.data_types else 'All'}")
                logger.info(f"Backfill stages: {[s.value for s in backfill_stages]}")
                
                orch_result = await orchestrator.orchestrate_backfill(
                    symbols=params.symbols,
                    start_date=params.start_date,
                    end_date=params.end_date,
                    intervals=params.intervals,
                    sources=params.sources,  # Fixed parameter name
                    user_requested_days=lookback_days  # Pass user's --days parameter
                )
                
                # Clean up orchestrator
                await orchestrator.cleanup()
                
                logger.info(f"Orchestrator completed: {orch_result.symbols_completed} symbols, {orch_result.records_loaded} records")
                
                # Convert orchestrator result to expected format
                results = {
                    'total_symbols': orch_result.total_symbols,
                    'symbols_processed': orch_result.symbols_completed,
                    'symbols_failed': orch_result.symbols_failed,
                    'records_downloaded': orch_result.records_loaded,
                    'data_types_processed': [s for s in orch_result.stages_completed],
                    'market_data_records': orch_result.stage_results.get('market_data', {}).get('records_loaded', 0),
                    'news_records': orch_result.stage_results.get('news', {}).get('records_loaded', 0),
                    'corporate_actions_records': orch_result.stage_results.get('corporate_actions', {}).get('records_loaded', 0),
                    'fundamentals_records': orch_result.stage_results.get('fundamentals', {}).get('records_loaded', 0),
                    'financial_records': orch_result.stage_results.get('fundamentals', {}).get('records_loaded', 0)  # Keep for backward compatibility
                }
                
            else:
                # Use traditional approach for smaller symbol sets
                info_message(
                    f"Processing {len(params.symbols)} symbols",
                    f"Sources: {[s.value for s in params.sources]}, Intervals: {[i.value for i in params.intervals]}"
                )
                
                results = await historical_manager.backfill_symbols(params)
            
            # Flush bulk loader if used
            if hasattr(historical_manager, 'bulk_loader') and historical_manager.bulk_loader:
                info_message("Flushing bulk loader", "Writing remaining buffered data...")
                flush_result = await historical_manager.bulk_loader.flush_all()
                if flush_result.success:
                    if flush_result.records_loaded > 0:
                        success_message(
                            "Bulk loader flush completed",
                            f"Loaded {flush_result.records_loaded} records for {len(flush_result.symbols_processed)} symbols"
                        )
                    metrics = historical_manager.bulk_loader.get_metrics()
                    info_message(
                        "Bulk loader metrics",
                        f"Total records: {metrics['total_records_loaded']}, "
                        f"Load time: {metrics['total_load_time']:.1f}s, "
                        f"Archive time: {metrics['total_archive_time']:.1f}s"
                    )
                else:
                    warning_message("Bulk loader flush had errors", "; ".join(flush_result.errors))
            
            # Run data archival if configured
            if old_config.get('storage', {}).get('lifecycle', {}).get('archive_on_backfill', True):
                info_message("Running data archival cycle", "Moving old data to cold storage...")
                try:
                    # Create DataLifecycleManager
                    lifecycle_manager = DataLifecycleManager(
                        config=old_config,
                        db_adapter=historical_manager.db_adapter,
                        archive=get_archive()
                    )
                    
                    # Run archival cycle
                    archival_results = await lifecycle_manager.run_archival_cycle(dry_run=False)
                    
                    if archival_results.get('status') == 'success':
                        records_archived = archival_results.get('records_archived', 0)
                        if records_archived > 0:
                            success_message(
                                "Data archival completed",
                                f"Archived {records_archived} records to cold storage"
                            )
                    elif archival_results.get('status') == 'noop':
                        info_message("Data archival", "No data eligible for archiving")
                    else:
                        warning_message(
                            "Data archival issue",
                            archival_results.get('message', 'Unknown error')
                        )
                except Exception as e:
                    error_message("Data archival failed", str(e))
                    logger.error(f"Failed to run data archival: {e}", exc_info=True)
            
            # Clean up resources
            await historical_manager.close()
            
            # Stop dual storage consumer
            await stop_dual_storage()
            
            # Success
            duration = backfill_timer.elapsed
            
            # Display detailed results
            info_message("Backfill Results Summary:")
            info_message(f"  Total symbols: {results.get('total_symbols', 0)}")
            info_message(f"  Symbols processed: {results.get('symbols_processed', 0)}")
            info_message(f"  Symbols failed: {results.get('symbols_failed', 0)}")
            info_message(f"  Total records downloaded: {results.get('records_downloaded', 0)}")
            
            # Show breakdown by data type (show all processed types even if 0 records)
            data_types_processed = results.get('data_types_processed', [])
            
            # Always show market data if it was processed
            if 'market_data' in data_types_processed or results.get('market_data_records', 0) > 0:
                info_message(f"  Market data: {results.get('market_data_records', 0)} records")
            
            # Always show news if it was processed
            if 'news' in data_types_processed or results.get('news_records', 0) > 0:
                info_message(f"  News: {results.get('news_records', 0)} records")
            
            # Always show corporate actions if it was processed
            if 'corporate_actions' in data_types_processed or results.get('corporate_actions_records', 0) > 0:
                info_message(f"  Corporate actions: {results.get('corporate_actions_records', 0)} records")
            
            # Show financials (renamed from fundamentals)
            if 'fundamentals' in data_types_processed or results.get('fundamentals_records', 0) > 0:
                info_message(f"  Financials: {results.get('fundamentals_records', 0)} records")
            
            info_message(f"  Data types processed: {results.get('data_types_processed', [])}")
            info_message(f"  Duration: {duration:.2f}s")
            
            if results.get('symbols_processed', 0) > 0:
                success_message(
                    "Historical backfill completed successfully",
                    f"Downloaded {results.get('records_downloaded', 0)} records for {results.get('symbols_processed', 0)} symbols"
                )
            else:
                warning_message(
                    "Historical backfill completed with no data",
                    "No records were downloaded - check logs for details"
                )
            
            return {
                'status': 'success',
                'symbols_processed': results.get('symbols_processed', 0),
                'records_downloaded': results.get('records_downloaded', 0),
                'duration_seconds': duration,
                'sources_used': [s.value for s in params.sources],
                'data_types_processed': results.get('data_types_processed', []),
                'detailed_results': results
            }
            
        except Exception as e:
            error_message("Historical backfill failed", str(e))
            logger.error(f"Historical backfill error: {e}", exc_info=True)
            
            # Try to clean up resources even on error
            try:
                if 'historical_manager' in locals():
                    await historical_manager.close()
                await stop_dual_storage()
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
                
            raise