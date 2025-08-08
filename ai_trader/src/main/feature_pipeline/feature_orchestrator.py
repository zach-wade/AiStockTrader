"""
Feature Orchestrator

Coordinates feature calculation across different data sources and time periods.
Integrates with scanner alerts to trigger feature updates when needed.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import hashlib
import json
import time

from main.config.config_manager import get_config
# HistoricalManager was refactored - using ETLService instead
from main.data_pipeline.historical.etl_service import ETLService
from main.data_pipeline.historical.data_fetch_service import DataFetchService
from main.interfaces.database import IAsyncDatabase
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.archive_initializer import get_archive
from main.data_pipeline.storage.repositories import get_repository_factory
from main.interfaces.repositories import IMarketDataRepository
from main.utils.resilience import ErrorRecoveryManager
from main.feature_pipeline.feature_store import FeatureStoreV2
from main.feature_pipeline.feature_store import FeatureStoreV2 as FeatureStoreRepository
from main.interfaces.events import IEventBus, EventType
from main.events.types import ScannerAlertEvent, FeatureRequestEvent
from main.data_pipeline.validation.core.validation_pipeline import ValidationPipeline, ValidationStage
from main.utils.data import (
    DataFrameStreamer, StreamingConfig, stream_process_dataframe, 
    streaming_context, optimize_feature_calculation
)
from main.utils.monitoring import get_memory_monitor, memory_profiled
from main.utils.cache import CacheType, MemoryBackend

logger = logging.getLogger(__name__)


# FeatureCacheAdapter removed - using MarketDataCache directly


class FeatureOrchestrator:
    """
    Orchestrates feature calculation and storage.
    
    Responsibilities:
    - Calculate features from market data
    - Respond to scanner alerts by updating features
    - Store calculated features in appropriate storage tier
    - Cache frequently accessed features
    """
    
    def __init__(self, config=None, event_bus: Optional[IEventBus] = None):
        """
        Initialize the feature orchestrator.
        
        Args:
            config: Configuration object (uses global config if not provided)
            event_bus: Optional event bus instance (for dependency injection)
        """
        self.config = config or get_config()
        # V3 Architecture: Use HistoricalManager for Data Lake access
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(self.config)
        
        # Initialize V3 dependencies for HistoricalManager
        self.data_archive = get_archive()
        repo_factory = get_repository_factory()
        self.market_data_repo = repo_factory.create_market_data_repository(self.db_adapter)
        self.feature_store_repo = FeatureStoreRepository(self.db_adapter)
        self.resilience = ResilienceStrategies(self.config)
        
        # Initialize FeatureStoreV2 for versioned HDF5 storage
        self.feature_store = FeatureStoreV2(config=self.config)
        
        # Create DataFetchService with required dependencies
        from main.data_pipeline.processing.standardizer import DataStandardizer
        standardizer = DataStandardizer(self.config)
        
        # DataFetchService will be initialized when clients are set
        self.data_fetch_service = None
        self.etl_service = None
        
        # Initialize services with proper dependencies
        # Note: clients will be injected from main orchestrator when needed
        
        # Use injected event bus or create one as fallback
        if event_bus:
            self.event_bus = event_bus
        else:
            # Lazy import to avoid circular dependency
            from main.events.core import EventBusFactory
            self.event_bus = EventBusFactory.create_test_instance()
        
        # Subscribe to scanner alerts and feature requests
        self.event_bus.subscribe(EventType.SCANNER_ALERT, self._on_scanner_alert)
        self.event_bus.subscribe(EventType.FEATURE_REQUEST, self._on_feature_request)
        
        # Initialize feature calculators
        self._init_calculators()
        
        # Initialize caching and parallelization
        self.cache_ttl = self.config.get('orchestrator.features.cache.ttl_seconds', 3600)
        self.cache = get_global_cache()
        
        # Thread pool for CPU-intensive feature calculations
        max_workers = self.config.get('orchestrator.features.parallel_processing.max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Progress tracking
        self.calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_tasks': 0,
            'errors': 0,
            'last_calculation_time': None,
            'scanner_triggered_calculations': 0,
            'feature_requests_processed': 0
        }
        
        # Alert-to-feature mapping
        self._alert_feature_mapping = self._init_alert_feature_mapping()
        
        # Batch processing queue for non-urgent feature requests
        self._batch_queue = asyncio.Queue()
        self._batch_processing_task = None
        
        # Initialize validation pipeline
        self.validation_pipeline = ValidationPipeline()
        
        # Initialize streaming processing capabilities
        streaming_chunk_size = self.config.get('orchestrator.features.streaming.chunk_size', 10000)
        streaming_memory_limit = self.config.get('orchestrator.features.streaming.max_memory_mb', 500.0)
        
        self.streaming_config = StreamingConfig(
            chunk_size=streaming_chunk_size,
            max_memory_mb=streaming_memory_limit,
            parallel_workers=max_workers,
            enable_gc_per_chunk=True,
            log_progress_every=5
        )
        
        # Memory monitoring
        self.memory_monitor = get_memory_monitor()
        
        # Streaming thresholds
        self.streaming_threshold_rows = self.config.get('orchestrator.features.streaming.threshold_rows', 50000)
        
        logger.info(f"FeatureOrchestrator initialized with {max_workers} parallel workers, "
                   f"{self.cache_ttl}s cache TTL, and streaming for datasets > {self.streaming_threshold_rows:,} rows")
    
    def set_data_clients(self, clients: Dict[str, Any]):
        """
        Set data source clients and initialize HistoricalManager.
        
        Args:
            clients: Dictionary of data source clients
        """
        if clients:
            # ETLService replaces HistoricalManager
            self.etl_service = ETLService(
                db_adapter=self.db_adapter,
                archive=self.data_archive,
                event_bus=None  # Optional event bus
            )
            self.data_fetch_service = DataFetchService(
                clients=clients,
                archive=self.data_archive
            )
            logger.info("ETL and DataFetch services initialized with data clients")
    
    def _init_calculators(self):
        """Initialize feature calculator instances."""
        # This would initialize actual feature calculators
        # For now, just placeholder
        self.calculators = {}
        logger.info("Feature calculators initialized")
    
    def _init_alert_feature_mapping(self) -> Dict[str, List[str]]:
        """Initialize mapping from alert types to required features."""
        return {
            # Layer 1 alerts (basic patterns)
            'high_volume': ['volume_features', 'price_features', 'volatility'],
            'price_breakout': ['price_features', 'trend_features', 'support_resistance'],
            'unusual_activity': ['volume_features', 'volatility', 'microstructure'],
            
            # Layer 2 alerts (advanced patterns)
            'momentum_shift': ['momentum_features', 'trend_features', 'volume_features'],
            'volatility_spike': ['volatility', 'option_features', 'microstructure'],
            'sentiment_surge': ['sentiment_features', 'news_features', 'social_features'],
            
            # Layer 3 alerts (composite signals)
            'catalyst_detected': ['all_features'],  # Compute all features
            'regime_change': ['regime_features', 'correlation_features', 'macro_features'],
            'opportunity_signal': ['all_features'],
            
            # Default features for unknown alert types
            'default': ['price_features', 'volume_features', 'volatility']
        }
    
    def _generate_cache_key(self, symbol: str, start_time: datetime, end_time: datetime, feature_sets: Optional[List[str]] = None) -> str:
        """Generate a cache key for feature calculation parameters."""
        key_data = {
            'symbol': symbol,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'feature_sets': sorted(feature_sets) if feature_sets else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def calculate_features(
        self, 
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        feature_sets: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate features for given symbols and time range with caching and parallelization.
        
        Args:
            symbols: List of symbols to calculate features for
            start_time: Start of time range
            end_time: End of time range
            feature_sets: Specific feature sets to calculate (None = all)
            
        Returns:
            Dictionary mapping symbol to feature DataFrame
        """
        logger.info(f"Calculating features for {len(symbols)} symbols")
        start_calc_time = time.time()
        
        # Check cache first
        cached_features = {}
        symbols_to_calculate = []
        
        for symbol in symbols:
            cache_key = self._generate_cache_key(symbol, start_time, end_time, feature_sets)
            cached_result = await self.cache.get(CacheType.FEATURES, cache_key)
            if cached_result is not None:
                cached_features[symbol] = cached_result
                self.calculation_stats['cache_hits'] += 1
                logger.debug(f"Using cached features for {symbol}")
            else:
                symbols_to_calculate.append(symbol)
                self.calculation_stats['cache_misses'] += 1
        
        # If all symbols are cached, return cached results
        if not symbols_to_calculate:
            logger.info(f"All {len(symbols)} symbols found in cache")
            return cached_features
        
        # Calculate features for uncached symbols using parallel processing
        logger.info(f"Calculating features for {len(symbols_to_calculate)} symbols ({len(cached_features)} cached)")
        
        # Create parallel tasks for feature calculation
        tasks = []
        for symbol in symbols_to_calculate:
            task = self._calculate_single_symbol_features(symbol, start_time, end_time, feature_sets)
            tasks.append(task)
        
        # Execute tasks concurrently with progress tracking
        calculated_features = await self._execute_parallel_feature_calculation(tasks, symbols_to_calculate)
        
        # Combine cached and calculated features
        all_features = {**cached_features, **calculated_features}
        
        # Update statistics
        self.calculation_stats['total_calculations'] += len(symbols_to_calculate)
        self.calculation_stats['last_calculation_time'] = datetime.now(timezone.utc)
        
        calc_duration = time.time() - start_calc_time
        logger.info(f"Feature calculation completed in {calc_duration:.2f}s - {len(all_features)} symbols total "
                   f"({len(cached_features)} cached, {len(calculated_features)} calculated)")
        
        return all_features
    
    async def _calculate_single_symbol_features(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime, 
        feature_sets: Optional[List[str]] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Calculate features for a single symbol.
        
        Args:
            symbol: Symbol to calculate features for
            start_time: Start of time range
            end_time: End of time range
            feature_sets: Specific feature sets to calculate
            
        Returns:
            Tuple of (symbol, feature_dataframe)
        """
        try:
            # Get market data for this symbol
            market_data = await self.market_data_repo.get_data_for_symbols_and_range(
                symbols=[symbol],
                start_time=start_time,
                end_time=end_time
            )
            
            if market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return symbol, pd.DataFrame()
            
            # Filter data for this symbol
            symbol_data = market_data[market_data.get('symbol', None) == symbol]
            
            if symbol_data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return symbol, pd.DataFrame()
            
            # Validate data before feature calculation
            validation_result = await self.validation_pipeline.validate_feature_ready(
                data=symbol_data,
                data_type='market_data',
                required_columns=['symbol', 'timestamp', 'close', 'volume']
            )
            
            if not validation_result.passed:
                logger.error(f"Feature validation failed for {symbol}: {validation_result.errors}")
                self.calculation_stats['errors'] += 1
                return symbol, pd.DataFrame()
            
            if validation_result.has_warnings:
                logger.warning(f"Feature validation warnings for {symbol}: {validation_result.warnings}")
            
            # Calculate features - use streaming for large datasets
            if len(symbol_data) > self.streaming_threshold_rows:
                logger.info(f"Using streaming processing for {symbol} ({len(symbol_data):,} rows)")
                feature_df = await self._calculate_features_streaming(symbol_data, feature_sets)
            else:
                # Use thread pool for smaller datasets
                loop = asyncio.get_event_loop()
                feature_df = await loop.run_in_executor(
                    self.executor,
                    self._calculate_basic_features,
                    symbol_data
                )
            
            # Cache the result
            cache_key = self._generate_cache_key(symbol, start_time, end_time, feature_sets)
            await self.cache.set(CacheType.FEATURES, cache_key, feature_df, self.cache_ttl)
            
            # Store calculated features in both hot and cold storage
            await self._store_features(symbol, feature_df, start_time, end_time)
            
            return symbol, feature_df
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}", exc_info=True)
            self.calculation_stats['errors'] += 1
            return symbol, pd.DataFrame()
    
    async def _execute_parallel_feature_calculation(
        self, 
        tasks: List[Any], 
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute feature calculation tasks in parallel with progress tracking.
        
        Args:
            tasks: List of feature calculation tasks
            symbols: List of symbols being processed
            
        Returns:
            Dictionary mapping symbol to feature DataFrame
        """
        results = {}
        completed_count = 0
        
        # Execute tasks concurrently
        max_concurrent = self.config.get('orchestrator.features.parallel_processing.max_concurrent_tasks', 10)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        # Create bounded tasks
        bounded_tasks = [bounded_task(task) for task in tasks]
        
        # Process results as they complete
        for coro in asyncio.as_completed(bounded_tasks):
            symbol, feature_df = await coro
            results[symbol] = feature_df
            completed_count += 1
            
            # Log progress
            if completed_count % 5 == 0 or completed_count == len(tasks):
                logger.info(f"Feature calculation progress: {completed_count}/{len(tasks)} symbols completed")
        
        self.calculation_stats['parallel_tasks'] += len(tasks)
        return results
    
    async def _store_features(
        self, 
        symbol: str, 
        feature_df: pd.DataFrame, 
        start_time: datetime, 
        end_time: datetime
    ):
        """
        Store calculated features in both hot and cold storage.
        
        Args:
            symbol: Symbol being stored
            feature_df: Feature DataFrame
            start_time: Start time of calculation
            end_time: End time of calculation
        """
        if feature_df.empty:
            return
        
        try:
            # Store to both hot and cold storage with proper versioning
            
            # Cold storage: Use FeatureStoreV2 for versioned HDF5 storage
            # This provides version management and automatic cleanup
            hdf5_path = await self.feature_store.save_features(
                symbol=symbol,
                features_df=feature_df,
                feature_type='technical_indicators',
                year=start_time.year,
                month=start_time.month,
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'feature_count': len(feature_df.columns),
                    'record_count': len(feature_df)
                }
            )
            logger.debug(f"Saved versioned features for {symbol} to HDF5: {hdf5_path}")
            
            # Hot storage: Store latest features in PostgreSQL for live trading
            # Store each row as a separate record for time series access
            if not feature_df.empty:
                # Vectorized feature records creation
                feature_records = [
                    {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'features': row.to_dict(),
                        'feature_set': 'technical',
                        'version': '1.0'
                    }
                    for timestamp, row in feature_df.iterrows()
                ]
                
                # Use batch storage for efficiency
                batch_result = await self.feature_store_repo.store_features_batch(feature_records)
                
                if batch_result.success:
                    logger.debug(f"Stored {len(feature_records)} feature records for {symbol} in PostgreSQL")
                else:
                    logger.error(f"Failed to store features for {symbol} in PostgreSQL: {batch_result.errors}")
                
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}", exc_info=True)
    
    async def _calculate_features_streaming(
        self, 
        data: pd.DataFrame, 
        feature_sets: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate features using streaming processing for large datasets
        
        Args:
            data: Market data DataFrame
            feature_sets: Feature sets to calculate
            
        Returns:
            Features DataFrame
        """
        logger.info(f"Starting streaming feature calculation for {len(data):,} rows")
        
        async with streaming_context(self.streaming_config) as streamer:
            
            async def feature_processor(chunk: pd.DataFrame) -> pd.DataFrame:
                """Process feature calculation for a data chunk"""
                return self._calculate_basic_features(chunk)
            
            # Process data in streaming fashion
            result = await streamer.process_stream(data, feature_processor)
            
            # Log streaming statistics
            stats = streamer.get_stats()
            logger.info(f"Streaming feature calculation completed: "
                       f"{stats.chunks_processed} chunks, "
                       f"{stats.total_rows:,} rows, "
                       f"{stats.processing_time:.2f}s, "
                       f"Peak memory: {stats.memory_peak_mb:.1f}MB")
            
            return result or pd.DataFrame()
    
    def _calculate_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic features from market data.
        
        This is a placeholder - real implementation would call
        actual feature calculators.
        """
        features = pd.DataFrame(index=data.index)
        
        # Simple moving averages
        if 'close' in data.columns:
            features['sma_10'] = data['close'].rolling(window=self.config.get('orchestrator.features.technical_indicators.sma_short_window', 10)).mean()
            features['sma_20'] = data['close'].rolling(window=self.config.get('orchestrator.features.technical_indicators.sma_long_window', 20)).mean()
            
        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(window=self.config.get('orchestrator.features.technical_indicators.volume_sma_window', 10)).mean()
            
        return features
    
    async def _on_scanner_alert(self, event: ScannerAlertEvent):
        """
        Handle scanner alerts by updating features for alerted symbols.
        
        Args:
            event: Scanner alert event
        """
        try:
            alert_data = event.data
            symbol = alert_data.get('symbol')
            alert_type = alert_data.get('alert_type')
            
            logger.info(f"Received scanner alert for {symbol}: {alert_type}")
            
            # Get required features for this alert type
            required_features = self._alert_feature_mapping.get(
                alert_type, 
                self._alert_feature_mapping['default']
            )
            
            # Calculate fresh features for the alerted symbol
            end_time = datetime.now(timezone.utc)
            start_time = end_time - pd.Timedelta(days=self.config.get('orchestrator.lookback_periods.feature_calculation_days', 30))
            
            await self.calculate_features(
                symbols=[symbol],
                start_time=start_time,
                end_time=end_time,
                feature_sets=required_features
            )
            
            self.calculation_stats['scanner_triggered_calculations'] += 1
            
        except Exception as e:
            logger.error(f"Error handling scanner alert: {e}", exc_info=True)
    
    async def _on_feature_request(self, event: FeatureRequestEvent):
        """
        Handle feature requests from scanner bridge.
        
        Args:
            event: Feature request event
        """
        try:
            request_data = event.data
            symbols = request_data.get('symbols', [])
            features = request_data.get('features', [])
            priority = request_data.get('priority', 5)
            
            logger.info(f"Received feature request: {len(symbols)} symbols, {len(features)} features, priority {priority}")
            
            # Expand 'all_features' to actual feature list
            if 'all_features' in features:
                features = self._get_all_features()
            
            # Calculate features with priority-based processing
            end_time = datetime.now(timezone.utc)
            start_time = end_time - pd.Timedelta(days=self.config.get('orchestrator.lookback_periods.feature_calculation_days', 30))
            
            # Process high-priority requests first
            if priority >= 8:
                logger.info(f"Processing high-priority feature request (priority {priority})")
                # Process immediately without batching
                await self.calculate_features(
                    symbols=symbols,
                    start_time=start_time,
                    end_time=end_time,
                    feature_sets=features
                )
            else:
                # Add to batch processing queue for lower priority
                await self._add_to_batch_queue(symbols, features, priority)
            
            self.calculation_stats['feature_requests_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error handling feature request: {e}", exc_info=True)
    
    def _get_all_features(self) -> List[str]:
        """Get list of all available features."""
        return [
            # Price features
            'price_features',
            'trend_features',
            'support_resistance',
            
            # Volume features
            'volume_features',
            'volume_profile',
            
            # Volatility features
            'volatility',
            'volatility_term_structure',
            
            # Market microstructure
            'microstructure',
            'order_flow',
            
            # Technical indicators
            'momentum_features',
            'mean_reversion_features',
            
            # Sentiment and news
            'sentiment_features',
            'news_features',
            'social_features',
            
            # Market regime
            'regime_features',
            'correlation_features',
            
            # Options data
            'option_features',
            'option_flow',
            
            # Macro features
            'macro_features',
            'sector_features'
        ]
    
    async def _add_to_batch_queue(self, symbols: List[str], features: List[str], priority: int):
        """Add feature request to batch processing queue."""
        batch_request = {
            'symbols': symbols,
            'features': features,
            'priority': priority,
            'timestamp': datetime.now(timezone.utc)
        }
        
        await self._batch_queue.put(batch_request)
        logger.debug(f"Added batch request to queue: {len(symbols)} symbols, priority {priority}")
        
        # Start batch processing task if not already running
        if self._batch_processing_task is None or self._batch_processing_task.done():
            self._batch_processing_task = asyncio.create_task(self._process_batch_queue())
    
    async def _process_batch_queue(self):
        """Process batch requests from queue."""
        batch_interval = self.config.get('orchestrator.features.batch_processing.interval_seconds', 5)
        
        while True:
            try:
                # Wait for requests to accumulate
                await asyncio.sleep(batch_interval)
                
                # Collect all pending requests
                pending_requests = []
                while not self._batch_queue.empty():
                    try:
                        request = self._batch_queue.get_nowait()
                        pending_requests.append(request)
                    except asyncio.QueueEmpty:
                        break
                
                if not pending_requests:
                    continue
                
                # Sort by priority (higher priority first)
                pending_requests.sort(key=lambda x: x['priority'], reverse=True)
                
                # Process batched requests
                for request in pending_requests:
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - pd.Timedelta(days=self.config.get('orchestrator.lookback_periods.feature_calculation_days', 30))
                    
                    await self.calculate_features(
                        symbols=request['symbols'],
                        start_time=start_time,
                        end_time=end_time,
                        feature_sets=request['features']
                    )
                    
                    logger.debug(f"Processed batch request: {len(request['symbols'])} symbols, priority {request['priority']}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing batch queue: {e}", exc_info=True)
    
    @memory_profiled(include_gc=True)
    async def run_batch_calculation(self, symbols: List[str]):
        """
        Run batch feature calculation for multiple symbols with enhanced parallelization and memory monitoring.
        
        Args:
            symbols: List of symbols to process
        """
        logger.info(f"Starting batch feature calculation for {len(symbols)} symbols")
        
        with self.memory_monitor.memory_context("batch_feature_calculation", gc_before=True, gc_after=True):
            # Process in batches to avoid overwhelming the system
            batch_size = self.config.get('orchestrator.features.batch_processing.default_batch_size', 50)
            
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} symbols)")
                
                with self.memory_monitor.memory_context(f"batch_{batch_num}", gc_before=True):
                    end_time = datetime.now(timezone.utc)
                    start_time = end_time - pd.Timedelta(days=self.config.get('orchestrator.lookback_periods.feature_calculation_days', 30))
                    
                    batch_start_time = time.time()
                    
                    # Calculate features for this batch (uses internal parallelization)
                    batch_features = await self.calculate_features(
                        symbols=batch,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    batch_duration = time.time() - batch_start_time
                    
                    logger.info(f"Batch {batch_num}/{total_batches} completed in {batch_duration:.2f}s "
                               f"({len(batch_features)} features calculated)")
                
                # Memory cleanup between batches
                if batch_num % 5 == 0:  # Every 5 batches
                    self.memory_monitor.tracker.force_garbage_collection()
                    logger.info(f"Memory cleanup after batch {batch_num}")
                
                # Small delay between batches
                await asyncio.sleep(self.config.get('orchestrator.features.batch_processing.processing_delay_seconds', 1))
            
            # Clean up expired cache entries after batch processing (handled automatically by MarketDataCache)
            logger.debug("Cache cleanup handled automatically by MarketDataCache")
            
            logger.info(f"Batch feature calculation completed for {len(symbols)} symbols")
            self._log_calculation_stats()
            
            # Final memory report
            memory_report = self.memory_monitor.get_memory_report()
            logger.info(f"Final memory usage: {memory_report['current']['rss_mb']:.1f}MB")
    
    def _log_calculation_stats(self):
        """Log feature calculation statistics."""
        stats = self.calculation_stats
        cache_hit_rate = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        logger.info(f"Feature calculation statistics:")
        logger.info(f"  Total calculations: {stats['total_calculations']}")
        logger.info(f"  Cache hits: {stats['cache_hits']}")
        logger.info(f"  Cache misses: {stats['cache_misses']}")
        logger.info(f"  Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(f"  Parallel tasks executed: {stats['parallel_tasks']}")
        logger.info(f"  Scanner-triggered calculations: {stats['scanner_triggered_calculations']}")
        logger.info(f"  Feature requests processed: {stats['feature_requests_processed']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info(f"  Last calculation: {stats['last_calculation_time']}")
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """
        Get feature calculation statistics.
        
        Returns:
            Dictionary containing calculation statistics
        """
        stats = self.calculation_stats.copy()
        cache_hit_rate = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        stats['cache_hit_rate'] = cache_hit_rate
        return stats
    
    async def clear_cache(self):
        """Clear the feature cache."""
        success = await self.cache.clear(CacheType.FEATURES)
        if success:
            logger.info("Feature cache cleared")
        else:
            logger.warning("Failed to clear feature cache")
    
    async def calculate_streaming_aggregations(
        self,
        symbols: List[str],
        aggregation_type: str = 'market_summary',
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Calculate streaming aggregations for large datasets
        
        Args:
            symbols: List of symbols to aggregate
            aggregation_type: Type of aggregation ('market_summary', 'correlation_matrix', etc.)
            lookback_days: Days of historical data to aggregate
            
        Returns:
            Aggregated results DataFrame
        """
        logger.info(f"Starting streaming aggregation for {len(symbols)} symbols: {aggregation_type}")
        
        from main.utils.data import StreamingAggregator
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.Timedelta(days=lookback_days)
        
        # Get all market data for symbols
        all_market_data = await self.market_data_repo.get_data_for_symbols_and_range(
            symbols=symbols,
            start_time=start_time,
            end_time=end_time
        )
        
        if all_market_data.empty:
            logger.warning("No market data available for streaming aggregation")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(all_market_data):,} rows for streaming aggregation")
        
        # Configure aggregations based on type
        if aggregation_type == 'market_summary':
            group_by = 'symbol'
            aggregations = {
                'close': ['mean', 'std', 'min', 'max'],
                'volume': ['mean', 'sum'],
                'high': 'max',
                'low': 'min'
            }
        elif aggregation_type == 'daily_summary':
            # Convert timestamp to date for daily grouping
            all_market_data['date'] = pd.to_datetime(all_market_data['timestamp']).dt.date
            group_by = ['symbol', 'date']
            aggregations = {
                'close': 'last',
                'high': 'max',
                'low': 'min',
                'volume': 'sum'
            }
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        
        # Use streaming aggregator for large datasets
        if len(all_market_data) > self.streaming_threshold_rows:
            aggregator = StreamingAggregator(self.streaming_config)
            
            result = await aggregator.aggregate_streaming(
                data_source=all_market_data,
                group_by=group_by,
                aggregations=aggregations
            )
        else:
            # Use regular pandas aggregation for smaller datasets
            result = all_market_data.groupby(group_by).agg(aggregations)
        
        logger.info(f"Streaming aggregation completed: {len(result)} aggregated records")
        return result
    
    async def process_large_dataset_streaming(
        self,
        data_source: Union[pd.DataFrame, str],
        processing_function: Callable[[pd.DataFrame], pd.DataFrame],
        output_path: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Process large datasets using streaming with custom processing function
        
        Args:
            data_source: DataFrame or file path to process
            processing_function: Function to apply to each chunk
            output_path: Optional output file path
            
        Returns:
            Processed DataFrame if output_path is None
        """
        logger.info("Starting large dataset streaming processing")
        
        async with streaming_context(self.streaming_config) as streamer:
            
            async def enhanced_processor(chunk: pd.DataFrame) -> pd.DataFrame:
                """Enhanced processor with memory monitoring"""
                with self.memory_monitor.memory_context(f"process_chunk_{len(chunk)}"):
                    return processing_function(chunk)
            
            result = await streamer.process_stream(
                data_source,
                enhanced_processor,
                output_path
            )
            
            # Log processing statistics
            stats = streamer.get_stats()
            logger.info(f"Large dataset processing completed: "
                       f"{stats.chunks_processed} chunks, "
                       f"{stats.total_rows:,} rows, "
                       f"{stats.processing_time:.2f}s")
            
            return result
    
    async def shutdown(self):
        """Shutdown the feature orchestrator and cleanup resources."""
        # Cancel batch processing task
        if self._batch_processing_task and not self._batch_processing_task.done():
            self._batch_processing_task.cancel()
            try:
                await self._batch_processing_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool executor
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        logger.info("FeatureOrchestrator shutdown completed")
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)