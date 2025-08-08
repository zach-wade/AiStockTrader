"""
Scanner data access utilities.

Provides helper functions and classes for efficient data retrieval
with hot/cold storage awareness.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import asyncio
from dataclasses import dataclass
import pandas as pd

from main.data_pipeline.storage.storage_router import StorageRouter, QueryType
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.data_pipeline.storage.repositories.scanner_data_repository import ScannerDataRepository
from main.interfaces.scanners import IScannerRepository
from main.utils.core import timer, create_task_safely
from main.utils.cache import get_global_cache, CacheType

logger = logging.getLogger(__name__)


@dataclass
class DataAccessConfig:
    """Configuration for scanner data access."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 60
    parallel_fetch_limit: int = 10
    timeout_seconds: float = 30.0
    prefer_hot_storage: bool = True
    fallback_on_error: bool = True


class ScannerDataAccess:
    """
    Helper class for scanner data access with optimizations.
    
    Features:
    - Automatic hot/cold storage routing
    - Result caching for performance
    - Parallel data fetching
    - Error handling with fallbacks
    """
    
    def __init__(
        self,
        repository: IScannerRepository,
        storage_router: StorageRouter,
        config: Optional[DataAccessConfig] = None
    ):
        """
        Initialize scanner data access helper.
        
        Args:
            repository: Scanner data repository
            storage_router: Storage routing logic
            config: Data access configuration
        """
        self.repository = repository
        self.storage_router = storage_router
        self.config = config or DataAccessConfig()
        
        # Initialize cache if enabled
        self.cache = None
        if self.config.enable_caching:
            self.cache = get_global_cache()
        
        # Semaphore for parallel fetch limiting
        self._fetch_semaphore = asyncio.Semaphore(self.config.parallel_fetch_limit)
    
    async def get_scanner_data_batch(
        self,
        symbols: List[str],
        data_types: List[str],
        lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch multiple data types for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols
            data_types: List of data types (market_data, volume_stats, etc.)
            lookback_hours: Hours of historical data
            
        Returns:
            Nested dict: {symbol: {data_type: data}}
        """
        with timer() as t:
            # Build query filter
            query_filter = QueryFilter(
                symbols=symbols,
                start_date=datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
                end_date=datetime.now(timezone.utc)
            )
            
            # Determine optimal query strategy
            query_type = self._determine_query_type(lookback_hours)
            routing_decision = self.storage_router.route_query(
                query_filter,
                query_type,
                prefer_performance=self.config.prefer_hot_storage
            )
            
            # Handle both enum and string types for primary_tier
            tier_name = routing_decision.primary_tier.value if hasattr(routing_decision.primary_tier, 'value') else str(routing_decision.primary_tier)
            logger.info(
                f"Fetching {len(data_types)} data types for {len(symbols)} symbols "
                f"using {tier_name} storage"
            )
            
            # Create fetch tasks
            tasks = []
            for data_type in data_types:
                task = create_task_safely(
                    self._fetch_data_type(symbols, data_type, query_filter)
                )
                tasks.append((data_type, task))
            
            # Execute fetches in parallel
            results = {}
            for data_type, task in tasks:
                try:
                    data = await asyncio.wait_for(
                        task,
                        timeout=self.config.timeout_seconds
                    )
                    results[data_type] = data
                except asyncio.TimeoutError:
                    logger.error(f"Timeout fetching {data_type}")
                    results[data_type] = {}
                except Exception as e:
                    logger.error(f"Error fetching {data_type}: {e}")
                    results[data_type] = {}
            
            # Reorganize by symbol
            symbol_data = {}
            for symbol in symbols:
                symbol_data[symbol] = {}
                for data_type, type_data in results.items():
                    symbol_data[symbol][data_type] = type_data.get(symbol, None)
            
            logger.info(f"Fetched scanner data in {t.elapsed_ms:.2f}ms")
            return symbol_data
    
    async def get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Get data from cache or fetch if not cached.
        
        Args:
            cache_key: Cache key
            fetch_func: Async function to fetch data
            *args, **kwargs: Arguments for fetch function
            
        Returns:
            Cached or fetched data
        """
        # Check cache first
        if self.cache and self.config.enable_caching:
            cached_data = await self.cache.get(
                cache_key,
                cache_type=CacheType.SCANNER_RESULTS
            )
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
        
        # Fetch data
        async with self._fetch_semaphore:
            data = await fetch_func(*args, **kwargs)
        
        # Cache result
        if self.cache and self.config.enable_caching and data is not None:
            await self.cache.set(
                cache_key,
                data,
                ttl=self.config.cache_ttl_seconds,
                cache_type=CacheType.SCANNER_RESULTS
            )
        
        return data
    
    async def get_market_snapshot(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current market snapshot for symbols.
        
        Optimized for real-time scanning with minimal latency.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dict with symbol as key and snapshot data as value
        """
        # Get latest prices and volume stats in parallel
        price_task = self.repository.get_latest_prices(symbols)
        volume_task = self.repository.get_volume_statistics(symbols, lookback_days=20)
        
        prices, volume_stats = await asyncio.gather(price_task, volume_task)
        
        # Combine into snapshot
        snapshots = {}
        for symbol in symbols:
            snapshot = {
                'price': prices.get(symbol, {}),
                'volume_stats': volume_stats.get(symbol, {}),
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Calculate relative volume if possible
            if snapshot['price'] and snapshot['volume_stats'].get('avg_volume'):
                current_volume = snapshot['price'].get('volume', 0)
                avg_volume = snapshot['volume_stats']['avg_volume']
                snapshot['relative_volume'] = current_volume / avg_volume if avg_volume > 0 else 0
            else:
                snapshot['relative_volume'] = 0
            
            snapshots[symbol] = snapshot
        
        return snapshots
    
    async def get_technical_data(
        self,
        symbols: List[str],
        indicators: List[str],
        lookback_days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Get market data suitable for technical indicator calculation.
        
        Args:
            symbols: List of symbols
            indicators: List of indicators needed (for optimization)
            lookback_days: Days of historical data
            
        Returns:
            Dict with symbol as key and DataFrame as value
        """
        # Determine required columns based on indicators
        required_columns = self._get_required_columns(indicators)
        
        # Build query filter
        query_filter = QueryFilter(
            symbols=symbols,
            start_date=datetime.now(timezone.utc) - timedelta(days=lookback_days),
            end_date=datetime.now(timezone.utc),
            additional_filters={'interval': '1day'}  # Daily for technical analysis
        )
        
        # Fetch data
        market_data = await self.repository.get_market_data(
            symbols,
            query_filter,
            columns=required_columns
        )
        
        return market_data
    
    async def _fetch_data_type(
        self,
        symbols: List[str],
        data_type: str,
        query_filter: QueryFilter
    ) -> Dict[str, Any]:
        """Fetch specific data type for symbols."""
        if data_type == 'market_data':
            return await self.repository.get_market_data(symbols, query_filter)
        elif data_type == 'volume_stats':
            return await self.repository.get_volume_statistics(symbols)
        elif data_type == 'latest_prices':
            return await self.repository.get_latest_prices(symbols)
        elif data_type == 'news_sentiment':
            return await self.repository.get_news_sentiment(symbols, query_filter)
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return {}
    
    def _determine_query_type(self, lookback_hours: int) -> QueryType:
        """Determine query type based on lookback period."""
        if lookback_hours <= 8:
            return QueryType.REAL_TIME
        elif lookback_hours <= 24:
            return QueryType.FEATURE_CALC
        else:
            return QueryType.ANALYSIS
    
    def _get_required_columns(self, indicators: List[str]) -> List[str]:
        """Get required data columns based on indicators."""
        # Base columns always needed
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Add indicator-specific columns
        indicator_requirements = {
            'vwap': ['vwap'],
            'money_flow': ['typical_price'],
            'spread': ['bid', 'ask'],
            'trades': ['trade_count']
        }
        
        for indicator in indicators:
            if indicator in indicator_requirements:
                columns.extend(indicator_requirements[indicator])
        
        return list(set(columns))  # Remove duplicates
    
    def invalidate_cache(self, symbols: Optional[List[str]] = None) -> None:
        """
        Invalidate cached data for symbols.
        
        Args:
            symbols: Symbols to invalidate (None for all)
        """
        if not self.cache:
            return
        
        # Build cache keys to invalidate
        if symbols:
            for symbol in symbols:
                # Invalidate all data types for symbol
                for data_type in ['market_data', 'volume_stats', 'latest_prices']:
                    cache_key = f"scanner:{data_type}:{symbol}"
                    asyncio.create_task(
                        self.cache.delete(cache_key, cache_type=CacheType.SCANNER_RESULTS)
                    )
        else:
            # Clear all scanner cache
            logger.info("Clearing all scanner cache")
            # This would need cache backend support for pattern deletion