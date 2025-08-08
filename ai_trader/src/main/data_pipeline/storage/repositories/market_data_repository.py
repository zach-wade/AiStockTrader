"""
Market Data Repository

Repository for market data storage and retrieval with hot/cold storage support.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import time

from main.interfaces.repositories.market_data import IMarketDataRepository
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import (
    RepositoryConfig,
    QueryFilter,
    OperationResult
)

from .base_repository import BaseRepository
from .helpers import (
    QueryBuilder,
    BatchProcessor,
    CrudExecutor,
    RepositoryMetricsCollector
)

from main.utils.core import get_logger, ensure_utc, chunk_list
from .constants import DEFAULT_BATCH_SIZE, DEFAULT_MAX_PARALLEL_WORKERS

logger = get_logger(__name__)


class MarketDataRepository(BaseRepository, IMarketDataRepository):
    """
    Repository for market data with specialized OHLCV operations.
    
    Provides optimized storage and retrieval for high-frequency market data
    with support for multiple intervals and hot/cold storage routing.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ):
        """
        Initialize the MarketDataRepository.
        
        Args:
            db_adapter: Database adapter
            config: Optional repository configuration
        """
        # Initialize with market_data_1h table (supports multiple intervals)
        super().__init__(db_adapter, type('MarketData', (), {'__tablename__': 'market_data_1h'}), config)
        
        # Additional components
        self.query_builder = QueryBuilder('market_data_1h')
        self.crud_executor = CrudExecutor(
            db_adapter,
            'market_data_1h',
            transaction_strategy=config.transaction_strategy if config else None
        )
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size if config else DEFAULT_BATCH_SIZE,
            max_parallel=config.max_parallel_workers if config else DEFAULT_MAX_PARALLEL_WORKERS
        )
        self.metrics = RepositoryMetricsCollector(
            'MarketDataRepository',
            enable_metrics=config.enable_metrics if config else True
        )
        
        # Interval mapping for table routing
        self.interval_tables = {
            '1min': 'market_data_1m',
            '5min': 'market_data_5m',
            '1hour': 'market_data_1h',
            '1day': 'market_data_1h'  # Daily data also stored in 1h table
        }
        
        logger.info("MarketDataRepository initialized with hot/cold storage support")
    
    # Required abstract methods from BaseRepository
    def get_required_fields(self) -> List[str]:
        """Get required fields for market data."""
        return ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate market data record."""
        errors = []
        
        # Check required fields
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate OHLC relationships
        if all(k in record for k in ['open', 'high', 'low', 'close']):
            o, h, l, c = record['open'], record['high'], record['low'], record['close']
            
            if None not in (o, h, l, c):
                if l > h:
                    errors.append("Low price cannot exceed high price")
                if o > h or o < l:
                    errors.append("Open price must be between low and high")
                if c > h or c < l:
                    errors.append("Close price must be between low and high")
        
        # Validate volume
        if 'volume' in record and record['volume'] is not None:
            if record['volume'] < 0:
                errors.append("Volume cannot be negative")
        
        return errors
    
    # IMarketDataRepository interface implementation
    async def get_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1day"
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        start_time = time.time()
        
        try:
            # Determine table based on interval
            table_name = self.interval_tables.get(interval, 'market_data_1h')
            
            # Build query
            query = f"""
                SELECT timestamp, open, high, low, close, volume, vwap, trades
                FROM {table_name}
                WHERE symbol = $1 
                AND interval = $2
                AND timestamp >= $3 
                AND timestamp <= $4
                ORDER BY timestamp ASC
            """
            
            params = [
                self._normalize_symbol(symbol),
                interval,
                ensure_utc(start_date),
                ensure_utc(end_date)
            ]
            
            # Check cache
            cache_key = self._get_cache_key(
                f"ohlcv_{symbol}",
                interval=interval,
                start=start_date.isoformat(),
                end=end_date.isoformat()
            )
            
            cached_data = await self._get_from_cache(cache_key)
            if cached_data is not None:
                await self.metrics.record_cache_access(hit=True)
                return pd.DataFrame(cached_data)
            
            await self.metrics.record_cache_access(hit=False)
            
            # Query database
            results = await self.db_adapter.fetch_all(query, *params)
            
            # Convert to DataFrame
            if results:
                df = pd.DataFrame([dict(r) for r in results])
                df.set_index('timestamp', inplace=True)
                df.index = pd.to_datetime(df.index)
                
                # Cache the result
                await self._set_in_cache(cache_key, df.to_dict('records'))
            else:
                df = pd.DataFrame()
            
            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation(
                'get_ohlcv',
                duration,
                success=True,
                records=len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            duration = time.time() - start_time
            await self.metrics.record_operation('get_ohlcv', duration, success=False)
            # Return empty DataFrame for data query methods
            return pd.DataFrame()
    
    async def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price data for a symbol."""
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume, vwap
                FROM market_data_1h
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            result = await self.db_adapter.fetch_one(query, self._normalize_symbol(symbol))
            
            if result:
                return {
                    'symbol': symbol,
                    'timestamp': result.get('timestamp'),
                    'price': result.get('close'),
                    'volume': result.get('volume'),
                    'high': result.get('high'),
                    'low': result.get('low'),
                    'vwap': result.get('vwap')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            raise
    
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest prices for multiple symbols."""
        try:
            # Use batch processing for efficiency
            placeholders = [f"${i+1}" for i in range(len(symbols))]
            
            query = f"""
                SELECT DISTINCT ON (symbol) 
                    symbol, timestamp, open, high, low, close, volume, vwap
                FROM market_data_1h
                WHERE symbol IN ({','.join(placeholders)})
                ORDER BY symbol, timestamp DESC
            """
            
            params = [self._normalize_symbol(s) for s in symbols]
            results = await self.db_adapter.fetch_all(query, *params)
            
            # Build result dictionary
            prices = {}
            for row in results:
                prices[row.get('symbol')] = {
                    'timestamp': row.get('timestamp'),
                    'price': row.get('close'),
                    'volume': row.get('volume'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'vwap': row.get('vwap')
                }
            
            return prices
            
        except Exception as e:
            logger.error(f"Error getting latest prices: {e}")
            raise
    
    async def store_ohlcv(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> OperationResult:
        """Store OHLCV data."""
        start_time = time.time()
        
        try:
            # Prepare records for insertion
            records = []
            for timestamp, row in data.iterrows():
                record = {
                    'symbol': self._normalize_symbol(symbol),
                    'timestamp': ensure_utc(timestamp),
                    'interval': interval,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'vwap': float(row.get('vwap', 0)),
                    'trades': int(row.get('trades', 0))
                }
                records.append(record)
            
            # Use batch processor for efficient insertion
            result = await self.batch_processor.process_batch(
                records,
                self._store_batch,
                validate_func=lambda r: len(self.validate_record(r)) == 0
            )
            
            # Invalidate cache for this symbol
            await self._invalidate_cache(f"ohlcv_{symbol}*")
            
            duration = time.time() - start_time
            
            return OperationResult(
                success=result['success'],
                records_affected=result['statistics']['succeeded'],
                records_created=result['statistics']['succeeded'],
                records_skipped=result['statistics']['failed'],
                duration_seconds=duration,
                metadata=result['statistics']
            )
            
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def get_price_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, float]]:
        """Get price range statistics for a period."""
        try:
            query = """
                SELECT 
                    MIN(low) as period_low,
                    MAX(high) as period_high,
                    AVG(close) as avg_close,
                    AVG(volume) as avg_volume
                FROM market_data_1h
                WHERE symbol = $1
                AND timestamp >= $2
                AND timestamp <= $3
            """
            
            result = await self.db_adapter.fetch_one(
                query,
                self._normalize_symbol(symbol),
                ensure_utc(start_date),
                ensure_utc(end_date)
            )
            
            if result and result['period_low'] is not None:
                return {
                    'low': float(result['period_low']),
                    'high': float(result['period_high']),
                    'average': float(result['avg_close']),
                    'avg_volume': float(result['avg_volume'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price range: {e}")
            raise
    
    async def get_volume_profile(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        bins: int = 10
    ) -> pd.DataFrame:
        """Get volume profile for price levels."""
        try:
            # Get OHLCV data
            df = await self.get_ohlcv(symbol, start_date, end_date)
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # Calculate volume at each price level
            volume_profile = []
            
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                
                # Volume for bars that overlap this price range
                mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
                bin_volume = df.loc[mask, 'volume'].sum()
                
                volume_profile.append({
                    'price_level': (bin_low + bin_high) / 2,
                    'price_low': bin_low,
                    'price_high': bin_high,
                    'volume': bin_volume,
                    'volume_pct': 0  # Will calculate after
                })
            
            profile_df = pd.DataFrame(volume_profile)
            
            # Calculate volume percentage
            total_volume = profile_df['volume'].sum()
            if total_volume > 0:
                profile_df['volume_pct'] = profile_df['volume'] / total_volume * 100
            
            return profile_df
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return pd.DataFrame()
    
    async def get_market_hours_data(
        self,
        symbol: str,
        date: datetime,
        extended_hours: bool = False
    ) -> pd.DataFrame:
        """Get market hours data for a specific date."""
        try:
            # Market hours (ET)
            market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = date.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if extended_hours:
                market_open = date.replace(hour=4, minute=0, second=0, microsecond=0)
                market_close = date.replace(hour=20, minute=0, second=0, microsecond=0)
            
            return await self.get_ohlcv(
                symbol,
                market_open,
                market_close,
                interval="5min"
            )
            
        except Exception as e:
            logger.error(f"Error getting market hours data: {e}")
            return pd.DataFrame()
    
    async def get_gaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        min_gap_percent: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Find price gaps in the data."""
        try:
            df = await self.get_ohlcv(symbol, start_date, end_date)
            
            if len(df) < 2:
                return []
            
            gaps = []
            
            for i in range(1, len(df)):
                prev_close = df.iloc[i-1]['close']
                curr_open = df.iloc[i]['open']
                
                gap_pct = abs((curr_open - prev_close) / prev_close * 100)
                
                if gap_pct >= min_gap_percent:
                    gaps.append({
                        'date': df.index[i],
                        'gap_type': 'up' if curr_open > prev_close else 'down',
                        'gap_percent': gap_pct,
                        'prev_close': prev_close,
                        'open': curr_open,
                        'filled': df.iloc[i]['low'] <= prev_close if curr_open > prev_close 
                                else df.iloc[i]['high'] >= prev_close
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error finding gaps: {e}")
            return []
    
    async def cleanup_old_data(
        self,
        days_to_keep: int,
        interval: Optional[str] = None
    ) -> OperationResult:
        """Clean up old market data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            if interval:
                query = """
                    DELETE FROM market_data_1h
                    WHERE timestamp < $1 AND interval = $2
                """
                params = [cutoff_date, interval]
            else:
                query = """
                    DELETE FROM market_data_1h
                    WHERE timestamp < $1
                """
                params = [cutoff_date]
            
            result = await self.crud_executor.execute_delete(query, params)
            
            # Clear cache after cleanup
            await self._invalidate_cache()
            
            logger.info(f"Cleaned up {result.records_deleted} old market data records")
            
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return OperationResult(success=False, error=str(e))
    
    # Private helper methods
    async def _store_batch(self, records: List[Dict[str, Any]]) -> Any:
        """Store a batch of records."""
        return await self.crud_executor.execute_bulk_insert(records)