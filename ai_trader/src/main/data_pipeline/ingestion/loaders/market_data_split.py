"""
Market data bulk loader with split table support and scanner qualification checks.

This module provides optimized bulk loading for market data with:
- Split tables by interval for better performance
- Scanner qualification checks to determine which data to store
- PostgreSQL COPY command for efficient loading
- Automatic partition management
"""

import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd

from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig, BulkLoadResult
from main.data_pipeline.services.storage import (
    QualificationService,
    TableRoutingService,
    PartitionManager
)
from main.utils.core import get_logger, timer
from .base import BaseBulkLoader

logger = get_logger(__name__)


class MarketDataSplitBulkLoader(BaseBulkLoader[Dict[str, Any]]):
    """
    Optimized bulk data loader for market data with split table support.
    
    Features:
    - Routes data to interval-specific tables via TableRoutingService
    - Checks scanner qualifications via QualificationService
    - Manages partitions via PartitionManager
    - Uses PostgreSQL COPY for maximum performance
    - Handles extended retention for Layer 2+ symbols
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        qualification_service: QualificationService,
        routing_service: TableRoutingService,
        partition_manager: PartitionManager,
        archive: Optional[Any] = None,
        config: Optional[BulkLoadConfig] = None
    ):
        """
        Initialize market data bulk loader with split table support.
        
        Args:
            db_adapter: Database adapter for operations
            qualification_service: Service for symbol qualification checks
            routing_service: Service for table routing decisions
            partition_manager: Service for partition management
            archive: Optional archive for cold storage
            config: Bulk loading configuration
        """
        super().__init__(
            db_adapter=db_adapter,
            archive=archive,
            config=config,
            data_type="market_data_split"
        )
        
        # Injected services
        self.qualification_service = qualification_service
        self.routing_service = routing_service
        self.partition_manager = partition_manager
        
        # Track seen timestamps to avoid duplicates within a flush cycle
        self._seen_timestamps: Dict[Tuple[str, str, datetime], bool] = {}
        
        # Track date ranges for partition checking
        self._min_timestamp: Optional[datetime] = None
        self._max_timestamp: Optional[datetime] = None
        
        # Track tables that will receive data
        self._tables_in_buffer: Set[str] = set()
        
        logger.info(
            f"MarketDataSplitBulkLoader initialized with services: "
            f"qualification={qualification_service.__class__.__name__}, "
            f"routing={routing_service.__class__.__name__}, "
            f"partitions={partition_manager.__class__.__name__}"
        )
    
    async def load(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        source: str = "polygon",
        **kwargs
    ) -> BulkLoadResult:
        """
        Load market data efficiently with scanner qualification checks.
        
        Args:
            data: DataFrame with market data (timestamp as index)
            symbol: Stock symbol
            interval: Time interval (e.g., '1minute', '5minute', '1hour', '1day')
            source: Data source name
            **kwargs: Additional parameters
            
        Returns:
            BulkLoadResult with operation details
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)
        
        if data.empty:
            result.success = True
            result.skip_reason = "Empty dataframe"
            return result
        
        symbol = symbol.upper()
        
        # Get symbol qualification
        qualification = await self.qualification_service.get_qualification(symbol)
        
        # Check if we should store this data
        if not qualification.should_store_interval(interval):
            logger.info(
                f"Skipping {interval} data for {symbol} - "
                f"Layer {qualification.layer_qualified} does not qualify"
            )
            result.success = True
            result.skipped = True
            result.skip_reason = f"Symbol not qualified for {interval} data"
            return result
        
        # Get target table via routing service
        table_name = self.routing_service.get_table_for_interval(interval)
        canonical_interval = self.routing_service.get_interval_for_table(table_name, interval)
        
        logger.debug(
            f"Loading {len(data)} {interval} records for {symbol} "
            f"to {table_name} as {canonical_interval}"
        )
        
        max_retries = self.config.max_retries if self.config.retry_on_failure else 1
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Prepare records with table routing
                records = self._prepare_records(
                    data=data,
                    symbol=symbol,
                    interval=canonical_interval,
                    source=source,
                    table_name=table_name
                )
                
                if not records:
                    result.success = True
                    result.skip_reason = "No valid records after preparation"
                    return result
                
                # Add to buffer
                self._add_to_buffer(records, symbol)
                self._tables_in_buffer.add(table_name)
                
                # Check if we should flush
                if self._should_flush():
                    flush_result = await self._flush_buffer()
                    result.records_loaded = flush_result.records_loaded
                    result.records_failed = flush_result.records_failed
                    result.symbols_processed = flush_result.symbols_processed
                    result.load_time_seconds = flush_result.load_time_seconds
                    result.archive_time_seconds = flush_result.archive_time_seconds
                    result.errors = flush_result.errors
                    result.success = flush_result.success
                    
                    # If flush failed and retry is enabled
                    if not flush_result.success and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(
                            f"Flush failed for {symbol}, retry {retry_count}/{max_retries}"
                        )
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        continue
                else:
                    # Data is buffered, will be written later
                    result.success = True
                    result.records_loaded = len(records)
                    result.symbols_processed = {symbol}
                    result.metadata["buffered"] = True
                    result.metadata["target_table"] = table_name
                
                return result
                
            except Exception as e:
                logger.error(f"Error loading {interval} data for {symbol}: {e}")
                result.errors.append(str(e))
                
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(2 ** retry_count)
                else:
                    return result
        
        return result
    
    def _prepare_records(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        source: str,
        table_name: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare records for bulk insertion with table routing.
        
        Args:
            data: Market data DataFrame
            symbol: Stock symbol
            interval: Canonical interval for the table
            source: Data source
            table_name: Target table name
            
        Returns:
            List of records ready for insertion
        """
        records = []
        current_time = datetime.now(timezone.utc)
        duplicates_skipped = 0
        
        # Reset index if timestamp is the index
        if data.index.name == 'timestamp' or isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
        
        # Ensure timestamp column exists
        if 'timestamp' not in data.columns and 't' in data.columns:
            data['timestamp'] = data['t']
        
        for _, row in data.iterrows():
            # Convert timestamp to timezone-aware if needed
            timestamp = row['timestamp']
            if isinstance(timestamp, pd.Timestamp):
                if timestamp.tz is None:
                    timestamp = timestamp.tz_localize('UTC')
                else:
                    timestamp = timestamp.tz_convert('UTC')
            elif isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp, utc=True)
            
            # Check for duplicates within the current buffer
            timestamp_key = (symbol, table_name, timestamp)
            if timestamp_key in self._seen_timestamps:
                duplicates_skipped += 1
                continue
            
            # Extract values with proper handling of None
            vwap_value = row.get('vwap', row.get('vw'))
            vwap = float(vwap_value) if pd.notna(vwap_value) else None
            
            trades_value = row.get('trades', row.get('trade_count', row.get('n')))
            trades = int(trades_value) if pd.notna(trades_value) else None
            
            # Validate prices
            open_price = float(row.get('open', row.get('o', 0)))
            close_price = float(row.get('close', row.get('c', 0)))
            
            if open_price <= 0 or close_price <= 0:
                logger.debug(f"Skipping invalid record for {symbol} at {timestamp}")
                continue
            
            record = {
                'symbol': symbol,
                'timestamp': timestamp,
                'interval': interval,  # Use canonical interval
                'open': open_price,
                'high': float(row.get('high', row.get('h', 0))),
                'low': float(row.get('low', row.get('l', 0))),
                'close': close_price,
                'volume': int(row.get('volume', row.get('v', 0))),
                'trades': trades,
                'vwap': vwap,
                'source': source,
                'created_at': current_time,
                'updated_at': current_time,
                '_table': table_name  # Special key for routing
            }
            
            # Mark this timestamp as seen
            self._seen_timestamps[timestamp_key] = True
            records.append(record)
            
            # Track min/max timestamps for partition checking
            if self._min_timestamp is None or timestamp < self._min_timestamp:
                self._min_timestamp = timestamp
            if self._max_timestamp is None or timestamp > self._max_timestamp:
                self._max_timestamp = timestamp
        
        if duplicates_skipped > 0:
            logger.info(
                f"Skipped {duplicates_skipped} duplicate timestamps for "
                f"{symbol} {interval} within current buffer"
            )
        
        return records
    
    async def _flush_buffer(self) -> BulkLoadResult:
        """
        Flush buffer with support for multiple tables.
        
        Groups records by table and performs separate COPY operations.
        Ensures partitions exist before loading.
        """
        result = BulkLoadResult(success=True, data_type=self.data_type)
        
        if not self._buffer:
            return result
        
        with timer("market_data_split_flush") as t:
            # Ensure partitions exist for the date range
            if self._min_timestamp and self._max_timestamp and self._tables_in_buffer:
                try:
                    table_ranges = {
                        table: (self._min_timestamp, self._max_timestamp)
                        for table in self._tables_in_buffer
                    }
                    
                    partitions_created = await self.partition_manager.ensure_partitions_for_tables(
                        table_ranges
                    )
                    
                    if partitions_created:
                        logger.info(
                            f"Created partitions: "
                            f"{', '.join(f'{k}={v}' for k, v in partitions_created.items() if v > 0)}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error ensuring partitions: {e}")
                    # Continue anyway - partition creation is not critical
            
            # Group records by table
            records_by_table = defaultdict(list)
            for record in self._buffer:
                table_name = record.pop('_table', 'market_data_1h')
                records_by_table[table_name].append(record)
            
            # Process each table
            for table_name, records in records_by_table.items():
                try:
                    loaded = await self._load_table_records(table_name, records)
                    result.records_loaded += loaded
                    logger.debug(f"Loaded {loaded} records to {table_name}")
                    
                except Exception as e:
                    # Extract unique symbols for error reporting
                    unique_symbols = set(r['symbol'] for r in records)
                    logger.error(
                        f"Error loading to {table_name}: {e}. "
                        f"Batch contained {len(records)} records for symbols: "
                        f"{sorted(unique_symbols)[:10]}..."
                    )
                    result.errors.append(f"{table_name}: {str(e)}")
                    result.records_failed += len(records)
                    result.success = False
            
            # Update result
            result.symbols_processed = self._symbols_in_buffer.copy()
            result.load_time_seconds = t.elapsed
            
            # Clear buffer and tracking
            self._buffer.clear()
            self._buffer_size_bytes = 0
            self._symbols_in_buffer.clear()
            self._tables_in_buffer.clear()
            self._last_flush_time = datetime.now(timezone.utc)
            self._seen_timestamps.clear()
            self._min_timestamp = None
            self._max_timestamp = None
        
        if result.success:
            logger.info(
                f"✓ Successfully flushed {result.records_loaded} records "
                f"for {len(result.symbols_processed)} symbols in {t.elapsed:.2f}s"
            )
        
        return result
    
    async def _load_table_records(
        self,
        table_name: str,
        records: List[Dict[str, Any]]
    ) -> int:
        """
        Load records to a specific table using COPY.
        
        Args:
            table_name: Target table
            records: Records to load
            
        Returns:
            Number of records loaded
        """
        if not records:
            return 0
        
        # Determine columns based on table
        if table_name == "market_data_1h":
            # 1h table includes interval column
            columns = [
                'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 'close',
                'volume', 'trades', 'vwap', 'source', 'created_at', 'updated_at'
            ]
            conflict_columns = "(symbol, timestamp, interval)"
        else:
            # Other tables don't need interval in conflict
            columns = [
                'symbol', 'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'trades', 'vwap', 'source', 'created_at', 'updated_at'
            ]
            conflict_columns = "(symbol, timestamp)"
        
        # Convert records to tuples for COPY
        copy_records = []
        for record in records:
            if table_name == "market_data_1h":
                copy_record = (
                    record['symbol'],
                    record['timestamp'],
                    record['interval'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume'],
                    record.get('trades'),
                    record.get('vwap'),
                    record['source'],
                    record['created_at'],
                    record['updated_at']
                )
            else:
                copy_record = (
                    record['symbol'],
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume'],
                    record.get('trades'),
                    record.get('vwap'),
                    record['source'],
                    record['created_at'],
                    record['updated_at']
                )
            copy_records.append(copy_record)
        
        async with self.db_adapter.acquire() as conn:
            # Create temp table
            temp_table = f"temp_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            await conn.execute(
                f"CREATE TEMP TABLE {temp_table} (LIKE {table_name} INCLUDING ALL)"
            )
            
            try:
                # Use COPY to load data to temp table
                await conn.copy_records_to_table(
                    temp_table,
                    records=copy_records,
                    columns=columns
                )
                
                # UPSERT from temp table to target table
                upsert_sql = f"""
                INSERT INTO {table_name}
                SELECT * FROM {temp_table}
                ON CONFLICT {conflict_columns}
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trades = EXCLUDED.trades,
                    vwap = EXCLUDED.vwap,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at
                """
                
                result = await conn.execute(upsert_sql)
                
                # Extract row count from result
                if result and result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        return int(parts[2])
                
                return len(records)
                
            finally:
                # Clean up temp table
                await conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
    
    async def _load_to_database(self, records: List[Dict[str, Any]]) -> int:
        """
        Override base method - not used in split loader.
        
        The split loader uses _load_table_records instead for
        table-specific loading.
        """
        # This method is required by base class but not used
        # in split loader implementation
        return 0
    
    async def _archive_records(self, records: List[Dict[str, Any]]) -> None:
        """Archive market data records to cold storage."""
        if not self.archive or not records:
            return
        
        # Group by symbol, date, and interval for efficient archiving
        groups = defaultdict(list)
        
        for record in records:
            symbol = record['symbol']
            date = record['timestamp'].date()
            interval = record.get('interval', '1hour')
            key = (symbol, date, interval)
            groups[key].append(record)
        
        # Archive each group
        for (symbol, date, interval), group_records in groups.items():
            try:
                # Convert to DataFrame for archiving
                df = pd.DataFrame(group_records)
                df.set_index('timestamp', inplace=True)
                
                # Remove internal routing field
                df.drop('_table', axis=1, errors='ignore', inplace=True)
                
                # Create RawDataRecord for archive
                from main.data_pipeline.storage.archive import RawDataRecord
                
                # Create metadata for the archive
                metadata = {
                    'data_type': 'market_data',
                    'symbol': symbol,
                    'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                    'interval': interval,
                    'record_count': len(group_records),
                    'start_time': df.index.min().isoformat() if not df.empty else None,
                    'end_time': df.index.max().isoformat() if not df.empty else None,
                    'has_vwap': any('vwap' in r and r['vwap'] is not None for r in group_records),
                    'source': group_records[0]['source'],
                    'table': f"market_data_{self._get_table_suffix(interval)}"
                }
                
                # Create the raw data record
                record = RawDataRecord(
                    source=group_records[0]['source'],
                    data_type='market_data',
                    symbol=f"{symbol}_{interval}_{date}",
                    timestamp=datetime.now(timezone.utc),
                    data={'market_data': df.to_dict('records')},
                    metadata=metadata
                )
                
                # Use archive's async save method
                await self.archive.save_raw_record_async(record)
                
            except Exception as e:
                logger.error(f"Failed to archive {symbol} {interval} data for {date}: {e}")
                # Continue with other groups even if one fails
    
    async def refresh_qualifications(self):
        """Refresh the qualification cache - useful after Layer updates."""
        await self.qualification_service.clear_cache()
        logger.info("Refreshed symbol qualification cache")