"""
Market data bulk loader for efficient backfill operations.

This module provides optimized bulk loading for market data (OHLCV),
using PostgreSQL COPY command and efficient batching strategies.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.ingestion import BulkLoadResult
from main.utils.core import get_logger

from .base import BaseBulkLoader

logger = get_logger(__name__)


class MarketDataBulkLoader(BaseBulkLoader[dict[str, Any]]):
    """
    Optimized bulk data loader for market data backfill operations.

    This loader uses PostgreSQL's COPY command for maximum performance
    during bulk operations.
    """

    def __init__(self, *args, **kwargs):
        """Initialize market data bulk loader."""
        super().__init__(*args, data_type="market_data", **kwargs)

    async def load(
        self, data: pd.DataFrame, symbol: str, interval: str, source: str = "polygon", **kwargs
    ) -> BulkLoadResult:
        """
        Load market data efficiently using bulk operations.

        Args:
            data: DataFrame with market data (timestamp as index)
            symbol: Stock symbol
            interval: Time interval (e.g., '1day', '1hour')
            source: Data source name

        Returns:
            BulkLoadResult with operation details
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)

        if data.empty:
            result.success = True
            result.skip_reason = "Empty dataframe"
            return result

        max_retries = self.config.max_retries if self.config.retry_on_failure else 1
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Convert DataFrame to records
                records = self._prepare_records(
                    data, symbol=symbol, interval=interval, source=source
                )

                if not records:
                    result.success = True
                    result.skip_reason = "No valid records after preparation"
                    return result

                # Add to buffer
                self._add_to_buffer(records, symbol)

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
                        await asyncio.sleep(2**retry_count)  # Exponential backoff
                        continue
                else:
                    # Data is buffered, will be written later
                    result.success = True
                    result.records_loaded = len(records)  # Track records added to buffer
                    result.metadata["buffered"] = True

                return result

            except Exception as e:
                retry_count += 1
                error_msg = (
                    f"Failed to load market data for {symbol} "
                    f"(attempt {retry_count}/{max_retries}): {e}"
                )
                logger.error(error_msg)
                result.errors.append(error_msg)

                if retry_count < max_retries:
                    await asyncio.sleep(2**retry_count)  # Exponential backoff
                else:
                    # Final failure - try to remove corrupted data
                    try:
                        self._remove_symbol_from_buffer(symbol)
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean buffer: {cleanup_error}")
                    return result

        return result

    def _prepare_records(
        self, data: pd.DataFrame, symbol: str, interval: str, source: str, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Prepare market data records from DataFrame.

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            interval: Time interval
            source: Data source

        Returns:
            List of record dictionaries
        """
        records = []
        current_time = datetime.now(UTC)

        for timestamp, row in data.iterrows():
            # Ensure timestamp is timezone-aware
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            elif isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp, utc=True)

            # Validate and clean data
            open_price = float(row.get("open", 0)) if pd.notna(row.get("open")) else 0
            high_price = float(row.get("high", 0)) if pd.notna(row.get("high")) else 0
            low_price = float(row.get("low", 0)) if pd.notna(row.get("low")) else 0
            close_price = float(row.get("close", 0)) if pd.notna(row.get("close")) else 0
            volume = int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0

            # Skip invalid records
            if open_price <= 0 or close_price <= 0:
                logger.debug(f"Skipping invalid record for {symbol} at {timestamp}")
                continue

            record = {
                "symbol": symbol.upper(),
                "timestamp": timestamp,
                "interval": interval,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "trades": int(row.get("trades", 0)) if pd.notna(row.get("trades")) else None,
                "vwap": float(row.get("vwap", 0)) if pd.notna(row.get("vwap")) else None,
                "source": source,
                "created_at": current_time,
                "updated_at": current_time,
            }
            records.append(record)

        logger.debug(f"Prepared {len(records)} records for {symbol}")
        return records

    def _estimate_record_size(self, record: dict[str, Any]) -> int:
        """Estimate size of a market data record."""
        # More accurate estimate for market data
        # symbol(10) + timestamps(32) + numbers(8*7) + interval(10) + source(10)
        # = ~130 bytes + overhead
        return 150

    def _remove_symbol_from_buffer(self, symbol: str):
        """Remove all records for a specific symbol from buffer."""
        if symbol not in self._symbols_in_buffer:
            return

        # Filter out records for this symbol
        original_size = len(self._buffer)
        self._buffer = [r for r in self._buffer if r.get("symbol") != symbol.upper()]
        removed_count = original_size - len(self._buffer)

        if removed_count > 0:
            # Update tracking
            self._symbols_in_buffer.discard(symbol)
            self._buffer_size_bytes -= removed_count * 150  # Use average size
            self._total_records_loaded -= removed_count  # Adjust count

            logger.info(f"Removed {removed_count} records for {symbol} from buffer")

    async def _load_to_database(self, records: list[dict[str, Any]]) -> int:
        """
        Load records to database using COPY or INSERT.

        Args:
            records: Market data records to load

        Returns:
            Number of records loaded
        """
        if not records:
            return 0

        if self.config.use_copy_command:
            try:
                return await self._load_with_copy(records)
            except Exception as e:
                logger.warning(f"COPY method failed: {e}, falling back to INSERT")
                return await self._load_with_insert(records)
        else:
            return await self._load_with_insert(records)

    async def _load_with_copy(self, records: list[dict[str, Any]]) -> int:
        """
        Load data using PostgreSQL COPY command.

        Args:
            records: Records to load

        Returns:
            Number of records loaded
        """
        columns = [
            "symbol",
            "timestamp",
            "interval",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trades",
            "vwap",
            "source",
            "created_at",
            "updated_at",
        ]

        # Convert records to tuples for asyncpg
        copy_records = []
        for record in records:
            copy_record = (
                record["symbol"],
                record["timestamp"],
                record["interval"],
                record["open"],
                record["high"],
                record["low"],
                record["close"],
                record["volume"],
                record.get("trades"),
                record.get("vwap"),
                record["source"],
                record["created_at"],
                record["updated_at"],
            )
            copy_records.append(copy_record)

        async with self.db_adapter.acquire() as conn:
            # Create temp table
            await conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS temp_market_data "
                "(LIKE market_data INCLUDING ALL)"
            )

            # Clear temp table
            await conn.execute("TRUNCATE temp_market_data")

            # Use asyncpg's copy_records_to_table
            await conn.copy_records_to_table(
                "temp_market_data", records=copy_records, columns=columns
            )

            # Upsert from temp table
            upsert_sql = """
            INSERT INTO market_data
            SELECT * FROM temp_market_data
            ON CONFLICT (symbol, timestamp, interval)
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

            # Clean up temp table
            await conn.execute("DROP TABLE temp_market_data")

            # Extract row count from result string
            if result and result.startswith("INSERT"):
                parts = result.split()
                if len(parts) >= 3:
                    return int(parts[2])

            return len(records)  # Fallback to record count

    async def _load_with_insert(self, records: list[dict[str, Any]]) -> int:
        """
        Load data using parameterized INSERT statements.

        Args:
            records: Records to load

        Returns:
            Number of records loaded
        """
        # Use smaller batches for INSERT to avoid parameter limit
        batch_size = 100
        total_loaded = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Build parameterized insert
            placeholders = []
            values = []
            param_count = 1

            for record in batch:
                placeholder = (
                    f"(${param_count}, ${param_count+1}, ${param_count+2}, "
                    f"${param_count+3}, ${param_count+4}, ${param_count+5}, "
                    f"${param_count+6}, ${param_count+7}, ${param_count+8}, "
                    f"${param_count+9}, ${param_count+10}, ${param_count+11}, "
                    f"${param_count+12})"
                )
                placeholders.append(placeholder)

                values.extend(
                    [
                        record["symbol"],
                        record["timestamp"],
                        record["interval"],
                        record["open"],
                        record["high"],
                        record["low"],
                        record["close"],
                        record["volume"],
                        record.get("trades"),
                        record.get("vwap"),
                        record["source"],
                        record["created_at"],
                        record["updated_at"],
                    ]
                )

                param_count += 13

            sql = f"""
            INSERT INTO market_data (
                symbol, timestamp, interval, open, high, low, close,
                volume, trades, vwap, source, created_at, updated_at
            )
            VALUES {','.join(placeholders)}
            ON CONFLICT (symbol, timestamp, interval)
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

            async with self.db_adapter.acquire() as conn:
                result = await conn.execute(sql, *values)
                if result and result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        total_loaded += int(parts[2])
                else:
                    # Assume all records were processed
                    total_loaded += len(batch)

        return total_loaded

    async def _archive_records(self, records: list[dict[str, Any]]) -> None:
        """Archive market data records to cold storage."""
        if not self.archive or not records:
            return

        # Group by symbol and date for efficient archiving
        symbol_date_groups = defaultdict(list)

        for record in records:
            symbol = record["symbol"]
            date = record["timestamp"].date()
            key = (symbol, date)
            symbol_date_groups[key].append(record)

        # Archive each group
        for (symbol, date), group_records in symbol_date_groups.items():
            try:
                # Convert to DataFrame for archiving
                df = pd.DataFrame(group_records)
                df.set_index("timestamp", inplace=True)

                # Create RawDataRecord for archive
                # Local imports
                from main.data_pipeline.storage.archive import RawDataRecord

                # Create metadata for the archive
                metadata = {
                    "data_type": "market_data",
                    "symbol": symbol,
                    "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                    "interval": group_records[0]["interval"],
                    "record_count": len(group_records),
                    "start_time": df.index.min().isoformat() if not df.empty else None,
                    "end_time": df.index.max().isoformat() if not df.empty else None,
                    "has_vwap": any("vwap" in r and r["vwap"] is not None for r in group_records),
                    "source": group_records[0]["source"],
                }

                # Create the raw data record
                record = RawDataRecord(
                    source=group_records[0]["source"],
                    data_type="market_data",
                    symbol=f"{symbol}_{group_records[0]['interval']}_{date}",
                    timestamp=datetime.now(UTC),
                    data={"market_data": df.to_dict("records")},
                    metadata=metadata,
                )

                # Use archive's async save method
                await self.archive.save_raw_record_async(record)

            except Exception as e:
                logger.error(f"Failed to archive {symbol} data for {date}: {e}")
                # Continue with other groups even if one fails
