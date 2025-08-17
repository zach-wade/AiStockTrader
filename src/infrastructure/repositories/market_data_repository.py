"""
Market Data Repository Implementation

PostgreSQL implementation of the IMarketDataRepository interface.
Handles persistence and retrieval of market data bars with optimized batch operations.
"""

# Standard library imports
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

# Local imports
from src.application.interfaces.exceptions import IntegrityError, RepositoryError
from src.application.interfaces.market_data import Bar
from src.application.interfaces.repositories import IMarketDataRepository
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol
from src.infrastructure.database.adapter import PostgreSQLAdapter

logger = logging.getLogger(__name__)


class MarketDataRepository(IMarketDataRepository):
    """
    PostgreSQL implementation of the market data repository.

    Handles storage and retrieval of market data bars with:
    - Batch insert optimization
    - Timezone conversion (stores as UTC)
    - Duplicate handling via unique constraints
    - Indexed queries for performance
    """

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize repository with database adapter.

        Args:
            adapter: PostgreSQL database adapter
        """
        self._adapter = adapter

    async def save_bar(self, bar: Bar) -> None:
        """
        Save a single market data bar.

        Args:
            bar: The bar data to save

        Raises:
            RepositoryError: If save operation fails
        """
        # Map timeframe to table name
        table_name = self._get_table_name(bar.timeframe)
        
        try:
            # Convert timestamp to UTC if needed
            timestamp_utc = self._ensure_utc(bar.timestamp)
            
            if bar.timeframe in ("1hour", "1day"):
                # market_data_1h has interval column and different unique constraint
                # nosec B608 - table name from controlled mapping
                query = f"""
                    INSERT INTO {table_name} (
                        symbol, timestamp,
                        open, high, low, close, volume,
                        vwap, trades, interval, source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp, interval)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        trades = EXCLUDED.trades,
                        updated_at = NOW()
                """
                
                # Map timeframe to interval value
                interval = "1day" if bar.timeframe == "1day" else "1hour"
                
                await self._adapter.execute_query(
                    query,
                    bar.symbol.value,
                    timestamp_utc,
                    float(bar.open.value),
                    float(bar.high.value),
                    float(bar.low.value),
                    float(bar.close.value),
                    bar.volume,
                    float(bar.vwap.value) if bar.vwap else None,
                    bar.trade_count,
                    interval,  # interval
                    "test",  # source
                )
            else:
                # Other tables don't have interval column
                # nosec B608 - table name from controlled mapping
                query = f"""
                    INSERT INTO {table_name} (
                        symbol, timestamp,
                        open, high, low, close, volume,
                        vwap, trades, source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        trades = EXCLUDED.trades,
                        updated_at = NOW()
                """
                
                await self._adapter.execute_query(
                    query,
                    bar.symbol.value,
                    timestamp_utc,
                    float(bar.open.value),
                    float(bar.high.value),
                    float(bar.low.value),
                    float(bar.close.value),
                    bar.volume,
                    float(bar.vwap.value) if bar.vwap else None,
                    bar.trade_count,
                    "test",  # source
                )

            logger.debug(f"Saved bar for {bar.symbol.value} at {timestamp_utc}")

        except IntegrityError as e:
            # This shouldn't happen due to ON CONFLICT, but handle just in case
            logger.warning(f"Duplicate bar ignored for {bar.symbol.value} at {bar.timestamp}: {e}")
        except Exception as e:
            logger.error(f"Failed to save bar for {bar.symbol.value}: {e}")
            raise RepositoryError(f"Failed to save bar: {e}") from e

    async def save_bars(self, bars: list[Bar]) -> None:
        """
        Save multiple market data bars in batch.

        Uses batch insert for improved performance.

        Args:
            bars: List of bars to save

        Raises:
            RepositoryError: If save operation fails
        """
        if not bars:
            return

        # Prepare batch data - structure depends on the table
        batch_data = []
        for bar in bars:
            timestamp_utc = self._ensure_utc(bar.timestamp)
            if bar.timeframe in ("1hour", "1day"):
                # For 1h table, include interval
                interval = "1day" if bar.timeframe == "1day" else "1hour"
                batch_data.append(
                    (
                        bar.symbol.value,
                        timestamp_utc,
                        float(bar.open.value),
                        float(bar.high.value),
                        float(bar.low.value),
                        float(bar.close.value),
                        bar.volume,
                        bar.trade_count,
                        float(bar.vwap.value) if bar.vwap else None,
                        interval,  # interval
                        "test",  # source
                    )
                )
            else:
                # For other tables, no interval column
                batch_data.append(
                    (
                        bar.symbol.value,
                        timestamp_utc,
                        float(bar.open.value),
                        float(bar.high.value),
                        float(bar.low.value),
                        float(bar.close.value),
                        bar.volume,
                        bar.trade_count,
                        float(bar.vwap.value) if bar.vwap else None,
                        "test",  # source
                    )
                )

        # Group bars by timeframe to insert into correct tables
        bars_by_timeframe = {}
        for bar, data in zip(bars, batch_data):
            if bar.timeframe not in bars_by_timeframe:
                bars_by_timeframe[bar.timeframe] = []
            bars_by_timeframe[bar.timeframe].append(data)

        try:
            # Insert each timeframe group into its corresponding table
            for timeframe, timeframe_data in bars_by_timeframe.items():
                table_name = self._get_table_name(timeframe)
                
                if timeframe in ("1hour", "1day"):
                    # market_data_1h has interval column
                    query = f"""
                        INSERT INTO {table_name} (
                            symbol, timestamp,
                            open, high, low, close, volume,
                            trades, vwap, interval, source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp, interval)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            trades = EXCLUDED.trades,
                            vwap = EXCLUDED.vwap,
                            updated_at = NOW()
                    """  # nosec B608
                else:
                    # Other tables don't have interval column
                    query = f"""
                        INSERT INTO {table_name} (
                            symbol, timestamp,
                            open, high, low, close, volume,
                            trades, vwap, source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            trades = EXCLUDED.trades,
                            vwap = EXCLUDED.vwap,
                            updated_at = NOW()
                    """  # nosec B608
                
                await self._adapter.execute_batch(query, timeframe_data)
            
            logger.info(f"Saved {len(bars)} bars in batch")
        except Exception as e:
            logger.error(f"Failed to save bars in batch: {e}")
            raise RepositoryError(f"Failed to save bars: {e}") from e

    async def get_latest_bar(self, symbol: str, timeframe: str = "1min") -> Bar | None:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: The trading symbol
            timeframe: Bar timeframe (default: "1min")

        Returns:
            The latest bar if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)
        # nosec B608 - table name from controlled mapping
        query = f"""
            SELECT symbol, timestamp, open, high, low, close,
                   volume, vwap, trades
            FROM {table_name}
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            row = await self._adapter.fetch_one(query, symbol)

            if row:
                return self._row_to_bar(row, timeframe)

            logger.debug(f"No bars found for {symbol} with timeframe {timeframe}")
            return None

        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            raise RepositoryError(f"Failed to get latest bar: {e}") from e

    async def get_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get bars for a symbol within a date range.

        Args:
            symbol: The trading symbol
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Bar timeframe (default: "1min")

        Returns:
            List of bars ordered by timestamp (ascending)

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)
        # nosec B608 - table name from controlled mapping
        query = f"""
            SELECT symbol, timestamp, open, high, low, close,
                   volume, vwap, trades
            FROM {table_name}
            WHERE symbol = %s
                AND timestamp >= %s
                AND timestamp <= %s
            ORDER BY timestamp ASC
        """

        try:
            # Ensure timestamps are UTC
            start_utc = self._ensure_utc(start)
            end_utc = self._ensure_utc(end)

            rows = await self._adapter.fetch_all(query, symbol, start_utc, end_utc)

            bars = [self._row_to_bar(row, timeframe) for row in rows]
            logger.debug(f"Retrieved {len(bars)} bars for {symbol} from {start_utc} to {end_utc}")

            return bars

        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            raise RepositoryError(f"Failed to get bars: {e}") from e

    async def get_bars_by_count(
        self, symbol: str, count: int, end: datetime | None = None, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get a specific number of most recent bars.

        Args:
            symbol: The trading symbol
            count: Number of bars to retrieve
            end: End datetime (defaults to now)
            timeframe: Bar timeframe

        Returns:
            List of bars ordered by timestamp (ascending)

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)
        
        if end is None:
            # Query without end date condition
            # nosec B608 - table name from controlled mapping
            query = f"""
                SELECT * FROM (
                    SELECT symbol, timestamp, open, high, low, close,
                           volume, vwap, trades
                    FROM {table_name}
                    WHERE symbol = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                ) AS recent_bars
                ORDER BY timestamp ASC
            """
            params = [symbol, count]
        else:
            # Query with end date condition
            end_utc = self._ensure_utc(end)
            # nosec B608 - table name from controlled mapping
            query = f"""
                SELECT * FROM (
                    SELECT symbol, timestamp, open, high, low, close,
                           volume, vwap, trades
                    FROM {table_name}
                    WHERE symbol = %s AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                ) AS recent_bars
                ORDER BY timestamp ASC
            """
            params = [symbol, end_utc, count]

        try:
            rows = await self._adapter.fetch_all(query, *params)

            bars = [self._row_to_bar(row, timeframe) for row in rows]
            logger.debug(f"Retrieved {len(bars)} bars for {symbol}")

            return bars

        except Exception as e:
            logger.error(f"Failed to get bars by count for {symbol}: {e}")
            raise RepositoryError(f"Failed to get bars by count: {e}") from e

    async def delete_bars_before(self, timestamp: datetime) -> int:
        """
        Delete bars older than a specified timestamp.

        Renamed from delete_old_bars to match interface.

        Args:
            timestamp: Delete bars before this time

        Returns:
            Number of bars deleted

        Raises:
            RepositoryError: If delete operation fails
        """
        # We need to delete from all timeframe tables
        # This is a maintenance operation, so we'll delete from all tables
        tables = ["market_data_1m", "market_data_5m", "market_data_15m", "market_data_30m", "market_data_1h"]
        total_deleted = 0
        
        try:
            timestamp_utc = self._ensure_utc(timestamp)
            
            for table in tables:
                # nosec B608 - table names from controlled list
                query = f"""
                    DELETE FROM {table}
                    WHERE timestamp < %s
                """
                result = await self._adapter.execute_query(query, timestamp_utc)
                
                # Extract row count from result string (e.g., "EXECUTE 42")
                if result and result.startswith("EXECUTE"):
                    deleted = int(result.split()[1])
                    total_deleted += deleted

            logger.info(f"Deleted {total_deleted} bars before {timestamp_utc} across all tables")
            return total_deleted

        except Exception as e:
            logger.error(f"Failed to delete old bars: {e}")
            raise RepositoryError(f"Failed to delete old bars: {e}") from e

    async def delete_old_bars(self, before: datetime) -> int:
        """
        Delete bars older than a specified timestamp.

        This is the method required by the interface.

        Args:
            before: Delete bars before this time

        Returns:
            Number of bars deleted

        Raises:
            RepositoryError: If delete operation fails
        """
        return await self.delete_bars_before(before)

    async def get_symbols_with_data(self) -> list[str]:
        """
        Get list of symbols that have stored data.

        Returns:
            List of symbol strings

        Raises:
            RepositoryError: If retrieval operation fails
        """
        # Query all timeframe tables for symbols
        query = """
            SELECT DISTINCT symbol FROM (
                SELECT symbol FROM market_data_1m
                UNION
                SELECT symbol FROM market_data_5m
                UNION
                SELECT symbol FROM market_data_15m
                UNION
                SELECT symbol FROM market_data_30m
                UNION
                SELECT symbol FROM market_data_1h
            ) AS all_symbols
            ORDER BY symbol
        """

        try:
            symbols = await self._adapter.fetch_values(query)
            logger.debug(f"Found {len(symbols)} symbols with data")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get symbols with data: {e}")
            raise RepositoryError(f"Failed to get symbols with data: {e}") from e

    async def get_data_range(self, symbol: str) -> tuple[datetime, datetime] | None:
        """
        Get the date range of available data for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp) or None if no data

        Raises:
            RepositoryError: If retrieval operation fails
        """
        # Check across all timeframe tables to find the full data range
        query = """
            SELECT MIN(all_timestamps.timestamp) as earliest, MAX(all_timestamps.timestamp) as latest
            FROM (
                SELECT timestamp FROM market_data_1m WHERE symbol = %s
                UNION ALL
                SELECT timestamp FROM market_data_5m WHERE symbol = %s
                UNION ALL
                SELECT timestamp FROM market_data_15m WHERE symbol = %s
                UNION ALL
                SELECT timestamp FROM market_data_30m WHERE symbol = %s
                UNION ALL
                SELECT timestamp FROM market_data_1h WHERE symbol = %s
            ) AS all_timestamps
        """

        try:
            # Pass symbol 5 times for each table in the UNION
            row = await self._adapter.fetch_one(query, symbol, symbol, symbol, symbol, symbol)

            if row and row["earliest"] and row["latest"]:
                return (row["earliest"], row["latest"])

            logger.debug(f"No data range found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Failed to get data range for {symbol}: {e}")
            raise RepositoryError(f"Failed to get data range: {e}") from e

    def _get_table_name(self, timeframe: str) -> str:
        """Map timeframe to table name."""
        timeframe_map = {
            "1min": "market_data_1m",
            "5min": "market_data_5m",
            "15min": "market_data_15m",
            "30min": "market_data_30m",
            "1hour": "market_data_1h",
            "1day": "market_data_1h",  # Daily data also goes in 1h table with interval='1day'
        }
        return timeframe_map.get(timeframe, "market_data_1m")
    
    def _ensure_utc(self, dt: datetime) -> datetime:
        """
        Ensure a datetime is in UTC timezone.

        Args:
            dt: Datetime to convert

        Returns:
            Datetime in UTC timezone
        """
        if dt.tzinfo is None:
            # Assume naive datetime is in local timezone
            return dt.replace(tzinfo=UTC)
        else:
            # Convert to UTC if not already
            return dt.astimezone(UTC)

    def _row_to_bar(self, row: Any, timeframe: str = "1min") -> Bar:
        """
        Convert a database row to a Bar object.

        Args:
            row: Database row
            timeframe: Bar timeframe (since not stored in DB)

        Returns:
            Bar object
        """
        return Bar(
            symbol=Symbol(row["symbol"]),
            timestamp=row["timestamp"],
            open=Price(row["open"]),
            high=Price(row["high"]),
            low=Price(row["low"]),
            close=Price(row["close"]),
            volume=row["volume"],
            vwap=Price(row["vwap"]) if row.get("vwap") is not None else None,
            trade_count=row.get("trades"),  # Database uses "trades" column
            timeframe=timeframe,  # Not stored in DB, passed as parameter
        )
