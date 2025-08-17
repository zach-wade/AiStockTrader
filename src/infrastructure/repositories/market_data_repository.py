"""
Market Data Repository Implementation

PostgreSQL implementation of the IMarketDataRepository interface.
Handles persistence and retrieval of market data bars with optimized batch operations.
"""

# Standard library imports
from datetime import UTC, datetime
import logging
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
        query = """
            INSERT INTO market_data (
                id, symbol, timeframe, timestamp,
                open, high, low, close, volume,
                vwap, trade_count, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap,
                trade_count = EXCLUDED.trade_count,
                updated_at = NOW()
        """

        try:
            # Convert timestamp to UTC if needed
            timestamp_utc = self._ensure_utc(bar.timestamp)

            await self._adapter.execute_query(
                query,
                str(uuid4()),
                bar.symbol.value,
                bar.timeframe,
                timestamp_utc,
                float(bar.open.value),
                float(bar.high.value),
                float(bar.low.value),
                float(bar.close.value),
                bar.volume,
                float(bar.vwap.value) if bar.vwap else None,
                bar.trade_count,
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

        # Prepare batch data
        batch_data = []
        for bar in bars:
            timestamp_utc = self._ensure_utc(bar.timestamp)
            batch_data.append(
                (
                    str(uuid4()),
                    bar.symbol.value,
                    bar.timeframe,
                    timestamp_utc,
                    float(bar.open.value),
                    float(bar.high.value),
                    float(bar.low.value),
                    float(bar.close.value),
                    bar.volume,
                    float(bar.vwap.value) if bar.vwap else None,
                    bar.trade_count,
                )
            )

        # Use a more efficient approach with COPY or batch insert
        query = """
            INSERT INTO market_data (
                id, symbol, timeframe, timestamp,
                open, high, low, close, volume,
                vwap, trade_count, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap,
                trade_count = EXCLUDED.trade_count,
                updated_at = NOW()
        """

        try:
            await self._adapter.execute_batch(query, batch_data)
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
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close,
                   volume, vwap, trade_count
            FROM market_data
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT 1
        """

        try:
            row = await self._adapter.fetch_one(query, symbol, timeframe)

            if row:
                return self._row_to_bar(row)

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
        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close,
                   volume, vwap, trade_count
            FROM market_data
            WHERE symbol = $1
                AND timeframe = $2
                AND timestamp >= $3
                AND timestamp <= $4
            ORDER BY timestamp ASC
        """

        try:
            # Ensure timestamps are UTC
            start_utc = self._ensure_utc(start)
            end_utc = self._ensure_utc(end)

            rows = await self._adapter.fetch_all(query, symbol, timeframe, start_utc, end_utc)

            bars = [self._row_to_bar(row) for row in rows]
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
        if end is None:
            # Query without end date condition
            query = """
                SELECT * FROM (
                    SELECT symbol, timeframe, timestamp, open, high, low, close,
                           volume, vwap, trade_count
                    FROM market_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                ) AS recent_bars
                ORDER BY timestamp ASC
            """
            params = [symbol, timeframe, count]
        else:
            # Query with end date condition
            end_utc = self._ensure_utc(end)
            query = """
                SELECT * FROM (
                    SELECT symbol, timeframe, timestamp, open, high, low, close,
                           volume, vwap, trade_count
                    FROM market_data
                    WHERE symbol = $1 AND timeframe = $2 AND timestamp <= $3
                    ORDER BY timestamp DESC
                    LIMIT $4
                ) AS recent_bars
                ORDER BY timestamp ASC
            """
            params = [symbol, timeframe, end_utc, count]

        try:
            rows = await self._adapter.fetch_all(query, *params)

            bars = [self._row_to_bar(row) for row in rows]
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
        query = """
            DELETE FROM market_data
            WHERE timestamp < $1
        """

        try:
            timestamp_utc = self._ensure_utc(timestamp)
            result = await self._adapter.execute_query(query, timestamp_utc)

            # Extract row count from result string (e.g., "EXECUTE 42")
            deleted_count = int(result.split()[1]) if result and result.startswith("EXECUTE") else 0

            logger.info(f"Deleted {deleted_count} bars before {timestamp_utc}")
            return deleted_count

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
        query = """
            SELECT DISTINCT symbol
            FROM market_data
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
        query = """
            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM market_data
            WHERE symbol = $1
        """

        try:
            row = await self._adapter.fetch_one(query, symbol)

            if row and row["earliest"] and row["latest"]:
                return (row["earliest"], row["latest"])

            logger.debug(f"No data range found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Failed to get data range for {symbol}: {e}")
            raise RepositoryError(f"Failed to get data range: {e}") from e

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

    def _row_to_bar(self, row: Any) -> Bar:
        """
        Convert a database row to a Bar object.

        Args:
            row: Database row

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
            vwap=Price(row["vwap"]) if row["vwap"] is not None else None,
            trade_count=row["trade_count"],
            timeframe=row["timeframe"],
        )
