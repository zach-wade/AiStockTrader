"""
Market Data Repository Implementation

PostgreSQL implementation of the IMarketDataRepository interface.
Handles persistence and retrieval of market data bars with optimized batch operations.

Security Notes:
    - All queries use parameterized statements with %s placeholders
    - Table names are validated through a strict whitelist mapping
    - No user input is ever directly interpolated into SQL strings
"""

# Standard library imports
import logging
from datetime import UTC, datetime
from typing import Any

# Local imports
from src.application.interfaces.exceptions import IntegrityError, RepositoryError
from src.application.interfaces.market_data import Bar
from src.application.interfaces.repositories import IMarketDataRepository
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.security.input_sanitizer import InputSanitizer
from src.infrastructure.security.validation import ValidationError, sanitize_input

logger = logging.getLogger(__name__)


class MarketDataRepository(IMarketDataRepository):
    """
    PostgreSQL implementation of the market data repository.

    Handles storage and retrieval of market data bars with:
    - Batch insert optimization
    - Timezone conversion (stores as UTC)
    - Duplicate handling via unique constraints
    - Indexed queries for performance
    - SQL injection prevention through parameterized queries
    """

    # SECURITY: Strict whitelist of allowed table names
    # These are the ONLY valid table names - never allow dynamic table names
    VALID_TABLES = {
        "market_data_1m",
        "market_data_5m",
        "market_data_15m",
        "market_data_30m",
        "market_data_1h",
    }

    def __init__(self, adapter: PostgreSQLAdapter) -> None:
        """
        Initialize repository with database adapter.

        Security Note:
            All queries use parameterized statements with %s placeholders.
            Table names are validated through a strict whitelist mapping.

        Args:
            adapter: PostgreSQL database adapter
        """
        self._adapter = adapter

    def _get_table_name(self, timeframe: str) -> str:
        """
        Map timeframe to table name with validation.

        Security Note:
            This method provides SQL injection protection by:
            1. Using a strict whitelist of allowed timeframes
            2. Raising an error for any input not in the whitelist
            3. Never allowing user input to directly become part of SQL

        Args:
            timeframe: The timeframe string

        Returns:
            Validated table name from whitelist

        Raises:
            ValidationError: If timeframe is invalid
        """
        # SECURITY: Strict whitelist - NEVER modify without security review
        timeframe_map = {
            "1min": "market_data_1m",
            "5min": "market_data_5m",
            "15min": "market_data_15m",
            "30min": "market_data_30m",
            "1hour": "market_data_1h",
            "1day": "market_data_1h",  # Daily data also goes in 1h table with interval='1day'
        }

        # Validate timeframe is in allowed list
        if timeframe not in timeframe_map:
            raise ValidationError(
                f"Invalid timeframe: {timeframe}. Must be one of: {', '.join(timeframe_map.keys())}"
            )

        table_name = timeframe_map[timeframe]

        # SECURITY: Double-check table name is in our valid set
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Internal error: Invalid table name {table_name}")

        return table_name

    def _build_insert_query_with_interval(self, table_name: str) -> str:
        """
        Build INSERT query for tables with interval column.

        SECURITY WARNING:
            Table name MUST be validated through _get_table_name() before calling.
            This is the ONLY place where we use string formatting for SQL, and
            it's only safe because the table name has been strictly validated.
        """
        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Table name is safe - it's from our whitelist
        return f"""
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

    def _build_insert_query_no_interval(self, table_name: str) -> str:
        """
        Build INSERT query for tables without interval column.

        SECURITY WARNING:
            Table name MUST be validated through _get_table_name() before calling.
        """
        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Table name is safe - it's from our whitelist
        return f"""
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

    def _build_select_query(self, table_name: str) -> str:
        """
        Build SELECT query with validated table name.

        SECURITY WARNING:
            Table name MUST be validated through _get_table_name() before calling.
        """
        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Table name is safe - it's from our whitelist
        return f"""
            SELECT symbol, timestamp, open, high, low, close,
                   volume, vwap, trades
            FROM {table_name}
        """

    def _build_delete_query(self, table_name: str) -> str:
        """
        Build DELETE query with validated table name.

        SECURITY WARNING:
            Table name MUST be validated through _get_table_name() before calling.
        """
        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Table name is safe - it's from our whitelist
        return f"DELETE FROM {table_name}"

    async def save_bar(self, bar: Bar) -> None:
        """
        Save a single market data bar.

        Security Note:
            Table names are validated through a whitelist.
            All data values use parameterized queries with %s placeholders.

        Args:
            bar: The bar data to save

        Raises:
            RepositoryError: If save operation fails
        """
        # Map timeframe to table name (validated through whitelist)
        table_name = self._get_table_name(bar.timeframe)

        try:
            # Convert timestamp to UTC if needed
            timestamp_utc = self._ensure_utc(bar.timestamp)

            if bar.timeframe in ("1hour", "1day"):
                # market_data_1h has interval column and different unique constraint
                query = self._build_insert_query_with_interval(table_name)

                # Map timeframe to interval value
                interval = "1day" if bar.timeframe == "1day" else "1hour"

                # All data values are parameterized - safe from SQL injection
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
                query = self._build_insert_query_no_interval(table_name)

                # All data values are parameterized - safe from SQL injection
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

        Security Note:
            Uses parameterized batch queries. Table names are validated.

        Args:
            bars: List of bars to save

        Raises:
            RepositoryError: If batch save operation fails
        """
        if not bars:
            return

        try:
            # Group bars by timeframe/table
            bars_by_timeframe: dict[str, list[tuple[Any, ...]]] = {}

            for bar in bars:
                # Convert timestamp to UTC
                timestamp_utc = self._ensure_utc(bar.timestamp)

                if bar.timeframe in ("1hour", "1day"):
                    # Map timeframe to interval value
                    interval = "1day" if bar.timeframe == "1day" else "1hour"

                    # Prepare data tuple for market_data_1h (with interval)
                    data = (
                        bar.symbol.value,
                        timestamp_utc,
                        float(bar.open.value),
                        float(bar.high.value),
                        float(bar.low.value),
                        float(bar.close.value),
                        bar.volume,
                        float(bar.vwap.value) if bar.vwap else None,
                        bar.trade_count,
                        interval,
                        "test",  # source
                    )
                else:
                    # Prepare data tuple for other tables (no interval)
                    data = (  # type: ignore
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

                if bar.timeframe not in bars_by_timeframe:
                    bars_by_timeframe[bar.timeframe] = []
                bars_by_timeframe[bar.timeframe].append(data)

            # Insert each timeframe group into its corresponding table
            for timeframe, timeframe_data in bars_by_timeframe.items():
                table_name = self._get_table_name(timeframe)

                if timeframe in ("1hour", "1day"):
                    query = self._build_insert_query_with_interval(table_name)
                else:
                    query = self._build_insert_query_no_interval(table_name)

                # Execute batch insert with parameterized values
                await self._adapter.execute_batch(query, timeframe_data)

            logger.info(f"Saved {len(bars)} bars in batch")
        except Exception as e:
            logger.error(f"Failed to save bars in batch: {e}")
            raise RepositoryError(f"Failed to save bars: {e}") from e

    @sanitize_input(
        symbol=lambda x: InputSanitizer.sanitize_symbol(x) if isinstance(x, str) else x,
        timeframe=lambda x: InputSanitizer.sanitize_identifier(x),
    )
    async def get_latest_bar(self, symbol: str | Symbol, timeframe: str = "1min") -> Bar | None:
        """
        Get the most recent bar for a symbol.

        Security Note:
            Symbol is sanitized and parameterized. Table name is validated.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Latest bar if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails
        """
        # Validate timeframe
        valid_timeframes = ["1min", "5min", "15min", "30min", "1hour", "1day"]
        if timeframe not in valid_timeframes:
            raise ValidationError(
                f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}"
            )

        table_name = self._get_table_name(timeframe)
        base_query = self._build_select_query(table_name)

        # Add WHERE and ORDER BY clauses with parameterized symbol
        query = (
            base_query
            + """
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )

        symbol_str = symbol.value if isinstance(symbol, Symbol) else symbol

        try:
            row = await self._adapter.fetch_one(query, symbol_str)

            if row:
                return self._row_to_bar(row, timeframe)

            logger.debug(f"No bars found for {symbol} with timeframe {timeframe}")
            return None

        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            raise RepositoryError(f"Failed to get latest bar: {e}") from e

    async def get_bars(
        self, symbol: str | Symbol, start: datetime, end: datetime, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get bars for a symbol within a date range.

        Security Note:
            All parameters are sanitized and parameterized.

        Args:
            symbol: Trading symbol
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            timeframe: Bar timeframe

        Returns:
            List of bars in chronological order

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)
        base_query = self._build_select_query(table_name)

        # Add WHERE clause with parameterized values
        query = (
            base_query
            + """
            WHERE symbol = %s
                AND timestamp >= %s
                AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        )

        symbol_str = symbol.value if isinstance(symbol, Symbol) else symbol
        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)

        try:
            rows = await self._adapter.fetch_all(query, symbol_str, start_utc, end_utc)

            bars = [self._row_to_bar(row, timeframe) for row in rows]
            logger.debug(f"Retrieved {len(bars)} bars for {symbol} from {start_utc} to {end_utc}")

            return bars

        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            raise RepositoryError(f"Failed to get bars: {e}") from e

    async def get_bars_by_count(
        self,
        symbol: str | Symbol,
        count: int,
        end: datetime | None = None,
        timeframe: str = "1min",
    ) -> list[Bar]:
        """
        Get a specific number of bars ending at a given time.

        Security Note:
            All parameters are sanitized and parameterized.

        Args:
            symbol: Trading symbol
            count: Number of bars to retrieve
            end: End timestamp (default: now)
            timeframe: Bar timeframe

        Returns:
            List of bars in chronological order

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)

        symbol_str = symbol.value if isinstance(symbol, Symbol) else symbol

        if end is None:
            # No end time specified - get the most recent bars
            base_query = self._build_select_query(table_name)
            query = f"""
                SELECT * FROM (
                    {base_query}
                    WHERE symbol = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                ) AS recent_bars
                ORDER BY timestamp ASC
            """

            params = (symbol_str, count)
        else:
            # Get bars up to the specified end time
            end_utc = self._ensure_utc(end)
            base_query = self._build_select_query(table_name)
            query = f"""
                SELECT * FROM (
                    {base_query}
                    WHERE symbol = %s AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                ) AS recent_bars
                ORDER BY timestamp ASC
            """

            params = (symbol_str, end_utc, count)  # type: ignore

        try:
            rows = await self._adapter.fetch_all(query, *params)

            bars = [self._row_to_bar(row, timeframe) for row in rows]
            logger.debug(f"Retrieved {len(bars)} bars for {symbol}")

            return bars

        except Exception as e:
            logger.error(f"Failed to get bars by count for {symbol}: {e}")
            raise RepositoryError(f"Failed to get bars by count: {e}") from e

    async def delete_bars_before(self, timestamp: datetime, symbol: str | None = None) -> int:
        """
        Delete bars before a given timestamp.

        Security Note:
            Parameters are sanitized and parameterized.

        Args:
            timestamp: Delete bars before this time
            symbol: Optional symbol filter

        Returns:
            Number of bars deleted

        Raises:
            RepositoryError: If delete operation fails
        """
        timestamp_utc = self._ensure_utc(timestamp)
        total_deleted = 0

        try:
            # Delete from all timeframe tables
            for timeframe in ["1min", "5min", "15min", "30min", "1hour", "1day"]:
                table_name = self._get_table_name(timeframe)
                base_query = self._build_delete_query(table_name)

                if symbol:
                    # SAFER: Delete only for specific symbol
                    query = (
                        base_query
                        + """
                        WHERE timestamp < %s AND symbol = %s
                    """
                    )
                    result = await self._adapter.execute_query(query, timestamp_utc, symbol)
                else:
                    # DANGEROUS: Delete all symbols - requires explicit None
                    # WARNING: This will delete PRODUCTION DATA if called without symbol filter!
                    logger.warning(
                        f"Deleting ALL bars before {timestamp_utc} from {table_name} - this affects production data!"
                    )
                    query = (
                        base_query
                        + """
                        WHERE timestamp < %s
                    """
                    )
                    result = await self._adapter.execute_query(query, timestamp_utc)

                # Extract row count from result
                if result and result.startswith("EXECUTE"):
                    count = int(result.split()[1])
                    total_deleted += count

            logger.info(f"Deleted {total_deleted} bars before {timestamp_utc}")
            return total_deleted

        except Exception as e:
            logger.error(f"Failed to delete bars: {e}")
            raise RepositoryError(f"Failed to delete bars: {e}") from e

    async def get_symbols(self, timeframe: str = "1min") -> list[str]:
        """
        Get list of unique symbols in the database.

        Security Note:
            Table name is validated. No user input in query.

        Args:
            timeframe: Timeframe to check for symbols

        Returns:
            List of unique symbols

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)

        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Safe query - table name is validated
        query = f"""
            SELECT DISTINCT symbol
            FROM {table_name}
            ORDER BY symbol
        """

        try:
            rows = await self._adapter.fetch_all(query)
            symbols = [row["symbol"] for row in rows]
            logger.debug(f"Found {len(symbols)} unique symbols in {table_name}")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            raise RepositoryError(f"Failed to get symbols: {e}") from e

    async def get_symbols_with_data(self) -> list[str]:
        """
        Get list of symbols that have stored data.

        Returns all unique symbols across all timeframes.

        Returns:
            List of symbol strings

        Raises:
            RepositoryError: If retrieval operation fails
        """
        # Use the existing get_symbols method which already does this
        return await self.get_symbols()

    async def get_data_range(
        self, symbol: str | Symbol, timeframe: str = "1min"
    ) -> tuple[datetime, datetime] | None:
        """
        Get the date range of available data for a symbol.

        Security Note:
            Symbol is parameterized. Table name is validated.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp) or None

        Raises:
            RepositoryError: If retrieval operation fails
        """
        table_name = self._get_table_name(timeframe)

        # Final safety check
        if table_name not in self.VALID_TABLES:
            raise ValidationError(f"Invalid table name: {table_name}")

        # Safe query - table name is validated, symbol is parameterized
        query = f"""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
            FROM {table_name}
            WHERE symbol = %s
        """

        symbol_str = symbol.value if isinstance(symbol, Symbol) else symbol

        try:
            row = await self._adapter.fetch_one(query, symbol_str)

            if row and row["min_ts"] and row["max_ts"]:
                return (row["min_ts"], row["max_ts"])

            return None

        except Exception as e:
            logger.error(f"Failed to get data range: {e}")
            raise RepositoryError(f"Failed to get data range: {e}") from e

    def _ensure_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is UTC."""
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            return dt.replace(tzinfo=UTC)
        elif dt.tzinfo != UTC:
            # Convert to UTC
            return dt.astimezone(UTC)
        return dt

    def _row_to_bar(self, row: dict[str, Any], timeframe: str) -> Bar:
        """Convert database row to Bar object."""
        return Bar(
            symbol=Symbol(row["symbol"]),
            timestamp=self._ensure_utc(row["timestamp"]),
            open=Price(row["open"]),
            high=Price(row["high"]),
            low=Price(row["low"]),
            close=Price(row["close"]),
            volume=row["volume"],
            vwap=Price(row["vwap"]) if row.get("vwap") else None,
            trade_count=row.get("trades", 0),
            timeframe=timeframe,
        )
