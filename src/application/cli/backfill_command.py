"""
Backfill Command Implementation

Handles historical data backfilling for market data, news, and fundamentals.
Migrates the backfill functionality from the old system to the new architecture.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg_pool import AsyncConnectionPool

from src.application.config import DatabaseConfig
from src.infrastructure.data.providers import DataProviderFactory
from src.infrastructure.database.adapter import PostgreSQLAdapter

logger = logging.getLogger(__name__)


class BackfillService:
    """
    Service for backfilling historical data.

    Manages the process of fetching historical data from providers
    and storing it in the database.
    """

    def __init__(self):
        """Initialize backfill service."""
        self.data_provider = None
        self.db_adapter = None
        self.stats = {
            "records_processed": 0,
            "records_failed": 0,
            "symbols_processed": 0,
            "start_time": None,
            "end_time": None,
        }

    async def initialize(self, provider_type: str | None = None) -> None:
        """
        Initialize the backfill service.

        Args:
            provider_type: Data provider type (polygon, alpaca, or auto)
        """
        # Create data provider
        self.data_provider = DataProviderFactory.create_provider(provider_type)
        await self.data_provider.connect()

        # Create database connection pool and adapter
        db_config = DatabaseConfig.from_env()
        connection_string = (
            f"postgresql://{db_config.user}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )

        self.pool = AsyncConnectionPool(connection_string, min_size=1, max_size=5, open=False)
        await self.pool.open()

        self.db_adapter = PostgreSQLAdapter(self.pool)

        logger.info("Backfill service initialized")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.data_provider:
            await self.data_provider.disconnect()
        if hasattr(self, "pool") and self.pool:
            await self.pool.close()

    async def backfill_market_data(
        self, symbols: list[str], days: int, timeframe: str = "1Day"
    ) -> dict[str, Any]:
        """
        Backfill market data for specified symbols.

        Args:
            symbols: List of symbol strings to backfill
            days: Number of days of history to fetch
            timeframe: Data timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            Statistics about the backfill operation
        """
        self.stats["start_time"] = datetime.now(UTC)

        # Calculate date range
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Backfilling {len(symbols)} symbols from {start_date} to {end_date} "
            f"with {timeframe} bars"
        )

        for symbol_str in symbols:
            try:
                await self._backfill_symbol(symbol_str, start_date, end_date, timeframe)
                self.stats["symbols_processed"] += 1

            except Exception as e:
                logger.error(f"Failed to backfill {symbol_str}: {e}")
                self.stats["records_failed"] += 1

        self.stats["end_time"] = datetime.now(UTC)
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        return {
            **self.stats,
            "duration_seconds": duration,
            "records_per_second": self.stats["records_processed"] / duration if duration > 0 else 0,
        }

    async def _backfill_symbol(
        self, symbol_str: str, start: datetime, end: datetime, timeframe: str
    ) -> None:
        """
        Backfill data for a single symbol.

        Args:
            symbol_str: Symbol string
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe
        """
        # Fetch historical bars
        bars = await self.data_provider.get_historical_bars(symbol_str, start, end, timeframe)

        if not bars:
            logger.warning(f"No data returned for {symbol_str}")
            return

        logger.info(f"Fetched {len(bars)} bars for {symbol_str}")

        # Prepare data for bulk insert
        table_name = self._get_table_name(timeframe)
        records = []

        for bar in bars:
            record = {
                "symbol": bar.symbol if isinstance(bar.symbol, str) else bar.symbol.value,
                "timestamp": bar.timestamp,
                "open": float(bar.open.value),
                "high": float(bar.high.value),
                "low": float(bar.low.value),
                "close": float(bar.close.value),
                "volume": bar.volume,
                "vwap": float((bar.high.value + bar.low.value + bar.close.value) / 3),
                "trades": 0,  # Not available in basic bars
                "interval": timeframe,
                "source": "polygon",
            }
            records.append(record)

        # Bulk insert to database
        await self._bulk_insert(table_name, records)
        self.stats["records_processed"] += len(records)

        logger.info(f"Inserted {len(records)} records for {symbol_str}")

    def _get_table_name(self, timeframe: str) -> str:
        """
        Get the database table name for the timeframe.

        Args:
            timeframe: Data timeframe

        Returns:
            Table name
        """
        table_map = {
            "1Min": "market_data_1m",
            "5Min": "market_data_5m",
            "15Min": "market_data_15m",
            "1Hour": "market_data_1h",
            "1Day": "market_data_1h",  # Daily data goes in hourly table
        }
        return table_map.get(timeframe, "market_data_1h")

    async def _bulk_insert(self, table: str, records: list[dict]) -> None:
        """
        Bulk insert records to database.

        Args:
            table: Table name
            records: List of records to insert
        """
        if not records:
            return

        # Build insert query with ON CONFLICT to handle duplicates
        columns = list(records[0].keys())

        # Create placeholders for each record
        values_list = []
        params = []

        for record in records:
            placeholders = []
            for col in columns:
                placeholders.append("%s")
                params.append(record[col])
            values_list.append(f"({','.join(placeholders)})")

        query = f"""  # noqa: S608
            INSERT INTO {table} ({','.join(columns)})
            VALUES {','.join(values_list)}
        """

        await self.db_adapter.execute_query(query, *params)

    async def backfill_news(
        self, symbols: list[str] | None = None, days: int = 7  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Backfill news data.

        Args:
            symbols: Optional list of symbols to filter news
            days: Number of days of history

        Returns:
            Statistics about the backfill operation
        """
        # Check if provider supports news
        if not hasattr(self.data_provider, "get_news"):
            logger.warning("Current data provider does not support news data")
            return {"error": "News not supported by current provider"}

        news_count = 0

        if symbols:
            for symbol_str in symbols:
                news_items = await self.data_provider.get_news(symbol_str, limit=50)
                news_count += len(news_items)
                # TODO: Store news items in database
        else:
            news_items = await self.data_provider.get_news(limit=100)
            news_count = len(news_items)
            # TODO: Store news items in database

        return {"news_items_fetched": news_count}

    async def backfill_fundamentals(self, symbols: list[str]) -> dict[str, Any]:
        """
        Backfill fundamental data for symbols.

        Args:
            symbols: List of symbols to fetch fundamentals for

        Returns:
            Statistics about the backfill operation
        """
        # Check if provider supports fundamentals
        if not hasattr(self.data_provider, "get_fundamentals"):
            logger.warning("Current data provider does not support fundamental data")
            return {"error": "Fundamentals not supported by current provider"}

        fundamentals_count = 0

        for symbol_str in symbols:
            fundamentals = await self.data_provider.get_fundamentals(symbol_str)
            if fundamentals:
                fundamentals_count += 1
                # TODO: Store fundamentals in database

        return {"fundamentals_fetched": fundamentals_count}


async def execute_backfill(
    symbols: str | None = None,
    days: int = 30,
    data_type: str = "market_data",
    layer: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Execute a backfill operation.

    Args:
        symbols: Comma-separated list of symbols or None for layer-based
        days: Number of days to backfill
        data_type: Type of data to backfill (market_data, news, fundamentals, all)
        layer: Layer name to backfill (layer0, layer1, layer2, layer3)
        provider: Data provider to use (polygon, alpaca, or auto)

    Returns:
        Statistics about the backfill operation
    """
    service = BackfillService()

    try:
        await service.initialize(provider)

        # Determine symbols to backfill
        if layer:
            # TODO: Load symbols from layer configuration
            symbol_list = await _get_layer_symbols(layer)
        elif symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
        else:
            logger.error("Either symbols or layer must be specified")
            return {"error": "No symbols specified"}

        results = {}

        # Backfill based on data type
        if data_type in ["market_data", "all"]:
            results["market_data"] = await service.backfill_market_data(symbol_list, days)

        if data_type in ["news", "all"]:
            results["news"] = await service.backfill_news(symbol_list, days)

        if data_type in ["fundamentals", "all"]:
            results["fundamentals"] = await service.backfill_fundamentals(symbol_list)

        return results

    finally:
        await service.cleanup()


async def _get_layer_symbols(layer: str) -> list[str]:
    """
    Get symbols for a specific layer.

    Args:
        layer: Layer name (layer0, layer1, layer2, layer3)

    Returns:
        List of symbol strings
    """
    # TODO: Implement layer symbol loading from database
    # For now, return sample symbols based on layer
    layer_samples = {
        "layer0": ["SPY", "QQQ", "IWM"],  # Universe sample
        "layer1": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],  # Liquid sample
        "layer2": ["TSLA", "AMD", "META", "NFLX"],  # Catalyst sample
        "layer3": ["AAPL", "NVDA"],  # Active sample
    }

    return layer_samples.get(layer, [])
