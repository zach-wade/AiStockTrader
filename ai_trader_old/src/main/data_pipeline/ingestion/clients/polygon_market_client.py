"""
Polygon Market Data Client - Refactored

Simplified client for fetching OHLCV market data from Polygon.io API.
Uses PolygonApiHandler for common functionality.
"""

# Standard library imports
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.services.ingestion.polygon_api_handler import PolygonApiHandler
from main.data_pipeline.types import TimeInterval
from main.utils.core import ensure_utc, get_logger
from main.utils.monitoring import MetricType, record_metric, timer

from .base_client import BaseIngestionClient, ClientConfig, FetchResult


class PolygonMarketClient(BaseIngestionClient[list[dict[str, Any]]]):
    """
    Simplified client for fetching market data from Polygon.io.

    Delegates common functionality to PolygonApiHandler.
    """

    def __init__(
        self, api_key: str, layer: DataLayer = DataLayer.BASIC, config: ClientConfig | None = None
    ):
        """Initialize the Polygon market client."""
        self.api_handler = PolygonApiHandler()

        # Create config using handler with layer-based configuration
        config = self.api_handler.create_polygon_config(api_key=api_key, layer=layer, config=config)

        super().__init__(config)
        self.layer = layer
        self.logger = get_logger(__name__)
        self.logger.info(f"PolygonMarketClient initialized with layer: {layer.name}")

    def get_base_url(self) -> str:
        """Get the base URL for Polygon API."""
        return self.config.base_url

    def get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return self.api_handler.get_standard_headers(self.config.api_key)

    async def validate_response(self, response) -> bool:
        """Validate Polygon API response."""
        return await self.api_handler.validate_http_response(response)

    async def parse_response(self, response) -> list[dict[str, Any]]:
        """Parse Polygon API response into standardized format."""
        results = await self.api_handler.parse_polygon_response(response)

        parsed_results = []
        for record in results:
            parsed_results.append(
                {
                    "timestamp": datetime.fromtimestamp(record["t"] / 1000, tz=UTC),
                    "open": float(record["o"]),
                    "high": float(record["h"]),
                    "low": float(record["l"]),
                    "close": float(record["c"]),
                    "volume": int(record["v"]),
                    "vwap": float(record.get("vw", 0)),
                    "trades": int(record.get("n", 0)),
                }
            )

        return parsed_results

    async def fetch_aggregates(
        self,
        symbol: str,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
        adjusted: bool = True,
    ) -> FetchResult[list[dict[str, Any]]]:
        """Fetch OHLCV aggregates for a symbol."""
        # Convert interval to Polygon format
        multiplier, timespan = self._convert_interval_to_polygon(interval)

        # Ensure dates are UTC
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)

        # Format dates for API
        from_date = int(start_date.timestamp() * 1000)
        to_date = int(end_date.timestamp() * 1000)

        # Build endpoint and params
        endpoint = f"v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": "50000"}

        # Track API call performance
        with timer(
            "polygon.market_data.fetch", tags={"symbol": symbol, "interval": interval.value}
        ):
            # Use handler for pagination
            result = await self.api_handler.fetch_with_pagination(
                self, endpoint, params, limit=None, max_pages=10
            )

        if result.success and result.data:
            # Add symbol and interval to each record
            for record in result.data:
                record["symbol"] = symbol
                record["interval"] = interval.value

            # Track records fetched
            record_metric(
                "polygon.market_data.records",
                len(result.data),
                MetricType.COUNTER,
                tags={"symbol": symbol, "interval": interval.value},
            )
        else:
            # Track API errors
            record_metric(
                "polygon.api.errors",
                1,
                MetricType.COUNTER,
                tags={
                    "data_type": "market_data",
                    "symbol": symbol,
                    "error": result.error or "unknown",
                },
            )

        return result

    async def fetch_latest_price(self, symbol: str) -> FetchResult[dict[str, Any]]:
        """Fetch the latest price for a symbol."""
        endpoint = f"v2/aggs/ticker/{symbol}/prev"

        # Track API call performance
        with timer("polygon.latest_price.fetch", tags={"symbol": symbol}):
            result = await self.fetch(endpoint, {})

        if result.success and result.data:
            if len(result.data) > 0:
                latest = result.data[0]
                latest["symbol"] = symbol

                # Track successful price fetch
                record_metric(
                    "polygon.latest_price.success", 1, MetricType.COUNTER, tags={"symbol": symbol}
                )

                # Track price as metric
                if "close" in latest:
                    record_metric(
                        "polygon.latest_price.value",
                        latest["close"],
                        MetricType.GAUGE,
                        tags={"symbol": symbol},
                    )

                return FetchResult(success=True, data=latest, metadata=result.metadata)

        # Track failures
        record_metric(
            "polygon.latest_price.failure",
            1,
            MetricType.COUNTER,
            tags={"symbol": symbol, "reason": "no_data"},
        )

        return FetchResult(success=False, error=f"No data found for {symbol}")

    async def fetch_multiple_symbols(
        self,
        symbols: list[str],
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
        max_concurrent: int = 5,
    ) -> dict[str, FetchResult[list[dict[str, Any]]]]:
        """Fetch data for multiple symbols concurrently using the handler."""

        async def fetch_symbol(symbol: str) -> FetchResult:
            return await self.fetch_aggregates(symbol, interval, start_date, end_date)

        # Track batch operation
        batch_start = datetime.now()
        record_metric(
            "polygon.batch.symbols_requested",
            len(symbols),
            MetricType.GAUGE,
            tags={"interval": interval.value},
        )

        # Use handler's batch_fetch
        results = await self.api_handler.batch_fetch(
            fetch_symbol, symbols, batch_size=50, max_concurrent=max_concurrent
        )

        # Calculate batch metrics
        batch_duration = (datetime.now() - batch_start).total_seconds()
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful

        # Record batch metrics
        record_metric(
            "polygon.batch.duration",
            batch_duration,
            MetricType.HISTOGRAM,
            tags={"interval": interval.value, "symbols": len(symbols)},
        )
        record_metric(
            "polygon.batch.success_rate",
            successful / len(symbols) if symbols else 0,
            MetricType.GAUGE,
            tags={"interval": interval.value},
        )

        if failed > 0:
            record_metric(
                "polygon.batch.failures",
                failed,
                MetricType.COUNTER,
                tags={"interval": interval.value},
            )
            self.logger.warning(f"Batch fetch failed for {failed}/{len(symbols)} symbols")

        return results

    def _convert_interval_to_polygon(self, interval: TimeInterval) -> tuple[int, str]:
        """Convert our TimeInterval enum to Polygon format."""
        mapping = {
            TimeInterval.ONE_MINUTE: (1, "minute"),
            TimeInterval.FIVE_MINUTES: (5, "minute"),
            TimeInterval.FIFTEEN_MINUTES: (15, "minute"),
            TimeInterval.THIRTY_MINUTES: (30, "minute"),
            TimeInterval.ONE_HOUR: (1, "hour"),
            TimeInterval.ONE_DAY: (1, "day"),
            TimeInterval.ONE_WEEK: (1, "week"),
            TimeInterval.ONE_MONTH: (1, "month"),
        }
        return mapping.get(interval, (1, "day"))
