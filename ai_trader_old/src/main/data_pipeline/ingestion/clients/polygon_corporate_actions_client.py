"""
Polygon Corporate Actions Client - Refactored

Simplified client for fetching corporate actions (dividends and splits) from Polygon.io API.
Uses PolygonApiHandler for common functionality.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.services.ingestion.polygon_api_handler import PolygonApiHandler
from main.utils.core import get_logger
from main.utils.monitoring import MetricType, record_metric, timer

from .base_client import BaseIngestionClient, ClientConfig, FetchResult


class PolygonCorporateActionsClient(BaseIngestionClient[list[dict[str, Any]]]):
    """
    Simplified client for fetching corporate actions from Polygon.io.

    Delegates common functionality to PolygonApiHandler.
    """

    def __init__(
        self, api_key: str, layer: DataLayer = DataLayer.BASIC, config: ClientConfig | None = None
    ):
        """Initialize the Polygon corporate actions client."""
        self.api_handler = PolygonApiHandler()

        # Create config using handler with layer-based configuration
        # Corporate actions use custom cache TTL regardless of layer
        config = self.api_handler.create_polygon_config(
            api_key=api_key,
            layer=layer,
            config=config,
            cache_ttl_seconds=3600,  # Cache corporate actions for 1 hour
        )

        super().__init__(config)
        self.layer = layer
        self.logger = get_logger(__name__)
        self.logger.info(f"PolygonCorporateActionsClient initialized with layer: {layer.name}")

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

        # Normalize based on endpoint (dividends or splits)
        normalized = []
        for record in results:
            # Check if it's a dividend or split based on fields present
            if "ex_dividend_date" in record:
                normalized.append(self._normalize_dividend(record))
            elif "execution_date" in record:
                normalized.append(self._normalize_split(record))
            else:
                normalized.append(record)

        return normalized

    async def fetch_dividends(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> FetchResult[list[dict[str, Any]]]:
        """Fetch dividend records for a symbol."""
        endpoint = "v3/reference/dividends"

        # Build params using handler
        params = self.api_handler.build_date_params(
            start_date, end_date, date_field="ex_dividend_date"
        )
        params.update(
            {
                "ticker": symbol.upper(),
                "limit": str(min(limit, 1000)),
                "order": "asc",
                "sort": "ex_dividend_date",
            }
        )

        # Track API call performance
        with timer("polygon.corporate_actions.dividends.fetch", tags={"symbol": symbol}):
            # Use handler for pagination
            result = await self.api_handler.fetch_with_pagination(
                self, endpoint, params, limit=limit, max_pages=10
            )

        if result.success and result.data:
            # Ensure all records are normalized as dividends
            for record in result.data:
                record["action_type"] = "dividend"

            # Track dividends fetched
            record_metric(
                "polygon.corporate_actions.dividends",
                len(result.data),
                MetricType.COUNTER,
                tags={"symbol": symbol},
            )

            # Track dividend yield if available
            if result.data:
                cash_amounts = [
                    r.get("cash_amount", 0) for r in result.data if r.get("cash_amount")
                ]
                if cash_amounts:
                    avg_dividend = sum(cash_amounts) / len(cash_amounts)
                    gauge(
                        "polygon.corporate_actions.avg_dividend",
                        avg_dividend,
                        tags={"symbol": symbol},
                    )
        else:
            # Track API errors
            record_metric(
                "polygon.api.errors",
                1,
                MetricType.COUNTER,
                tags={
                    "data_type": "dividends",
                    "symbol": symbol,
                    "error": result.error or "unknown",
                },
            )

        return result

    async def fetch_splits(
        self, symbol: str, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> FetchResult[list[dict[str, Any]]]:
        """Fetch stock split records for a symbol."""
        endpoint = "v3/reference/splits"

        # Build params using handler
        params = self.api_handler.build_date_params(
            start_date, end_date, date_field="execution_date"
        )
        params.update(
            {
                "ticker": symbol.upper(),
                "limit": str(min(limit, 1000)),
                "order": "asc",
                "sort": "execution_date",
            }
        )

        # Track API call performance
        with timer("polygon.corporate_actions.splits.fetch", tags={"symbol": symbol}):
            # Use handler for pagination
            result = await self.api_handler.fetch_with_pagination(
                self, endpoint, params, limit=limit, max_pages=10
            )

        if result.success and result.data:
            # Ensure all records are normalized as splits
            for record in result.data:
                record["action_type"] = "split"

            # Track splits fetched
            record_metric(
                "polygon.corporate_actions.splits",
                len(result.data),
                MetricType.COUNTER,
                tags={"symbol": symbol},
            )

            # Track split ratios
            for split in result.data:
                if split.get("split_from") and split.get("split_to"):
                    ratio = (
                        split["split_to"] / split["split_from"] if split["split_from"] > 0 else 0
                    )
                    record_metric(
                        "polygon.corporate_actions.split_ratio",
                        ratio,
                        MetricType.HISTOGRAM,
                        tags={"symbol": symbol},
                    )
        else:
            # Track API errors
            record_metric(
                "polygon.api.errors",
                1,
                MetricType.COUNTER,
                tags={"data_type": "splits", "symbol": symbol, "error": result.error or "unknown"},
            )

        return result

    async def fetch_corporate_actions(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        include_dividends: bool = True,
        include_splits: bool = True,
    ) -> FetchResult[list[dict[str, Any]]]:
        """Fetch all corporate actions for a symbol."""
        all_actions = []
        errors = []

        # Track combined fetch
        with timer("polygon.corporate_actions.combined.fetch", tags={"symbol": symbol}):
            # Fetch dividends and splits in parallel
            tasks = []
            if include_dividends:
                tasks.append(self.fetch_dividends(symbol, start_date, end_date))
            if include_splits:
                tasks.append(self.fetch_splits(symbol, start_date, end_date))

            if not tasks:
                return FetchResult(success=False, error="No action types selected")

            # Execute tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif isinstance(result, FetchResult):
                    if result.success and result.data:
                        all_actions.extend(result.data)
                    elif result.error:
                        errors.append(result.error)

            # Sort by date
            all_actions.sort(key=lambda x: x.get("ex_date", ""))

        # Track metrics
        dividend_count = len([a for a in all_actions if a.get("action_type") == "dividend"])
        split_count = len([a for a in all_actions if a.get("action_type") == "split"])

        gauge("polygon.corporate_actions.total_actions", len(all_actions), tags={"symbol": symbol})
        gauge("polygon.corporate_actions.dividend_count", dividend_count, tags={"symbol": symbol})
        gauge("polygon.corporate_actions.split_count", split_count, tags={"symbol": symbol})

        if errors:
            record_metric(
                "polygon.corporate_actions.errors",
                len(errors),
                MetricType.COUNTER,
                tags={"symbol": symbol},
            )

        return FetchResult(
            success=len(all_actions) > 0,
            data=all_actions,
            metadata={
                "symbol": symbol,
                "dividend_count": dividend_count,
                "split_count": split_count,
                "errors": errors if errors else None,
            },
        )

    async def fetch_multiple_symbols(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        include_dividends: bool = True,
        include_splits: bool = True,
        max_concurrent: int = 5,
    ) -> dict[str, FetchResult[list[dict[str, Any]]]]:
        """Fetch corporate actions for multiple symbols using the handler."""

        async def fetch_symbol(symbol: str) -> FetchResult:
            return await self.fetch_corporate_actions(
                symbol, start_date, end_date, include_dividends, include_splits
            )

        # Track batch operation
        batch_start = datetime.now()
        gauge("polygon.corporate_actions.batch_symbols", len(symbols))

        # Use handler's batch_fetch
        results = await self.api_handler.batch_fetch(
            fetch_symbol, symbols, batch_size=50, max_concurrent=max_concurrent
        )

        # Calculate batch metrics
        batch_duration = (datetime.now() - batch_start).total_seconds()
        successful = sum(1 for r in results.values() if r.success)
        total_actions = sum(len(r.data) for r in results.values() if r.success and r.data)

        # Record batch metrics
        record_metric(
            "polygon.corporate_actions.batch_duration",
            batch_duration,
            MetricType.HISTOGRAM,
            tags={"symbols": len(symbols)},
        )
        record_metric(
            "polygon.corporate_actions.batch_success_rate",
            successful / len(symbols) if symbols else 0,
            MetricType.GAUGE,
        )
        record_metric(
            "polygon.corporate_actions.batch_total_actions", total_actions, MetricType.COUNTER
        )

        if len(symbols) - successful > 0:
            self.logger.warning(
                f"Corporate actions batch fetch failed for {len(symbols) - successful}/{len(symbols)} symbols"
            )

        return results

    def _normalize_dividend(self, dividend: dict[str, Any]) -> dict[str, Any]:
        """Normalize a dividend record."""
        # Parse dates
        ex_date = self._parse_date(dividend.get("ex_dividend_date"))
        record_date = self._parse_date(dividend.get("record_date"))
        payment_date = self._parse_date(dividend.get("pay_date"))
        declaration_date = self._parse_date(dividend.get("declaration_date"))

        return {
            "ticker": dividend.get("ticker", ""),
            "action_type": "dividend",
            "ex_date": ex_date,
            "ex_dividend_date": ex_date,  # Keep original field name
            "record_date": record_date,
            "payment_date": payment_date,
            "declaration_date": declaration_date,
            "cash_amount": dividend.get("cash_amount"),
            "dividend_type": dividend.get("dividend_type"),
            "frequency": dividend.get("frequency"),
            "currency": dividend.get("currency", "USD"),
            "raw_data": dividend,
        }

    def _normalize_split(self, split: dict[str, Any]) -> dict[str, Any]:
        """Normalize a stock split record."""
        # Parse dates
        execution_date = self._parse_date(split.get("execution_date"))

        return {
            "ticker": split.get("ticker", ""),
            "action_type": "split",
            "ex_date": execution_date,
            "execution_date": execution_date,  # Keep original field name
            "split_from": split.get("split_from"),
            "split_to": split.get("split_to"),
            "raw_data": split,
        }

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse a date string to datetime."""
        if not date_str:
            return None

        try:
            # Handle both date and datetime formats
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                # Parse date and add UTC timezone
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return dt.replace(tzinfo=UTC)
        except Exception as e:
            self.logger.debug(f"Error parsing date {date_str}: {e}")
            return None
