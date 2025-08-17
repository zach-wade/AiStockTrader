"""
Data Fetch Service

Service responsible for fetching data from various sources.
Part of the service-oriented architecture for historical data processing.
"""

# Standard library imports
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.interfaces.database import IAsyncDatabase
from main.utils.core import RateLimiter, get_logger, timer
from main.utils.processing.historical import ProcessingUtils

logger = get_logger(__name__)


@dataclass
class FetchRequest:
    """Configuration for a data fetch request."""

    symbol: str
    data_type: str
    start_date: datetime
    end_date: datetime
    source: str = "polygon"
    intervals: list[str] | None = None
    layer: int = 1


@dataclass
class FetchResult:
    """Result of a data fetch operation."""

    success: bool
    symbol: str
    data_type: str
    records_fetched: int = 0
    errors: list[str] = None
    data: Any = None
    metadata: dict[str, Any] = None


class DataFetchService:
    """
    Service for fetching data from various sources.

    Responsibilities:
    - Coordinate with data source clients
    - Handle rate limiting and retries
    - Validate fetched data
    - Cache frequently accessed data
    """

    def __init__(
        self, db_adapter: IAsyncDatabase | None = None, config: dict[str, Any] | None = None
    ):
        """
        Initialize the data fetch service.

        Args:
            db_adapter: Database adapter for metadata queries
            config: Service configuration
        """
        self.logger = get_logger(__name__)
        self.db_adapter = db_adapter
        self.config = config or {}

        # Initialize components
        self._clients = {}
        self._rate_limiters = {}
        self._cache = {}

        # Processing utilities
        self.processing_utils = ProcessingUtils()

        self.logger.info("DataFetchService initialized")

    async def initialize(self):
        """Initialize data source clients."""
        # Import clients here to avoid circular dependencies
        try:
            # Initialize clients based on config
            # Standard library imports
            import os

            # Local imports
            from main.data_pipeline.ingestion.clients import (
                PolygonCorporateActionsClient,
                PolygonFundamentalsClient,
                PolygonMarketClient,
                PolygonNewsClient,
            )

            polygon_api_key = os.getenv("POLYGON_API_KEY")

            if polygon_api_key:  # Only initialize if API key is available
                self._clients["polygon_market"] = PolygonMarketClient(
                    api_key=polygon_api_key,
                    layer=DataLayer.BASIC,  # Default to BASIC for historical fetching
                )
                self._clients["polygon_news"] = PolygonNewsClient(
                    api_key=polygon_api_key,
                    layer=DataLayer.BASIC,  # Default to BASIC for historical fetching
                )
                self._clients["polygon_fundamentals"] = PolygonFundamentalsClient(
                    api_key=polygon_api_key,
                    layer=DataLayer.BASIC,  # Default to BASIC for historical fetching
                )
                self._clients["polygon_corporate_actions"] = PolygonCorporateActionsClient(
                    api_key=polygon_api_key,
                    layer=DataLayer.BASIC,  # Default to BASIC for historical fetching
                )
            else:
                self.logger.warning("POLYGON_API_KEY not found in environment variables")

                # Initialize rate limiter for Polygon
                self._rate_limiters["polygon"] = RateLimiter(
                    calls_per_minute=polygon_config.get("rate_limit", 300)
                )

            self.logger.info(f"Initialized {len(self._clients)} data source clients")

        except ImportError as e:
            self.logger.warning(f"Could not import data clients: {e}")
            # Service can still function with limited capabilities

    @timer
    async def fetch_data(self, request: FetchRequest) -> FetchResult:
        """
        Fetch data based on request parameters.

        Args:
            request: Fetch request configuration

        Returns:
            FetchResult with fetched data
        """
        result = FetchResult(
            success=False, symbol=request.symbol, data_type=request.data_type, errors=[]
        )

        try:
            # Select appropriate client
            client = self._get_client(request.source, request.data_type)
            if not client:
                error_msg = f"No client available for {request.source}/{request.data_type}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
                return result

            # Apply rate limiting
            rate_limiter = self._rate_limiters.get(request.source)
            if rate_limiter:
                await rate_limiter.acquire()

            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self._cache:
                cached_data, cached_time = self._cache[cache_key]
                cache_age = (datetime.now(UTC) - cached_time).total_seconds()

                # Use cache if fresh enough
                cache_ttl = self.config.get("cache_ttl", 3600)
                if cache_age < cache_ttl:
                    self.logger.debug(f"Using cached data for {cache_key}")
                    result.success = True
                    result.data = cached_data
                    result.records_fetched = (
                        len(cached_data) if isinstance(cached_data, list) else 1
                    )
                    result.metadata = {"from_cache": True, "cache_age": cache_age}
                    return result

            # Fetch data based on type
            if request.data_type == "market_data":
                data = await self._fetch_market_data(client, request)
            elif request.data_type == "news":
                data = await self._fetch_news_data(client, request)
            elif request.data_type == "fundamentals":
                data = await self._fetch_fundamentals_data(client, request)
            elif request.data_type == "corporate_actions":
                data = await self._fetch_corporate_actions(client, request)
            else:
                error_msg = f"Unsupported data type: {request.data_type}"
                result.errors.append(error_msg)
                return result

            # Validate fetched data
            if data:
                validation_errors = self._validate_data(data, request.data_type)
                if validation_errors:
                    result.errors.extend(validation_errors)
                    self.logger.warning(f"Data validation errors: {validation_errors}")

                # Cache successful fetch
                self._cache[cache_key] = (data, datetime.now(UTC))

                result.success = True
                result.data = data
                result.records_fetched = len(data) if isinstance(data, list) else 1
                result.metadata = {"from_cache": False}
            else:
                result.errors.append("No data returned from source")

        except Exception as e:
            error_msg = f"Error fetching data: {e!s}"
            result.errors.append(error_msg)
            self.logger.error(error_msg, exc_info=True)

        return result

    async def fetch_batch(
        self, requests: list[FetchRequest], max_concurrent: int = 5
    ) -> list[FetchResult]:
        """
        Fetch data for multiple requests concurrently.

        Args:
            requests: List of fetch requests
            max_concurrent: Maximum concurrent requests

        Returns:
            List of fetch results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(request):
            async with semaphore:
                return await self.fetch_data(request)

        tasks = [fetch_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    FetchResult(
                        success=False,
                        symbol=requests[i].symbol,
                        data_type=requests[i].data_type,
                        errors=[str(result)],
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def _get_client(self, source: str, data_type: str) -> Any | None:
        """Get appropriate client for source and data type."""
        client_key = f"{source}_{data_type.replace('_data', '')}"
        return self._clients.get(client_key)

    def _get_cache_key(self, request: FetchRequest) -> str:
        """Generate cache key for request."""
        intervals = request.intervals or []
        return f"{request.symbol}:{request.data_type}:{request.source}:{request.start_date.date()}:{request.end_date.date()}:{','.join(intervals)}"

    async def _fetch_market_data(self, client: Any, request: FetchRequest) -> list[dict] | None:
        """Fetch market data from client."""
        try:
            data = []

            # Fetch for each interval
            intervals = request.intervals or ["1day"]
            for interval in intervals:
                interval_data = await client.fetch_historical_bars(
                    symbol=request.symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    interval=interval,
                )

                if interval_data:
                    # Add interval metadata to each record
                    for record in interval_data:
                        record["interval"] = interval
                    data.extend(interval_data)

            return data

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise

    async def _fetch_news_data(self, client: Any, request: FetchRequest) -> list[dict] | None:
        """Fetch news data from client."""
        try:
            return await client.fetch_news(
                symbol=request.symbol, start_date=request.start_date, end_date=request.end_date
            )
        except Exception as e:
            self.logger.error(f"Error fetching news data: {e}")
            raise

    async def _fetch_fundamentals_data(
        self, client: Any, request: FetchRequest
    ) -> list[dict] | None:
        """Fetch fundamentals data from client."""
        try:
            return await client.fetch_financials(
                symbol=request.symbol, start_date=request.start_date, end_date=request.end_date
            )
        except Exception as e:
            self.logger.error(f"Error fetching fundamentals: {e}")
            raise

    async def _fetch_corporate_actions(
        self, client: Any, request: FetchRequest
    ) -> list[dict] | None:
        """Fetch corporate actions from client."""
        try:
            return await client.fetch_corporate_actions(
                symbol=request.symbol, start_date=request.start_date, end_date=request.end_date
            )
        except Exception as e:
            self.logger.error(f"Error fetching corporate actions: {e}")
            raise

    def _validate_data(self, data: Any, data_type: str) -> list[str]:
        """Validate fetched data."""
        errors = []

        if not data:
            errors.append("Empty data returned")
            return errors

        # Type-specific validation
        if data_type == "market_data":
            # Validate OHLCV data
            required_fields = ["open", "high", "low", "close", "volume"]
            if isinstance(data, list) and data:
                sample = data[0]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    errors.append(f"Missing required fields: {missing}")

        elif data_type == "news":
            # Validate news articles
            if isinstance(data, list) and data:
                sample = data[0]
                if "title" not in sample and "headline" not in sample:
                    errors.append("News articles missing title/headline")

        return errors

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "clients_initialized": len(self._clients),
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())[:10],  # Sample of cache keys
        }
