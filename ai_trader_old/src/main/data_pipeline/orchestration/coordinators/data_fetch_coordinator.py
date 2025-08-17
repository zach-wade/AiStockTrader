"""
Data Fetch Coordinator

Coordinates data fetching across different clients and stages.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Local imports
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.types import RawDataRecord
from main.utils.core import get_logger


class DataStage(Enum):
    """Data processing stages."""

    MARKET_DATA = "market_data"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    CORPORATE_ACTIONS = "corporate_actions"


@dataclass
class FetchRequest:
    """Request for data fetching."""

    symbols: list[str]
    stages: list[DataStage]
    start_date: datetime
    end_date: datetime
    batch_size: int = 50
    max_concurrent: int = 5


@dataclass
class FetchResult:
    """Result of data fetching."""

    stage: DataStage
    symbols_processed: int
    records_fetched: int
    errors: list[str]
    success: bool


class DataFetchCoordinator:
    """
    Coordinates data fetching from various sources.

    Manages batching, concurrency, and delegates to appropriate
    clients for each data type.
    """

    def __init__(
        self,
        archive: DataArchive,
        market_client: Any | None = None,
        news_client: Any | None = None,
        fundamentals_client: Any | None = None,
        corporate_actions_client: Any | None = None,
    ):
        """
        Initialize the data fetch coordinator.

        Args:
            archive: Data archive for storing results
            market_client: Client for market data
            news_client: Client for news data
            fundamentals_client: Client for fundamentals
            corporate_actions_client: Client for corporate actions
        """
        self.archive = archive
        self.market_client = market_client
        self.news_client = news_client
        self.fundamentals_client = fundamentals_client
        self.corporate_actions_client = corporate_actions_client
        self.logger = get_logger(__name__)

    async def fetch_data(self, request: FetchRequest) -> list[FetchResult]:
        """
        Fetch data according to the request.

        Args:
            request: Fetch request with parameters

        Returns:
            List of fetch results for each stage
        """
        results = []

        for stage in request.stages:
            self.logger.info(f"Fetching {stage.value} for {len(request.symbols)} symbols")

            result = await self._fetch_stage(
                stage=stage,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                batch_size=request.batch_size,
                max_concurrent=request.max_concurrent,
            )

            results.append(result)

            self.logger.info(f"Completed {stage.value}: {result.records_fetched} records")

        return results

    async def _fetch_stage(
        self,
        stage: DataStage,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int,
        max_concurrent: int,
    ) -> FetchResult:
        """
        Fetch data for a specific stage.

        Args:
            stage: Stage to fetch
            symbols: Symbols to fetch
            start_date: Start date
            end_date: End date
            batch_size: Batch size for processing
            max_concurrent: Max concurrent requests

        Returns:
            Fetch result for the stage
        """
        errors = []
        records_fetched = 0

        try:
            if stage == DataStage.MARKET_DATA and self.market_client:
                records_fetched = await self._fetch_market_data(
                    symbols, start_date, end_date, batch_size, max_concurrent
                )

            elif stage == DataStage.NEWS and self.news_client:
                records_fetched = await self._fetch_news(
                    symbols, start_date, end_date, batch_size, max_concurrent
                )

            elif stage == DataStage.FUNDAMENTALS and self.fundamentals_client:
                records_fetched = await self._fetch_fundamentals(
                    symbols, start_date, end_date, batch_size, max_concurrent
                )

            elif stage == DataStage.CORPORATE_ACTIONS and self.corporate_actions_client:
                records_fetched = await self._fetch_corporate_actions(
                    symbols, start_date, end_date, batch_size, max_concurrent
                )
            else:
                errors.append(f"No client available for {stage.value}")

        except Exception as e:
            self.logger.error(f"Error fetching {stage.value}: {e}")
            errors.append(str(e))

        return FetchResult(
            stage=stage,
            symbols_processed=len(symbols),
            records_fetched=records_fetched,
            errors=errors,
            success=len(errors) == 0,
        )

    async def _fetch_market_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int,
        max_concurrent: int,
    ) -> int:
        """Fetch market data for symbols."""
        total_records = 0

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]

            results = await self.market_client.fetch_batch(
                symbols=batch,
                interval="1hour",
                start_date=start_date,
                end_date=end_date,
                max_concurrent=max_concurrent,
            )

            # Archive results
            for symbol, result in results.items():
                if result.success and result.data:
                    await self._archive_data("market_data", symbol, result.data)
                    total_records += len(result.data)

        return total_records

    async def _fetch_news(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int,
        max_concurrent: int,
    ) -> int:
        """Fetch news for symbols."""
        results = await self.news_client.fetch_news_batch(
            symbols=symbols, start_date=start_date, end_date=end_date, limit_per_symbol=100
        )

        total_records = 0
        for symbol, result in results.items():
            if result.success and result.data:
                await self._archive_data("news", symbol, result.data)
                total_records += len(result.data)

        return total_records

    async def _fetch_fundamentals(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int,
        max_concurrent: int,
    ) -> int:
        """Fetch fundamentals for symbols."""
        results = await self.fundamentals_client.fetch_multiple_symbols(
            symbols=symbols,
            timeframe="quarterly",
            start_date=start_date,
            end_date=end_date,
            max_concurrent=max_concurrent,
        )

        total_records = 0
        for symbol, result in results.items():
            if result.success and result.data:
                await self._archive_data("fundamentals", symbol, result.data)
                total_records += len(result.data)

        return total_records

    async def _fetch_corporate_actions(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        batch_size: int,
        max_concurrent: int,
    ) -> int:
        """Fetch corporate actions for symbols."""
        results = await self.corporate_actions_client.fetch_batch(
            symbols=symbols, start_date=start_date, end_date=end_date, batch_size=batch_size
        )

        total_records = 0
        for symbol, result in results.items():
            if result.success and result.data:
                await self._archive_data("corporate_actions", symbol, result.data)
                total_records += len(result.data)

        return total_records

    async def _archive_data(self, data_type: str, symbol: str, data: Any):
        """Archive data to the data lake."""
        record = RawDataRecord(
            source="polygon",
            data_type=data_type,
            timestamp=datetime.now(UTC),
            symbol=symbol,
            data={"data": data},
            metadata={"symbol": symbol},
        )

        await self.archive.save_raw_record_async(record)
