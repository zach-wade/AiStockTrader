"""
Market Data Use Cases

Handles market data retrieval and processing.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from src.application.interfaces.market_data import IMarketDataProvider
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.domain.value_objects.converter import ValueObjectConverter
from src.domain.value_objects.symbol import Symbol

from .base import TransactionalUseCase, UseCaseResponse
from .base_request import BaseRequestDTO

# UTC timezone already imported from datetime above


# Request/Response DTOs
@dataclass
class GetMarketDataRequest(BaseRequestDTO):
    """Request to get market data."""

    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "1min"


@dataclass
class GetMarketDataResponse(UseCaseResponse):
    """Response with market data."""

    bars: list[dict[str, Any]] | None = None
    count: int = 0

    def __post_init__(self) -> None:
        if self.bars is None:
            self.bars = []


@dataclass
class GetLatestPriceRequest(BaseRequestDTO):
    """Request to get latest price."""

    symbol: str


@dataclass
class GetLatestPriceResponse(UseCaseResponse):
    """Response with latest price."""

    symbol: str | None = None
    price: Decimal | None = None
    timestamp: datetime | None = None
    volume: int | None = None


@dataclass
class GetHistoricalDataRequest(BaseRequestDTO):
    """Request to get historical market data."""

    symbol: str
    days: int = 30
    timeframe: str = "1day"


@dataclass
class GetHistoricalDataResponse(UseCaseResponse):
    """Response with historical data."""

    data: list[dict[str, Any]] | None = None
    statistics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = []


# Use Case Implementations
class GetMarketDataUseCase(TransactionalUseCase[GetMarketDataRequest, GetMarketDataResponse]):
    """
    Retrieves market data for a specific time range.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        market_data_provider: IMarketDataProvider | None = None,
    ):
        """Initialize get market data use case."""
        super().__init__(unit_of_work, "GetMarketDataUseCase")
        self.market_data_provider = market_data_provider

    async def validate(self, request: GetMarketDataRequest) -> str | None:
        """Validate the request."""
        if request.start_date >= request.end_date:
            return "Start date must be before end date"

        valid_timeframes = ["1min", "5min", "15min", "30min", "1hour", "1day"]
        if request.timeframe not in valid_timeframes:
            return f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"

        return None

    async def process(self, request: GetMarketDataRequest) -> GetMarketDataResponse:
        """Process the market data request."""
        # Get from repository first
        market_repo = self.unit_of_work.market_data

        bars = await market_repo.get_bars_by_date_range(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
        )

        # If no data and we have a provider, fetch from provider
        if not bars and self.market_data_provider:
            # Note: IMarketDataProvider doesn't have get_bars method
            # This would need to be implemented or use get_current_price instead
            pass

        # Convert to response format
        bars_data = []
        for bar in bars:
            bars_data.append(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(ValueObjectConverter.extract_value(bar.open)),
                    "high": float(ValueObjectConverter.extract_value(bar.high)),
                    "low": float(ValueObjectConverter.extract_value(bar.low)),
                    "close": float(ValueObjectConverter.extract_value(bar.close)),
                    "volume": bar.volume,
                    "vwap": (
                        float(ValueObjectConverter.extract_value(bar.vwap)) if bar.vwap else None
                    ),
                }
            )

        return GetMarketDataResponse(
            success=True, bars=bars_data, count=len(bars_data), request_id=request.request_id
        )


class GetLatestPriceUseCase(TransactionalUseCase[GetLatestPriceRequest, GetLatestPriceResponse]):
    """
    Gets the latest price for a symbol.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        market_data_provider: IMarketDataProvider | None = None,
    ):
        """Initialize get latest price use case."""
        super().__init__(unit_of_work, "GetLatestPriceUseCase")
        self.market_data_provider = market_data_provider

    async def validate(self, request: GetLatestPriceRequest) -> str | None:
        """Validate the request."""
        if not request.symbol:
            return "Symbol is required"
        return None

    async def process(self, request: GetLatestPriceRequest) -> GetLatestPriceResponse:
        """Process the latest price request."""
        # Get from repository
        market_repo = self.unit_of_work.market_data

        latest_bar = await market_repo.get_latest_bar(symbol=request.symbol, timeframe="1min")

        # If no data and we have a provider, fetch current price and create bar
        if not latest_bar and self.market_data_provider:
            current_price = await self.market_data_provider.get_current_price(request.symbol)
            # Create a simplified bar from current price
            from src.application.interfaces.market_data import Bar

            latest_bar = Bar(
                symbol=Symbol(request.symbol),
                timestamp=datetime.now(UTC),
                open=current_price,
                high=current_price,
                low=current_price,
                close=current_price,
                volume=0,
                timeframe="1min",
            )

            # Save to repository
            if latest_bar:
                await market_repo.save_bar(latest_bar)

        if latest_bar:
            return GetLatestPriceResponse(
                success=True,
                symbol=request.symbol,
                price=ValueObjectConverter.extract_value(latest_bar.close),
                timestamp=latest_bar.timestamp,
                volume=latest_bar.volume,
                request_id=request.request_id,
            )
        else:
            return GetLatestPriceResponse(
                success=False,
                error=f"No price data available for {request.symbol}",
                request_id=request.request_id or uuid4(),
            )


class GetHistoricalDataUseCase(
    TransactionalUseCase[GetHistoricalDataRequest, GetHistoricalDataResponse]
):
    """
    Gets historical market data with statistics.
    """

    def __init__(
        self,
        unit_of_work: IUnitOfWork,
        market_data_provider: IMarketDataProvider | None = None,
    ):
        """Initialize get historical data use case."""
        super().__init__(unit_of_work, "GetHistoricalDataUseCase")
        self.market_data_provider = market_data_provider

    async def validate(self, request: GetHistoricalDataRequest) -> str | None:
        """Validate the request."""
        if request.days <= 0:
            return "Days must be positive"

        if request.days > 365:
            return "Maximum 365 days of historical data"

        return None

    async def process(self, request: GetHistoricalDataRequest) -> GetHistoricalDataResponse:
        """Process the historical data request."""
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        # Get from repository
        market_repo = self.unit_of_work.market_data

        bars = await market_repo.get_bars_by_date_range(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=request.timeframe,
        )

        # Convert to response format
        data = []
        prices = []

        for bar in bars:
            data.append(
                {
                    "date": bar.timestamp.date().isoformat(),
                    "open": float(ValueObjectConverter.extract_value(bar.open)),
                    "high": float(ValueObjectConverter.extract_value(bar.high)),
                    "low": float(ValueObjectConverter.extract_value(bar.low)),
                    "close": float(ValueObjectConverter.extract_value(bar.close)),
                    "volume": bar.volume,
                }
            )
            prices.append(float(ValueObjectConverter.extract_value(bar.close)))

        # Calculate statistics
        statistics = None
        if prices:
            from statistics import mean, stdev

            statistics = {
                "mean": mean(prices),
                "std_dev": stdev(prices) if len(prices) > 1 else 0,
                "min": min(prices),
                "max": max(prices),
                "range": max(prices) - min(prices),
                "count": len(prices),
            }

        return GetHistoricalDataResponse(
            success=True, data=data, statistics=statistics, request_id=request.request_id
        )
