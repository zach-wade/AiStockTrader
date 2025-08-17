"""
Market Data Interface Definitions

Defines the contracts for market data providers and related data structures.
Following clean architecture principles with protocol-based interfaces.
"""

# Standard library imports
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

# Local imports
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol


@dataclass(frozen=True)
class Bar:
    """
    Represents a single OHLCV bar for market data.

    Attributes:
        symbol: The trading symbol
        timestamp: Bar timestamp (start of period)
        open: Opening price for the period
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price for the period
        volume: Trading volume during the period
        vwap: Volume-weighted average price (optional)
        trade_count: Number of trades (optional)
        timeframe: Bar timeframe (e.g., "1min", "5min", "1hour", "1day")
    """

    symbol: Symbol
    timestamp: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: int
    vwap: Price | None = None
    trade_count: int | None = None
    timeframe: str = "1min"

    def __post_init__(self) -> None:
        """Validate bar data after initialization."""
        if self.high < self.low:
            raise ValueError(f"High price {self.high} cannot be less than low price {self.low}")
        if self.high < self.open or self.high < self.close:
            raise ValueError("High price must be >= open and close prices")
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low price must be <= open and close prices")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError(f"Trade count cannot be negative: {self.trade_count}")

    @property
    def range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high.value - self.low.value

    @property
    def is_bullish(self) -> bool:
        """Check if this is a bullish bar (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if this is a bearish bar (close < open)."""
        return self.close < self.open

    @property
    def body_size(self) -> Decimal:
        """Calculate the body size (abs(close - open))."""
        return abs(self.close.value - self.open.value)


@dataclass(frozen=True)
class PriceUpdate:
    """
    Represents a real-time price update from market data stream.

    Attributes:
        symbol: The trading symbol
        timestamp: Update timestamp
        price: Current price
        size: Size of the update (shares/contracts)
        conditions: Trade conditions (optional)
        exchange: Exchange where trade occurred (optional)
    """

    symbol: Symbol
    timestamp: datetime
    price: Price
    size: int | None = None
    conditions: list[str] | None = None
    exchange: str | None = None

    def __post_init__(self) -> None:
        """Validate price update after initialization."""
        if self.size is not None and self.size < 0:
            raise ValueError(f"Size cannot be negative: {self.size}")


@dataclass(frozen=True)
class Quote:
    """
    Represents a market quote with bid/ask prices.

    Attributes:
        symbol: The trading symbol
        timestamp: Quote timestamp
        bid_price: Best bid price
        bid_size: Size at best bid
        ask_price: Best ask price
        ask_size: Size at best ask
        bid_exchange: Exchange for best bid (optional)
        ask_exchange: Exchange for best ask (optional)
    """

    symbol: Symbol
    timestamp: datetime
    bid_price: Price
    bid_size: int
    ask_price: Price
    ask_size: int
    bid_exchange: str | None = None
    ask_exchange: str | None = None

    def __post_init__(self) -> None:
        """Validate quote after initialization."""
        if self.bid_price > self.ask_price:
            raise ValueError(
                f"Bid price {self.bid_price} cannot be greater than ask price {self.ask_price}"
            )
        if self.bid_size < 0:
            raise ValueError(f"Bid size cannot be negative: {self.bid_size}")
        if self.ask_size < 0:
            raise ValueError(f"Ask size cannot be negative: {self.ask_size}")

    @property
    def spread(self) -> Decimal:
        """Calculate the bid-ask spread."""
        return self.ask_price.value - self.bid_price.value

    @property
    def spread_percentage(self) -> Decimal:
        """Calculate the bid-ask spread as a percentage of midpoint."""
        return self.bid_price.calculate_spread_percentage(self.ask_price)

    @property
    def midpoint(self) -> Price:
        """Calculate the midpoint price."""
        return Price.from_bid_ask(self.bid_price.value, self.ask_price.value)


class IMarketDataProvider(Protocol):
    """
    Market data provider interface.

    Defines the contract for retrieving market data from external sources.
    Implementations should handle rate limiting, error recovery, and data validation.
    """

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Price:
        """
        Get the current market price for a symbol.

        Args:
            symbol: The trading symbol (e.g., "AAPL", "BTC-USD")

        Returns:
            Current market price

        Raises:
            MarketDataError: If unable to fetch price
            SymbolNotFoundError: If symbol is not valid/found
            RateLimitError: If rate limit exceeded
        """
        ...

    @abstractmethod
    async def get_current_quote(self, symbol: str) -> Quote:
        """
        Get the current bid/ask quote for a symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Current market quote with bid/ask

        Raises:
            MarketDataError: If unable to fetch quote
            SymbolNotFoundError: If symbol is not valid/found
            RateLimitError: If rate limit exceeded
        """
        ...

    @abstractmethod
    async def get_historical_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1min"
    ) -> list[Bar]:
        """
        Get historical price bars for a symbol.

        Args:
            symbol: The trading symbol
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            timeframe: Bar timeframe (e.g., "1min", "5min", "1hour", "1day")

        Returns:
            List of price bars ordered by timestamp (ascending)

        Raises:
            MarketDataError: If unable to fetch bars
            SymbolNotFoundError: If symbol is not valid/found
            InvalidTimeframeError: If timeframe is not supported
            RateLimitError: If rate limit exceeded
        """
        ...

    @abstractmethod
    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[PriceUpdate]:
        """
        Stream real-time price updates for multiple symbols.

        Args:
            symbols: List of trading symbols to stream

        Yields:
            Real-time price updates as they occur

        Raises:
            MarketDataError: If unable to establish stream
            SymbolNotFoundError: If any symbol is not valid/found
            RateLimitError: If rate limit exceeded
            ConnectionError: If stream connection fails
        """
        ...

    @abstractmethod
    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """
        Stream real-time quote updates for multiple symbols.

        Args:
            symbols: List of trading symbols to stream

        Yields:
            Real-time quote updates as they occur

        Raises:
            MarketDataError: If unable to establish stream
            SymbolNotFoundError: If any symbol is not valid/found
            RateLimitError: If rate limit exceeded
            ConnectionError: If stream connection fails
        """
        ...

    @abstractmethod
    async def is_market_open(self) -> bool:
        """
        Check if the market is currently open for trading.

        Returns:
            True if market is open, False otherwise

        Raises:
            MarketDataError: If unable to determine market status
        """
        ...

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> dict:
        """
        Get detailed information about a trading symbol.

        Args:
            symbol: The trading symbol

        Returns:
            Dictionary with symbol information (varies by provider)

        Raises:
            MarketDataError: If unable to fetch info
            SymbolNotFoundError: If symbol is not valid/found
            RateLimitError: If rate limit exceeded
        """
        ...


# Market Data Errors
class MarketDataError(Exception):
    """Base exception for market data operations."""

    pass


class SymbolNotFoundError(MarketDataError):
    """Exception raised when a symbol is not found or invalid."""

    pass


class InvalidTimeframeError(MarketDataError):
    """Exception raised when an invalid timeframe is specified."""

    pass


class RateLimitError(MarketDataError):
    """Exception raised when rate limit is exceeded."""

    pass


class ConnectionError(MarketDataError):
    """Exception raised when connection to market data provider fails."""

    pass
