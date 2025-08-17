"""
Market Data Repository Interface

Interface for market data storage and retrieval operations.
"""

# Standard library imports
from abc import abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

from .base import IRepository, OperationResult


class IMarketDataRepository(IRepository):
    """Interface for market data repositories."""

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1day"
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (1min, 5min, 1hour, 1day)

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get latest price data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price data or None
        """
        pass

    @abstractmethod
    async def get_latest_prices(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """
        Get latest prices for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary mapping symbols to price data
        """
        pass

    @abstractmethod
    async def store_ohlcv(self, data: pd.DataFrame, symbol: str, interval: str) -> OperationResult:
        """
        Store OHLCV data.

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol
            interval: Data interval

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_price_range(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> dict[str, float] | None:
        """
        Get price range statistics for a period.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with high, low, avg prices or None
        """
        pass

    @abstractmethod
    async def get_volume_profile(
        self, symbol: str, start_date: datetime, end_date: datetime, bins: int = 10
    ) -> pd.DataFrame:
        """
        Get volume profile for price levels.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            bins: Number of price bins

        Returns:
            DataFrame with volume by price level
        """
        pass

    @abstractmethod
    async def get_market_hours_data(
        self, symbol: str, date: datetime, extended_hours: bool = False
    ) -> pd.DataFrame:
        """
        Get market hours data for a specific date.

        Args:
            symbol: Stock symbol
            date: Trading date
            extended_hours: Include pre/post market data

        Returns:
            DataFrame with market hours data
        """
        pass

    @abstractmethod
    async def get_gaps(
        self, symbol: str, start_date: datetime, end_date: datetime, min_gap_percent: float = 1.0
    ) -> list[dict[str, Any]]:
        """
        Find price gaps in the data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            min_gap_percent: Minimum gap percentage to report

        Returns:
            List of gap occurrences
        """
        pass

    @abstractmethod
    async def cleanup_old_data(
        self, days_to_keep: int, interval: str | None = None
    ) -> OperationResult:
        """
        Clean up old market data.

        Args:
            days_to_keep: Number of days of data to keep
            interval: Optional specific interval to clean

        Returns:
            Operation result with deletion count
        """
        pass
