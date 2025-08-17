"""
Repository Interfaces

Defines contracts for data repositories to ensure clean separation
and consistent access patterns across the data pipeline.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd


class IRepository(ABC):
    """Base interface for all data repositories."""

    @abstractmethod
    async def get_data(
        self, symbol: str, start_date: datetime, end_date: datetime, **kwargs
    ) -> pd.DataFrame:
        """
        Retrieve data for a symbol within a date range.

        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional query parameters

        Returns:
            DataFrame with requested data
        """
        pass

    @abstractmethod
    async def store_data(self, data: pd.DataFrame, **kwargs) -> int:
        """
        Store data in the repository.

        Args:
            data: Data to store
            **kwargs: Additional storage parameters

        Returns:
            Number of records stored
        """
        pass

    @abstractmethod
    async def query(
        self, filters: dict[str, Any], limit: int | None = None, offset: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """
        Query data with filters.

        Args:
            filters: Query filters
            limit: Maximum number of records
            offset: Number of records to skip
            **kwargs: Additional query parameters

        Returns:
            Query results as DataFrame
        """
        pass


class IMarketDataRepository(IRepository):
    """Interface for market data repositories."""

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, start_date: datetime, end_date: datetime, interval: str = "1d"
    ) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass

    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float | None:
        """Get latest price for a symbol."""
        pass


class IFeatureRepository(IRepository):
    """Interface for feature storage repositories."""

    @abstractmethod
    async def store_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Store calculated features.

        Args:
            symbol: Stock symbol
            timestamp: Feature calculation timestamp
            features: Dictionary of feature name to value
            metadata: Optional metadata about features

        Returns:
            True if stored successfully
        """
        pass

    @abstractmethod
    async def get_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve stored features.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            feature_names: Optional list of specific features to retrieve

        Returns:
            DataFrame with features
        """
        pass


class IRepositoryFactory(ABC):
    """Factory interface for creating repositories."""

    @abstractmethod
    def create_repository(self, repo_type: str) -> IRepository:
        """
        Create a repository instance.

        Args:
            repo_type: Type of repository to create

        Returns:
            Repository instance

        Raises:
            ValueError: If repo_type is unknown
        """
        pass

    @abstractmethod
    def get_available_repositories(self) -> list[str]:
        """
        Get list of available repository types.

        Returns:
            List of repository type names
        """
        pass

    @abstractmethod
    def register_repository(self, repo_type: str, repo_class: type, override: bool = False) -> None:
        """
        Register a new repository type.

        Args:
            repo_type: Type name for the repository
            repo_class: Repository class (must implement IRepository)
            override: Whether to override existing registration

        Raises:
            ValueError: If repo_type exists and override is False
        """
        pass
