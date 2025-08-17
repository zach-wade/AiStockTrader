"""
News Repository Interface

Interface for news data storage and retrieval operations.
"""

# Standard library imports
from abc import abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

from .base import IRepository, OperationResult


class INewsRepository(IRepository):
    """Interface for news data repositories."""

    @abstractmethod
    async def store_news_article(
        self,
        article_id: str,
        symbol: str,
        headline: str,
        content: str,
        source: str,
        published_at: datetime,
        sentiment_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationResult:
        """
        Store a news article.

        Args:
            article_id: Unique article identifier
            symbol: Stock symbol
            headline: Article headline
            content: Article content
            source: News source
            published_at: Publication timestamp
            sentiment_score: Optional sentiment
            metadata: Optional additional data

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        sources: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get news articles for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            sources: Optional source filter

        Returns:
            DataFrame with news articles
        """
        pass

    @abstractmethod
    async def get_latest_news(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Get latest news for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum articles

        Returns:
            DataFrame with latest news
        """
        pass

    @abstractmethod
    async def search_news(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Search news articles.

        Args:
            query: Search query
            start_date: Optional start date
            end_date: Optional end date
            limit: Maximum results

        Returns:
            DataFrame with matching articles
        """
        pass

    @abstractmethod
    async def get_news_sentiment_summary(
        self, symbol: str, lookback_days: int = 7
    ) -> dict[str, Any]:
        """
        Get news sentiment summary.

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            Summary statistics dictionary
        """
        pass

    @abstractmethod
    async def get_breaking_news(
        self, symbols: list[str] | None = None, lookback_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Get recent breaking news.

        Args:
            symbols: Optional symbol filter
            lookback_minutes: Minutes to look back

        Returns:
            DataFrame with breaking news
        """
        pass

    @abstractmethod
    async def deduplicate_news(self, articles: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate news articles.

        Args:
            articles: DataFrame with articles

        Returns:
            DataFrame without duplicates
        """
        pass

    @abstractmethod
    async def get_news_volume(
        self, symbol: str, start_date: datetime, end_date: datetime, aggregation: str = "daily"
    ) -> pd.DataFrame:
        """
        Get news volume over time.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            aggregation: Time aggregation

        Returns:
            DataFrame with volume data
        """
        pass

    @abstractmethod
    async def get_related_symbols(self, article_id: str) -> list[str]:
        """
        Get symbols mentioned in an article.

        Args:
            article_id: Article identifier

        Returns:
            List of related symbols
        """
        pass

    @abstractmethod
    async def batch_store_news(self, articles: pd.DataFrame) -> OperationResult:
        """
        Store multiple news articles.

        Args:
            articles: DataFrame with articles

        Returns:
            Operation result with statistics
        """
        pass

    @abstractmethod
    async def get_news_analytics(self, symbol: str, lookback_days: int = 30) -> dict[str, Any]:
        """
        Get comprehensive news analytics.

        Args:
            symbol: Stock symbol
            lookback_days: Analysis period

        Returns:
            Analytics dictionary
        """
        pass
