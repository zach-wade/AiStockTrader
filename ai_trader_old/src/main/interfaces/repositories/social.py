"""
Social Sentiment Repository Interface

Interface for social media sentiment data operations.
"""

# Standard library imports
from abc import abstractmethod
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

from .base import IRepository, OperationResult


class ISocialSentimentRepository(IRepository):
    """Interface for social sentiment repositories."""

    @abstractmethod
    async def store_social_post(
        self,
        platform: str,
        post_id: str,
        symbol: str,
        timestamp: datetime,
        content: str,
        sentiment_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> OperationResult:
        """
        Store a social media post with sentiment.

        Args:
            platform: Social platform (twitter, reddit, stocktwits)
            post_id: Unique post identifier
            symbol: Stock symbol mentioned
            timestamp: Post timestamp
            content: Post content
            sentiment_score: Calculated sentiment
            metadata: Optional additional data

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def get_social_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        platforms: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get social sentiment data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            platforms: Optional platform filter

        Returns:
            DataFrame with social sentiment
        """
        pass

    @abstractmethod
    async def get_trending_symbols(
        self, platforms: list[str] | None = None, lookback_hours: int = 24, top_n: int = 20
    ) -> list[dict[str, Any]]:
        """
        Get trending symbols on social media.

        Args:
            platforms: Optional platform filter
            lookback_hours: Hours to look back
            top_n: Number of top symbols

        Returns:
            List of trending symbol data
        """
        pass

    @abstractmethod
    async def get_social_volume(
        self, symbol: str, start_date: datetime, end_date: datetime, aggregation: str = "hourly"
    ) -> pd.DataFrame:
        """
        Get social media mention volume.

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
    async def detect_duplicates(self, posts: list[dict[str, Any]]) -> list[str]:
        """
        Detect duplicate posts.

        Args:
            posts: List of posts to check

        Returns:
            List of duplicate post IDs
        """
        pass

    @abstractmethod
    async def get_influencer_sentiment(
        self, symbol: str, min_followers: int = 10000, lookback_days: int = 7
    ) -> pd.DataFrame:
        """
        Get sentiment from influential accounts.

        Args:
            symbol: Stock symbol
            min_followers: Minimum follower count
            lookback_days: Days to look back

        Returns:
            DataFrame with influencer sentiment
        """
        pass

    @abstractmethod
    async def get_sentiment_changes(self, symbol: str, period_hours: int = 24) -> dict[str, Any]:
        """
        Detect significant sentiment changes.

        Args:
            symbol: Stock symbol
            period_hours: Period to analyze

        Returns:
            Dictionary with change metrics
        """
        pass

    @abstractmethod
    async def batch_store_posts(self, posts: pd.DataFrame) -> OperationResult:
        """
        Store multiple social posts.

        Args:
            posts: DataFrame with post data

        Returns:
            Operation result with statistics
        """
        pass

    @abstractmethod
    async def cleanup_old_posts(
        self, days_to_keep: int, platforms: list[str] | None = None
    ) -> OperationResult:
        """
        Clean up old social posts.

        Args:
            days_to_keep: Days of data to keep
            platforms: Optional platform filter

        Returns:
            Operation result with deletion count
        """
        pass

    @abstractmethod
    async def get_cross_platform_correlation(
        self, symbol: str, platforms: list[str], lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Analyze sentiment correlation across platforms.

        Args:
            symbol: Stock symbol
            platforms: Platforms to compare
            lookback_days: Analysis period

        Returns:
            Correlation matrix DataFrame
        """
        pass
