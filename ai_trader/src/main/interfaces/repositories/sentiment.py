"""
Sentiment Repository Interface

Interface for sentiment data storage and aggregation operations.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from .base import IRepository, OperationResult


class ISentimentRepository(IRepository):
    """Interface for sentiment data repositories."""
    
    @abstractmethod
    async def store_sentiment(
        self,
        symbol: str,
        source: str,
        timestamp: datetime,
        sentiment_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """
        Store sentiment data.
        
        Args:
            symbol: Stock symbol
            source: Sentiment source (news, social, analyst)
            timestamp: Sentiment timestamp
            sentiment_score: Sentiment value (-1 to 1)
            metadata: Optional additional data
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            sources: Optional filter by sources
            
        Returns:
            DataFrame with sentiment data
        """
        pass
    
    @abstractmethod
    async def get_aggregated_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        aggregation: str = "daily",
        sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get aggregated sentiment over time periods.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            aggregation: Aggregation period (hourly, daily, weekly)
            sources: Optional source filter
            
        Returns:
            DataFrame with aggregated sentiment
        """
        pass
    
    @abstractmethod
    async def get_latest_sentiment(
        self,
        symbol: str,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            source: Optional specific source
            
        Returns:
            Latest sentiment data or None
        """
        pass
    
    @abstractmethod
    async def get_sentiment_trends(
        self,
        symbol: str,
        lookback_days: int = 30,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate sentiment trends.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to look back
            sources: Optional source filter
            
        Returns:
            Dictionary with trend metrics
        """
        pass
    
    @abstractmethod
    async def batch_store_sentiment(
        self,
        sentiment_data: pd.DataFrame
    ) -> OperationResult:
        """
        Store sentiment for multiple records.
        
        Args:
            sentiment_data: DataFrame with sentiment records
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_multi_symbol_sentiment(
        self,
        symbols: List[str],
        date: datetime,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sentiment for multiple symbols on a date.
        
        Args:
            symbols: List of symbols
            date: Date to check
            sources: Optional source filter
            
        Returns:
            Nested dict of symbol -> sentiment data
        """
        pass
    
    @abstractmethod
    async def get_sentiment_extremes(
        self,
        start_date: datetime,
        end_date: datetime,
        top_n: int = 10,
        sources: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find extreme sentiment values.
        
        Args:
            start_date: Start date
            end_date: End date
            top_n: Number of extremes to return
            sources: Optional source filter
            
        Returns:
            Dict with 'positive' and 'negative' extremes
        """
        pass