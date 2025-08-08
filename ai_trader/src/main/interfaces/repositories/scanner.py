"""
Scanner Data Repository Interface

Interface for scanner data with technical indicators and screening.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from .base import IRepository, OperationResult


class IScannerDataRepository(IRepository):
    """Interface for scanner data repositories."""
    
    @abstractmethod
    async def get_scanner_data(
        self,
        symbols: List[str],
        date: datetime,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get scanner data for multiple symbols.
        
        Args:
            symbols: List of symbols
            date: Data date
            indicators: Optional specific indicators
            
        Returns:
            DataFrame with scanner data
        """
        pass
    
    @abstractmethod
    async def store_scanner_data(
        self,
        data: pd.DataFrame,
        date: datetime
    ) -> OperationResult:
        """
        Store scanner data.
        
        Args:
            data: Scanner data DataFrame
            date: Data date
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_technical_indicators(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        indicators: List[str]
    ) -> pd.DataFrame:
        """
        Get technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            indicators: List of indicator names
            
        Returns:
            DataFrame with indicator values
        """
        pass
    
    @abstractmethod
    async def scan_by_criteria(
        self,
        criteria: Dict[str, Any],
        date: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Scan for symbols matching criteria.
        
        Args:
            criteria: Scan criteria dictionary
            date: Optional scan date (default: latest)
            limit: Maximum results
            
        Returns:
            DataFrame with matching symbols
        """
        pass
    
    @abstractmethod
    async def get_relative_strength(
        self,
        symbols: List[str],
        benchmark: str,
        period_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate relative strength vs benchmark.
        
        Args:
            symbols: List of symbols
            benchmark: Benchmark symbol (e.g., SPY)
            period_days: Calculation period
            
        Returns:
            Dictionary of symbol -> RS value
        """
        pass
    
    @abstractmethod
    async def get_momentum_scores(
        self,
        symbols: List[str],
        lookback_periods: List[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate momentum scores.
        
        Args:
            symbols: List of symbols
            lookback_periods: Periods for momentum calculation
            
        Returns:
            DataFrame with momentum scores
        """
        pass
    
    @abstractmethod
    async def get_volatility_metrics(
        self,
        symbol: str,
        period_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate volatility metrics.
        
        Args:
            symbol: Stock symbol
            period_days: Calculation period
            
        Returns:
            Dictionary with volatility metrics
        """
        pass
    
    @abstractmethod
    async def get_support_resistance(
        self,
        symbol: str,
        lookback_days: int = 100
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            symbol: Stock symbol
            lookback_days: Analysis period
            
        Returns:
            Dict with support and resistance levels
        """
        pass
    
    @abstractmethod
    async def get_pattern_signals(
        self,
        symbol: str,
        patterns: List[str],
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Detect technical patterns.
        
        Args:
            symbol: Stock symbol
            patterns: Patterns to detect
            lookback_days: Analysis period
            
        Returns:
            List of detected patterns
        """
        pass
    
    @abstractmethod
    async def get_scanner_rankings(
        self,
        metric: str,
        date: Optional[datetime] = None,
        top_n: int = 50,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Get rankings by a specific metric.
        
        Args:
            metric: Metric to rank by
            date: Optional date (default: latest)
            top_n: Number of top results
            ascending: Sort order
            
        Returns:
            DataFrame with ranked symbols
        """
        pass
    
    @abstractmethod
    async def get_hot_cold_storage_data(
        self,
        symbol: str,
        lookback_days: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data from hot and cold storage.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to retrieve
            
        Returns:
            Tuple of (hot_data, cold_data) DataFrames
        """
        pass