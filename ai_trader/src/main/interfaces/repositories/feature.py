"""
Feature Repository Interface

Interface for ML feature storage and retrieval operations.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from .base import IRepository, OperationResult


class IFeatureRepository(IRepository):
    """Interface for feature storage repositories."""
    
    @abstractmethod
    async def store_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """
        Store calculated features.
        
        Args:
            symbol: Stock symbol
            timestamp: Feature calculation timestamp
            features: Dictionary of feature name to value
            metadata: Optional metadata about features
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None
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
    
    @abstractmethod
    async def get_latest_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest features for a symbol.
        
        Args:
            symbol: Stock symbol
            feature_names: Optional specific features
            
        Returns:
            Latest features or None
        """
        pass
    
    @abstractmethod
    async def batch_store_features(
        self,
        features_data: pd.DataFrame
    ) -> OperationResult:
        """
        Store features for multiple symbols/timestamps.
        
        Args:
            features_data: DataFrame with symbol, timestamp, and feature columns
            
        Returns:
            Operation result with storage statistics
        """
        pass
    
    @abstractmethod
    async def get_feature_statistics(
        self,
        symbol: str,
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for features over a period.
        
        Args:
            symbol: Stock symbol
            feature_names: Features to analyze
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with statistics per feature
        """
        pass
    
    @abstractmethod
    async def get_feature_correlation(
        self,
        symbol: str,
        features: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for features.
        
        Args:
            symbol: Stock symbol
            features: Features to correlate
            start_date: Start date
            end_date: End date
            
        Returns:
            Correlation matrix as DataFrame
        """
        pass
    
    @abstractmethod
    async def cleanup_old_features(
        self,
        days_to_keep: int,
        feature_names: Optional[List[str]] = None
    ) -> OperationResult:
        """
        Clean up old feature data.
        
        Args:
            days_to_keep: Number of days to keep
            feature_names: Optional specific features to clean
            
        Returns:
            Operation result with deletion count
        """
        pass
    
    @abstractmethod
    async def get_feature_availability(
        self,
        symbols: List[str],
        feature_names: List[str],
        date: datetime
    ) -> Dict[str, Dict[str, bool]]:
        """
        Check feature availability for symbols on a date.
        
        Args:
            symbols: List of symbols
            feature_names: Features to check
            date: Date to check
            
        Returns:
            Nested dict of symbol -> feature -> availability
        """
        pass
    
    @abstractmethod
    async def get_missing_features(
        self,
        symbol: str,
        expected_features: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Find missing features in a date range.
        
        Args:
            symbol: Stock symbol
            expected_features: Features that should exist
            start_date: Start date
            end_date: End date
            
        Returns:
            List of missing feature records
        """
        pass