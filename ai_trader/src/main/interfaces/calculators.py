"""
Calculator Interfaces

Defines contracts for feature calculators to ensure clean separation
between data_pipeline and feature_pipeline modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class CalculatorConfig:
    """Configuration for calculators."""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class FeatureResult:
    """Result from a feature calculation."""
    features: pd.DataFrame
    metadata: Dict[str, Any]
    calculation_time: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success(self) -> bool:
        """Check if calculation was successful."""
        return len(self.errors) == 0


class IFeatureCalculator(ABC):
    """Base interface for all feature calculators."""
    
    @abstractmethod
    def calculate(
        self, 
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> FeatureResult:
        """
        Calculate features from input data.
        
        Args:
            data: Input data DataFrame
            symbols: Optional list of symbols to calculate for
            start_date: Optional start date for calculation
            end_date: Optional end date for calculation
            **kwargs: Additional calculator-specific parameters
            
        Returns:
            FeatureResult containing calculated features and metadata
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this calculator produces.
        
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required input columns.
        
        Returns:
            List of column names required in input data
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns and format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        pass


class ITechnicalCalculator(IFeatureCalculator):
    """Interface for technical indicator calculators."""
    
    @abstractmethod
    def calculate_indicators(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            price_data: OHLC price data
            volume_data: Optional volume data
            indicator_config: Configuration for specific indicators
            
        Returns:
            DataFrame with calculated indicators
        """
        pass


class ISentimentCalculator(IFeatureCalculator):
    """Interface for sentiment analysis calculators."""
    
    @abstractmethod
    def calculate_sentiment(
        self,
        text_data: Union[pd.DataFrame, List[str]],
        sentiment_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment features.
        
        Args:
            text_data: Text data to analyze
            sentiment_config: Configuration for sentiment analysis
            
        Returns:
            DataFrame with sentiment scores and features
        """
        pass
    
    @abstractmethod
    def get_sentiment_model_info(self) -> Dict[str, Any]:
        """
        Get information about the sentiment model being used.
        
        Returns:
            Dictionary with model name, version, and capabilities
        """
        pass


class ICalculatorFactory(ABC):
    """Factory interface for creating calculators."""
    
    @abstractmethod
    def create_technical_calculator(
        self, 
        config: Union[Dict[str, Any], CalculatorConfig]
    ) -> ITechnicalCalculator:
        """
        Create a technical indicator calculator.
        
        Args:
            config: Configuration for the calculator
            
        Returns:
            Technical calculator instance
        """
        pass
    
    @abstractmethod
    def create_sentiment_calculator(
        self,
        config: Union[Dict[str, Any], CalculatorConfig]
    ) -> ISentimentCalculator:
        """
        Create a sentiment analysis calculator.
        
        Args:
            config: Configuration for the calculator
            
        Returns:
            Sentiment calculator instance
        """
        pass
    
    @abstractmethod
    def get_available_calculators(self) -> Dict[str, List[str]]:
        """
        Get list of available calculator types.
        
        Returns:
            Dictionary mapping calculator type to list of implementations
        """
        pass
    
    @abstractmethod
    def register_calculator(
        self,
        calculator_type: str,
        calculator_class: type,
        name: Optional[str] = None
    ) -> None:
        """
        Register a new calculator implementation.
        
        Args:
            calculator_type: Type of calculator (technical, sentiment, etc.)
            calculator_class: Class implementing the calculator interface
            name: Optional name for the calculator
        """
        pass