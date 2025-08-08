"""
Sentiment Calculator Adapter

Adapts the existing SentimentFeaturesCalculator to implement ISentimentCalculator interface.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd

from main.interfaces.calculators import (
    ISentimentCalculator,
    FeatureResult,
    CalculatorConfig
)
from .sentiment_features import SentimentFeaturesCalculator

logger = logging.getLogger(__name__)


class SentimentCalculatorAdapter(ISentimentCalculator):
    """
    Adapter that makes SentimentFeaturesCalculator implement ISentimentCalculator.
    
    This allows the existing calculator to work with the new interface system
    without modifying the original implementation.
    """
    
    def __init__(self, config: Union[Dict[str, Any], CalculatorConfig]):
        """Initialize the adapter with a sentiment calculator."""
        if isinstance(config, CalculatorConfig):
            self._calculator = SentimentFeaturesCalculator(config.parameters)
            self._config = config
        else:
            self._calculator = SentimentFeaturesCalculator(config)
            self._config = CalculatorConfig(
                name="sentiment_features",
                enabled=True,
                parameters=config
            )
    
    def calculate(
        self, 
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> FeatureResult:
        """
        Calculate sentiment features from input data.
        
        Args:
            data: Input data with text content
            symbols: Optional list of symbols to calculate for
            start_date: Optional start date for calculation
            end_date: Optional end date for calculation
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with calculated sentiment features
        """
        start_time = time.time()
        errors = []
        
        try:
            # Filter data if date range specified
            if start_date or end_date:
                mask = pd.Series(True, index=data.index)
                if start_date and 'timestamp' in data.columns:
                    mask &= data['timestamp'] >= start_date
                if end_date and 'timestamp' in data.columns:
                    mask &= data['timestamp'] <= end_date
                data = data[mask]
            
            # Filter by symbols if specified
            if symbols and 'symbol' in data.columns:
                data = data[data['symbol'].isin(symbols)]
            
            # Calculate features using the underlying calculator
            features_df = self._calculator.calculate(data, **kwargs)
            
            metadata = {
                'calculator': 'sentiment_features',
                'config': self._config.parameters,
                'input_shape': data.shape,
                'output_shape': features_df.shape,
                'symbols_processed': symbols or data['symbol'].unique().tolist() if 'symbol' in data.columns else []
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")
            errors.append(str(e))
            features_df = pd.DataFrame()
            metadata = {'error': str(e)}
        
        calculation_time = time.time() - start_time
        
        return FeatureResult(
            features=features_df,
            metadata=metadata,
            calculation_time=calculation_time,
            errors=errors
        )
    
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
        # Convert list to DataFrame if needed
        if isinstance(text_data, list):
            text_data = pd.DataFrame({'text': text_data})
        
        # Use underlying calculator
        return self._calculator.calculate(text_data, **(sentiment_config or {}))
    
    def get_sentiment_model_info(self) -> Dict[str, Any]:
        """
        Get information about the sentiment model being used.
        
        Returns:
            Dictionary with model name, version, and capabilities
        """
        return {
            'model_name': 'sentiment_features',
            'version': '1.0',
            'capabilities': [
                'news_sentiment',
                'social_sentiment', 
                'text_classification',
                'entity_extraction'
            ],
            'supported_languages': ['en'],
            'max_text_length': 512
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this calculator produces.
        
        Returns:
            List of feature column names
        """
        # Get from underlying calculator if it has the method
        if hasattr(self._calculator, 'get_feature_names'):
            return self._calculator.get_feature_names()
        
        # Otherwise return a default list based on typical sentiment features
        return [
            'sentiment_score', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_magnitude', 'sentiment_confidence',
            'news_count', 'news_relevance', 'news_novelty',
            'social_volume', 'social_engagement', 'social_reach'
        ]
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required input columns.
        
        Returns:
            List of column names required in input data
        """
        # Sentiment calculator can work with various text columns
        return []  # No strictly required columns, but needs at least one text column
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required columns and format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, raises exception if invalid
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check for at least one text-like column
        text_columns = ['text', 'content', 'headline', 'title', 'body', 'message']
        has_text = any(col in data.columns for col in text_columns)
        
        if not has_text:
            # Check if any column contains string data
            string_cols = [col for col in data.columns if pd.api.types.is_string_dtype(data[col])]
            if not string_cols:
                raise ValueError(
                    f"No text columns found. Expected one of: {text_columns} "
                    "or any column with string data"
                )
        
        return True