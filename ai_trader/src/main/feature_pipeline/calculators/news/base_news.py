"""
Base News Calculator

Shared utilities and base functionality for all news-based calculators.
Provides common methods for news processing, text analysis, and feature creation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import re
from datetime import datetime, timedelta
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

from ..base_calculator import BaseFeatureCalculator
from ..helpers import (
    safe_divide, validate_news_data, ensure_datetime_index,
    create_rolling_features, postprocess_features
)
from .news_config import NewsConfig
from main.utils.core import get_logger, ErrorHandlingMixin, ensure_utc

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except Exception as e:
        logger.warning(f"Could not download NLTK stopwords: {e}")


class BaseNewsCalculator(BaseFeatureCalculator, ErrorHandlingMixin):
    """
    Base class for all news-based feature calculators.
    
    Provides shared utilities for:
    - News data preprocessing and validation
    - Text processing and sentiment analysis
    - Time window calculations
    - Source credibility weighting
    - Common feature creation patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base news calculator."""
        super().__init__(config)
        
        # Initialize news configuration
        self.news_config = NewsConfig(
            **config.get('news', {}) if config else {}
        )
        
        # News data storage
        self.news_data = None
        
        # Text processing utilities
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            logger.warning("Could not load NLTK stopwords, using empty set")
        
        # TF-IDF vectorizer for topic analysis
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Numerical tolerance for calculations
        self.numerical_tolerance = 1e-10
    
    def set_news_data(self, news_data: pd.DataFrame):
        """
        Set news data for feature calculation.
        
        Args:
            news_data: DataFrame with news articles including timestamp, text, source columns
        """
        if news_data is None or news_data.empty:
            logger.warning("Empty news data provided")
            self.news_data = pd.DataFrame()
            return
        
        self.news_data = news_data.copy()
        
        # Preprocess news data
        self._preprocess_news_data()
    
    def _preprocess_news_data(self):
        """Preprocess news data for analysis."""
        if self.news_data is None or self.news_data.empty:
            return
        
        try:
            # Use centralized validation
            is_valid, errors = validate_news_data(self.news_data)
            if not is_valid:
                logger.warning(f"News data validation failed: {errors}")
                if self.news_config.strict_validation:
                    self.news_data = pd.DataFrame()
                    return
            
            # Ensure datetime index
            self.news_data = ensure_datetime_index(self.news_data, datetime_column='timestamp')
            
            # Ensure UTC timezone
            if self.news_data.index.tz is None:
                self.news_data.index = self.news_data.index.tz_localize('UTC')
            else:
                self.news_data.index = self.news_data.index.tz_convert('UTC')
            
            # Add source weight if not present
            if 'source' in self.news_data.columns and 'weight' not in self.news_data.columns:
                self.news_data['weight'] = self.news_data['source'].map(
                    lambda x: self.news_config.source_weights.get(str(x).lower(), 0.5)
                )
            
            # Sort by timestamp for efficient time window processing
            self.news_data = self.news_data.sort_values('timestamp')
            
            logger.debug(f"Preprocessed {len(self.news_data)} news articles")
            
        except Exception as e:
            logger.error(f"Error preprocessing news data: {e}")
    
    def get_news_in_window(self, end_time: pd.Timestamp, window_hours: int) -> pd.DataFrame:
        """
        Get news articles within a time window.
        
        Args:
            end_time: End time of the window
            window_hours: Window size in hours
            
        Returns:
            DataFrame with news articles in the time window
        """
        if self.news_data is None or self.news_data.empty:
            return pd.DataFrame()
        
        try:
            window_start = end_time - timedelta(hours=window_hours)
            
            # Filter news in window
            mask = (
                (self.news_data['timestamp'] >= window_start) & 
                (self.news_data['timestamp'] <= end_time)
            )
            
            return self.news_data[mask].copy()
            
        except Exception as e:
            logger.error(f"Error filtering news in window: {e}")
            return pd.DataFrame()
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            if not text or pd.isna(text):
                return {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'compound': 0.0
                }
            
            blob = TextBlob(str(text))
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity,  # 0 (objective) to 1 (subjective)
                'compound': blob.sentiment.polarity * blob.sentiment.subjectivity  # Combined score
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'compound': 0.0
            }
    
    def calculate_news_weights(self, news_df: pd.DataFrame, reference_time: pd.Timestamp) -> pd.Series:
        """
        Calculate weights based on source credibility and time decay.
        
        Args:
            news_df: DataFrame with news articles
            reference_time: Reference time for decay calculation
            
        Returns:
            Series with calculated weights
        """
        if news_df.empty:
            return pd.Series(dtype=float)
        
        try:
            weights = pd.Series(1.0, index=news_df.index)
            
            # Source credibility weight
            if 'weight' in news_df.columns:
                weights *= news_df['weight']
            elif 'source' in news_df.columns:
                source_weights = news_df['source'].map(
                    lambda x: self.news_config.source_weights.get(str(x).lower(), 0.5)
                )
                weights *= source_weights
            
            # Time decay weight
            if 'timestamp' in news_df.columns:
                time_diff = (reference_time - news_df['timestamp']).dt.total_seconds() / 3600  # Hours
                decay_weight = np.exp(-np.log(2) * time_diff / self.news_config.decay_half_life)
                weights *= decay_weight
            
            # Normalize weights
            total_weight = weights.sum()
            if total_weight > self.numerical_tolerance:
                return weights / total_weight
            else:
                return weights
                
        except Exception as e:
            logger.error(f"Error calculating news weights: {e}")
            return pd.Series(1.0, index=news_df.index)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        try:
            if not text or pd.isna(text):
                return ""
            
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove special characters and digits, keep letters and spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove stopwords if available
            if self.stop_words:
                words = text.split()
                words = [word for word in words if word not in self.stop_words and len(word) > 2]
                text = ' '.join(words)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error preprocessing text: {e}")
            return str(text) if text else ""
    
    def extract_topics(self, texts: pd.Series) -> List[List[str]]:
        """
        Extract topics from news texts using TF-IDF.
        
        Args:
            texts: Series of text content
            
        Returns:
            List of topic word lists
        """
        try:
            if texts.empty:
                return []
            
            # Preprocess texts
            processed_texts = texts.apply(self.preprocess_text)
            processed_texts = processed_texts[processed_texts.str.len() > 0]
            
            if processed_texts.empty:
                return []
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(processed_texts)
            
            # Get feature names
            feature_names = self.tfidf.get_feature_names_out()
            
            # Extract top terms for topics
            topics = []
            n_topics = min(self.news_config.n_topics, len(feature_names))
            
            # Get mean TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top features
            top_indices = mean_scores.argsort()[-n_topics:][::-1]
            
            for i in range(0, len(top_indices), self.news_config.min_topic_words):
                topic_indices = top_indices[i:i + self.news_config.min_topic_words]
                topic_words = [feature_names[idx] for idx in topic_indices if mean_scores[idx] > 0]
                
                if topic_words:
                    topics.append(topic_words)
            
            return topics
            
        except Exception as e:
            logger.warning(f"Error extracting topics: {e}")
            return []
    
    def calculate_rolling_news_metric(self, data: pd.DataFrame, window_hours: int, 
                                    metric_func, default_value: float = 0.0) -> pd.Series:
        """
        Calculate a rolling news metric over time windows.
        
        Args:
            data: DataFrame with price/market data (index is timestamps)
            window_hours: Window size in hours
            metric_func: Function that takes news DataFrame and returns a metric value
            default_value: Default value when no news is available
            
        Returns:
            Series with calculated metric values
        """
        try:
            result = []
            
            for timestamp in data.index:
                try:
                    # Get news in window
                    window_news = self.get_news_in_window(timestamp, window_hours)
                    
                    if window_news.empty:
                        result.append(default_value)
                    else:
                        # Calculate metric
                        metric_value = metric_func(window_news)
                        result.append(metric_value if not pd.isna(metric_value) else default_value)
                        
                except Exception as e:
                    logger.warning(f"Error calculating metric for {timestamp}: {e}")
                    result.append(default_value)
            
            return pd.Series(result, index=data.index)
            
        except Exception as e:
            logger.error(f"Error in rolling news metric calculation: {e}")
            return pd.Series(default_value, index=data.index)
    
    def create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """Create empty features DataFrame with given index."""
        try:
            feature_names = self.get_feature_names()
            features = pd.DataFrame(index=index, columns=feature_names, dtype=float)
            features[:] = np.nan
            return features
        except Exception as e:
            logger.error(f"Error creating empty features: {e}")
            return pd.DataFrame(index=index)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names. Should be implemented by subclasses."""
        return []
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for news analysis."""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data for news calculations."""
        try:
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for sufficient data
            if len(data) < 1:
                logger.error("Insufficient data for news analysis")
                return False
            
            # Check for valid numeric data
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logger.error(f"Column '{col}' must be numeric")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False