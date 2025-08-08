"""
News Sentiment Calculator

Analyzes sentiment in financial news articles using multiple approaches
including lexicon-based methods, TextBlob analysis, and custom finance-specific
sentiment scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from textblob import TextBlob

from .base_news import BaseNewsCalculator
from ..helpers import (
    create_feature_dataframe, safe_divide, calculate_rolling_mean,
    calculate_rolling_std, aggregate_features, create_rolling_features
)

from main.utils.core import get_logger, RateLimiter

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class SentimentCalculator(BaseNewsCalculator):
    """
    Calculates sentiment-based features from news data.
    
    Features include:
    - Basic sentiment scores (polarity, subjectivity)
    - Time-weighted sentiment
    - Source-weighted sentiment
    - Sentiment momentum and acceleration
    - Sentiment volatility
    - Positive/negative sentiment ratios
    - Sentiment extremes and distribution
    - Cross-temporal sentiment correlations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment calculator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Sentiment-specific configuration
        self.sentiment_windows = config.get('sentiment_windows', [1, 6, 24, 72])
        self.extreme_threshold = config.get('extreme_threshold', 0.8)
        self.neutral_range = config.get('neutral_range', (-0.1, 0.1))
        
        # Finance-specific sentiment adjustments
        self.finance_sentiment_words = {
            'positive': {
                'beat': 0.3, 'exceed': 0.3, 'upgrade': 0.4, 'raise': 0.2,
                'strong': 0.2, 'growth': 0.2, 'profit': 0.1, 'gain': 0.2,
                'bullish': 0.4, 'outperform': 0.3, 'buy': 0.3, 'surge': 0.3,
                'rally': 0.3, 'breakout': 0.2, 'momentum': 0.1, 'recovery': 0.2
            },
            'negative': {
                'miss': -0.3, 'below': -0.2, 'downgrade': -0.4, 'cut': -0.3,
                'weak': -0.2, 'loss': -0.3, 'decline': -0.2, 'fall': -0.2,
                'bearish': -0.4, 'underperform': -0.3, 'sell': -0.3, 'plunge': -0.3,
                'crash': -0.4, 'breakdown': -0.3, 'recession': -0.3, 'warning': -0.2
            }
        }
        
        # Rate limiter for sentiment API calls (if using external service)
        self.rate_limiter = RateLimiter(
            max_calls=100,
            time_window=60  # 100 calls per minute
        )
        
        logger.info("Initialized SentimentCalculator")
    
    def get_feature_names(self) -> List[str]:
        """Get list of sentiment feature names."""
        features = []
        
        # Basic sentiment scores for each time window
        for window in self.time_windows:
            window_suffix = self._get_window_suffix(window)
            
            features.extend([
                f'sentiment_mean_{window_suffix}',
                f'sentiment_weighted_{window_suffix}',
                f'sentiment_std_{window_suffix}',
                f'sentiment_skew_{window_suffix}',
                f'sentiment_positive_ratio_{window_suffix}',
                f'sentiment_negative_ratio_{window_suffix}',
                f'sentiment_neutral_ratio_{window_suffix}',
                f'sentiment_extreme_ratio_{window_suffix}',
                f'subjectivity_mean_{window_suffix}',
                f'subjectivity_std_{window_suffix}'
            ])
        
        # Sentiment momentum and dynamics
        features.extend([
            'sentiment_momentum_1h_6h',
            'sentiment_momentum_6h_24h',
            'sentiment_momentum_24h_72h',
            'sentiment_acceleration',
            'sentiment_volatility_24h',
            'sentiment_trend_strength'
        ])
        
        # Sentiment distribution
        features.extend([
            'sentiment_range',
            'sentiment_iqr',
            'sentiment_entropy',
            'sentiment_concentration',
            'sentiment_divergence'
        ])
        
        # Composite scores
        features.extend([
            'sentiment_composite_score',
            'sentiment_confidence_score',
            'sentiment_agreement_score',
            'sentiment_intensity_score'
        ])
        
        # Finance-specific sentiment
        features.extend([
            'finance_adjusted_sentiment',
            'earnings_sentiment_score',
            'analyst_sentiment_score',
            'market_sentiment_alignment'
        ])
        
        return features
    
    def calculate(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment features from news data.
        
        Args:
            news_data: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            # Validate and prepare data
            validated_data = self.validate_and_prepare_data(news_data)
            if validated_data.empty:
                return self._create_empty_features(news_data.index)
            
            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)
            
            # Calculate base sentiment scores
            sentiment_data = self._calculate_base_sentiment(validated_data)
            
            # Calculate time-windowed features
            for window in self.time_windows:
                window_features = self._calculate_window_sentiment(
                    sentiment_data, window
                )
                features = pd.concat([features, window_features], axis=1)
            
            # Calculate sentiment dynamics
            dynamics_features = self._calculate_sentiment_dynamics(sentiment_data)
            features = pd.concat([features, dynamics_features], axis=1)
            
            # Calculate sentiment distribution
            distribution_features = self._calculate_sentiment_distribution(
                sentiment_data
            )
            features = pd.concat([features, distribution_features], axis=1)
            
            # Calculate composite scores
            composite_features = self._calculate_composite_scores(
                sentiment_data, validated_data
            )
            features = pd.concat([features, composite_features], axis=1)
            
            # Calculate finance-specific sentiment
            finance_features = self._calculate_finance_sentiment(
                validated_data, sentiment_data
            )
            features = pd.concat([features, finance_features], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")
            return self._create_empty_features(news_data.index)
    
    def _calculate_base_sentiment(
        self,
        news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate base sentiment scores for each article."""
        sentiment_data = news_data.copy()
        
        # Calculate sentiment for each article
        sentiments = []
        subjectivities = []
        
        for idx, row in news_data.iterrows():
            # Combine headline and content
            text = f"{row.get('headline', '')} {row.get('content', '')}"
            
            # Calculate TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Apply finance-specific adjustments
            adjusted_polarity = self._adjust_finance_sentiment(text, polarity)
            
            sentiments.append(adjusted_polarity)
            subjectivities.append(subjectivity)
        
        sentiment_data['sentiment'] = sentiments
        sentiment_data['subjectivity'] = subjectivities
        
        # Calculate source credibility weights
        sentiment_data['credibility_weight'] = sentiment_data['source'].apply(
            self.get_source_credibility
        )
        
        # Calculate time decay weights
        sentiment_data['time_weight'] = self.calculate_time_decay(
            sentiment_data['timestamp']
        )
        
        # Combined weight
        sentiment_data['combined_weight'] = (
            sentiment_data['credibility_weight'] * sentiment_data['time_weight']
        )
        
        return sentiment_data
    
    def _calculate_window_sentiment(
        self,
        sentiment_data: pd.DataFrame,
        window_hours: int
    ) -> pd.DataFrame:
        """Calculate sentiment features for a specific time window."""
        features = pd.DataFrame(index=sentiment_data.index)
        window_suffix = self._get_window_suffix(window_hours)
        
        # Filter data within window
        window_data = self.filter_by_time_window(sentiment_data, window_hours)
        
        if not window_data.empty:
            # Basic statistics
            features[f'sentiment_mean_{window_suffix}'] = window_data.groupby(
                window_data.index
            )['sentiment'].transform('mean')
            
            # Weighted sentiment
            features[f'sentiment_weighted_{window_suffix}'] = self._calculate_weighted_sentiment(
                window_data
            )
            
            # Sentiment variability
            features[f'sentiment_std_{window_suffix}'] = window_data.groupby(
                window_data.index
            )['sentiment'].transform('std').fillna(0)
            
            # Sentiment skewness
            features[f'sentiment_skew_{window_suffix}'] = window_data.groupby(
                window_data.index
            )['sentiment'].transform('skew').fillna(0)
            
            # Sentiment ratios
            features[f'sentiment_positive_ratio_{window_suffix}'] = \
                self._calculate_sentiment_ratio(window_data, 'positive')
            features[f'sentiment_negative_ratio_{window_suffix}'] = \
                self._calculate_sentiment_ratio(window_data, 'negative')
            features[f'sentiment_neutral_ratio_{window_suffix}'] = \
                self._calculate_sentiment_ratio(window_data, 'neutral')
            
            # Extreme sentiment ratio
            features[f'sentiment_extreme_ratio_{window_suffix}'] = \
                self._calculate_extreme_ratio(window_data)
            
            # Subjectivity statistics
            features[f'subjectivity_mean_{window_suffix}'] = window_data.groupby(
                window_data.index
            )['subjectivity'].transform('mean')
            
            features[f'subjectivity_std_{window_suffix}'] = window_data.groupby(
                window_data.index
            )['subjectivity'].transform('std').fillna(0)
        
        return features.fillna(0)
    
    def _calculate_sentiment_dynamics(
        self,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate sentiment momentum and dynamics."""
        features = pd.DataFrame(index=sentiment_data.index)
        
        # Calculate average sentiment for different windows
        sent_1h = self._get_window_average(sentiment_data, 1, 'sentiment')
        sent_6h = self._get_window_average(sentiment_data, 6, 'sentiment')
        sent_24h = self._get_window_average(sentiment_data, 24, 'sentiment')
        sent_72h = self._get_window_average(sentiment_data, 72, 'sentiment')
        
        # Sentiment momentum
        features['sentiment_momentum_1h_6h'] = safe_divide(
            sent_1h - sent_6h, sent_6h.abs() + 0.1
        )
        features['sentiment_momentum_6h_24h'] = safe_divide(
            sent_6h - sent_24h, sent_24h.abs() + 0.1
        )
        features['sentiment_momentum_24h_72h'] = safe_divide(
            sent_24h - sent_72h, sent_72h.abs() + 0.1
        )
        
        # Sentiment acceleration
        features['sentiment_acceleration'] = (
            features['sentiment_momentum_1h_6h'] - 
            features['sentiment_momentum_6h_24h']
        )
        
        # Sentiment volatility (24h rolling)
        # Create DataFrame for rolling features
        sent_df = pd.DataFrame({'sentiment': sentiment_data['sentiment']}, index=sentiment_data.index)
        rolling_features = create_rolling_features(
            sent_df,
            columns=['sentiment'],
            windows=['24H'],
            operations=['std'],
            min_periods=1
        )
        features['sentiment_volatility_24h'] = rolling_features['sentiment_rolling_std_24H'].fillna(0)
        
        # Sentiment trend strength
        features['sentiment_trend_strength'] = self._calculate_trend_strength(
            sentiment_data
        )
        
        return features
    
    def _calculate_sentiment_distribution(
        self,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate sentiment distribution metrics."""
        features = pd.DataFrame(index=sentiment_data.index)
        
        # Group by timestamp for calculations
        grouped = sentiment_data.groupby(sentiment_data.index)['sentiment']
        
        # Sentiment range
        features['sentiment_range'] = grouped.transform(
            lambda x: x.max() - x.min()
        ).fillna(0)
        
        # Interquartile range
        features['sentiment_iqr'] = grouped.transform(
            lambda x: x.quantile(0.75) - x.quantile(0.25) if len(x) > 1 else 0
        ).fillna(0)
        
        # Sentiment entropy (diversity)
        features['sentiment_entropy'] = grouped.transform(
            self._calculate_sentiment_entropy
        ).fillna(0)
        
        # Sentiment concentration (Herfindahl index)
        features['sentiment_concentration'] = grouped.transform(
            self._calculate_sentiment_concentration
        ).fillna(0)
        
        # Sentiment divergence from mean
        features['sentiment_divergence'] = grouped.transform(
            lambda x: (x - x.mean()).abs().mean() if len(x) > 1 else 0
        ).fillna(0)
        
        return features
    
    def _calculate_composite_scores(
        self,
        sentiment_data: pd.DataFrame,
        news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite sentiment scores."""
        features = pd.DataFrame(index=sentiment_data.index)
        
        # Composite sentiment score (weighted average across time windows)
        sent_scores = []
        weights = [0.4, 0.3, 0.2, 0.1]  # Recent news weighted more
        
        for window, weight in zip([1, 6, 24, 72], weights):
            window_sent = self._get_window_average(sentiment_data, window, 'sentiment')
            sent_scores.append(window_sent * weight)
        
        features['sentiment_composite_score'] = sum(sent_scores)
        
        # Confidence score (based on volume and credibility)
        volume_factor = news_data.groupby(news_data.index).size()
        volume_factor = np.clip(safe_divide(volume_factor, 10, default_value=0), 0, 1)  # Normalize
        
        avg_credibility = sentiment_data.groupby(
            sentiment_data.index
        )['credibility_weight'].transform('mean')
        
        features['sentiment_confidence_score'] = volume_factor * avg_credibility
        
        # Agreement score (inverse of standard deviation)
        sentiment_std = sentiment_data.groupby(
            sentiment_data.index
        )['sentiment'].transform('std').fillna(0)
        
        features['sentiment_agreement_score'] = safe_divide(1, (1 + sentiment_std), default_value=1.0)
        
        # Intensity score (absolute sentiment * volume)
        abs_sentiment = sentiment_data.groupby(
            sentiment_data.index
        )['sentiment'].transform(lambda x: x.abs().mean())
        
        features['sentiment_intensity_score'] = abs_sentiment * volume_factor
        
        return features
    
    def _calculate_finance_sentiment(
        self,
        news_data: pd.DataFrame,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate finance-specific sentiment features."""
        features = pd.DataFrame(index=news_data.index)
        
        # Finance-adjusted sentiment (already calculated in base)
        features['finance_adjusted_sentiment'] = sentiment_data.groupby(
            sentiment_data.index
        )['sentiment'].transform('mean')
        
        # Earnings-related sentiment
        earnings_mask = news_data['headline'].str.contains(
            'earnings|revenue|profit|loss|guidance',
            case=False,
            na=False
        )
        features['earnings_sentiment_score'] = self._calculate_category_sentiment(
            sentiment_data, earnings_mask
        )
        
        # Analyst-related sentiment
        analyst_mask = news_data['headline'].str.contains(
            'analyst|upgrade|downgrade|rating|target',
            case=False,
            na=False
        )
        features['analyst_sentiment_score'] = self._calculate_category_sentiment(
            sentiment_data, analyst_mask
        )
        
        # Market sentiment alignment (correlation with market terms)
        market_terms = ['bull', 'bear', 'rally', 'sell-off', 'correction']
        market_sentiment = 0
        
        for term in market_terms:
            term_mask = news_data['headline'].str.contains(term, case=False, na=False)
            if term_mask.any():
                term_sent = sentiment_data[term_mask].groupby(
                    sentiment_data[term_mask].index
                )['sentiment'].transform('mean')
                market_sentiment += term_sent.reindex(features.index, fill_value=0)
        
        features['market_sentiment_alignment'] = safe_divide(market_sentiment, len(market_terms), default_value=0.0)
        
        return features
    
    def _adjust_finance_sentiment(
        self,
        text: str,
        base_sentiment: float
    ) -> float:
        """Adjust sentiment based on finance-specific terms."""
        text_lower = text.lower()
        adjustment = 0
        
        # Check positive terms
        for word, score in self.finance_sentiment_words['positive'].items():
            if word in text_lower:
                adjustment += score
        
        # Check negative terms
        for word, score in self.finance_sentiment_words['negative'].items():
            if word in text_lower:
                adjustment += score
        
        # Combine with base sentiment
        adjusted = base_sentiment + adjustment * 0.5  # Moderate the adjustment
        
        # Clip to valid range
        return np.clip(adjusted, -1, 1)
    
    def _calculate_weighted_sentiment(
        self,
        window_data: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted sentiment score."""
        grouped = window_data.groupby(window_data.index)
        
        def weighted_mean(group):
            if len(group) == 0:
                return 0
            weights = group['combined_weight']
            values = group['sentiment']
            if weights.sum() == 0:
                return values.mean()
            return safe_divide((weights * values).sum(), weights.sum(), default_value=0.0)
        
        return grouped.apply(weighted_mean)
    
    def _calculate_sentiment_ratio(
        self,
        window_data: pd.DataFrame,
        sentiment_type: str
    ) -> pd.Series:
        """Calculate ratio of specific sentiment type."""
        grouped = window_data.groupby(window_data.index)
        
        def calc_ratio(group):
            if len(group) == 0:
                return 0
            
            if sentiment_type == 'positive':
                count = (group['sentiment'] > self.neutral_range[1]).sum()
            elif sentiment_type == 'negative':
                count = (group['sentiment'] < self.neutral_range[0]).sum()
            else:  # neutral
                count = ((group['sentiment'] >= self.neutral_range[0]) & 
                        (group['sentiment'] <= self.neutral_range[1])).sum()
            
            return safe_divide(count, len(group), default_value=0.0)
        
        return grouped.apply(calc_ratio)
    
    def _calculate_extreme_ratio(
        self,
        window_data: pd.DataFrame
    ) -> pd.Series:
        """Calculate ratio of extreme sentiments."""
        grouped = window_data.groupby(window_data.index)
        
        def calc_extreme(group):
            if len(group) == 0:
                return 0
            extreme_count = (group['sentiment'].abs() > self.extreme_threshold).sum()
            return safe_divide(extreme_count, len(group), default_value=0.0)
        
        return grouped.apply(calc_extreme)
    
    def _calculate_trend_strength(
        self,
        sentiment_data: pd.DataFrame
    ) -> pd.Series:
        """Calculate sentiment trend strength."""
        # Use 24h rolling window for trend
        sent_df = pd.DataFrame({'sentiment': sentiment_data['sentiment']}, index=sentiment_data.index)
        rolling_features = create_rolling_features(
            sent_df,
            columns=['sentiment'],
            windows=['24H'],
            operations=['mean'],
            min_periods=1
        )
        rolling_mean = rolling_features['sentiment_rolling_mean_24H']
        
        # Calculate slope of trend
        def calc_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            return slope
        
        # Calculate trend using custom logic
        trend_strength = pd.Series(index=rolling_mean.index, dtype=float)
        for i in range(len(rolling_mean)):
            if i < 1:  # min_periods=2
                trend_strength.iloc[i] = 0
            else:
                window_start = max(0, i - 23)  # 24 window
                window_data = rolling_mean.iloc[window_start:i+1]
                if len(window_data) >= 2:
                    trend_strength.iloc[i] = calc_slope(window_data.values)
                else:
                    trend_strength.iloc[i] = 0
        
        return trend_strength.fillna(0)
    
    def _calculate_sentiment_entropy(self, sentiments: pd.Series) -> float:
        """Calculate entropy of sentiment distribution."""
        if len(sentiments) < 2:
            return 0
        
        # Discretize sentiments into bins
        bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
        counts, _ = np.histogram(sentiments, bins=bins)
        
        # Calculate entropy
        probs = safe_divide(counts, counts.sum(), default_value=0.0)
        probs = probs[probs > 0]  # Remove zeros
        
        if len(probs) == 0:
            return 0
        
        entropy = -np.sum(probs * np.log2(probs))
        return safe_divide(entropy, np.log2(len(bins) - 1), default_value=0.0)  # Normalize
    
    def _calculate_sentiment_concentration(
        self,
        sentiments: pd.Series
    ) -> float:
        """Calculate concentration of sentiments (Herfindahl index)."""
        if len(sentiments) < 2:
            return 1
        
        # Discretize sentiments
        bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
        counts, _ = np.histogram(sentiments, bins=bins)
        
        # Calculate Herfindahl index
        shares = safe_divide(counts, counts.sum(), default_value=0.0)
        hhi = np.sum(shares ** 2)
        
        return hhi
    
    def _calculate_category_sentiment(
        self,
        sentiment_data: pd.DataFrame,
        category_mask: pd.Series
    ) -> pd.Series:
        """Calculate sentiment for specific category of news."""
        result = pd.Series(index=sentiment_data.index, dtype=float)
        
        if category_mask.any():
            category_data = sentiment_data[category_mask]
            grouped_sentiment = category_data.groupby(
                category_data.index
            )['sentiment'].mean()
            result.update(grouped_sentiment)
        
        return result.fillna(0)
    
    def _get_window_average(
        self,
        data: pd.DataFrame,
        window_hours: int,
        column: str
    ) -> pd.Series:
        """Get average value for a time window."""
        window_data = self.filter_by_time_window(data, window_hours)
        if window_data.empty:
            return pd.Series(0, index=data.index)
        
        return window_data.groupby(window_data.index)[column].transform('mean')