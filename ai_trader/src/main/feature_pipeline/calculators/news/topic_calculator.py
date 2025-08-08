"""
News Topic Calculator

Specialized calculator for news topic modeling and categorization including:
- TF-IDF based topic extraction
- Topic presence scoring across time windows
- Topic diversity and concentration metrics
- Dynamic topic tracking and evolution
- Entity and keyword extraction
- Topic sentiment correlation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import timedelta
from collections import Counter
import re

from .base_news import BaseNewsCalculator
from ..helpers import create_rolling_features

logger = logging.getLogger(__name__)


class NewsTopicCalculator(BaseNewsCalculator):
    """Calculator for news topic modeling and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize news topic calculator."""
        super().__init__(config)
        
        # Topic-specific parameters
        self.topic_window_days = self.config.get('topic_window_days', 7)
        self.min_articles_for_topics = self.config.get('min_articles_for_topics', 10)
        self.topic_score_window = self.config.get('topic_score_window', 24)  # Hours
        
        # Predefined category keywords
        self.category_keywords = {
            'earnings': ['earnings', 'revenue', 'eps', 'guidance', 'quarter', 'beat', 'miss', 'profit', 'loss'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover', 'deal', 'consolidation'],
            'regulatory': ['fda', 'approval', 'regulation', 'compliance', 'investigation', 'lawsuit'],
            'management': ['ceo', 'cfo', 'executive', 'leadership', 'management', 'board', 'resign'],
            'product': ['product', 'launch', 'release', 'innovation', 'patent', 'technology'],
            'financial': ['debt', 'credit', 'loan', 'financing', 'capital', 'ipo', 'dividend'],
            'market': ['market', 'sector', 'industry', 'competition', 'share', 'growth'],
            'crisis': ['crisis', 'scandal', 'fraud', 'bankruptcy', 'default', 'emergency']
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of news topic feature names."""
        feature_names = []
        
        # Dynamic topic features (based on n_topics from config)
        for i in range(self.news_config.n_topics):
            feature_names.extend([
                f'news_topic_{i}_score_24h',
                f'news_topic_{i}_trend_24h',
                f'news_topic_{i}_intensity'
            ])
        
        # Predefined category features
        for category in self.category_keywords.keys():
            feature_names.extend([
                f'news_category_{category}_24h',
                f'news_category_{category}_72h',
                f'news_category_{category}_intensity'
            ])
        
        # Topic aggregation features
        feature_names.extend([
            'news_topic_diversity_24h',
            'news_topic_concentration_24h',
            'news_dominant_topic_score',
            'news_topic_stability_24h',
            'news_topic_emergence_24h',
            'news_entity_mentions_24h',
            'news_keyword_density_24h',
            'news_topic_sentiment_correlation',
            'news_category_breadth_24h',
            'news_content_complexity_24h'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate news topic features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with news topic features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)
            
            if self.news_data is None or self.news_data.empty:
                logger.warning("No news data available for topic calculation")
                return features
            
            # Extract topics from recent news
            self._extract_dynamic_topics(data)
            
            # Calculate dynamic topic features
            features = self._calculate_dynamic_topic_features(data, features)
            
            # Calculate predefined category features
            features = self._calculate_category_features(data, features)
            
            # Calculate topic aggregation features
            features = self._calculate_topic_aggregation_features(features)
            
            # Calculate advanced topic metrics
            features = self._calculate_advanced_topic_metrics(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating news topic features: {e}")
            return self.create_empty_features(data.index)
    
    def _extract_dynamic_topics(self, data: pd.DataFrame):
        """Extract topics from recent news data."""
        try:
            # Get recent news for topic modeling
            end_time = data.index[-1] if len(data) > 0 else pd.Timestamp.now()
            start_time = end_time - timedelta(days=self.topic_window_days)
            
            recent_news = self.news_data[
                (self.news_data['timestamp'] >= start_time) &
                (self.news_data['timestamp'] <= end_time)
            ]
            
            if len(recent_news) < self.min_articles_for_topics:
                logger.warning(f"Insufficient articles ({len(recent_news)}) for topic modeling")
                self.topics = []
                return
            
            # Extract topics using base class method
            if 'text' in recent_news.columns:
                self.topics = self.extract_topics(recent_news['text'])
            else:
                logger.warning("No text column found in news data")
                self.topics = []
            
            logger.debug(f"Extracted {len(self.topics)} topics from {len(recent_news)} articles")
            
        except Exception as e:
            logger.error(f"Error extracting dynamic topics: {e}")
            self.topics = []
    
    def _calculate_dynamic_topic_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for dynamically extracted topics."""
        
        if not hasattr(self, 'topics') or not self.topics:
            # Fill with zeros if no topics extracted
            for i in range(self.news_config.n_topics):
                features[f'news_topic_{i}_score_24h'] = 0.0
                features[f'news_topic_{i}_trend_24h'] = 0.0
                features[f'news_topic_{i}_intensity'] = 0.0
            return features
        
        # Calculate topic presence for each timestamp
        for i, topic_words in enumerate(self.topics[:self.news_config.n_topics]):
            
            def topic_score_func(timestamp):
                return self._calculate_topic_score(timestamp, topic_words, self.topic_score_window)
            
            # Calculate topic scores
            topic_scores = [topic_score_func(ts) for ts in data.index]
            features[f'news_topic_{i}_score_24h'] = topic_scores
            
            # Calculate topic trend
            topic_series = pd.Series(topic_scores, index=data.index)
            features[f'news_topic_{i}_trend_24h'] = self._calculate_trend(topic_series, window=24)
            
            # Calculate topic intensity (weighted by sentiment)
            def topic_intensity_func(timestamp):
                return self._calculate_topic_intensity(timestamp, topic_words, self.topic_score_window)
            
            topic_intensities = [topic_intensity_func(ts) for ts in data.index]
            features[f'news_topic_{i}_intensity'] = topic_intensities
        
        return features
    
    def _calculate_category_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for predefined news categories."""
        
        for category, keywords in self.category_keywords.items():
            
            # 24h category presence
            def category_24h_func(timestamp):
                return self._calculate_topic_score(timestamp, keywords, 24)
            
            category_scores_24h = [category_24h_func(ts) for ts in data.index]
            features[f'news_category_{category}_24h'] = category_scores_24h
            
            # 72h category presence
            def category_72h_func(timestamp):
                return self._calculate_topic_score(timestamp, keywords, 72)
            
            category_scores_72h = [category_72h_func(ts) for ts in data.index]
            features[f'news_category_{category}_72h'] = category_scores_72h
            
            # Category intensity (weighted)
            def category_intensity_func(timestamp):
                return self._calculate_topic_intensity(timestamp, keywords, 24)
            
            category_intensities = [category_intensity_func(ts) for ts in data.index]
            features[f'news_category_{category}_intensity'] = category_intensities
        
        return features
    
    def _calculate_topic_aggregation_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated topic features."""
        
        # Get topic score columns
        topic_score_cols = [col for col in features.columns if 'topic_' in col and '_score_24h' in col]
        category_cols = [col for col in features.columns if 'category_' in col and '_24h' in col]
        
        if topic_score_cols:
            # Topic diversity (standard deviation of topic scores)
            features['news_topic_diversity_24h'] = features[topic_score_cols].std(axis=1)
            
            # Topic concentration (max topic score)
            features['news_topic_concentration_24h'] = features[topic_score_cols].max(axis=1)
            
            # Dominant topic score
            features['news_dominant_topic_score'] = features[topic_score_cols].max(axis=1)
            
            # Topic stability (how much topic distribution changes)
            topic_matrix = features[topic_score_cols].values
            topic_stability = []
            for i in range(len(topic_matrix)):
                if i == 0:
                    topic_stability.append(1.0)  # Perfect stability for first point
                else:
                    # Calculate cosine similarity between consecutive topic distributions
                    prev_dist = topic_matrix[i-1]
                    curr_dist = topic_matrix[i]
                    
                    # Normalize distributions
                    prev_norm = np.linalg.norm(prev_dist) + self.numerical_tolerance
                    curr_norm = np.linalg.norm(curr_dist) + self.numerical_tolerance
                    
                    if prev_norm > 0 and curr_norm > 0:
                        similarity = np.dot(prev_dist, curr_dist) / (prev_norm * curr_norm)
                        topic_stability.append(max(0, similarity))
                    else:
                        topic_stability.append(0.0)
            
            features['news_topic_stability_24h'] = topic_stability
            
            # Topic emergence (sudden increase in topic presence)
            emergence_scores = []
            for col in topic_score_cols:
                series = features[col]
                # Create DataFrame for rolling features
                series_df = pd.DataFrame({col: series}, index=series.index)
                rolling_features_calc = create_rolling_features(
                    series_df,
                    columns=[col],
                    windows=[6],
                    operations=['mean'],
                    min_periods=2
                )
                rolling_mean = rolling_features_calc[f'{col}_rolling_mean_6']
                current_vs_baseline = series / (rolling_mean + self.numerical_tolerance)
                emergence_scores.append(current_vs_baseline)
            
            if emergence_scores:
                features['news_topic_emergence_24h'] = pd.concat(emergence_scores, axis=1).max(axis=1)
        
        # Category breadth (how many categories are active)
        if category_cols:
            # Count categories with significant presence (> threshold)
            category_threshold = 0.1
            active_categories = (features[category_cols] > category_threshold).sum(axis=1)
            features['news_category_breadth_24h'] = active_categories
        
        return features
    
    def _calculate_advanced_topic_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced topic-related metrics."""
        
        # Entity mentions (simplified - count of proper nouns and financial terms)
        def entity_mentions_func(timestamp):
            return self._count_entity_mentions(timestamp, 24)
        
        entity_counts = [entity_mentions_func(ts) for ts in data.index]
        features['news_entity_mentions_24h'] = entity_counts
        
        # Keyword density (average keywords per article)
        def keyword_density_func(timestamp):
            return self._calculate_keyword_density(timestamp, 24)
        
        keyword_densities = [keyword_density_func(ts) for ts in data.index]
        features['news_keyword_density_24h'] = keyword_densities
        
        # Topic-sentiment correlation
        if hasattr(self, 'topics') and self.topics:
            def topic_sentiment_corr_func(timestamp):
                return self._calculate_topic_sentiment_correlation(timestamp, 24)
            
            correlations = [topic_sentiment_corr_func(ts) for ts in data.index]
            features['news_topic_sentiment_correlation'] = correlations
        else:
            features['news_topic_sentiment_correlation'] = 0.0
        
        # Content complexity (based on text length and vocabulary diversity)
        def content_complexity_func(timestamp):
            return self._calculate_content_complexity(timestamp, 24)
        
        complexities = [content_complexity_func(ts) for ts in data.index]
        features['news_content_complexity_24h'] = complexities
        
        return features
    
    def _calculate_topic_score(self, timestamp: pd.Timestamp, keywords: List[str], window_hours: int) -> float:
        """Calculate topic presence score for given keywords and time window."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            # Calculate topic presence for each article
            topic_scores = []
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                text_lower = str(text).lower()
                
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                score = matches / len(keywords) if keywords else 0.0
                topic_scores.append(score)
            
            if not topic_scores:
                return 0.0
            
            # Weight by news credibility
            weights = self.calculate_news_weights(window_news, timestamp)
            if len(weights) > 0 and weights.sum() > 0:
                weighted_score = sum(score * weight for score, weight in zip(topic_scores, weights)) / weights.sum()
            else:
                weighted_score = np.mean(topic_scores)
            
            return weighted_score
            
        except Exception as e:
            logger.warning(f"Error calculating topic score: {e}")
            return 0.0
    
    def _calculate_topic_intensity(self, timestamp: pd.Timestamp, keywords: List[str], window_hours: int) -> float:
        """Calculate topic intensity (topic score weighted by sentiment)."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            topic_scores = []
            sentiment_weights = []
            
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                text_lower = str(text).lower()
                
                # Calculate topic score
                matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                topic_score = matches / len(keywords) if keywords else 0.0
                
                if topic_score > 0:
                    # Get sentiment for weighting
                    sentiment = self.calculate_sentiment(text)
                    sentiment_magnitude = abs(sentiment['polarity']) * sentiment['subjectivity']
                    
                    topic_scores.append(topic_score)
                    sentiment_weights.append(sentiment_magnitude)
            
            if not topic_scores:
                return 0.0
            
            # Calculate intensity as topic score weighted by sentiment magnitude
            intensity = sum(score * weight for score, weight in zip(topic_scores, sentiment_weights))
            intensity = intensity / len(topic_scores) if topic_scores else 0.0
            
            return intensity
            
        except Exception as e:
            logger.warning(f"Error calculating topic intensity: {e}")
            return 0.0
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend for a time series."""
        def trend_func(values):
            if len(values) < 3:
                return 0.0
            try:
                x = np.arange(len(values))
                y = values.values
                trend = np.polyfit(x, y, 1)[0]  # Slope
                return trend
            except (ValueError, TypeError, np.linalg.LinAlgError):
                return 0.0
        
        # Use custom implementation for trend calculation
        trend_values = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            if i < 2:  # min_periods=3
                trend_values.iloc[i] = 0.0
            else:
                window_start = max(0, i - window + 1)
                window_data = series.iloc[window_start:i+1]
                if len(window_data) >= 3:
                    trend_values.iloc[i] = trend_func(window_data.values)
                else:
                    trend_values.iloc[i] = 0.0
        
        return trend_values
    
    def _count_entity_mentions(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Count entity mentions (simplified approach)."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            total_entities = 0
            total_articles = 0
            
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                # Simple entity detection - count capitalized words
                words = str(text).split()
                entities = sum(1 for word in words if word[0].isupper() and len(word) > 2)
                
                total_entities += entities
                total_articles += 1
            
            return total_entities / total_articles if total_articles > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error counting entity mentions: {e}")
            return 0.0
    
    def _calculate_keyword_density(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate keyword density (keywords per article)."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            # Combine all category keywords
            all_keywords = []
            for keywords in self.category_keywords.values():
                all_keywords.extend(keywords)
            
            total_keywords = 0
            total_articles = 0
            
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                text_lower = str(text).lower()
                keyword_count = sum(1 for keyword in all_keywords if keyword in text_lower)
                
                total_keywords += keyword_count
                total_articles += 1
            
            return total_keywords / total_articles if total_articles > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating keyword density: {e}")
            return 0.0
    
    def _calculate_topic_sentiment_correlation(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate correlation between topic presence and sentiment."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or len(window_news) < 3 or 'text' not in window_news.columns:
                return 0.0
            
            if not hasattr(self, 'topics') or not self.topics:
                return 0.0
            
            # Calculate topic scores and sentiments for each article
            topic_scores = []
            sentiments = []
            
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                # Calculate average topic score across all topics
                text_lower = str(text).lower()
                article_topic_scores = []
                
                for topic_words in self.topics:
                    matches = sum(1 for word in topic_words if word.lower() in text_lower)
                    score = matches / len(topic_words) if topic_words else 0.0
                    article_topic_scores.append(score)
                
                if article_topic_scores:
                    avg_topic_score = np.mean(article_topic_scores)
                    sentiment = self.calculate_sentiment(text)
                    
                    topic_scores.append(avg_topic_score)
                    sentiments.append(sentiment['polarity'])
            
            if len(topic_scores) < 3:
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(topic_scores, sentiments)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating topic-sentiment correlation: {e}")
            return 0.0
    
    def _calculate_content_complexity(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate content complexity based on text characteristics."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            total_complexity = 0.0
            total_articles = 0
            
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                
                text_str = str(text)
                
                # Text length factor
                length_factor = min(len(text_str) / 1000.0, 1.0)  # Normalize to [0, 1]
                
                # Vocabulary diversity (unique words / total words)
                words = text_str.lower().split()
                if len(words) > 0:
                    unique_words = len(set(words))
                    vocab_diversity = unique_words / len(words)
                else:
                    vocab_diversity = 0.0
                
                # Sentence complexity (average sentence length)
                sentences = text_str.split('.')
                avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                sentence_complexity = min(avg_sentence_length / 20.0, 1.0)  # Normalize
                
                # Combined complexity score
                complexity = (length_factor + vocab_diversity + sentence_complexity) / 3.0
                total_complexity += complexity
                total_articles += 1
            
            return total_complexity / total_articles if total_articles > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating content complexity: {e}")
            return 0.0