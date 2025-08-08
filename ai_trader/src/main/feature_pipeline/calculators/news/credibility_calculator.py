"""
News Credibility Calculator

Specialized calculator for news source credibility and quality analysis including:
- Source diversity and entropy metrics
- Mainstream vs alternative source ratios
- Source credibility weighting and distribution
- Quality indicators and reputation scoring
- Cross-source validation and consensus metrics
- Trust and reliability trends
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import timedelta
from collections import Counter

from .base_news import BaseNewsCalculator

logger = logging.getLogger(__name__)


class NewsCredibilityCalculator(BaseNewsCalculator):
    """Calculator for news source credibility and quality analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize news credibility calculator."""
        super().__init__(config)
        
        # Credibility-specific parameters
        self.diversity_window = self.config.get('diversity_window', 24)  # Hours
        self.consensus_window = self.config.get('consensus_window', 24)  # Hours
        self.quality_baseline_window = self.config.get('quality_baseline_window', 168)  # Hours
        
        # Source categorizations
        self.mainstream_sources = self.config.get('mainstream_sources', [
            'reuters', 'bloomberg', 'wsj', 'marketwatch', 'cnbc', 'ap', 'ft'
        ])
        
        self.alternative_sources = self.config.get('alternative_sources', [
            'reddit', 'twitter', 'seekingalpha', 'stocktwits', 'benzinga', 'yahoo'
        ])
        
        self.premium_sources = self.config.get('premium_sources', [
            'reuters', 'bloomberg', 'wsj', 'ft'
        ])
    
    def get_feature_names(self) -> List[str]:
        """Return list of news credibility feature names."""
        feature_names = [
            # Source diversity features
            'news_source_entropy_24h',
            'news_source_diversity_24h',
            'news_source_concentration_24h',
            'news_unique_sources_24h',
            
            # Source type ratios
            'news_mainstream_ratio_24h',
            'news_alternative_ratio_24h',
            'news_premium_ratio_24h',
            'news_mainstream_vs_alternative',
            
            # Credibility metrics
            'news_source_consensus_24h',
            'news_weighted_credibility_24h',
            'news_credibility_distribution',
            'news_source_reliability_score',
            
            # Quality indicators
            'news_source_quality_24h',
            'news_coverage_breadth_24h',
            'news_cross_source_validation',
            'news_source_agreement_ratio',
            
            # Trust metrics
            'news_trust_score_24h',
            'news_source_reputation_24h',
            'news_credibility_trend_24h',
            'news_source_consistency',
            
            # Coverage analysis
            'news_source_coverage_overlap',
            'news_story_source_count',
            'news_exclusive_coverage_ratio',
            'news_redundant_coverage_ratio',
            
            # Advanced credibility features
            'news_source_weight_variance',
            'news_credibility_confidence',
            'news_source_network_effect',
            'news_information_quality_score'
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate news credibility features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with news credibility features
        """
        try:
            # Create features DataFrame with proper index
            features = self.create_empty_features(data.index)
            
            if self.news_data is None or self.news_data.empty:
                logger.warning("No news data available for credibility calculation")
                return features
            
            # Calculate source diversity features
            features = self._calculate_source_diversity_features(data, features)
            
            # Calculate source type ratio features
            features = self._calculate_source_type_features(data, features)
            
            # Calculate credibility metrics
            features = self._calculate_credibility_metrics(data, features)
            
            # Calculate quality indicators
            features = self._calculate_quality_indicators(data, features)
            
            # Calculate trust metrics
            features = self._calculate_trust_metrics(data, features)
            
            # Calculate coverage analysis features
            features = self._calculate_coverage_features(data, features)
            
            # Calculate advanced features
            features = self._calculate_advanced_credibility_features(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating news credibility features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_source_diversity_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate source diversity and entropy features."""
        
        def source_entropy_func(timestamp):
            return self._calculate_source_entropy(timestamp, self.diversity_window)
        
        def source_diversity_func(timestamp):
            return self._calculate_source_diversity(timestamp, self.diversity_window)
        
        def source_concentration_func(timestamp):
            return self._calculate_source_concentration(timestamp, self.diversity_window)
        
        def unique_sources_func(timestamp):
            return self._count_unique_sources(timestamp, self.diversity_window)
        
        # Calculate for all timestamps
        source_entropy = [source_entropy_func(ts) for ts in data.index]
        source_diversity = [source_diversity_func(ts) for ts in data.index]
        source_concentration = [source_concentration_func(ts) for ts in data.index]
        unique_sources = [unique_sources_func(ts) for ts in data.index]
        
        features['news_source_entropy_24h'] = source_entropy
        features['news_source_diversity_24h'] = source_diversity
        features['news_source_concentration_24h'] = source_concentration
        features['news_unique_sources_24h'] = unique_sources
        
        return features
    
    def _calculate_source_type_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate source type ratio features."""
        
        def mainstream_ratio_func(timestamp):
            return self._calculate_source_type_ratio(timestamp, self.mainstream_sources, self.diversity_window)
        
        def alternative_ratio_func(timestamp):
            return self._calculate_source_type_ratio(timestamp, self.alternative_sources, self.diversity_window)
        
        def premium_ratio_func(timestamp):
            return self._calculate_source_type_ratio(timestamp, self.premium_sources, self.diversity_window)
        
        mainstream_ratios = [mainstream_ratio_func(ts) for ts in data.index]
        alternative_ratios = [alternative_ratio_func(ts) for ts in data.index]
        premium_ratios = [premium_ratio_func(ts) for ts in data.index]
        
        features['news_mainstream_ratio_24h'] = mainstream_ratios
        features['news_alternative_ratio_24h'] = alternative_ratios
        features['news_premium_ratio_24h'] = premium_ratios
        
        # Mainstream vs alternative balance
        mainstream_series = pd.Series(mainstream_ratios, index=data.index)
        alternative_series = pd.Series(alternative_ratios, index=data.index)
        
        features['news_mainstream_vs_alternative'] = (
            mainstream_series / (alternative_series + self.numerical_tolerance)
        )
        
        return features
    
    def _calculate_credibility_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate credibility scoring metrics."""
        
        def consensus_func(timestamp):
            return self._calculate_source_consensus(timestamp, self.consensus_window)
        
        def weighted_credibility_func(timestamp):
            return self._calculate_weighted_credibility(timestamp, self.consensus_window)
        
        def credibility_distribution_func(timestamp):
            return self._calculate_credibility_distribution(timestamp, self.consensus_window)
        
        def reliability_score_func(timestamp):
            return self._calculate_source_reliability_score(timestamp, self.consensus_window)
        
        consensus = [consensus_func(ts) for ts in data.index]
        weighted_credibility = [weighted_credibility_func(ts) for ts in data.index]
        credibility_dist = [credibility_distribution_func(ts) for ts in data.index]
        reliability = [reliability_score_func(ts) for ts in data.index]
        
        features['news_source_consensus_24h'] = consensus
        features['news_weighted_credibility_24h'] = weighted_credibility
        features['news_credibility_distribution'] = credibility_dist
        features['news_source_reliability_score'] = reliability
        
        return features
    
    def _calculate_quality_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate news quality indicator features."""
        
        def quality_func(timestamp):
            return self._calculate_source_quality(timestamp, self.diversity_window)
        
        def coverage_breadth_func(timestamp):
            return self._calculate_coverage_breadth(timestamp, self.diversity_window)
        
        def cross_validation_func(timestamp):
            return self._calculate_cross_source_validation(timestamp, self.diversity_window)
        
        def agreement_ratio_func(timestamp):
            return self._calculate_source_agreement_ratio(timestamp, self.diversity_window)
        
        quality = [quality_func(ts) for ts in data.index]
        coverage_breadth = [coverage_breadth_func(ts) for ts in data.index]
        cross_validation = [cross_validation_func(ts) for ts in data.index]
        agreement_ratio = [agreement_ratio_func(ts) for ts in data.index]
        
        features['news_source_quality_24h'] = quality
        features['news_coverage_breadth_24h'] = coverage_breadth
        features['news_cross_source_validation'] = cross_validation
        features['news_source_agreement_ratio'] = agreement_ratio
        
        return features
    
    def _calculate_trust_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate trust and reputation metrics."""
        
        def trust_score_func(timestamp):
            return self._calculate_trust_score(timestamp, self.diversity_window)
        
        def reputation_func(timestamp):
            return self._calculate_source_reputation(timestamp, self.diversity_window)
        
        def consistency_func(timestamp):
            return self._calculate_source_consistency(timestamp, self.quality_baseline_window)
        
        trust_scores = [trust_score_func(ts) for ts in data.index]
        reputation = [reputation_func(ts) for ts in data.index]
        consistency = [consistency_func(ts) for ts in data.index]
        
        features['news_trust_score_24h'] = trust_scores
        features['news_source_reputation_24h'] = reputation
        features['news_source_consistency'] = consistency
        
        # Credibility trend
        trust_series = pd.Series(trust_scores, index=data.index)
        features['news_credibility_trend_24h'] = trust_series.diff(periods=24).fillna(0)
        
        return features
    
    def _calculate_coverage_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate coverage analysis features."""
        
        def coverage_overlap_func(timestamp):
            return self._calculate_coverage_overlap(timestamp, self.diversity_window)
        
        def story_source_count_func(timestamp):
            return self._calculate_story_source_count(timestamp, self.diversity_window)
        
        def exclusive_ratio_func(timestamp):
            return self._calculate_exclusive_coverage_ratio(timestamp, self.diversity_window)
        
        def redundant_ratio_func(timestamp):
            return self._calculate_redundant_coverage_ratio(timestamp, self.diversity_window)
        
        coverage_overlap = [coverage_overlap_func(ts) for ts in data.index]
        story_source_count = [story_source_count_func(ts) for ts in data.index]
        exclusive_ratio = [exclusive_ratio_func(ts) for ts in data.index]
        redundant_ratio = [redundant_ratio_func(ts) for ts in data.index]
        
        features['news_source_coverage_overlap'] = coverage_overlap
        features['news_story_source_count'] = story_source_count
        features['news_exclusive_coverage_ratio'] = exclusive_ratio
        features['news_redundant_coverage_ratio'] = redundant_ratio
        
        return features
    
    def _calculate_advanced_credibility_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced credibility features."""
        
        def weight_variance_func(timestamp):
            return self._calculate_source_weight_variance(timestamp, self.diversity_window)
        
        def confidence_func(timestamp):
            return self._calculate_credibility_confidence(timestamp, self.diversity_window)
        
        def network_effect_func(timestamp):
            return self._calculate_source_network_effect(timestamp, self.diversity_window)
        
        def info_quality_func(timestamp):
            return self._calculate_information_quality_score(timestamp, self.diversity_window)
        
        weight_variance = [weight_variance_func(ts) for ts in data.index]
        confidence = [confidence_func(ts) for ts in data.index]
        network_effect = [network_effect_func(ts) for ts in data.index]
        info_quality = [info_quality_func(ts) for ts in data.index]
        
        features['news_source_weight_variance'] = weight_variance
        features['news_credibility_confidence'] = confidence
        features['news_source_network_effect'] = network_effect
        features['news_information_quality_score'] = info_quality
        
        return features
    
    def _calculate_source_entropy(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate Shannon entropy of source distribution."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0.0
            
            # Get source distribution
            source_counts = window_news['source'].value_counts()
            
            if len(source_counts) <= 1:
                return 0.0
            
            # Calculate Shannon entropy
            probabilities = source_counts / source_counts.sum()
            entropy = -sum(p * np.log(p + self.numerical_tolerance) for p in probabilities)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(source_counts))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return normalized_entropy
            
        except Exception as e:
            logger.warning(f"Error calculating source entropy: {e}")
            return 0.0
    
    def _calculate_source_diversity(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source diversity index."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0.0
            
            source_counts = window_news['source'].value_counts()
            total_articles = len(window_news)
            
            if total_articles == 0 or len(source_counts) <= 1:
                return 0.0
            
            # Simpson's diversity index: 1 - sum(p_i^2)
            probabilities = source_counts / total_articles
            simpson_index = 1 - sum(p**2 for p in probabilities)
            
            return simpson_index
            
        except Exception as e:
            logger.warning(f"Error calculating source diversity: {e}")
            return 0.0
    
    def _calculate_source_concentration(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source concentration (Herfindahl index)."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 1.0  # Maximum concentration
            
            source_counts = window_news['source'].value_counts()
            total_articles = len(window_news)
            
            if total_articles == 0:
                return 1.0
            
            # Herfindahl concentration index
            probabilities = source_counts / total_articles
            herfindahl = sum(p**2 for p in probabilities)
            
            return herfindahl
            
        except Exception as e:
            logger.warning(f"Error calculating source concentration: {e}")
            return 1.0
    
    def _count_unique_sources(self, timestamp: pd.Timestamp, window_hours: int) -> int:
        """Count unique sources in time window."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0
            
            return len(window_news['source'].unique())
            
        except Exception as e:
            logger.warning(f"Error counting unique sources: {e}")
            return 0
    
    def _calculate_source_type_ratio(self, timestamp: pd.Timestamp, source_list: List[str], window_hours: int) -> float:
        """Calculate ratio of articles from specific source types."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0.0
            
            total_articles = len(window_news)
            if total_articles == 0:
                return 0.0
            
            # Count articles from specified sources
            source_count = window_news['source'].str.lower().isin([s.lower() for s in source_list]).sum()
            
            return source_count / total_articles
            
        except Exception as e:
            logger.warning(f"Error calculating source type ratio: {e}")
            return 0.0
    
    def _calculate_source_consensus(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source consensus (inverse of entropy)."""
        try:
            entropy = self._calculate_source_entropy(timestamp, window_hours)
            # Consensus is high when entropy is low
            consensus = 1.0 - entropy
            return max(0.0, consensus)
            
        except Exception as e:
            logger.warning(f"Error calculating source consensus: {e}")
            return 0.0
    
    def _calculate_weighted_credibility(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate credibility-weighted average."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty:
                return 0.0
            
            # Get credibility weights
            weights = self.calculate_news_weights(window_news, timestamp)
            
            if len(weights) == 0 or weights.sum() == 0:
                return 0.0
            
            # Calculate weighted average credibility
            # Vectorized approach: map sources to credibility scores
            sources_lower = window_news['source'].fillna('').str.lower()
            credibility_scores = sources_lower.map(lambda x: self.news_config.source_weights.get(x, 0.5)).values
            
            if not credibility_scores:
                return 0.0
            
            weighted_credibility = sum(c * w for c, w in zip(credibility_scores, weights)) / weights.sum()
            return weighted_credibility
            
        except Exception as e:
            logger.warning(f"Error calculating weighted credibility: {e}")
            return 0.0
    
    def _calculate_credibility_distribution(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate variance in credibility scores."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0.0
            
            credibility_scores = []
            for source in window_news['source']:
                source_lower = str(source).lower()
                credibility = self.news_config.source_weights.get(source_lower, 0.5)
                credibility_scores.append(credibility)
            
            if len(credibility_scores) < 2:
                return 0.0
            
            return np.var(credibility_scores)
            
        except Exception as e:
            logger.warning(f"Error calculating credibility distribution: {e}")
            return 0.0
    
    def _calculate_source_reliability_score(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate overall source reliability score."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'source' not in window_news.columns:
                return 0.0
            
            # Combine multiple factors
            mainstream_ratio = self._calculate_source_type_ratio(timestamp, self.mainstream_sources, window_hours)
            premium_ratio = self._calculate_source_type_ratio(timestamp, self.premium_sources, window_hours)
            weighted_credibility = self._calculate_weighted_credibility(timestamp, window_hours)
            
            # Composite score
            reliability = (mainstream_ratio * 0.4 + premium_ratio * 0.3 + weighted_credibility * 0.3)
            
            return reliability
            
        except Exception as e:
            logger.warning(f"Error calculating source reliability score: {e}")
            return 0.0
    
    def _calculate_source_quality(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source quality based on multiple factors."""
        try:
            # Combine diversity, credibility, and coverage
            diversity = self._calculate_source_diversity(timestamp, window_hours)
            credibility = self._calculate_weighted_credibility(timestamp, window_hours)
            unique_sources = self._count_unique_sources(timestamp, window_hours)
            
            # Normalize unique sources (assume max 20 sources)
            normalized_sources = min(unique_sources / 20.0, 1.0)
            
            quality = (diversity * 0.4 + credibility * 0.4 + normalized_sources * 0.2)
            
            return quality
            
        except Exception as e:
            logger.warning(f"Error calculating source quality: {e}")
            return 0.0
    
    def _calculate_coverage_breadth(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate breadth of news coverage."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty:
                return 0.0
            
            # Simple measure: ratio of unique sources to total articles
            total_articles = len(window_news)
            unique_sources = self._count_unique_sources(timestamp, window_hours)
            
            if total_articles == 0:
                return 0.0
            
            breadth = unique_sources / total_articles
            return min(breadth, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating coverage breadth: {e}")
            return 0.0
    
    def _calculate_cross_source_validation(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate cross-source validation score."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 0.0
            
            # Simple validation: stories covered by multiple sources
            # Vectorized approach: process text and source columns together
            valid_articles = window_news[window_news['text'].notna()].copy()
            
            if valid_articles.empty:
                return 0.0
                
            # Create snippets and sources in vectorized way
            snippets = valid_articles['text'].astype(str).str[:100].str.lower()
            sources = valid_articles['source'].fillna('').astype(str).str.lower()
            
            # Group by snippet to find sources for each story
            story_snippets = {}
            for snippet, source in zip(snippets, sources):
                if snippet not in story_snippets:
                    story_snippets[snippet] = set()
                story_snippets[snippet].add(source)
            
            if not story_snippets:
                return 0.0
            
            # Calculate ratio of stories with multiple sources
            multi_source_stories = sum(1 for sources in story_snippets.values() if len(sources) > 1)
            total_stories = len(story_snippets)
            
            validation_ratio = multi_source_stories / total_stories if total_stories > 0 else 0.0
            
            return validation_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating cross-source validation: {e}")
            return 0.0
    
    def _calculate_source_agreement_ratio(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate ratio of sources in agreement."""
        try:
            # Simplified: use sentiment agreement as proxy
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty or 'text' not in window_news.columns:
                return 1.0  # Assume agreement if no data
            
            sentiments = []
            for text in window_news['text']:
                if pd.isna(text):
                    continue
                sentiment = self.calculate_sentiment(text)
                sentiments.append(sentiment['polarity'])
            
            if len(sentiments) < 2:
                return 1.0
            
            # Calculate agreement based on sentiment consistency
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # Agreement is high when most sentiments are in the same direction
            total = len(sentiments)
            max_agreement = max(positive_count, negative_count, neutral_count)
            agreement_ratio = max_agreement / total if total > 0 else 1.0
            
            return agreement_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating source agreement ratio: {e}")
            return 1.0
    
    def _calculate_trust_score(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate overall trust score."""
        try:
            # Combine multiple trust factors
            reliability = self._calculate_source_reliability_score(timestamp, window_hours)
            quality = self._calculate_source_quality(timestamp, window_hours)
            validation = self._calculate_cross_source_validation(timestamp, window_hours)
            agreement = self._calculate_source_agreement_ratio(timestamp, window_hours)
            
            trust_score = (reliability * 0.3 + quality * 0.3 + validation * 0.2 + agreement * 0.2)
            
            return trust_score
            
        except Exception as e:
            logger.warning(f"Error calculating trust score: {e}")
            return 0.0
    
    def _calculate_source_reputation(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source reputation based on credibility weights."""
        try:
            return self._calculate_weighted_credibility(timestamp, window_hours)
            
        except Exception as e:
            logger.warning(f"Error calculating source reputation: {e}")
            return 0.0
    
    def _calculate_source_consistency(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate source consistency over time."""
        try:
            # Look at source distribution consistency over time windows
            current_window = self.get_news_in_window(timestamp, 24)
            baseline_window = self.get_news_in_window(timestamp, window_hours)
            
            if current_window.empty or baseline_window.empty:
                return 1.0  # Assume consistency if no data
            
            if 'source' not in current_window.columns or 'source' not in baseline_window.columns:
                return 1.0
            
            # Compare source distributions
            current_dist = current_window['source'].value_counts(normalize=True)
            baseline_dist = baseline_window['source'].value_counts(normalize=True)
            
            # Calculate overlap in source distributions
            common_sources = set(current_dist.index) & set(baseline_dist.index)
            
            if not common_sources:
                return 0.0
            
            # Calculate similarity (simplified cosine similarity)
            similarity_scores = []
            for source in common_sources:
                current_prop = current_dist.get(source, 0)
                baseline_prop = baseline_dist.get(source, 0)
                similarity_scores.append(min(current_prop, baseline_prop))
            
            consistency = sum(similarity_scores) if similarity_scores else 0.0
            
            return min(consistency, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating source consistency: {e}")
            return 1.0
    
    def _calculate_coverage_overlap(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate coverage overlap between sources."""
        try:
            return self._calculate_cross_source_validation(timestamp, window_hours)
            
        except Exception as e:
            logger.warning(f"Error calculating coverage overlap: {e}")
            return 0.0
    
    def _calculate_story_source_count(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate average number of sources per story."""
        try:
            window_news = self.get_news_in_window(timestamp, window_hours)
            
            if window_news.empty:
                return 0.0
            
            total_articles = len(window_news)
            unique_sources = self._count_unique_sources(timestamp, window_hours)
            
            # Average sources per story (inverse of coverage breadth)
            if unique_sources == 0:
                return 0.0
            
            avg_sources = total_articles / unique_sources
            
            return avg_sources
            
        except Exception as e:
            logger.warning(f"Error calculating story source count: {e}")
            return 0.0
    
    def _calculate_exclusive_coverage_ratio(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate ratio of exclusive coverage."""
        try:
            validation = self._calculate_cross_source_validation(timestamp, window_hours)
            # Exclusive coverage is inverse of validation
            exclusive_ratio = 1.0 - validation
            
            return exclusive_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating exclusive coverage ratio: {e}")
            return 0.0
    
    def _calculate_redundant_coverage_ratio(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate ratio of redundant coverage."""
        try:
            return self._calculate_cross_source_validation(timestamp, window_hours)
            
        except Exception as e:
            logger.warning(f"Error calculating redundant coverage ratio: {e}")
            return 0.0
    
    def _calculate_source_weight_variance(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate variance in source weights."""
        try:
            return self._calculate_credibility_distribution(timestamp, window_hours)
            
        except Exception as e:
            logger.warning(f"Error calculating source weight variance: {e}")
            return 0.0
    
    def _calculate_credibility_confidence(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate confidence in credibility assessment."""
        try:
            # Confidence is higher with more sources and less variance
            unique_sources = self._count_unique_sources(timestamp, window_hours)
            weight_variance = self._calculate_source_weight_variance(timestamp, window_hours)
            
            # Normalize unique sources (assume max 20 sources)
            source_factor = min(unique_sources / 20.0, 1.0)
            
            # Lower variance means higher confidence
            variance_factor = 1.0 - min(weight_variance, 1.0)
            
            confidence = (source_factor + variance_factor) / 2.0
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating credibility confidence: {e}")
            return 0.0
    
    def _calculate_source_network_effect(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate network effect of sources."""
        try:
            # Network effect is stronger with more diverse, credible sources
            diversity = self._calculate_source_diversity(timestamp, window_hours)
            credibility = self._calculate_weighted_credibility(timestamp, window_hours)
            unique_sources = self._count_unique_sources(timestamp, window_hours)
            
            # Normalize sources
            source_factor = min(unique_sources / 15.0, 1.0)
            
            network_effect = (diversity * credibility * source_factor) ** (1/3)  # Geometric mean
            
            return network_effect
            
        except Exception as e:
            logger.warning(f"Error calculating source network effect: {e}")
            return 0.0
    
    def _calculate_information_quality_score(self, timestamp: pd.Timestamp, window_hours: int) -> float:
        """Calculate overall information quality score."""
        try:
            # Combine multiple quality factors
            source_quality = self._calculate_source_quality(timestamp, window_hours)
            trust_score = self._calculate_trust_score(timestamp, window_hours)
            confidence = self._calculate_credibility_confidence(timestamp, window_hours)
            network_effect = self._calculate_source_network_effect(timestamp, window_hours)
            
            # Weighted combination
            info_quality = (
                source_quality * 0.3 + 
                trust_score * 0.3 + 
                confidence * 0.2 + 
                network_effect * 0.2
            )
            
            return info_quality
            
        except Exception as e:
            logger.warning(f"Error calculating information quality score: {e}")
            return 0.0