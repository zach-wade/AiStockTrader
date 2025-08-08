"""
News Feature Facade

Unified interface for all news-based feature calculations, orchestrating
multiple specialized calculators to provide comprehensive news analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_news import BaseNewsCalculator
from .sentiment_calculator import SentimentCalculator
from .volume_calculator import VolumeCalculator
from .monetary_calculator import MonetaryImpactCalculator
from .topic_calculator import NewsTopicCalculator as TopicCalculator
from .event_calculator import NewsEventCalculator as EventCalculator
from .credibility_calculator import NewsCredibilityCalculator as CredibilityCalculator

from ..helpers import create_feature_dataframe, safe_divide, aggregate_features

from main.utils.core import get_logger, process_in_batches

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class NewsFeatureFacade(BaseNewsCalculator):
    """
    Facade for comprehensive news feature calculation.
    
    Orchestrates all news calculators to provide:
    - Complete news feature set (247+ features)
    - Parallel calculation for performance
    - Cross-calculator feature interactions
    - Unified configuration management
    - Composite news signals
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize news feature facade.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize all calculators
        self.calculators = {
            'sentiment': SentimentCalculator(config),
            'volume': VolumeCalculator(config),
            'monetary': MonetaryImpactCalculator(config),
            'topic': TopicCalculator(config),
            'event': EventCalculator(config),
            'credibility': CredibilityCalculator(config)
        }
        
        # Parallel processing configuration
        self.max_workers = config.get('max_workers', 4)
        self.batch_size = config.get('batch_size', 1000)
        
        # Feature importance weights for composite scores
        self.feature_weights = config.get('feature_weights', {
            'sentiment': 0.25,
            'volume': 0.15,
            'monetary': 0.20,
            'event': 0.20,
            'credibility': 0.10,
            'topic': 0.10
        })
        
        logger.info(f"Initialized NewsFeatureFacade with {len(self.calculators)} calculators")
    
    def get_feature_names(self) -> List[str]:
        """Get complete list of news feature names."""
        features = []
        
        # Collect features from all calculators
        for name, calculator in self.calculators.items():
            calc_features = calculator.get_feature_names()
            # Prefix with calculator name for clarity
            features.extend([f"{name}_{feat}" for feat in calc_features])
        
        # Add facade-specific composite features
        features.extend([
            'news_composite_score',
            'news_signal_strength',
            'news_divergence_score',
            'information_quality_index',
            'market_relevance_score',
            'news_momentum_indicator',
            'sentiment_volume_interaction',
            'event_impact_multiplier',
            'news_alpha_signal',
            'information_edge_score'
        ])
        
        return features
    
    def calculate(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all news features using parallel processing.
        
        Args:
            news_data: DataFrame with news articles
            
        Returns:
            DataFrame with all news features
        """
        try:
            # Validate and prepare data
            validated_data = self.validate_and_prepare_data(news_data)
            if validated_data.empty:
                return self._create_empty_features(news_data.index)
            
            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)
            
            # Calculate features in parallel
            if self.max_workers > 1:
                calc_features = self._calculate_parallel(validated_data)
            else:
                calc_features = self._calculate_sequential(validated_data)
            
            # Merge calculator features
            for calc_name, calc_df in calc_features.items():
                # Prefix column names
                calc_df.columns = [f"{calc_name}_{col}" for col in calc_df.columns]
                features = pd.concat([features, calc_df], axis=1)
            
            # Calculate cross-calculator interactions
            interaction_features = self._calculate_interactions(calc_features)
            features = pd.concat([features, interaction_features], axis=1)
            
            # Calculate composite scores
            composite_features = self._calculate_composite_scores(calc_features)
            features = pd.concat([features, composite_features], axis=1)
            
            # Calculate advanced signals
            signal_features = self._calculate_advanced_signals(
                calc_features, validated_data
            )
            features = pd.concat([features, signal_features], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in NewsFeatureFacade calculation: {e}")
            return self._create_empty_features(news_data.index)
    
    def _calculate_parallel(
        self,
        news_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate features using parallel processing."""
        calc_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit calculation tasks
            future_to_calc = {
                executor.submit(
                    self._safe_calculate,
                    calc_name,
                    calculator,
                    news_data
                ): calc_name
                for calc_name, calculator in self.calculators.items()
            }
            
            # Collect results
            for future in as_completed(future_to_calc):
                calc_name = future_to_calc[future]
                try:
                    result = future.result()
                    if result is not None:
                        calc_results[calc_name] = result
                except Exception as e:
                    logger.error(f"Error in {calc_name} calculation: {e}")
                    calc_results[calc_name] = create_feature_dataframe(news_data.index)
        
        return calc_results
    
    def _calculate_sequential(
        self,
        news_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate features sequentially."""
        calc_results = {}
        
        for calc_name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(news_data)
                calc_results[calc_name] = result
            except Exception as e:
                logger.error(f"Error in {calc_name} calculation: {e}")
                calc_results[calc_name] = create_feature_dataframe(news_data.index)
        
        return calc_results
    
    def _safe_calculate(
        self,
        calc_name: str,
        calculator: BaseNewsCalculator,
        news_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Safely calculate features with error handling."""
        try:
            logger.debug(f"Calculating {calc_name} features")
            return calculator.calculate(news_data)
        except Exception as e:
            logger.error(f"Failed to calculate {calc_name} features: {e}")
            return None
    
    def _calculate_interactions(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate cross-calculator interaction features."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        # Sentiment-Volume interaction
        if 'sentiment' in calc_features and 'volume' in calc_features:
            sent_score = calc_features['sentiment'].get('sentiment_composite_score', 0)
            vol_intensity = calc_features['volume'].get('news_intensity_score', 0)
            
            features['sentiment_volume_interaction'] = sent_score * vol_intensity
        
        # Event impact multiplier
        if 'event' in calc_features and 'credibility' in calc_features:
            event_score = calc_features['event'].get('event_composite_score', 0)
            cred_score = calc_features['credibility'].get('credibility_weighted_score', 1)
            
            features['event_impact_multiplier'] = event_score * cred_score
        
        return features
    
    def _calculate_composite_scores(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate composite news scores."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        # News composite score (weighted average of all calculators)
        composite_components = []
        
        for calc_name, weight in self.feature_weights.items():
            if calc_name in calc_features:
                # Get main score from each calculator
                if calc_name == 'sentiment':
                    score = calc_features[calc_name].get('sentiment_composite_score', 0)
                elif calc_name == 'volume':
                    score = calc_features[calc_name].get('news_intensity_score', 0)
                elif calc_name == 'monetary':
                    score = calc_features[calc_name].get('monetary_impact_score', 0)
                elif calc_name == 'event':
                    score = calc_features[calc_name].get('event_composite_score', 0)
                elif calc_name == 'credibility':
                    score = calc_features[calc_name].get('credibility_weighted_score', 0)
                elif calc_name == 'topic':
                    score = calc_features[calc_name].get('topic_diversity_score', 0)
                else:
                    score = 0
                
                composite_components.append(score * weight)
        
        features['news_composite_score'] = sum(composite_components)
        
        # News signal strength (combination of volume and sentiment extremity)
        sentiment_abs = abs(calc_features.get('sentiment', {}).get(
            'sentiment_composite_score', 0
        ))
        volume_score = calc_features.get('volume', {}).get(
            'news_intensity_score', 0
        )
        
        features['news_signal_strength'] = (sentiment_abs + volume_score) / 2
        
        # News divergence score (disagreement across sources)
        features['news_divergence_score'] = self._calculate_divergence_score(
            calc_features
        )
        
        # Information quality index
        quality_components = [
            calc_features.get('credibility', {}).get('credibility_weighted_score', 0),
            calc_features.get('credibility', {}).get('source_diversity_score', 0),
            1 - calc_features.get('volume', {}).get('redundancy_score', 0),
            calc_features.get('topic', {}).get('topic_coherence_score', 0)
        ]
        features['information_quality_index'] = np.mean(
            [c for c in quality_components if c != 0]
        )
        
        # Market relevance score
        relevance_components = [
            calc_features.get('monetary', {}).get('analyst_action_intensity', 0),
            calc_features.get('event', {}).get('breaking_news_score', 0),
            calc_features.get('topic', {}).get('earnings_topic_score', 0),
            calc_features.get('sentiment', {}).get('finance_adjusted_sentiment', 0)
        ]
        features['market_relevance_score'] = np.mean(
            [abs(c) for c in relevance_components if c != 0]
        )
        
        return features
    
    def _calculate_advanced_signals(
        self,
        calc_features: Dict[str, pd.DataFrame],
        news_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate advanced trading signals from news."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        # News momentum indicator
        momentum_components = [
            calc_features.get('volume', {}).get('news_momentum', 0),
            calc_features.get('sentiment', {}).get('sentiment_momentum_1h_6h', 0),
            calc_features.get('event', {}).get('event_momentum', 0)
        ]
        features['news_momentum_indicator'] = np.mean(
            [c for c in momentum_components if c != 0]
        )
        
        # News alpha signal (predictive power)
        alpha_components = []
        
        # High sentiment with low volume (contrarian signal)
        sent_score = abs(calc_features.get('sentiment', {}).get(
            'sentiment_composite_score', 0
        ))
        vol_score = calc_features.get('volume', {}).get('news_intensity_score', 0)
        if vol_score > 0:
            alpha_components.append(sent_score / vol_score)
        
        # Monetary impact with high credibility
        mon_impact = calc_features.get('monetary', {}).get(
            'monetary_impact_score', 0
        )
        cred_score = calc_features.get('credibility', {}).get(
            'credibility_weighted_score', 1
        )
        alpha_components.append(mon_impact * cred_score)
        
        # Event surprise factor
        event_score = calc_features.get('event', {}).get('event_surprise_score', 0)
        alpha_components.append(event_score)
        
        features['news_alpha_signal'] = np.mean(alpha_components) if alpha_components else 0
        
        # Information edge score (exclusive/early information)
        edge_components = [
            calc_features.get('volume', {}).get('exclusive_coverage_ratio', 0),
            calc_features.get('credibility', {}).get('premium_source_ratio', 0),
            calc_features.get('event', {}).get('breaking_news_score', 0),
            1 - calc_features.get('volume', {}).get('news_saturation_index', 0)
        ]
        features['information_edge_score'] = np.mean(
            [c for c in edge_components if c != 0]
        )
        
        return features
    
    def _calculate_divergence_score(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Calculate divergence/disagreement score across news sources."""
        divergence_components = []
        
        # Sentiment divergence
        if 'sentiment' in calc_features:
            sent_std = calc_features['sentiment'].get('sentiment_std_24h', 0)
            divergence_components.append(sent_std)
        
        # Source disagreement
        if 'credibility' in calc_features:
            source_entropy = calc_features['credibility'].get('source_entropy', 0)
            consensus = calc_features['credibility'].get('consensus_score', 1)
            divergence_components.append(source_entropy * (1 - consensus))
        
        # Topic fragmentation
        if 'topic' in calc_features:
            topic_div = calc_features['topic'].get('topic_diversity_score', 0)
            divergence_components.append(topic_div)
        
        return np.mean(divergence_components) if divergence_components else 0
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores for all news features."""
        importance_data = []
        
        # Define importance scores for key features
        feature_importance = {
            # High importance
            'sentiment_composite_score': 0.9,
            'monetary_impact_score': 0.9,
            'event_composite_score': 0.85,
            'news_composite_score': 0.85,
            'breaking_news_score': 0.8,
            
            # Medium importance
            'news_intensity_score': 0.7,
            'credibility_weighted_score': 0.7,
            'analyst_net_score': 0.7,
            'sentiment_momentum_1h_6h': 0.65,
            'news_alpha_signal': 0.65,
            
            # Lower importance
            'topic_diversity_score': 0.5,
            'source_entropy': 0.5,
            'redundancy_score': 0.4,
            'periodicity_score': 0.3
        }
        
        # Build importance DataFrame
        for feature, importance in feature_importance.items():
            # Find which calculator owns this feature
            for calc_name in self.calculators:
                full_name = f"{calc_name}_{feature}"
                importance_data.append({
                    'feature': full_name,
                    'importance': importance,
                    'calculator': calc_name,
                    'category': self._get_feature_category(feature)
                })
        
        return pd.DataFrame(importance_data).sort_values(
            'importance', ascending=False
        )
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Categorize feature by type."""
        if 'sentiment' in feature_name:
            return 'sentiment'
        elif 'volume' in feature_name or 'count' in feature_name:
            return 'volume'
        elif 'monetary' in feature_name or 'price_target' in feature_name:
            return 'monetary'
        elif 'event' in feature_name or 'breaking' in feature_name:
            return 'event'
        elif 'credibility' in feature_name or 'source' in feature_name:
            return 'credibility'
        elif 'topic' in feature_name:
            return 'topic'
        elif 'composite' in feature_name or 'score' in feature_name:
            return 'composite'
        else:
            return 'other'