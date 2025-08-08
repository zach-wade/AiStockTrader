"""
Unusual Activity Calculator

Specialized calculator for unusual options activity detection including:
- Volume spike detection and analysis
- Block trade identification
- Unusual volume vs average comparison
- Premium flow anomaly detection
- Smart money vs retail unusual activity
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_options import BaseOptionsCalculator
from ..helpers import safe_divide, safe_log

from main.utils.core import get_logger

logger = get_logger(__name__)


class UnusualActivityCalculator(BaseOptionsCalculator):
    """Calculator for unusual options activity detection and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unusual activity calculator."""
        super().__init__(config)
        
        # Unusual activity specific parameters
        self.volume_config = self.options_config.get_volume_config()
        self.unusual_threshold = self.volume_config['unusual_threshold']
        self.block_size = self.volume_config['block_size']
        self.large_size = self.volume_config.get('large_size', 500)
        self.volume_window = self.volume_config['window']
        
        # Detection parameters
        self.premium_threshold = self.options_config.premium_threshold
        self.volume_spike_threshold = 3.0  # 3x normal volume
        self.oi_ratio_threshold = 0.5      # Volume/OI ratio threshold
        
        logger.debug(f"Initialized UnusualActivityCalculator with {self.unusual_threshold}x threshold")
    
    def get_feature_names(self) -> List[str]:
        """Return list of unusual activity feature names."""
        feature_names = [
            # Volume anomaly detection
            'unusual_option_count',
            'unusual_option_volume',
            'unusual_volume_ratio',
            'volume_spike_score',
            'volume_anomaly_strength',
            
            # Directional unusual activity
            'unusual_call_volume',
            'unusual_put_volume',
            'unusual_call_count',
            'unusual_put_count',
            'unusual_sentiment',
            
            # Block trade analysis
            'block_trade_count',
            'block_trade_volume',
            'block_trade_ratio',
            'large_trade_count',
            'large_trade_volume',
            'mega_trade_count',
            
            # Premium flow anomalies
            'unusual_premium_flow',
            'top10_premium_volume',
            'top10_premium_total',
            'premium_concentration_score',
            
            # Smart money indicators
            'smart_money_unusual_ratio',
            'institutional_flow_score',
            'retail_fomo_score',
            
            # Advanced anomaly metrics
            'activity_burst_intensity',
            'volume_momentum_score',
            'flow_disruption_index',
            'market_attention_score'
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate unusual activity features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with unusual activity features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            processed_data = self.preprocess_data(data)
            
            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            features = self.create_empty_features(data.index)
            
            # Calculate volume anomaly detection
            features = self._calculate_volume_anomalies(processed_data, features)
            
            # Calculate directional unusual activity
            features = self._calculate_directional_unusual_activity(processed_data, features)
            
            # Calculate block trade analysis
            features = self._calculate_block_trade_analysis(processed_data, features)
            
            # Calculate premium flow anomalies
            features = self._calculate_premium_flow_anomalies(processed_data, features)
            
            # Calculate smart money indicators
            features = self._calculate_smart_money_indicators(processed_data, features)
            
            # Calculate advanced anomaly metrics
            features = self._calculate_advanced_anomaly_metrics(processed_data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating unusual activity features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_volume_anomalies(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume spike detection and anomalies."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate unusual activity when no options chain
                features = self._estimate_volume_anomalies(data, features)
                return features
            
            # Check if we have average volume data
            has_avg_volume = 'avgVolume' in self.options_chain.columns
            
            total_volume = self.options_chain['volume'].sum()
            
            # Vectorized unusual activity detection
            if has_avg_volume:
                # Calculate volume ratios for all options
                avg_volume = self.options_chain['avgVolume'].fillna(0)
                volume_ratios = safe_divide(self.options_chain['volume'], avg_volume, default_value=0.0)
                unusual_mask = volume_ratios >= self.unusual_threshold
            else:
                # Use volume/OI ratio as proxy for unusual activity
                oi = self.options_chain['openInterest'].fillna(0)
                volume_oi_ratios = safe_divide(self.options_chain['volume'], oi, default_value=0.0)
                unusual_mask = volume_oi_ratios >= self.oi_ratio_threshold
            
            # Get unusual options and calculate total unusual volume
            unusual_options = self.options_chain[unusual_mask]
            total_unusual_volume = unusual_options['volume'].sum()
            
            # Handle case where no average volume data but large volume
            if not has_avg_volume and total_unusual_volume == 0:
                # Use block size as fallback for large volume detection
                large_volume_mask = self.options_chain['volume'] >= self.block_size
                unusual_options = self.options_chain[large_volume_mask]
                total_unusual_volume = unusual_options['volume'].sum()
            
            # Calculate basic unusual activity metrics
            features['unusual_option_count'] = len(unusual_options)
            features['unusual_option_volume'] = total_unusual_volume
            
            # Unusual volume ratio
            unusual_ratio = safe_divide(total_unusual_volume, total_volume, default_value=0.0)
            features['unusual_volume_ratio'] = unusual_ratio
            
            # Volume spike score (intensity of unusual activity)
            if unusual_options and has_avg_volume:
                spike_scores = []
                for option in unusual_options:
                    avg_vol = option.get('avgVolume', 1)
                    if avg_vol > 0:
                        spike_score = option['volume'] / avg_vol
                        spike_scores.append(spike_score)
                
                if spike_scores:
                    avg_spike_score = np.mean(spike_scores)
                    features['volume_spike_score'] = avg_spike_score
                else:
                    features['volume_spike_score'] = 1.0
            else:
                features['volume_spike_score'] = 1.0
            
            # Volume anomaly strength (weighted by volume)
            if total_unusual_volume > 0 and total_volume > 0:
                anomaly_strength = (total_unusual_volume / total_volume) * len(unusual_options)
                features['volume_anomaly_strength'] = min(anomaly_strength, 10.0)  # Cap at 10
            else:
                features['volume_anomaly_strength'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volume anomalies: {e}")
            return features
    
    def _estimate_volume_anomalies(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate unusual activity when options chain unavailable."""
        try:
            # Use stock volume patterns to estimate options unusual activity
            if 'volume' in data.columns and len(data) >= self.volume_window:
                stock_volume = data['volume'].iloc[-1]
                avg_stock_volume = data['volume'].iloc[-self.volume_window:].mean()
                
                stock_volume_ratio = safe_divide(stock_volume, avg_stock_volume, default_value=1.0)
                
                # Scale to options activity
                if stock_volume_ratio >= 2.0:  # High stock volume
                    features['unusual_option_count'] = 15
                    features['unusual_option_volume'] = stock_volume * 0.1
                    features['unusual_volume_ratio'] = 0.4
                    features['volume_spike_score'] = stock_volume_ratio * 0.8
                    features['volume_anomaly_strength'] = min(stock_volume_ratio, 5.0)
                else:
                    features['unusual_option_count'] = 5
                    features['unusual_option_volume'] = stock_volume * 0.05
                    features['unusual_volume_ratio'] = 0.2
                    features['volume_spike_score'] = 1.0
                    features['volume_anomaly_strength'] = 1.0
            else:
                # Default low activity
                features['unusual_option_count'] = 3
                features['unusual_option_volume'] = 1000
                features['unusual_volume_ratio'] = 0.1
                features['volume_spike_score'] = 1.0
                features['volume_anomaly_strength'] = 0.5
            
            return features
            
        except Exception as e:
            logger.warning(f"Error estimating volume anomalies: {e}")
            return features
    
    def _calculate_directional_unusual_activity(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate directional (calls vs puts) unusual activity."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate directional activity
                total_unusual = features.get('unusual_option_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                
                # Use market sentiment to estimate call/put split
                if len(data) >= 5:
                    recent_return = data['close'].pct_change().iloc[-5:].mean()
                    if recent_return > 0.01:  # Bullish
                        call_bias = 0.7
                    elif recent_return < -0.01:  # Bearish
                        call_bias = 0.3
                    else:  # Neutral
                        call_bias = 0.5
                else:
                    call_bias = 0.5
                
                features['unusual_call_volume'] = total_unusual * call_bias
                features['unusual_put_volume'] = total_unusual * (1 - call_bias)
                features['unusual_call_count'] = 5 * call_bias
                features['unusual_put_count'] = 5 * (1 - call_bias)
                
                return features
            
            # Separate unusual activity by option type
            unusual_calls = []
            unusual_puts = []
            
            # Use the same logic as volume anomalies but separate by type
            has_avg_volume = 'avgVolume' in self.options_chain.columns
            
            # Vectorized unusual activity detection by type
            if has_avg_volume:
                avg_volume = self.options_chain['avgVolume'].fillna(0)
                volume_ratios = safe_divide(self.options_chain['volume'], avg_volume, default_value=0.0)
                unusual_mask = volume_ratios >= self.unusual_threshold
            else:
                oi = self.options_chain['openInterest'].fillna(0)
                volume_oi_ratios = safe_divide(self.options_chain['volume'], oi, default_value=0.0)
                unusual_mask = (volume_oi_ratios >= self.oi_ratio_threshold) | (self.options_chain['volume'] >= self.block_size)
            
            unusual_options = self.options_chain[unusual_mask]
            unusual_calls = unusual_options[unusual_options['optionType'].str.lower() == 'call']
            unusual_puts = unusual_options[unusual_options['optionType'].str.lower() == 'put']
            
            # Calculate directional metrics using vectorized operations
            unusual_call_volume = unusual_calls['volume'].sum()
            unusual_put_volume = unusual_puts['volume'].sum()
            
            features['unusual_call_volume'] = unusual_call_volume
            features['unusual_put_volume'] = unusual_put_volume
            features['unusual_call_count'] = len(unusual_calls)
            features['unusual_put_count'] = len(unusual_puts)
            
            # Unusual sentiment (bullish vs bearish unusual activity)
            total_unusual_volume = unusual_call_volume + unusual_put_volume
            unusual_sentiment = safe_divide((unusual_call_volume - unusual_put_volume), total_unusual_volume, default_value=0.0)
            features['unusual_sentiment'] = unusual_sentiment
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating directional unusual activity: {e}")
            return features
    
    def _calculate_block_trade_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate block trade and large trade analysis."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate block trades
                total_volume = features.get('unusual_option_volume', pd.Series([1000])).iloc[-1] if len(features) > 0 else 1000
                
                features['block_trade_count'] = max(1, int(total_volume / self.block_size * 0.1))
                features['block_trade_volume'] = total_volume * 0.3
                features['block_trade_ratio'] = 0.3
                features['large_trade_count'] = max(1, int(total_volume / self.large_size * 0.05))
                features['large_trade_volume'] = total_volume * 0.15
                features['mega_trade_count'] = max(0, int(total_volume / (self.large_size * 2) * 0.02))
                
                return features
            
            # Identify different sizes of trades
            block_trades = self.options_chain[self.options_chain['volume'] >= self.block_size]
            large_trades = self.options_chain[self.options_chain['volume'] >= self.large_size]
            mega_trades = self.options_chain[self.options_chain['volume'] >= self.large_size * 2]
            
            total_volume = self.options_chain['volume'].sum()
            
            # Block trade metrics
            block_count = len(block_trades)
            block_volume = block_trades['volume'].sum() if not block_trades.empty else 0
            block_ratio = safe_divide(block_volume, total_volume, default_value=0.0)
            
            features['block_trade_count'] = block_count
            features['block_trade_volume'] = block_volume
            features['block_trade_ratio'] = block_ratio
            
            # Large trade metrics
            large_count = len(large_trades)
            large_volume = large_trades['volume'].sum() if not large_trades.empty else 0
            
            features['large_trade_count'] = large_count
            features['large_trade_volume'] = large_volume
            
            # Mega trade metrics (very large trades)
            mega_count = len(mega_trades)
            features['mega_trade_count'] = mega_count
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating block trade analysis: {e}")
            return features
    
    def _calculate_premium_flow_anomalies(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate premium flow anomaly detection."""
        try:
            if self.options_chain is None or self.options_chain.empty or 'lastPrice' not in self.options_chain.columns:
                # Estimate premium flow when unavailable
                unusual_volume = features.get('unusual_option_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                stock_price = data['close'].iloc[-1] if len(data) > 0 else 100
                
                estimated_avg_premium = stock_price * 0.03  # 3% of stock price
                estimated_premium_flow = unusual_volume * estimated_avg_premium * 100
                
                features['unusual_premium_flow'] = estimated_premium_flow
                features['top10_premium_volume'] = unusual_volume * 0.5
                features['top10_premium_total'] = estimated_premium_flow * 0.5
                features['premium_concentration_score'] = 0.3
                
                return features
            
            # Calculate premium for each option
            self.options_chain['premium_value'] = (
                self.options_chain['lastPrice'] * 
                self.options_chain['volume'] * 
                100  # Options multiplier
            )
            
            # Focus on unusual volume options for premium analysis
            unusual_volume_threshold = self.options_chain['volume'].quantile(0.9)  # Top 10% by volume
            high_volume_options = self.options_chain[
                self.options_chain['volume'] >= unusual_volume_threshold
            ]
            
            if not high_volume_options.empty:
                # Unusual premium flow
                unusual_premium = high_volume_options['premium_value'].sum()
                features['unusual_premium_flow'] = unusual_premium
                
                # Top 10 by premium
                top_premium_options = self.options_chain.nlargest(10, 'premium_value')
                features['top10_premium_volume'] = top_premium_options['volume'].sum()
                features['top10_premium_total'] = top_premium_options['premium_value'].sum()
                
                # Premium concentration (how much premium is in top options)
                total_premium = self.options_chain['premium_value'].sum()
                concentration = safe_divide(top_premium_options['premium_value'].sum(), total_premium, default_value=0.0)
                features['premium_concentration_score'] = concentration
            else:
                features['unusual_premium_flow'] = 0.0
                features['top10_premium_volume'] = 0.0
                features['top10_premium_total'] = 0.0
                features['premium_concentration_score'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating premium flow anomalies: {e}")
            return features
    
    def _calculate_smart_money_indicators(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate smart money vs retail unusual activity indicators."""
        try:
            if self.options_chain is None or self.options_chain.empty:
                # Estimate smart money activity
                unusual_volume = features.get('unusual_option_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                
                # Assume 30% of unusual activity is smart money
                features['smart_money_unusual_ratio'] = 0.3
                features['institutional_flow_score'] = unusual_volume * 0.3
                features['retail_fomo_score'] = unusual_volume * 0.7
                
                return features
            
            # Classify trades by size (proxy for smart money)
            total_unusual_volume = features.get('unusual_option_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            
            if total_unusual_volume > 0:
                # Large trades are more likely smart money
                large_unusual_volume = 0
                small_unusual_volume = 0
                
                # Use the same unusual detection logic but separate by size
                has_avg_volume = 'avgVolume' in self.options_chain.columns
                
                # Vectorized smart money detection
                if has_avg_volume:
                    avg_volume = self.options_chain['avgVolume'].fillna(0)
                    volume_ratios = safe_divide(self.options_chain['volume'], avg_volume, default_value=0.0)
                    unusual_mask = volume_ratios >= self.unusual_threshold
                else:
                    oi = self.options_chain['openInterest'].fillna(0)
                    volume_oi_ratios = safe_divide(self.options_chain['volume'], oi, default_value=0.0)
                    unusual_mask = (volume_oi_ratios >= self.oi_ratio_threshold) | (self.options_chain['volume'] >= self.block_size)
                
                unusual_options = self.options_chain[unusual_mask]
                
                # Separate large and small unusual volume
                large_unusual_volume = unusual_options[unusual_options['volume'] >= self.block_size]['volume'].sum()
                small_unusual_volume = unusual_options[unusual_options['volume'] < self.block_size]['volume'].sum()
                
                # Smart money ratio
                smart_money_ratio = safe_divide(large_unusual_volume, total_unusual_volume, default_value=0.0)
                features['smart_money_unusual_ratio'] = smart_money_ratio
                
                # Institutional flow score
                features['institutional_flow_score'] = large_unusual_volume
                
                # Retail FOMO score
                features['retail_fomo_score'] = small_unusual_volume
                
            else:
                features['smart_money_unusual_ratio'] = 0.0
                features['institutional_flow_score'] = 0.0
                features['retail_fomo_score'] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating smart money indicators: {e}")
            return features
    
    def _calculate_advanced_anomaly_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced anomaly and attention metrics."""
        try:
            # Activity burst intensity
            unusual_count = features.get('unusual_option_count', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            unusual_volume = features.get('unusual_option_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            
            # Intensity based on count and volume together
            avg_unusual_volume = safe_divide(unusual_volume, unusual_count, default_value=0.0)
            if unusual_count > 0:
                burst_intensity = min(unusual_count * safe_log(1 + safe_divide(avg_unusual_volume, 1000)), 10.0)
                features['activity_burst_intensity'] = burst_intensity
            else:
                features['activity_burst_intensity'] = 0.0
            
            # Volume momentum score (rate of increase in unusual activity)
            volume_spike_score = features.get('volume_spike_score', pd.Series([1.0])).iloc[-1] if len(features) > 0 else 1.0
            volume_momentum = max(0, (volume_spike_score - 1.0) * 2)  # Scale above 1
            features['volume_momentum_score'] = min(volume_momentum, 5.0)
            
            # Flow disruption index (how much unusual activity disrupts normal flow)
            unusual_ratio = features.get('unusual_volume_ratio', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            block_ratio = features.get('block_trade_ratio', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            
            disruption_index = safe_divide((unusual_ratio + block_ratio), 2, default_value=0.0)
            features['flow_disruption_index'] = min(disruption_index, 1.0)
            
            # Market attention score (combined measure of unusual activity)
            attention_components = [
                unusual_ratio,
                safe_divide(volume_spike_score, 5, default_value=0.0),  # Normalize
                min(safe_divide(unusual_count, 20, default_value=0.0), 1.0),  # Normalize count
                block_ratio
            ]
            
            attention_score = np.mean(attention_components)
            features['market_attention_score'] = min(attention_score, 1.0)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced anomaly metrics: {e}")
            return features