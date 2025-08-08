"""
Regime Correlation Calculator

Specialized calculator for regime-dependent correlation analysis including:
- High/low volatility regime correlations
- Up/down trend regime correlations
- Crisis vs normal period correlations
- Regime transition detection and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_correlation import BaseCorrelationCalculator

logger = logging.getLogger(__name__)


class RegimeCorrelationCalculator(BaseCorrelationCalculator):
    """Calculator for regime-dependent correlation analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime correlation calculator."""
        super().__init__(config)
        
        # Regime-specific parameters
        self.regime_config = self.correlation_config.get_window_config('regime')
        self.volatility_lookback = self.regime_config.get('volatility_lookback', 20)
        self.volatility_percentiles = self.regime_config.get('volatility_percentiles', [0.33, 0.67])
        self.trend_lookback = self.regime_config.get('trend_lookback', 20)
        self.trend_threshold = self.regime_config.get('trend_threshold', 0.0)
        
        logger.debug(f"Initialized RegimeCorrelationCalculator with {self.volatility_lookback}d lookback")
    
    def get_feature_names(self) -> List[str]:
        """Return list of regime correlation feature names."""
        feature_names = [
            # Volatility regime correlations
            'corr_high_vol',
            'corr_low_vol', 
            'corr_regime_diff',
            'vol_regime_correlation_stability',
            
            # Trend regime correlations
            'corr_uptrend',
            'corr_downtrend',
            'corr_trend_diff',
            'trend_regime_correlation_stability',
            
            # Crisis vs normal correlations
            'corr_crisis_periods',
            'corr_normal_periods',
            'crisis_correlation_spike',
            'correlation_contagion_score',
            
            # Regime transition features
            'regime_transition_frequency',
            'correlation_regime_persistence',
            'regime_correlation_momentum',
            'correlation_regime_clustering',
            
            # Advanced regime features
            'correlation_tail_dependence',
            'regime_correlation_asymmetry',
            'structural_regime_indicator',
            'correlation_stress_sensitivity'
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime-dependent correlation features.
        
        Args:
            data: DataFrame with symbol, timestamp, close columns
            
        Returns:
            DataFrame with regime correlation features
        """
        try:
            # Validate and preprocess data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            processed_data = self.preprocess_data(data)
            
            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            unique_timestamps = processed_data['timestamp'].unique()
            features = self.create_empty_features(pd.Index(unique_timestamps))
            
            # Calculate volatility regime correlations
            features = self._calculate_volatility_regime_correlations(processed_data, features)
            
            # Calculate trend regime correlations
            features = self._calculate_trend_regime_correlations(processed_data, features)
            
            # Calculate crisis regime correlations
            features = self._calculate_crisis_correlations(processed_data, features)
            
            # Calculate regime transition features
            features = self._calculate_regime_transitions(processed_data, features)
            
            # Calculate advanced regime features
            features = self._calculate_advanced_regime_features(processed_data, features)
            
            # Align features with original data
            if len(features) != len(data):
                features = self._align_features_with_data(features, data)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating regime correlation features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_volatility_regime_correlations(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations in different volatility regimes."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            # Get market proxy for regime definition
            market_returns = self.get_market_proxy(data)
            
            if market_returns.empty:
                return features
            
            # Calculate market volatility
            market_volatility = market_returns.rolling(window=self.volatility_lookback).std()
            
            # Define volatility regimes
            low_vol_threshold = market_volatility.quantile(self.volatility_percentiles[0])
            high_vol_threshold = market_volatility.quantile(self.volatility_percentiles[1])
            
            # Create regime masks
            low_vol_mask = market_volatility <= low_vol_threshold
            high_vol_mask = market_volatility >= high_vol_threshold
            
            # Calculate correlations in each regime
            high_vol_correlations = []
            low_vol_correlations = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:  # Skip market proxy
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # High volatility regime correlation
                if high_vol_mask.any():
                    high_vol_symbol = symbol_returns[high_vol_mask]
                    high_vol_market = market_returns[high_vol_mask]
                    
                    if len(high_vol_symbol.dropna()) > 10:
                        high_vol_corr = self.safe_correlation(high_vol_symbol, high_vol_market)
                        high_vol_correlations.append(high_vol_corr)
                
                # Low volatility regime correlation
                if low_vol_mask.any():
                    low_vol_symbol = symbol_returns[low_vol_mask]
                    low_vol_market = market_returns[low_vol_mask]
                    
                    if len(low_vol_symbol.dropna()) > 10:
                        low_vol_corr = self.safe_correlation(low_vol_symbol, low_vol_market)
                        low_vol_correlations.append(low_vol_corr)
            
            # Aggregate regime correlations
            if high_vol_correlations:
                avg_high_vol_corr = np.mean(high_vol_correlations)
                features['corr_high_vol'] = avg_high_vol_corr
            
            if low_vol_correlations:
                avg_low_vol_corr = np.mean(low_vol_correlations)
                features['corr_low_vol'] = avg_low_vol_corr
            
            # Regime difference
            if high_vol_correlations and low_vol_correlations:
                regime_diff = np.mean(high_vol_correlations) - np.mean(low_vol_correlations)
                features['corr_regime_diff'] = regime_diff
                
                # Volatility regime stability
                high_vol_std = np.std(high_vol_correlations)
                low_vol_std = np.std(low_vol_correlations)
                stability = 1.0 / (1.0 + high_vol_std + low_vol_std)
                features['vol_regime_correlation_stability'] = stability
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volatility regime correlations: {e}")
            return features
    
    def _calculate_trend_regime_correlations(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations in different trend regimes."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            market_returns = self.get_market_proxy(data)
            
            if market_returns.empty:
                return features
            
            # Calculate market trend
            market_trend = market_returns.rolling(window=self.trend_lookback).mean()
            
            # Define trend regimes
            uptrend_mask = market_trend > self.trend_threshold
            downtrend_mask = market_trend < -abs(self.trend_threshold)
            
            # Calculate correlations in each trend regime
            uptrend_correlations = []
            downtrend_correlations = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # Uptrend correlation
                if uptrend_mask.any():
                    uptrend_symbol = symbol_returns[uptrend_mask]
                    uptrend_market = market_returns[uptrend_mask]
                    
                    if len(uptrend_symbol.dropna()) > 10:
                        uptrend_corr = self.safe_correlation(uptrend_symbol, uptrend_market)
                        uptrend_correlations.append(uptrend_corr)
                
                # Downtrend correlation
                if downtrend_mask.any():
                    downtrend_symbol = symbol_returns[downtrend_mask]
                    downtrend_market = market_returns[downtrend_mask]
                    
                    if len(downtrend_symbol.dropna()) > 10:
                        downtrend_corr = self.safe_correlation(downtrend_symbol, downtrend_market)
                        downtrend_correlations.append(downtrend_corr)
            
            # Aggregate trend regime correlations
            if uptrend_correlations:
                avg_uptrend_corr = np.mean(uptrend_correlations)
                features['corr_uptrend'] = avg_uptrend_corr
            
            if downtrend_correlations:
                avg_downtrend_corr = np.mean(downtrend_correlations)
                features['corr_downtrend'] = avg_downtrend_corr
            
            # Trend regime difference
            if uptrend_correlations and downtrend_correlations:
                trend_diff = np.mean(downtrend_correlations) - np.mean(uptrend_correlations)
                features['corr_trend_diff'] = trend_diff
                
                # Trend regime stability
                uptrend_std = np.std(uptrend_correlations)
                downtrend_std = np.std(downtrend_correlations)
                stability = 1.0 / (1.0 + uptrend_std + downtrend_std)
                features['trend_regime_correlation_stability'] = stability
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating trend regime correlations: {e}")
            return features
    
    def _calculate_crisis_correlations(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlations during crisis vs normal periods."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            market_returns = self.get_market_proxy(data)
            
            if market_returns.empty:
                return features
            
            # Define crisis periods using multiple criteria
            market_volatility = market_returns.rolling(window=20).std()
            market_drawdown = self._calculate_drawdown(market_returns.cumsum())
            
            # Crisis indicators
            high_vol_crisis = market_volatility > market_volatility.quantile(0.9)
            drawdown_crisis = market_drawdown < -0.1  # 10% drawdown
            extreme_return_crisis = abs(market_returns) > market_returns.std() * 3
            
            # Combine crisis indicators
            crisis_mask = high_vol_crisis | drawdown_crisis | extreme_return_crisis
            normal_mask = ~crisis_mask
            
            # Calculate correlations in each period
            crisis_correlations = []
            normal_correlations = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # Crisis period correlation
                if crisis_mask.any():
                    crisis_symbol = symbol_returns[crisis_mask]
                    crisis_market = market_returns[crisis_mask]
                    
                    if len(crisis_symbol.dropna()) > 5:
                        crisis_corr = self.safe_correlation(crisis_symbol, crisis_market)
                        crisis_correlations.append(crisis_corr)
                
                # Normal period correlation
                if normal_mask.any():
                    normal_symbol = symbol_returns[normal_mask]
                    normal_market = market_returns[normal_mask]
                    
                    if len(normal_symbol.dropna()) > 10:
                        normal_corr = self.safe_correlation(normal_symbol, normal_market)
                        normal_correlations.append(normal_corr)
            
            # Aggregate crisis correlations
            if crisis_correlations:
                avg_crisis_corr = np.mean(crisis_correlations)
                features['corr_crisis_periods'] = avg_crisis_corr
            
            if normal_correlations:
                avg_normal_corr = np.mean(normal_correlations)
                features['corr_normal_periods'] = avg_normal_corr
            
            # Crisis correlation spike
            if crisis_correlations and normal_correlations:
                correlation_spike = np.mean(crisis_correlations) - np.mean(normal_correlations)
                features['crisis_correlation_spike'] = correlation_spike
                
                # Contagion score (how much correlations increase during crisis)
                if avg_normal_corr > 0:
                    contagion_score = correlation_spike / avg_normal_corr
                    features['correlation_contagion_score'] = contagion_score
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating crisis correlations: {e}")
            return features
    
    def _calculate_regime_transitions(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime transition features."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            market_returns = self.get_market_proxy(data)
            
            if market_returns.empty:
                return features
            
            # Calculate regime indicators
            market_volatility = market_returns.rolling(window=20).std()
            vol_threshold = market_volatility.quantile(0.67)
            
            # Volatility regime (0 = low, 1 = high)
            vol_regime = (market_volatility > vol_threshold).astype(int)
            
            # Detect regime transitions
            regime_changes = vol_regime.diff().abs()
            
            # Transition frequency
            transition_frequency = regime_changes.rolling(window=60).mean()
            features['regime_transition_frequency'] = transition_frequency.mean()
            
            # Regime persistence (average regime duration)
            regime_durations = self._calculate_regime_durations(vol_regime)
            if regime_durations:
                avg_persistence = np.mean(regime_durations)
                features['correlation_regime_persistence'] = avg_persistence
            
            # Regime momentum (tendency to stay in current regime)
            def momentum_function(values):
                if len(values) < 5:
                    return 0.0
                current_regime = values.iloc[-1]
                recent_regime = values.iloc[-5:].mean()
                return abs(current_regime - recent_regime)
            
            regime_momentum = vol_regime.rolling(window=10).apply(momentum_function)
            features['regime_correlation_momentum'] = regime_momentum.mean()
            
            # Regime clustering (how clustered transitions are)
            transition_indices = regime_changes[regime_changes > 0].index
            if len(transition_indices) > 1:
                transition_gaps = [(transition_indices[i] - transition_indices[i-1]).days 
                                 for i in range(1, len(transition_indices))]
                if transition_gaps:
                    clustering_score = np.std(transition_gaps) / (np.mean(transition_gaps) + 1)
                    features['correlation_regime_clustering'] = clustering_score
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating regime transitions: {e}")
            return features
    
    def _calculate_advanced_regime_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced regime-dependent features."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty:
                return features
            
            market_returns = self.get_market_proxy(data)
            
            if market_returns.empty:
                return features
            
            # Tail dependence (correlation in extreme events)
            extreme_down_threshold = market_returns.quantile(0.05)
            extreme_up_threshold = market_returns.quantile(0.95)
            
            extreme_down_mask = market_returns <= extreme_down_threshold
            extreme_up_mask = market_returns >= extreme_up_threshold
            
            tail_correlations = []
            
            for symbol in returns_pivot.columns:
                if symbol in market_returns.index:
                    continue
                
                symbol_returns = returns_pivot[symbol]
                
                # Downside tail dependence
                if extreme_down_mask.any():
                    down_tail_corr = self.safe_correlation(
                        symbol_returns[extreme_down_mask], 
                        market_returns[extreme_down_mask]
                    )
                    tail_correlations.append(abs(down_tail_corr))
            
            if tail_correlations:
                avg_tail_dependence = np.mean(tail_correlations)
                features['correlation_tail_dependence'] = avg_tail_dependence
            
            # Regime correlation asymmetry
            if ('corr_uptrend' in features.columns and 'corr_downtrend' in features.columns and
                'corr_high_vol' in features.columns and 'corr_low_vol' in features.columns):
                
                trend_asymmetry = abs(features['corr_uptrend'].iloc[0] - features['corr_downtrend'].iloc[0])
                vol_asymmetry = abs(features['corr_high_vol'].iloc[0] - features['corr_low_vol'].iloc[0])
                
                total_asymmetry = (trend_asymmetry + vol_asymmetry) / 2
                features['regime_correlation_asymmetry'] = total_asymmetry
            
            # Structural regime indicator
            # Based on stability of regime correlations
            regime_stability_features = [
                'vol_regime_correlation_stability',
                'trend_regime_correlation_stability'
            ]
            
            stability_scores = []
            for feature in regime_stability_features:
                if feature in features.columns:
                    score = features[feature].iloc[0] if len(features) > 0 else 0.0
                    stability_scores.append(score)
            
            if stability_scores:
                structural_indicator = np.mean(stability_scores)
                features['structural_regime_indicator'] = structural_indicator
            
            # Correlation stress sensitivity
            # How much correlations change during stressed periods
            if ('corr_crisis_periods' in features.columns and 
                'corr_normal_periods' in features.columns):
                
                crisis_corr = features['corr_crisis_periods'].iloc[0] if len(features) > 0 else 0.0
                normal_corr = features['corr_normal_periods'].iloc[0] if len(features) > 0 else 0.0
                
                if normal_corr != 0:
                    stress_sensitivity = abs(crisis_corr - normal_corr) / abs(normal_corr)
                    features['correlation_stress_sensitivity'] = stress_sensitivity
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced regime features: {e}")
            return features
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate rolling maximum drawdown."""
        try:
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown
            
        except Exception as e:
            logger.warning(f"Error calculating drawdown: {e}")
            return pd.Series(0.0, index=cumulative_returns.index)
    
    def _calculate_regime_durations(self, regime_series: pd.Series) -> List[int]:
        """Calculate durations of each regime period."""
        try:
            durations = []
            current_regime = regime_series.iloc[0]
            current_duration = 1
            
            for i in range(1, len(regime_series)):
                if regime_series.iloc[i] == current_regime:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_regime = regime_series.iloc[i]
                    current_duration = 1
            
            # Add the last regime duration
            durations.append(current_duration)
            
            return durations
            
        except Exception as e:
            logger.warning(f"Error calculating regime durations: {e}")
            return []
    
    def _align_features_with_data(self, features: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Align features DataFrame with original data structure."""
        try:
            if 'timestamp' in original_data.columns:
                expanded_features = original_data[['timestamp']].merge(
                    features.reset_index().rename(columns={'index': 'timestamp'}),
                    on='timestamp',
                    how='left'
                )
                
                expanded_features = expanded_features.drop('timestamp', axis=1)
                expanded_features.index = original_data.index
                expanded_features = expanded_features.fillna(0.0)
                
                return expanded_features
            else:
                return features.reindex(original_data.index, fill_value=0.0)
                
        except Exception as e:
            logger.warning(f"Error aligning features with data: {e}")
            return features