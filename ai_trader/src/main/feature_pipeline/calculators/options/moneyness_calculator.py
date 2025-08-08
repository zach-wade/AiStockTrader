"""
Moneyness Calculator

Specialized calculator for strike-based analysis and moneyness categorization including:
- ITM/ATM/OTM volume and open interest analysis
- Moneyness-based Put/Call ratios
- Strike distribution analysis
- Moneyness concentration metrics
- Dynamic moneyness classification
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_options import BaseOptionsCalculator
from ..helpers import safe_divide, safe_sqrt

from main.utils.core import get_logger

logger = get_logger(__name__)


class MoneynessCalculator(BaseOptionsCalculator):
    """Calculator for moneyness-based analysis and strike distribution."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize moneyness calculator."""
        super().__init__(config)
        
        # Moneyness-specific parameters
        self.moneyness_config = self.options_config.get_moneyness_config()
        self.atm_threshold = self.options_config.atm_threshold
        self.itm_threshold = self.options_config.itm_threshold
        self.otm_threshold = self.options_config.otm_threshold
        
        # Strike analysis parameters
        self.max_strike_range = self.options_config.max_strike_range
        
        logger.debug(f"Initialized MoneynessCalculator with ATM threshold: ±{self.atm_threshold:.1%}")
    
    def get_feature_names(self) -> List[str]:
        """Return list of moneyness feature names."""
        feature_names = [
            # Volume by moneyness
            'itm_call_volume',
            'itm_put_volume', 
            'atm_call_volume',
            'atm_put_volume',
            'otm_call_volume',
            'otm_put_volume',
            
            # Open interest by moneyness
            'itm_call_oi',
            'itm_put_oi',
            'atm_call_oi', 
            'atm_put_oi',
            'otm_call_oi',
            'otm_put_oi',
            
            # Put/Call ratios by moneyness
            'itm_pc_ratio',
            'atm_pc_ratio',
            'otm_pc_ratio',
            'pc_moneyness_spread',
            'pc_moneyness_skew',
            
            # Volume concentration metrics
            'volume_concentration_itm',
            'volume_concentration_atm',
            'volume_concentration_otm',
            'moneyness_concentration_index',
            
            # Strike distribution analysis
            'strike_range_coverage',
            'strike_density_atm',
            'weighted_average_strike',
            'strike_dispersion',
            
            # Advanced moneyness metrics
            'moneyness_momentum',
            'moneyness_reversion_tendency',
            'speculative_ratio',
            'hedging_ratio',
            'moneyness_efficiency_score'
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moneyness features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with moneyness features
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
            
            # Calculate volume by moneyness
            features = self._calculate_volume_by_moneyness(processed_data, features)
            
            # Calculate open interest by moneyness
            features = self._calculate_oi_by_moneyness(processed_data, features)
            
            # Calculate P/C ratios by moneyness
            features = self._calculate_pc_ratios_by_moneyness(processed_data, features)
            
            # Calculate concentration metrics
            features = self._calculate_concentration_metrics(processed_data, features)
            
            # Calculate strike distribution analysis
            features = self._calculate_strike_distribution(processed_data, features)
            
            # Calculate advanced moneyness metrics
            features = self._calculate_advanced_moneyness_metrics(processed_data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating moneyness features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_volume_by_moneyness(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume distribution by moneyness."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            if self.options_chain is None or self.options_chain.empty:
                # Estimate volume distribution when no options chain
                features = self._estimate_volume_by_moneyness(data, features)
                return features
            
            # Classify each option by moneyness using vectorized operations
            volume_by_moneyness = {
                'itm_call': 0, 'itm_put': 0,
                'atm_call': 0, 'atm_put': 0,
                'otm_call': 0, 'otm_put': 0
            }
            
            # Calculate moneyness for all options vectorized
            moneyness_series = self.options_chain.apply(
                lambda row: self.calculate_moneyness(row['strike'], current_price, row['optionType']),
                axis=1
            )
            
            # Create masks for each moneyness category
            itm_mask = moneyness_series.isin(['deep_itm', 'itm'])
            atm_mask = moneyness_series == 'atm'
            otm_mask = moneyness_series.isin(['otm', 'deep_otm'])
            
            # Calculate volume sums for each category
            for option_type in ['call', 'put']:
                type_mask = self.options_chain['optionType'].str.lower() == option_type
                
                volume_by_moneyness[f'itm_{option_type}'] = self.options_chain.loc[itm_mask & type_mask, 'volume'].sum()
                volume_by_moneyness[f'atm_{option_type}'] = self.options_chain.loc[atm_mask & type_mask, 'volume'].sum()
                volume_by_moneyness[f'otm_{option_type}'] = self.options_chain.loc[otm_mask & type_mask, 'volume'].sum()
            
            # Store volume features
            for key, volume in volume_by_moneyness.items():
                features[f'{key}_volume'] = volume
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating volume by moneyness: {e}")
            return features
    
    def _estimate_volume_by_moneyness(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate volume distribution when options chain unavailable."""
        try:
            # Use market conditions to estimate distribution
            if len(data) >= 20:
                recent_returns = data['close'].pct_change().iloc[-20:].mean()
                vol = data['close'].pct_change().iloc[-20:].std() * safe_sqrt(252)
                
                # Base volume estimate
                base_volume = 1000  # Arbitrary base
                
                # Distribute based on market conditions
                if recent_returns > 0.01:  # Bullish
                    # More call volume, especially OTM calls
                    call_bias = 1.3
                    otm_bias = 1.2
                elif recent_returns < -0.01:  # Bearish
                    # More put volume, especially OTM puts
                    call_bias = 0.7
                    otm_bias = 1.2
                else:  # Neutral
                    call_bias = 1.0
                    otm_bias = 1.0
                
                # High volatility increases OTM activity
                if vol > 0.3:
                    otm_bias *= 1.3
                
                # Distribute volume
                features['itm_call_volume'] = base_volume * 0.2 * call_bias
                features['itm_put_volume'] = base_volume * 0.2 / call_bias
                features['atm_call_volume'] = base_volume * 0.4 * call_bias
                features['atm_put_volume'] = base_volume * 0.4 / call_bias
                features['otm_call_volume'] = base_volume * 0.4 * call_bias * otm_bias
                features['otm_put_volume'] = base_volume * 0.4 / call_bias * otm_bias
            
            else:
                # Default neutral distribution
                base_volume = 1000
                features['itm_call_volume'] = base_volume * 0.2
                features['itm_put_volume'] = base_volume * 0.2
                features['atm_call_volume'] = base_volume * 0.4
                features['atm_put_volume'] = base_volume * 0.4
                features['otm_call_volume'] = base_volume * 0.4
                features['otm_put_volume'] = base_volume * 0.4
            
            return features
            
        except Exception as e:
            logger.warning(f"Error estimating volume by moneyness: {e}")
            return features
    
    def _calculate_oi_by_moneyness(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate open interest distribution by moneyness."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            if self.options_chain is None or self.options_chain.empty or 'openInterest' not in self.options_chain.columns:
                # Estimate OI as multiple of volume
                oi_multiple = 5.0  # Typical OI is 5x daily volume
                
                for moneyness in ['itm', 'atm', 'otm']:
                    for option_type in ['call', 'put']:
                        volume_key = f'{moneyness}_{option_type}_volume'
                        oi_key = f'{moneyness}_{option_type}_oi'
                        
                        if volume_key in features.columns:
                            volume = features[volume_key].iloc[-1] if len(features) > 0 else 0
                            features[oi_key] = volume * oi_multiple
                        else:
                            features[oi_key] = 0
                
                return features
            
            # Calculate actual OI distribution using vectorized operations
            oi_by_moneyness = {
                'itm_call': 0, 'itm_put': 0,
                'atm_call': 0, 'atm_put': 0,
                'otm_call': 0, 'otm_put': 0
            }
            
            # Calculate moneyness for all options vectorized
            moneyness_series = self.options_chain.apply(
                lambda row: self.calculate_moneyness(row['strike'], current_price, row['optionType']),
                axis=1
            )
            
            # Create masks for each moneyness category
            itm_mask = moneyness_series.isin(['deep_itm', 'itm'])
            atm_mask = moneyness_series == 'atm'
            otm_mask = moneyness_series.isin(['otm', 'deep_otm'])
            
            # Calculate OI sums for each category
            for option_type in ['call', 'put']:
                type_mask = self.options_chain['optionType'].str.lower() == option_type
                
                oi_by_moneyness[f'itm_{option_type}'] = self.options_chain.loc[itm_mask & type_mask, 'openInterest'].sum()
                oi_by_moneyness[f'atm_{option_type}'] = self.options_chain.loc[atm_mask & type_mask, 'openInterest'].sum()
                oi_by_moneyness[f'otm_{option_type}'] = self.options_chain.loc[otm_mask & type_mask, 'openInterest'].sum()
            
            # Store OI features
            for key, oi in oi_by_moneyness.items():
                features[f'{key}_oi'] = oi
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating OI by moneyness: {e}")
            return features
    
    def _calculate_pc_ratios_by_moneyness(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Put/Call ratios by moneyness category."""
        try:
            # Calculate P/C ratios for each moneyness category
            for moneyness in ['itm', 'atm', 'otm']:
                call_volume_key = f'{moneyness}_call_volume'
                put_volume_key = f'{moneyness}_put_volume'
                pc_ratio_key = f'{moneyness}_pc_ratio'
                
                if call_volume_key in features.columns and put_volume_key in features.columns:
                    call_volume = features[call_volume_key].iloc[-1] if len(features) > 0 else 0
                    put_volume = features[put_volume_key].iloc[-1] if len(features) > 0 else 0
                    
                    pc_ratio = safe_divide(put_volume, call_volume, default_value=1.0)
                    features[pc_ratio_key] = pc_ratio
                else:
                    features[pc_ratio_key] = 1.0  # Neutral
            
            # Calculate moneyness spread and skew
            if all(f'{m}_pc_ratio' in features.columns for m in ['itm', 'atm', 'otm']):
                itm_pc = features['itm_pc_ratio'].iloc[-1] if len(features) > 0 else 1.0
                atm_pc = features['atm_pc_ratio'].iloc[-1] if len(features) > 0 else 1.0
                otm_pc = features['otm_pc_ratio'].iloc[-1] if len(features) > 0 else 1.0
                
                # Spread between OTM and ITM P/C ratios
                pc_moneyness_spread = otm_pc - itm_pc
                features['pc_moneyness_spread'] = pc_moneyness_spread
                
                # Skew (asymmetry around ATM)
                left_skew = atm_pc - itm_pc
                right_skew = otm_pc - atm_pc
                pc_moneyness_skew = right_skew - left_skew
                features['pc_moneyness_skew'] = pc_moneyness_skew
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating P/C ratios by moneyness: {e}")
            return features
    
    def _calculate_concentration_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume concentration metrics by moneyness."""
        try:
            # Calculate volume concentration for each moneyness category
            total_volume = 0
            moneyness_volumes = {}
            
            for moneyness in ['itm', 'atm', 'otm']:
                call_volume = features.get(f'{moneyness}_call_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                put_volume = features.get(f'{moneyness}_put_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                
                moneyness_volume = call_volume + put_volume
                moneyness_volumes[moneyness] = moneyness_volume
                total_volume += moneyness_volume
            
            # Calculate concentration percentages
            if total_volume > 0:
                for moneyness, volume in moneyness_volumes.items():
                    concentration = safe_divide(volume, total_volume, default_value=0.0)
                    features[f'volume_concentration_{moneyness}'] = concentration
                
                # Calculate Herfindahl concentration index
                concentrations = [safe_divide(v, total_volume, default_value=0.0) for v in moneyness_volumes.values()]
                concentration_index = sum(c**2 for c in concentrations)
                features['moneyness_concentration_index'] = concentration_index
            else:
                # Default equal distribution
                for moneyness in ['itm', 'atm', 'otm']:
                    features[f'volume_concentration_{moneyness}'] = 1/3
                features['moneyness_concentration_index'] = 1/3
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating concentration metrics: {e}")
            return features
    
    def _calculate_strike_distribution(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate strike distribution analysis."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            if self.options_chain is None or self.options_chain.empty:
                # Estimate strike distribution metrics
                features['strike_range_coverage'] = 0.8  # 80% coverage
                features['strike_density_atm'] = 10      # 10 strikes around ATM
                features['weighted_average_strike'] = current_price
                features['strike_dispersion'] = current_price * 0.2  # 20% dispersion
                return features
            
            strikes = self.options_chain['strike'].unique()
            volumes = []
            
            # Get volume-weighted strikes
            for strike in strikes:
                strike_options = self.options_chain[self.options_chain['strike'] == strike]
                total_volume = strike_options['volume'].sum()
                volumes.append(total_volume)
            
            if len(strikes) > 0 and sum(volumes) > 0:
                # Strike range coverage
                min_strike = strikes.min()
                max_strike = strikes.max()
                range_coverage = safe_divide((max_strike - min_strike), current_price, default_value=0.0)
                features['strike_range_coverage'] = range_coverage
                
                # ATM strike density (strikes within ±5% of current price)
                atm_range_low = current_price * 0.95
                atm_range_high = current_price * 1.05
                atm_strikes = strikes[(strikes >= atm_range_low) & (strikes <= atm_range_high)]
                features['strike_density_atm'] = len(atm_strikes)
                
                # Volume-weighted average strike
                total_volume = sum(volumes)
                weighted_strike = safe_divide(sum(strike * volume for strike, volume in zip(strikes, volumes)), total_volume, default_value=current_price)
                features['weighted_average_strike'] = weighted_strike
                
                # Strike dispersion (standard deviation)
                if len(strikes) > 1:
                    weights = np.array(volumes) / total_volume
                    variance = sum(weights * (strikes - weighted_strike)**2)
                    strike_dispersion = safe_sqrt(variance)
                    features['strike_dispersion'] = strike_dispersion
                else:
                    features['strike_dispersion'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating strike distribution: {e}")
            return features
    
    def _calculate_advanced_moneyness_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced moneyness-based metrics."""
        try:
            # Moneyness momentum (shift in activity toward OTM vs ITM)
            if all(f'{m}_call_volume' in features.columns for m in ['itm', 'atm', 'otm']):
                itm_volume = (features['itm_call_volume'].iloc[-1] + features['itm_put_volume'].iloc[-1]) if len(features) > 0 else 0
                otm_volume = (features['otm_call_volume'].iloc[-1] + features['otm_put_volume'].iloc[-1]) if len(features) > 0 else 0
                total_volume = itm_volume + otm_volume
                
                moneyness_momentum = safe_divide((otm_volume - itm_volume), total_volume, default_value=0.0)
                features['moneyness_momentum'] = moneyness_momentum
            
            # Mean reversion tendency (preference for ATM options)
            if 'volume_concentration_atm' in features.columns:
                atm_concentration = features['volume_concentration_atm'].iloc[-1] if len(features) > 0 else 0.33
                # Higher ATM concentration suggests mean reversion bias
                reversion_tendency = atm_concentration * 2 - 1  # Scale to [-1, 1]
                features['moneyness_reversion_tendency'] = max(-1, min(1, reversion_tendency))
            else:
                features['moneyness_reversion_tendency'] = 0.0
            
            # Speculative ratio (OTM vs total activity)
            otm_call_vol = features.get('otm_call_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            otm_put_vol = features.get('otm_put_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            total_otm = otm_call_vol + otm_put_vol
            
            total_volume = sum([
                features.get(f'{m}_{t}_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
                for m in ['itm', 'atm', 'otm'] 
                for t in ['call', 'put']
            ])
            
            speculative_ratio = safe_divide(total_otm, total_volume, default_value=0.0)
            features['speculative_ratio'] = speculative_ratio
            
            # Hedging ratio (ITM activity, which is often hedging)
            itm_call_vol = features.get('itm_call_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            itm_put_vol = features.get('itm_put_volume', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            total_itm = itm_call_vol + itm_put_vol
            
            hedging_ratio = safe_divide(total_itm, total_volume, default_value=0.0)
            features['hedging_ratio'] = hedging_ratio
            
            # Moneyness efficiency score
            # Higher score indicates efficient distribution across moneyness levels
            if 'moneyness_concentration_index' in features.columns:
                concentration_index = features['moneyness_concentration_index'].iloc[-1] if len(features) > 0 else 0.33
                # Lower concentration = higher efficiency
                efficiency_score = 1 - concentration_index
                features['moneyness_efficiency_score'] = efficiency_score
            else:
                features['moneyness_efficiency_score'] = 0.67  # Moderate efficiency
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced moneyness metrics: {e}")
            return features