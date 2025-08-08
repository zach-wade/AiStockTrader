"""
Greeks Calculator

Specialized calculator for options Greeks computation including:
- Delta, Gamma, Theta, Vega calculations
- Aggregate Greeks exposure analysis
- Pin risk and gamma exposure analysis
- Greeks-based portfolio risk metrics
- Estimated Greeks using Black-Scholes model
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_options import BaseOptionsCalculator
from ..helpers import safe_divide, safe_sqrt

logger = logging.getLogger(__name__)


class GreeksCalculator(BaseOptionsCalculator):
    """Calculator for options Greeks computation and risk exposure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Greeks calculator."""
        super().__init__(config)
        
        # Greeks-specific parameters
        self.greeks_config = self.options_config.get_blackscholes_config()
        self.pin_risk_range = self.options_config.pin_risk_range
        self.max_delta_exposure = self.options_config.max_delta_exposure
        self.max_gamma_exposure = self.options_config.max_gamma_exposure
        self.max_vega_exposure = self.options_config.max_vega_exposure
        
        # Calculation parameters
        self.spot_shift = self.greeks_config['greeks_shifts']['spot']
        self.vol_shift = self.greeks_config['greeks_shifts']['vol']
        self.time_shift = self.greeks_config['greeks_shifts']['time']
        
        logger.debug(f"Initialized GreeksCalculator with pin risk range: ±{self.pin_risk_range:.1%}")
    
    def get_feature_names(self) -> List[str]:
        """Return list of Greeks feature names."""
        feature_names = [
            # Aggregate exposure metrics
            'net_delta_exposure',
            'total_gamma_exposure', 
            'total_vega_exposure',
            'total_theta_exposure',
            
            # Pin risk analysis
            'pin_risk_gamma',
            'pin_risk_level',
            'max_pain_level',
            'gamma_concentration',
            
            # Delta analysis
            'call_delta_exposure',
            'put_delta_exposure',
            'delta_hedge_ratio',
            'delta_sensitivity',
            
            # Gamma analysis
            'positive_gamma_exposure',
            'negative_gamma_exposure',
            'gamma_dollar_exposure',
            'gamma_risk_score',
            
            # Vega analysis
            'call_vega_exposure',
            'put_vega_exposure',
            'vega_dollar_exposure',
            'iv_sensitivity_score',
            
            # Theta analysis
            'theta_decay_daily',
            'theta_decay_weekly',
            'time_decay_risk',
            
            # Cross-Greeks and advanced metrics
            'charm_exposure',  # Delta-theta cross derivative
            'vomma_exposure',  # Vega-volatility sensitivity
            'vera_exposure',   # Vega-rho sensitivity
            'speed_exposure',  # Gamma-spot sensitivity
            
            # Portfolio Greeks ratios
            'delta_gamma_ratio',
            'vega_theta_ratio',
            'greeks_diversification_score',
            'options_leverage_ratio'
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Greeks features
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
            
            # Calculate aggregate Greeks exposure
            features = self._calculate_aggregate_greeks(processed_data, features)
            
            # Calculate pin risk analysis
            features = self._calculate_pin_risk_analysis(processed_data, features)
            
            # Calculate individual Greeks analysis
            features = self._calculate_individual_greeks_analysis(processed_data, features)
            
            # Calculate cross-Greeks and advanced metrics
            features = self._calculate_advanced_greeks_metrics(processed_data, features)
            
            # Calculate portfolio Greeks ratios
            features = self._calculate_greeks_ratios(processed_data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating Greeks features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_aggregate_greeks(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate Greeks exposure metrics."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            if self.options_chain is None or self.options_chain.empty:
                # Estimate Greeks when options chain unavailable
                features = self._estimate_aggregate_greeks(data, features)
                return features
            
            # Check if Greeks are available in options chain
            has_greeks = all(col in self.options_chain.columns for col in ['delta', 'gamma', 'theta', 'vega'])
            
            if has_greeks:
                # Use provided Greeks
                features = self._calculate_from_chain_greeks(features)
            else:
                # Calculate Greeks using Black-Scholes
                features = self._calculate_estimated_greeks(data, features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating aggregate Greeks: {e}")
            return features
    
    def _calculate_from_chain_greeks(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate metrics from options chain Greeks."""
        try:
            # Separate calls and puts
            calls = self.options_chain[self.options_chain['optionType'] == 'call']
            puts = self.options_chain[self.options_chain['optionType'] == 'put']
            
            # Weight by open interest (position size)
            call_weights = calls['openInterest']
            put_weights = puts['openInterest']
            
            # Aggregate Delta exposure
            call_delta_exposure = (calls['delta'] * call_weights).sum()
            put_delta_exposure = (puts['delta'] * put_weights).sum()
            net_delta_exposure = call_delta_exposure + put_delta_exposure
            
            features['call_delta_exposure'] = call_delta_exposure
            features['put_delta_exposure'] = put_delta_exposure
            features['net_delta_exposure'] = net_delta_exposure
            
            # Aggregate Gamma exposure
            total_gamma_exposure = (self.options_chain['gamma'] * self.options_chain['openInterest']).sum()
            features['total_gamma_exposure'] = total_gamma_exposure
            
            # Separate positive and negative gamma
            positive_gamma = self.options_chain[self.options_chain['gamma'] > 0]
            negative_gamma = self.options_chain[self.options_chain['gamma'] < 0]
            
            features['positive_gamma_exposure'] = (positive_gamma['gamma'] * positive_gamma['openInterest']).sum()
            features['negative_gamma_exposure'] = (negative_gamma['gamma'] * negative_gamma['openInterest']).sum()
            
            # Aggregate Vega exposure
            call_vega_exposure = (calls['vega'] * call_weights).sum()
            put_vega_exposure = (puts['vega'] * put_weights).sum()
            total_vega_exposure = call_vega_exposure + put_vega_exposure
            
            features['call_vega_exposure'] = call_vega_exposure
            features['put_vega_exposure'] = put_vega_exposure
            features['total_vega_exposure'] = total_vega_exposure
            
            # Aggregate Theta exposure
            total_theta_exposure = (self.options_chain['theta'] * self.options_chain['openInterest']).sum()
            features['total_theta_exposure'] = total_theta_exposure
            
            # Time decay metrics
            features['theta_decay_daily'] = total_theta_exposure
            features['theta_decay_weekly'] = total_theta_exposure * 7
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating from chain Greeks: {e}")
            return features
    
    def _calculate_estimated_greeks(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate estimated Greeks using Black-Scholes model."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            # Estimate IV for Greeks calculation
            estimated_iv = 0.25  # Default 25% IV
            if len(data) >= 20:
                hist_vol = data['close'].pct_change().rolling(20).std().iloc[-1] * safe_sqrt(252)
                estimated_iv = hist_vol * 1.2  # IV premium
            
            # Use options chain for strikes and expiries if available
            if self.options_chain is not None and not self.options_chain.empty:
                try:
                    # Calculate Greeks for all options using vectorized operations
                    time_to_expiry = safe_divide(self.options_chain.get('daysToExpiration', 30).fillna(30), 365.25, default_value=0.0)
                    open_interest = self.options_chain.get('openInterest', 1).fillna(1)
                    
                    # Calculate delta for all options
                    delta_values = self.options_chain.apply(
                        lambda row: self.calculate_delta(
                            current_price, row['strike'], time_to_expiry[row.name],
                            self.greeks_config['risk_free_rate'], estimated_iv, row['optionType']
                        ), axis=1
                    )
                    
                    # Calculate gamma for all options
                    gamma_values = self.options_chain.apply(
                        lambda row: self.calculate_gamma(
                            current_price, row['strike'], time_to_expiry[row.name],
                            self.greeks_config['risk_free_rate'], estimated_iv
                        ), axis=1
                    )
                    
                    # Calculate vega for all options
                    vega_values = self.options_chain.apply(
                        lambda row: self.calculate_vega(
                            current_price, row['strike'], time_to_expiry[row.name],
                            self.greeks_config['risk_free_rate'], estimated_iv
                        ), axis=1
                    )
                    
                    # Calculate theta for all options
                    theta_values = self.options_chain.apply(
                        lambda row: self.calculate_theta(
                            current_price, row['strike'], time_to_expiry[row.name],
                            self.greeks_config['risk_free_rate'], estimated_iv, row['optionType']
                        ), axis=1
                    )
                    
                    # Weight by open interest and aggregate
                    aggregate_delta = (delta_values * open_interest).sum()
                    aggregate_gamma = (gamma_values * open_interest).sum()
                    aggregate_vega = (vega_values * open_interest).sum()
                    aggregate_theta = (theta_values * open_interest).sum()
                    
                except Exception as e:
                    logger.debug(f"Error calculating Greeks for options chain: {e}")
                    # Fall back to zero aggregates
                    aggregate_delta = aggregate_gamma = aggregate_vega = aggregate_theta = 0
                
                features['net_delta_exposure'] = aggregate_delta
                features['total_gamma_exposure'] = aggregate_gamma
                features['total_vega_exposure'] = aggregate_vega
                features['total_theta_exposure'] = aggregate_theta
                
            else:
                # Rough estimates for ATM options
                time_to_expiry = safe_divide(30, 365.25, default_value=0.0)  # 30 days
                
                atm_delta = self.calculate_delta(
                    current_price, current_price, time_to_expiry,
                    self.greeks_config['risk_free_rate'], estimated_iv, 'call'
                )
                
                atm_gamma = self.calculate_gamma(
                    current_price, current_price, time_to_expiry,
                    self.greeks_config['risk_free_rate'], estimated_iv
                )
                
                atm_vega = self.calculate_vega(
                    current_price, current_price, time_to_expiry,
                    self.greeks_config['risk_free_rate'], estimated_iv
                )
                
                atm_theta = self.calculate_theta(
                    current_price, current_price, time_to_expiry,
                    self.greeks_config['risk_free_rate'], estimated_iv, 'call'
                )
                
                # Assume some reasonable position size
                position_multiplier = 100
                
                features['net_delta_exposure'] = atm_delta * position_multiplier
                features['total_gamma_exposure'] = atm_gamma * position_multiplier
                features['total_vega_exposure'] = atm_vega * position_multiplier
                features['total_theta_exposure'] = atm_theta * position_multiplier
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating estimated Greeks: {e}")
            return features
    
    def _estimate_aggregate_greeks(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Estimate aggregate Greeks when no options data available."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            # Estimate based on typical options market characteristics
            # These would be more sophisticated in production
            
            # Estimate delta exposure (market neutral assumption)
            features['net_delta_exposure'] = 0.0
            features['call_delta_exposure'] = current_price * 50  # Estimate
            features['put_delta_exposure'] = -current_price * 50  # Estimate
            
            # Estimate gamma exposure
            features['total_gamma_exposure'] = 100.0
            features['positive_gamma_exposure'] = 120.0
            features['negative_gamma_exposure'] = -20.0
            
            # Estimate vega exposure
            features['total_vega_exposure'] = current_price * 2
            features['call_vega_exposure'] = current_price * 1.2
            features['put_vega_exposure'] = current_price * 0.8
            
            # Estimate theta exposure
            features['total_theta_exposure'] = -current_price * 0.1
            features['theta_decay_daily'] = -current_price * 0.1
            features['theta_decay_weekly'] = -current_price * 0.7
            
            return features
            
        except Exception as e:
            logger.warning(f"Error estimating aggregate Greeks: {e}")
            return features
    
    def _calculate_pin_risk_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate pin risk and gamma concentration analysis."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            if self.options_chain is None or self.options_chain.empty:
                # Estimate pin risk
                features['pin_risk_gamma'] = 50.0  # Moderate pin risk
                features['pin_risk_level'] = 0.3   # 30% pin risk
                features['max_pain_level'] = current_price
                features['gamma_concentration'] = 0.5
                return features
            
            # Define pin risk range around current price
            lower_bound = current_price * (1 - self.pin_risk_range)
            upper_bound = current_price * (1 + self.pin_risk_range)
            
            # Filter options near the money
            near_money_options = self.options_chain[
                (self.options_chain['strike'] >= lower_bound) &
                (self.options_chain['strike'] <= upper_bound)
            ]
            
            # Calculate pin risk gamma
            if 'gamma' in self.options_chain.columns:
                pin_risk_gamma = (
                    near_money_options['gamma'] * 
                    near_money_options['openInterest']
                ).sum()
                features['pin_risk_gamma'] = pin_risk_gamma
                
                # Total gamma for concentration calculation
                total_gamma = (
                    self.options_chain['gamma'] * 
                    self.options_chain['openInterest']
                ).sum()
                
                # Gamma concentration (how much gamma is near current price)
                gamma_concentration = safe_divide(abs(pin_risk_gamma), abs(total_gamma), default_value=0.0)
                features['gamma_concentration'] = gamma_concentration
                
            else:
                # Estimate gamma concentration
                features['pin_risk_gamma'] = 50.0
                features['gamma_concentration'] = 0.3
            
            # Pin risk level (probability of pinning)
            # This is a simplified calculation - production would be more sophisticated
            total_volume_near_money = near_money_options['volume'].sum()
            total_volume = self.options_chain['volume'].sum()
            
            if total_volume > 0:
                pin_risk_level = safe_divide(total_volume_near_money, total_volume, default_value=0.0)
                features['pin_risk_level'] = pin_risk_level
            else:
                features['pin_risk_level'] = 0.3
            
            # Max pain calculation (strike with maximum open interest)
            strike_oi = self.options_chain.groupby('strike')['openInterest'].sum()
            if not strike_oi.empty:
                max_pain_strike = strike_oi.idxmax()
                features['max_pain_level'] = max_pain_strike
            else:
                features['max_pain_level'] = current_price
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating pin risk analysis: {e}")
            return features
    
    def _calculate_individual_greeks_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate detailed analysis for individual Greeks."""
        try:
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            
            # Delta analysis
            if 'net_delta_exposure' in features.columns:
                net_delta = features['net_delta_exposure'].iloc[-1] if len(features) > 0 else 0
                
                # Delta hedge ratio (how much stock needed to hedge)
                features['delta_hedge_ratio'] = safe_divide(abs(net_delta), current_price, default_value=0.0)
                
                # Delta sensitivity (change in P&L per $1 stock move)
                features['delta_sensitivity'] = abs(net_delta)
            
            # Gamma analysis
            if 'total_gamma_exposure' in features.columns:
                total_gamma = features['total_gamma_exposure'].iloc[-1] if len(features) > 0 else 0
                
                # Gamma dollar exposure (change in delta per $1 stock move)
                features['gamma_dollar_exposure'] = abs(total_gamma) * current_price
                
                # Gamma risk score (0-1, where 1 is highest risk)
                normalized_gamma = min(safe_divide(abs(total_gamma), self.max_gamma_exposure, default_value=0.0), 1.0)
                features['gamma_risk_score'] = normalized_gamma
            
            # Vega analysis
            if 'total_vega_exposure' in features.columns:
                total_vega = features['total_vega_exposure'].iloc[-1] if len(features) > 0 else 0
                
                # Vega dollar exposure (change in P&L per 1% IV move)
                features['vega_dollar_exposure'] = abs(total_vega)
                
                # IV sensitivity score
                normalized_vega = min(safe_divide(abs(total_vega), self.max_vega_exposure, default_value=0.0), 1.0)
                features['iv_sensitivity_score'] = normalized_vega
            
            # Theta analysis
            if 'total_theta_exposure' in features.columns:
                total_theta = features['total_theta_exposure'].iloc[-1] if len(features) > 0 else 0
                
                # Time decay risk (how much P&L decays per day)
                features['time_decay_risk'] = abs(total_theta)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating individual Greeks analysis: {e}")
            return features
    
    def _calculate_advanced_greeks_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced Greeks and cross-derivatives."""
        try:
            # These would be calculated using more sophisticated models in production
            # For now, providing reasonable estimates
            
            # Charm (delta-theta sensitivity)
            delta_exposure = features.get('net_delta_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            theta_exposure = features.get('total_theta_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            
            # Estimate charm as interaction between delta and theta
            charm_estimate = delta_exposure * theta_exposure * 0.001  # Scaling factor
            features['charm_exposure'] = charm_estimate
            
            # Vomma (vega-volatility sensitivity)
            vega_exposure = features.get('total_vega_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            vomma_estimate = vega_exposure * 0.1  # Rough estimate
            features['vomma_exposure'] = vomma_estimate
            
            # Vera (vega-rho sensitivity) - interest rate sensitivity of vega
            vera_estimate = vega_exposure * 0.05  # Rough estimate
            features['vera_exposure'] = vera_estimate
            
            # Speed (gamma-spot sensitivity)
            gamma_exposure = features.get('total_gamma_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            speed_estimate = gamma_exposure * 0.01  # Rough estimate
            features['speed_exposure'] = speed_estimate
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced Greeks metrics: {e}")
            return features
    
    def _calculate_greeks_ratios(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Greeks ratios and portfolio metrics."""
        try:
            # Extract Greeks values
            delta = features.get('net_delta_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            gamma = features.get('total_gamma_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            vega = features.get('total_vega_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            theta = features.get('total_theta_exposure', pd.Series([0])).iloc[-1] if len(features) > 0 else 0
            
            # Delta-Gamma ratio
            delta_gamma_ratio = safe_divide(abs(delta), abs(gamma), default_value=0.0)
            features['delta_gamma_ratio'] = delta_gamma_ratio
            
            # Vega-Theta ratio
            vega_theta_ratio = safe_divide(abs(vega), abs(theta), default_value=0.0)
            features['vega_theta_ratio'] = vega_theta_ratio
            
            # Greeks diversification score
            # Higher score = more balanced Greeks exposure
            greeks_values = [abs(delta), abs(gamma), abs(vega), abs(theta)]
            if sum(greeks_values) > 0:
                # Calculate Herfindahl index (concentration)
                normalized_greeks = [safe_divide(g, sum(greeks_values), default_value=0.0) for g in greeks_values]
                concentration = sum(g**2 for g in normalized_greeks)
                diversification_score = 1 - concentration  # Higher = more diversified
                features['greeks_diversification_score'] = diversification_score
            else:
                features['greeks_diversification_score'] = 0.0
            
            # Options leverage ratio
            current_price = data['close'].iloc[-1] if len(data) > 0 else 100
            notional_exposure = abs(delta) * current_price
            
            # Estimate options premium invested
            if self.options_chain is not None and not self.options_chain.empty and 'lastPrice' in self.options_chain.columns:
                total_premium = (
                    self.options_chain['lastPrice'] * 
                    self.options_chain['openInterest'] * 
                    100
                ).sum()
                
                leverage_ratio = safe_divide(notional_exposure, total_premium, default_value=1.0)
                features['options_leverage_ratio'] = leverage_ratio
            else:
                # Estimate typical leverage
                features['options_leverage_ratio'] = 5.0  # Typical options leverage
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating Greeks ratios: {e}")
            return features