"""
Black-Scholes Calculator

Implements Black-Scholes option pricing model and Greeks calculations
for comprehensive options analytics and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from scipy import stats

from .base_options import BaseOptionsCalculator
from ..helpers import (
    create_feature_dataframe, safe_divide, safe_log, safe_sqrt, calculate_rolling_mean,
    calculate_rolling_std, normalize_series, create_rolling_features
)

from main.utils.core import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class BlackScholesCalculator(BaseOptionsCalculator):
    """
    Calculates Black-Scholes theoretical values and Greeks.
    
    Features include:
    - Theoretical option prices
    - Delta, Gamma, Theta, Vega, Rho
    - Higher-order Greeks (Vanna, Charm, Vomma)
    - Implied volatility calculations
    - Price sensitivities
    - Risk metrics
    - Greeks surfaces and term structures
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Black-Scholes calculator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Greeks calculation parameters
        self.calculate_higher_order = config.get('calculate_higher_order_greeks', True)
        self.iv_iterations = config.get('iv_iterations', 100)
        self.iv_tolerance = config.get('iv_tolerance', 1e-6)
        
        # Sensitivity analysis parameters
        self.spot_bumps = config.get('spot_bumps', [-0.1, -0.05, 0.05, 0.1])
        self.vol_bumps = config.get('vol_bumps', [-0.2, -0.1, 0.1, 0.2])
        self.time_horizons = config.get('time_horizons', [1, 7, 30])
        
        logger.info("Initialized BlackScholesCalculator")
    
    def get_feature_names(self) -> List[str]:
        """Get list of Black-Scholes feature names."""
        features = []
        
        # Theoretical prices and mispricing
        features.extend([
            'bs_call_price',
            'bs_put_price',
            'call_mispricing',
            'put_mispricing',
            'call_mispricing_pct',
            'put_mispricing_pct',
            'avg_mispricing'
        ])
        
        # First-order Greeks
        features.extend([
            'call_delta',
            'put_delta',
            'gamma',
            'call_theta',
            'put_theta',
            'vega',
            'call_rho',
            'put_rho'
        ])
        
        # Greeks ratios and normalized metrics
        features.extend([
            'delta_hedge_ratio',
            'gamma_scalp_potential',
            'theta_decay_rate',
            'vega_exposure',
            'dollar_gamma',
            'dollar_vega',
            'dollar_theta'
        ])
        
        # Higher-order Greeks
        if self.calculate_higher_order:
            features.extend([
                'vanna',  # dDelta/dVol
                'charm',  # dDelta/dTime
                'vomma',  # dVega/dVol
                'speed',  # dGamma/dSpot
                'zomma',  # dGamma/dVol
                'color'   # dGamma/dTime
            ])
        
        # Implied volatility metrics
        features.extend([
            'implied_volatility',
            'iv_premium',
            'iv_rank',
            'iv_percentile',
            'realized_iv_spread'
        ])
        
        # Sensitivity analysis
        features.extend([
            'spot_sensitivity',
            'vol_sensitivity',
            'time_sensitivity',
            'convexity_measure'
        ])
        
        return features
    
    def calculate(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Black-Scholes features from options data.
        
        Args:
            options_data: DataFrame with options data
            
        Returns:
            DataFrame with Black-Scholes features
        """
        try:
            # Validate options data
            validated_data = self.validate_options_data(options_data)
            if validated_data.empty:
                return self._create_empty_features(options_data.index)
            
            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)
            
            # Calculate theoretical prices
            price_features = self._calculate_theoretical_prices(validated_data)
            features = pd.concat([features, price_features], axis=1)
            
            # Calculate Greeks
            greeks_features = self._calculate_greeks(validated_data)
            features = pd.concat([features, greeks_features], axis=1)
            
            # Calculate higher-order Greeks if enabled
            if self.calculate_higher_order:
                higher_greeks = self._calculate_higher_order_greeks(validated_data)
                features = pd.concat([features, higher_greeks], axis=1)
            
            # Calculate implied volatility metrics
            iv_features = self._calculate_iv_metrics(validated_data, price_features)
            features = pd.concat([features, iv_features], axis=1)
            
            # Calculate sensitivity analysis
            sensitivity_features = self._calculate_sensitivities(
                validated_data, greeks_features
            )
            features = pd.concat([features, sensitivity_features], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating Black-Scholes features: {e}")
            return self._create_empty_features(options_data.index)
    
    def _calculate_theoretical_prices(
        self,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Black-Scholes theoretical prices."""
        features = pd.DataFrame(index=options_data.index)
        
        # Extract required data
        spot = options_data['underlying_price']
        strike = options_data['strike']
        time_to_expiry = safe_divide(options_data['days_to_expiry'], 365.0, default_value=0.0)
        volatility = options_data.get('implied_volatility', 0.2)
        risk_free_rate = self.config.risk_free_rate
        
        # Calculate BS prices
        bs_call, bs_put = self.black_scholes_price(
            spot, strike, time_to_expiry, volatility, risk_free_rate
        )
        
        features['bs_call_price'] = bs_call
        features['bs_put_price'] = bs_put
        
        # Calculate mispricing
        if 'call_price' in options_data.columns:
            features['call_mispricing'] = options_data['call_price'] - bs_call
            features['call_mispricing_pct'] = safe_divide(
                features['call_mispricing'], bs_call
            ) * 100
        
        if 'put_price' in options_data.columns:
            features['put_mispricing'] = options_data['put_price'] - bs_put
            features['put_mispricing_pct'] = safe_divide(
                features['put_mispricing'], bs_put
            ) * 100
        
        # Average mispricing
        if 'call_mispricing_pct' in features and 'put_mispricing_pct' in features:
            features['avg_mispricing'] = safe_divide(
                (features['call_mispricing_pct'].abs() + 
                features['put_mispricing_pct'].abs()), 2, default_value=0.0
            )
        
        return features
    
    def _calculate_greeks(
        self,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate first-order Greeks."""
        features = pd.DataFrame(index=options_data.index)
        
        # Extract required data
        spot = options_data['underlying_price']
        strike = options_data['strike']
        time_to_expiry = safe_divide(options_data['days_to_expiry'], 365.0, default_value=0.0)
        volatility = options_data.get('implied_volatility', 0.2)
        risk_free_rate = self.config.risk_free_rate
        
        # Calculate d1 and d2 for Greeks
        d1 = self._calculate_d1(spot, strike, time_to_expiry, volatility, risk_free_rate)
        d2 = d1 - volatility * safe_sqrt(time_to_expiry)
        
        # Delta
        features['call_delta'] = stats.norm.cdf(d1)
        features['put_delta'] = features['call_delta'] - 1
        
        # Gamma (same for calls and puts)
        features['gamma'] = self._calculate_gamma(
            spot, strike, time_to_expiry, volatility, d1
        )
        
        # Theta
        call_theta, put_theta = self._calculate_theta(
            spot, strike, time_to_expiry, volatility, risk_free_rate, d1, d2
        )
        features['call_theta'] = safe_divide(call_theta, 365, default_value=0.0)  # Convert to daily
        features['put_theta'] = safe_divide(put_theta, 365, default_value=0.0)
        
        # Vega (same for calls and puts)
        features['vega'] = safe_divide(
            self._calculate_vega(spot, time_to_expiry, d1), 100, default_value=0.0
        )  # Per 1% change in volatility
        
        # Rho
        features['call_rho'] = safe_divide(
            self._calculate_call_rho(strike, time_to_expiry, risk_free_rate, d2), 100, default_value=0.0
        )  # Per 1% change in rate
        features['put_rho'] = safe_divide(
            self._calculate_put_rho(strike, time_to_expiry, risk_free_rate, d2), 100, default_value=0.0
        )
        
        # Greeks ratios and dollar Greeks
        features['delta_hedge_ratio'] = -features['call_delta']
        features['gamma_scalp_potential'] = features['gamma'] * volatility * spot
        features['theta_decay_rate'] = safe_divide(features['call_theta'], spot, default_value=0.0)
        features['vega_exposure'] = features['vega'] * volatility
        
        # Dollar Greeks
        features['dollar_gamma'] = safe_divide(features['gamma'] * spot * spot, 100, default_value=0.0)
        features['dollar_vega'] = features['vega'] * spot
        features['dollar_theta'] = features['call_theta'] * spot
        
        return features
    
    def _calculate_higher_order_greeks(
        self,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate higher-order Greeks."""
        features = pd.DataFrame(index=options_data.index)
        
        # Extract required data
        spot = options_data['underlying_price']
        strike = options_data['strike']
        time_to_expiry = safe_divide(options_data['days_to_expiry'], 365.0, default_value=0.0)
        volatility = options_data.get('implied_volatility', 0.2)
        risk_free_rate = self.config.risk_free_rate
        
        # Calculate d1 for higher-order Greeks
        d1 = self._calculate_d1(spot, strike, time_to_expiry, volatility, risk_free_rate)
        
        # Vanna (dDelta/dVol)
        features['vanna'] = self._calculate_vanna(
            spot, time_to_expiry, volatility, d1
        )
        
        # Charm (dDelta/dTime)
        features['charm'] = self._calculate_charm(
            spot, strike, time_to_expiry, volatility, risk_free_rate, d1
        )
        
        # Vomma (dVega/dVol)
        features['vomma'] = self._calculate_vomma(
            spot, time_to_expiry, volatility, d1
        )
        
        # Speed (dGamma/dSpot)
        features['speed'] = self._calculate_speed(
            spot, strike, time_to_expiry, volatility, d1
        )
        
        # Zomma (dGamma/dVol)
        features['zomma'] = self._calculate_zomma(
            spot, time_to_expiry, volatility, d1
        )
        
        # Color (dGamma/dTime)
        features['color'] = self._calculate_color(
            spot, strike, time_to_expiry, volatility, risk_free_rate, d1
        )
        
        return features
    
    def _calculate_iv_metrics(
        self,
        options_data: pd.DataFrame,
        price_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate implied volatility metrics."""
        features = pd.DataFrame(index=options_data.index)
        
        # Calculate IV for market prices
        if 'call_price' in options_data.columns:
            features['implied_volatility'] = self._calculate_implied_volatility(
                options_data, 'call'
            )
        elif 'put_price' in options_data.columns:
            features['implied_volatility'] = self._calculate_implied_volatility(
                options_data, 'put'
            )
        else:
            features['implied_volatility'] = options_data.get('implied_volatility', 0.2)
        
        # IV premium over historical volatility
        if 'historical_volatility' in options_data.columns:
            features['iv_premium'] = (
                features['implied_volatility'] - options_data['historical_volatility']
            )
        
        # IV rank (percentile over past year)
        features['iv_rank'] = self._calculate_iv_rank(features['implied_volatility'])
        
        # IV percentile
        features['iv_percentile'] = self._calculate_iv_percentile(
            features['implied_volatility']
        )
        
        # Realized-Implied spread
        if 'realized_volatility' in options_data.columns:
            features['realized_iv_spread'] = (
                options_data['realized_volatility'] - features['implied_volatility']
            )
        
        return features
    
    def _calculate_sensitivities(
        self,
        options_data: pd.DataFrame,
        greeks_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate price sensitivities."""
        features = pd.DataFrame(index=options_data.index)
        
        # Spot sensitivity (using delta and gamma)
        delta = greeks_features.get('call_delta', 0)
        gamma = greeks_features.get('gamma', 0)
        
        spot_sens = 0
        for bump in self.spot_bumps:
            price_change = delta * bump + 0.5 * gamma * bump**2
            spot_sens += abs(price_change)
        features['spot_sensitivity'] = safe_divide(spot_sens, len(self.spot_bumps), default_value=0.0)
        
        # Volatility sensitivity (using vega and vomma if available)
        vega = greeks_features.get('vega', 0)
        vomma = greeks_features.get('vomma', 0) if self.calculate_higher_order else 0
        
        vol_sens = 0
        for bump in self.vol_bumps:
            price_change = vega * bump + 0.5 * vomma * bump**2
            vol_sens += abs(price_change)
        features['vol_sensitivity'] = safe_divide(vol_sens, len(self.vol_bumps), default_value=0.0)
        
        # Time sensitivity (using theta and charm if available)
        theta = greeks_features.get('call_theta', 0)
        charm = greeks_features.get('charm', 0) if self.calculate_higher_order else 0
        
        time_sens = 0
        for days in self.time_horizons:
            price_change = theta * days + 0.5 * charm * days**2
            time_sens += abs(price_change)
        features['time_sensitivity'] = safe_divide(time_sens, len(self.time_horizons), default_value=0.0)
        
        # Convexity measure (gamma-based)
        features['convexity_measure'] = gamma * options_data['underlying_price']**2
        
        return features
    
    def _calculate_d1(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        rate: float
    ) -> pd.Series:
        """Calculate d1 for Black-Scholes formula."""
        return safe_divide(
            (safe_log(safe_divide(spot, strike, default_value=1.0)) + (rate + 0.5 * volatility**2) * time),
            (volatility * safe_sqrt(time)), default_value=0.0
        )
    
    def _calculate_gamma(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate gamma."""
        return safe_divide(stats.norm.pdf(d1), (spot * volatility * safe_sqrt(time)), default_value=0.0)
    
    def _calculate_theta(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        rate: float,
        d1: pd.Series,
        d2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate theta for calls and puts."""
        # Call theta
        call_theta = -safe_divide(
            (spot * stats.norm.pdf(d1) * volatility), (2 * safe_sqrt(time)), default_value=0.0
        ) - rate * strike * np.exp(-rate * time) * stats.norm.cdf(d2)
        
        # Put theta
        put_theta = -safe_divide(
            (spot * stats.norm.pdf(d1) * volatility), (2 * safe_sqrt(time)), default_value=0.0
        ) + rate * strike * np.exp(-rate * time) * stats.norm.cdf(-d2)
        
        return call_theta, put_theta
    
    def _calculate_vega(
        self,
        spot: pd.Series,
        time: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate vega."""
        return spot * stats.norm.pdf(d1) * safe_sqrt(time)
    
    def _calculate_call_rho(
        self,
        strike: pd.Series,
        time: pd.Series,
        rate: float,
        d2: pd.Series
    ) -> pd.Series:
        """Calculate rho for calls."""
        return strike * time * np.exp(-rate * time) * stats.norm.cdf(d2)
    
    def _calculate_put_rho(
        self,
        strike: pd.Series,
        time: pd.Series,
        rate: float,
        d2: pd.Series
    ) -> pd.Series:
        """Calculate rho for puts."""
        return -strike * time * np.exp(-rate * time) * stats.norm.cdf(-d2)
    
    def _calculate_vanna(
        self,
        spot: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate vanna (dDelta/dVol)."""
        d2 = d1 - volatility * safe_sqrt(time)
        return safe_divide(-stats.norm.pdf(d1) * d2, volatility, default_value=0.0)
    
    def _calculate_charm(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        rate: float,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate charm (dDelta/dTime)."""
        d2 = d1 - volatility * safe_sqrt(time)
        
        charm = safe_divide(
            -stats.norm.pdf(d1) * (2 * rate * time - d2 * volatility * safe_sqrt(time)),
            (2 * time * volatility * safe_sqrt(time)), default_value=0.0
        )
        
        return charm
    
    def _calculate_vomma(
        self,
        spot: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate vomma (dVega/dVol)."""
        d2 = d1 - volatility * safe_sqrt(time)
        vega = spot * stats.norm.pdf(d1) * safe_sqrt(time)
        return safe_divide(vega * d1 * d2, volatility, default_value=0.0)
    
    def _calculate_speed(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate speed (dGamma/dSpot)."""
        gamma = self._calculate_gamma(spot, strike, time, volatility, d1)
        return safe_divide(
            -gamma * (d1 + volatility * safe_sqrt(time)),
            (spot * volatility * safe_sqrt(time)), default_value=0.0
        )
    
    def _calculate_zomma(
        self,
        spot: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate zomma (dGamma/dVol)."""
        d2 = d1 - volatility * safe_sqrt(time)
        gamma = safe_divide(stats.norm.pdf(d1), (spot * volatility * safe_sqrt(time)), default_value=0.0)
        return safe_divide(gamma * (d1 * d2 - 1), volatility, default_value=0.0)
    
    def _calculate_color(
        self,
        spot: pd.Series,
        strike: pd.Series,
        time: pd.Series,
        volatility: pd.Series,
        rate: float,
        d1: pd.Series
    ) -> pd.Series:
        """Calculate color (dGamma/dTime)."""
        d2 = d1 - volatility * safe_sqrt(time)
        
        color = safe_divide(
            -stats.norm.pdf(d1) * (2 * rate * time - d2 * volatility * safe_sqrt(time) - d1 * (1 + d1 * d2)),
            (2 * spot * time * volatility * safe_sqrt(time)), default_value=0.0
        )
        
        return color
    
    def _calculate_implied_volatility(
        self,
        options_data: pd.DataFrame,
        option_type: str
    ) -> pd.Series:
        """Calculate implied volatility using Newton-Raphson method."""
        iv_series = pd.Series(index=options_data.index, dtype=float)
        
        for idx, row in options_data.iterrows():
            spot = row['underlying_price']
            strike = row['strike']
            time = safe_divide(row['days_to_expiry'], 365.0, default_value=0.0)
            market_price = row[f'{option_type}_price']
            
            # Initial guess
            iv = 0.2
            
            # Newton-Raphson iterations
            for _ in range(self.iv_iterations):
                if option_type == 'call':
                    bs_price, _ = self.black_scholes_price(
                        spot, strike, time, iv, self.config.risk_free_rate
                    )
                else:
                    _, bs_price = self.black_scholes_price(
                        spot, strike, time, iv, self.config.risk_free_rate
                    )
                
                # Calculate vega for Newton-Raphson
                d1 = self._calculate_d1(spot, strike, time, iv, self.config.risk_free_rate)
                vega = spot * stats.norm.pdf(d1) * safe_sqrt(time)
                
                # Update IV
                price_diff = market_price - bs_price
                if abs(price_diff) < self.iv_tolerance:
                    break
                
                iv += safe_divide(price_diff, (vega + 1e-10), default_value=0.0)  # Add small value to avoid division by zero
                iv = max(0.01, min(iv, 5.0))  # Bound IV between 1% and 500%
            
            iv_series[idx] = iv
        
        return iv_series
    
    def _calculate_iv_rank(self, iv: pd.Series) -> pd.Series:
        """Calculate IV rank over past year."""
        # Simplified: use rolling window
        window = 252  # Trading days in a year
        
        # Create DataFrame for rolling features
        iv_df = pd.DataFrame({'iv': iv})
        
        # Use create_rolling_features for min/max calculation
        rolling_features = create_rolling_features(
            iv_df,
            columns=['iv'],
            windows=[window],
            operations=['min', 'max'],
            min_periods=20
        )
        
        iv_min = rolling_features[f'iv_rolling_min_{window}']
        iv_max = rolling_features[f'iv_rolling_max_{window}']
        
        iv_rank = safe_divide(iv - iv_min, iv_max - iv_min, default_value=0.0) * 100
        
        return iv_rank.fillna(50)  # Default to middle rank
    
    def _calculate_iv_percentile(self, iv: pd.Series) -> pd.Series:
        """Calculate IV percentile over past year."""
        window = 252
        
        def percentile_rank(x):
            if len(x) < 20:
                return 50
            return stats.percentileofscore(x, x.iloc[-1])
        
        iv_percentile = iv.rolling(window=window, min_periods=20).apply(
            percentile_rank, raw=False
        )
        
        return iv_percentile.fillna(50)