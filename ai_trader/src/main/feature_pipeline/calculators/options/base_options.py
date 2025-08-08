"""
Base Options Calculator

Shared utilities and base functionality for all options analysis calculators.
Provides common data preprocessing, validation, options data handling,
and mathematical utilities used across specialized options calculators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats

from ..base_calculator import BaseFeatureCalculator
from ..helpers import (
    safe_divide, safe_log, safe_sqrt, create_feature_dataframe, 
    validate_price_data, validate_options_data, postprocess_features
)
from .options_config import OptionsConfig

from main.utils.core import get_logger, ErrorHandlingMixin, ensure_utc, is_market_open

logger = get_logger(__name__)


class BaseOptionsCalculator(BaseFeatureCalculator, ErrorHandlingMixin, ABC):
    """
    Abstract base class for options analysis calculators.
    
    Provides shared functionality including:
    - Options data preprocessing and validation
    - Common options computation utilities
    - Black-Scholes mathematical utilities
    - Error handling and numerical stability
    - Configuration management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base options calculator."""
        super().__init__(config)
        
        # Initialize options-specific configuration
        options_config_dict = self.config.get('options', {})
        self.options_config = OptionsConfig(**options_config_dict)
        
        # Validate configuration
        validation = self.options_config.validate_configuration()
        if not validation['valid']:
            logger.error(f"Invalid options configuration: {validation['errors']}")
            
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"Options config warning: {warning}")
        
        # Numerical stability parameters
        self.numerical_tolerance = 1e-8
        self.max_iterations = 100
        
        # Options data storage
        self.options_chain = None
        self.historical_iv = None
        
        # Cache available via BaseFeatureCalculator
        
        logger.debug(f"Initialized {self.__class__.__name__} with {len(self.options_config.expiry_windows)} expiry windows")
    
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for options calculations."""
        return ['close']  # Minimum requirement, individual calculators may need more
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input market data for options calculations.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            True if data is valid for options analysis
        """
        if data.empty:
            logger.warning("Empty data provided for options analysis")
            return False
        
        # Check required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns for options analysis: {missing_cols}")
            return False
        
        # Validate price data
        if 'close' in data.columns:
            if (data['close'] <= 0).any():
                logger.warning("Found non-positive prices in options data")
                if self.options_config.strict_validation:
                    return False
        
        # Check minimum data length
        min_required = 30  # Minimum periods for meaningful options analysis
        if len(data) < min_required:
            logger.warning(f"Insufficient data for options analysis: {len(data)} < {min_required}")
            if self.options_config.strict_validation:
                return False
        
        return True
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Legacy method name for backward compatibility."""
        return self.validate_input_data(data)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for options calculations.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Cleaned and validated DataFrame
        """
        if data.empty:
            return data
        
        # Start with base preprocessing
        processed_data = super().preprocess_data(data)
        
        # Options-specific preprocessing
        if 'close' in processed_data.columns:
            # Ensure positive prices
            processed_data['close'] = processed_data['close'].clip(lower=0.01)
            
            # Calculate returns for volatility estimation
            processed_data['returns'] = processed_data['close'].pct_change()
        
        # Handle OHLCV data if present
        ohlc_cols = ['open', 'high', 'low']
        for col in ohlc_cols:
            if col in processed_data.columns:
                processed_data[col] = processed_data[col].clip(lower=0.01)
        
        if 'volume' in processed_data.columns:
            processed_data['volume'] = processed_data['volume'].fillna(0).clip(lower=0)
        
        return processed_data
    
    def validate_options_chain(self, options_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean options chain data.
        
        Args:
            options_chain: Raw options chain data
            
        Returns:
            Cleaned options chain DataFrame
        """
        if options_chain.empty:
            return options_chain
        
        # Use centralized validation helper
        is_valid, errors = validate_options_data(options_chain, min_rows=self.options_config.min_options_required)
        if not is_valid:
            logger.warning(f"Options chain validation failed: {errors}")
            if self.options_config.strict_validation:
                return pd.DataFrame()
        
        cleaned_chain = options_chain.copy()
        
        # Required columns for options analysis
        required_cols = ['volume', 'openInterest', 'strike']
        available_cols = [col for col in required_cols if col in cleaned_chain.columns]
        
        if len(available_cols) < 2:
            logger.warning("Options chain missing critical columns")
            return pd.DataFrame()
        
        # Clean numerical columns
        numerical_cols = ['volume', 'openInterest', 'lastPrice', 'bid', 'ask', 'impliedVolatility']
        for col in numerical_cols:
            if col in cleaned_chain.columns:
                cleaned_chain[col] = pd.to_numeric(cleaned_chain[col], errors='coerce')
                cleaned_chain[col] = cleaned_chain[col].fillna(0).clip(lower=0)
        
        # Filter by minimum volume and open interest
        if all(col in cleaned_chain.columns for col in ['volume', 'openInterest']):
            valid_mask = (
                (cleaned_chain['volume'] >= self.options_config.min_volume) |
                (cleaned_chain['openInterest'] >= self.options_config.min_open_interest)
            )
            cleaned_chain = cleaned_chain[valid_mask]
        
        # Filter by price range if available
        if 'lastPrice' in cleaned_chain.columns:
            price_mask = (
                (cleaned_chain['lastPrice'] >= self.options_config.min_option_price) &
                (cleaned_chain['lastPrice'] <= self.options_config.max_option_price)
            )
            cleaned_chain = cleaned_chain[price_mask]
        
        # Validate option types
        if 'optionType' in cleaned_chain.columns:
            valid_types = cleaned_chain['optionType'].str.lower().isin(['call', 'put'])
            cleaned_chain = cleaned_chain[valid_types]
        
        # Filter by days to expiration if available
        if 'daysToExpiration' in cleaned_chain.columns:
            expiry_mask = (
                (cleaned_chain['daysToExpiration'] >= self.options_config.min_days_to_expiry) &
                (cleaned_chain['daysToExpiration'] <= self.options_config.max_days_to_expiry)
            )
            cleaned_chain = cleaned_chain[expiry_mask]
        
        return cleaned_chain
    
    def calculate_moneyness(self, strike: float, spot: float, option_type: str) -> str:
        """
        Calculate moneyness category for an option.
        
        Args:
            strike: Strike price
            spot: Current spot price
            option_type: 'call' or 'put'
            
        Returns:
            Moneyness category: 'deep_itm', 'itm', 'atm', 'otm', 'deep_otm'
        """
        if spot <= 0:
            return 'unknown'
        
        moneyness_ratio = safe_divide((strike - spot), spot)
        
        if option_type.lower() == 'call':
            if moneyness_ratio < -0.15:
                return 'deep_itm'
            elif moneyness_ratio < -self.options_config.itm_threshold:
                return 'itm'
            elif abs(moneyness_ratio) <= self.options_config.atm_threshold:
                return 'atm'
            elif moneyness_ratio < 0.20:
                return 'otm'
            else:
                return 'deep_otm'
        else:  # put
            if moneyness_ratio > 0.15:
                return 'deep_itm'
            elif moneyness_ratio > self.options_config.itm_threshold:
                return 'itm'
            elif abs(moneyness_ratio) <= self.options_config.atm_threshold:
                return 'atm'
            elif moneyness_ratio > -0.20:
                return 'otm'
            else:
                return 'deep_otm'
    
    def calculate_time_to_expiry(self, expiry_date: Union[str, datetime], 
                                current_date: Optional[datetime] = None) -> float:
        """
        Calculate time to expiry in years.
        
        Args:
            expiry_date: Expiration date
            current_date: Current date (defaults to now)
            
        Returns:
            Time to expiry in years
        """
        if current_date is None:
            current_date = ensure_utc(datetime.now())
        else:
            current_date = ensure_utc(current_date)
        
        if isinstance(expiry_date, str):
            expiry_date = ensure_utc(pd.to_datetime(expiry_date))
        else:
            expiry_date = ensure_utc(expiry_date)
        
        time_diff = (expiry_date - current_date).total_seconds()
        return max(time_diff / (365.25 * 24 * 3600), 1/365.25)  # Minimum 1 day
    
    # Removed duplicate safe_divide - using centralized helper instead
    
    def calculate_historical_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate historical volatility from price series.
        
        Args:
            prices: Price series
            window: Rolling window for volatility calculation
            
        Returns:
            Annualized historical volatility series
        """
        returns = prices.pct_change().dropna()
        return returns.rolling(window=window).std() * safe_sqrt(252)
    
    def estimate_implied_volatility(self, price_data: pd.Series, window: int = 20) -> pd.Series:
        """
        Estimate implied volatility from historical price data.
        
        Args:
            price_data: Historical price series
            window: Window for volatility calculation
            
        Returns:
            Estimated implied volatility series
        """
        hist_vol = self.calculate_historical_volatility(price_data, window)
        # IV typically trades at a premium to historical volatility
        estimated_iv = hist_vol * 1.2  # 20% premium assumption
        return estimated_iv.clip(
            self.options_config.min_volatility,
            self.options_config.max_volatility
        )
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float,
                            sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Theoretical option price
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = safe_divide((safe_log(safe_divide(S, K)) + (r + 0.5 * sigma**2) * T), (sigma * safe_sqrt(T)))
            d2 = d1 - sigma * safe_sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
            return max(price, 0.0)  # Option price cannot be negative
            
        except Exception as e:
            logger.warning(f"Black-Scholes calculation error: {e}")
            return 0.0
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = 'call') -> float:
        """Calculate option delta."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = safe_divide((safe_log(safe_divide(S, K)) + (r + 0.5 * sigma**2) * T), (sigma * safe_sqrt(T)))
            
            if option_type.lower() == 'call':
                return stats.norm.cdf(d1)
            else:
                return -stats.norm.cdf(-d1)
                
        except Exception:
            return 0.0
    
    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = safe_divide((safe_log(safe_divide(S, K)) + (r + 0.5 * sigma**2) * T), (sigma * safe_sqrt(T)))
            return safe_divide(stats.norm.pdf(d1), (S * sigma * safe_sqrt(T)))
        except Exception:
            return 0.0
    
    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = safe_divide((safe_log(safe_divide(S, K)) + (r + 0.5 * sigma**2) * T), (sigma * safe_sqrt(T)))
            return safe_divide(S * stats.norm.pdf(d1) * safe_sqrt(T), 100)  # Per 1% vol change
        except Exception:
            return 0.0
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str = 'call') -> float:
        """Calculate option theta."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        try:
            d1 = safe_divide((safe_log(safe_divide(S, K)) + (r + 0.5 * sigma**2) * T), (sigma * safe_sqrt(T)))
            d2 = d1 - sigma * safe_sqrt(T)
            
            if option_type.lower() == 'call':
                theta = safe_divide((
                    -S * stats.norm.pdf(d1) * sigma / (2 * safe_sqrt(T))
                    - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                ), 365)  # Per day
            else:
                theta = safe_divide((
                    -S * stats.norm.pdf(d1) * sigma / (2 * safe_sqrt(T))
                    + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                ), 365)  # Per day
            
            return theta
            
        except Exception:
            return 0.0
    
    def create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """
        Create empty features DataFrame with proper index.
        
        Args:
            index: Index for the features DataFrame
            
        Returns:
            Empty DataFrame with correct index
        """
        return pd.DataFrame(index=index)
    
    def _align_features_with_data(self, features: pd.DataFrame, 
                                 original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align features DataFrame with original data structure.
        
        Args:
            features: Calculated features DataFrame
            original_data: Original input data
            
        Returns:
            Aligned features DataFrame
        """
        if len(features) == len(original_data):
            features.index = original_data.index
            return features
        
        # Handle length mismatch by reindexing
        return features.reindex(original_data.index, fill_value=0.0)
    
    def set_options_data(self, options_chain: pd.DataFrame):
        """Set options chain data for analysis."""
        self.options_chain = self.validate_options_chain(options_chain)
        logger.debug(f"Set options chain with {len(self.options_chain)} valid options")
    
    def set_historical_iv(self, historical_iv: pd.DataFrame):
        """Set historical implied volatility data."""
        self.historical_iv = historical_iv
        logger.debug(f"Set historical IV data with {len(historical_iv)} periods")
    
    def clear_cache(self):
        """Clear calculation cache."""
        # Cache clearing handled by BaseFeatureCalculator
        logger.debug("Cleared options calculation cache")