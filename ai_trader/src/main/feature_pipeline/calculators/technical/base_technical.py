"""
Base Technical Calculator

Provides shared utilities and base functionality for all technical indicator calculators.
Contains common helper methods, validation logic, and preprocessing steps.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import talib
from scipy import stats

from ..base_calculator import BaseFeatureCalculator


class BaseTechnicalCalculator(BaseFeatureCalculator):
    """Base class for technical indicator calculators with shared utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base technical calculator with configuration."""
        super().__init__(config)
        
        # Common technical indicator parameters
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20, 50, 200])
        self.adaptive_enabled = self.config.get('adaptive_enabled', True)
        
        # Bollinger Band parameters
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        # RSI parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        
        # ATR parameters
        self.atr_period = self.config.get('atr_period', 14)
        
        # Volume parameters
        self.volume_sma_period = self.config.get('volume_sma_period', 20)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for technical indicator calculations."""
        return ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data for technical indicator calculations."""
        try:
            # Check if DataFrame is empty
            if data.empty:
                return False
            
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for sufficient data
            min_required = max(self.lookback_periods) + 50 if self.lookback_periods else 100
            if len(data) < min_required:
                print(f"Insufficient data: {len(data)} < {min_required}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating inputs: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for technical indicator calculations."""
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Clean the data
            processed_data = data.copy()
            
            # Handle missing values by forward filling
            processed_data[required_cols] = processed_data[required_cols].fillna(method='ffill')
            
            # Remove any remaining NaN rows
            processed_data = processed_data.dropna(subset=required_cols)
            
            # Validate OHLC relationships
            invalid_ohlc = (
                (processed_data['high'] < processed_data['low']) |
                (processed_data['high'] < processed_data['open']) |
                (processed_data['high'] < processed_data['close']) |
                (processed_data['low'] > processed_data['open']) |
                (processed_data['low'] > processed_data['close'])
            )
            
            if invalid_ohlc.any():
                print(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
                # Fix invalid OHLC by using close price as fallback
                mask = invalid_ohlc
                processed_data.loc[mask, 'high'] = processed_data.loc[mask, ['high', 'open', 'close']].max(axis=1)
                processed_data.loc[mask, 'low'] = processed_data.loc[mask, ['low', 'open', 'close']].min(axis=1)
            
            # Ensure positive prices
            for col in required_cols[:-1]:  # Exclude volume
                if (processed_data[col] <= 0).any():
                    print(f"Found non-positive prices in {col}, filtering out")
                    processed_data = processed_data[processed_data[col] > 0]
            
            # Handle volume (can be zero but not negative)
            processed_data['volume'] = processed_data['volume'].clip(lower=0)
            
            # Ensure symbol and timestamp columns exist
            if 'symbol' not in processed_data.columns:
                processed_data['symbol'] = 'UNKNOWN'
                
            if 'timestamp' not in processed_data.columns:
                if hasattr(processed_data.index, 'to_pydatetime'):
                    processed_data['timestamp'] = processed_data.index
                else:
                    processed_data['timestamp'] = pd.date_range(
                        start='2023-01-01', periods=len(processed_data), freq='D'
                    )
            
            return processed_data
            
        except Exception as e:
            print(f"Error preprocessing data for technical analysis: {e}")
            return pd.DataFrame()
    
    def postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Postprocess technical indicator features."""
        try:
            if features.empty:
                return features
            
            processed_features = features.copy()
            
            # Handle infinite values
            processed_features = processed_features.replace([np.inf, -np.inf], np.nan)
            
            # Handle specific feature types
            
            # Moving averages should be positive
            ma_cols = [col for col in processed_features.columns if any(ma in col for ma in ['sma_', 'ema_', 'wma_'])]
            for col in ma_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].clip(lower=0.01)
            
            # Price ratios should be positive and reasonable
            ratio_cols = [col for col in processed_features.columns if 'price_to_' in col or '_ratio' in col]
            for col in ratio_cols:
                if col in processed_features.columns:
                    if 'price_to_' in col:
                        processed_features[col] = processed_features[col].clip(0.1, 10)  # Price ratios
                    else:
                        processed_features[col] = processed_features[col].clip(0, 100)  # Other ratios
            
            # RSI should be in [0, 100]
            rsi_cols = [col for col in processed_features.columns if col.startswith('rsi_')]
            for col in rsi_cols:
                if col in processed_features.columns and not col.endswith(('_oversold', '_overbought')):
                    processed_features[col] = processed_features[col].clip(0, 100)
            
            # Boolean indicators should be 0 or 1
            bool_cols = [col for col in processed_features.columns if any(suffix in col for suffix in ['_oversold', '_overbought', '_cross', '_above_', '_below_'])]
            for col in bool_cols:
                if col in processed_features.columns:
                    processed_features[col] = processed_features[col].fillna(0).astype(int).clip(0, 1)
            
            # Forward fill then backward fill remaining NaN values
            processed_features = processed_features.fillna(method='ffill').fillna(method='bfill')
            
            return processed_features
            
        except Exception as e:
            print(f"Error postprocessing technical indicator features: {e}")
            return features
    
    # Shared utility methods for technical indicators
    
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average."""
        def weighted_average(vals):
            weights = np.arange(1, len(vals) + 1)
            return np.average(vals, weights=weights)
        
        return series.rolling(window=period, min_periods=1).apply(weighted_average, raw=True)
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(series, period)
        std = series.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        exp1 = self.calculate_ema(series, fast)
        exp2 = self.calculate_ema(series, slow)
        
        macd_line = exp1 - exp2
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        highest_high = high.rolling(window=k_period).max()
        lowest_low = low.rolling(window=k_period).min()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def calculate_donchian_channels(self, high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Donchian Channels."""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    def detect_crossover(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Detect crossover between two series (1 when series1 crosses above series2)."""
        above = series1 > series2
        above_prev = above.shift(1)
        crossover = above & ~above_prev
        return crossover.astype(int)
    
    def detect_crossunder(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Detect crossunder between two series (1 when series1 crosses below series2)."""
        below = series1 < series2
        below_prev = below.shift(1)
        crossunder = below & ~below_prev
        return crossunder.astype(int)
    
    def calculate_efficiency_ratio(self, close: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio."""
        change = np.abs(close - close.shift(period))
        volatility = np.abs(close.diff()).rolling(window=period).sum()
        
        efficiency_ratio = change / volatility
        return efficiency_ratio.fillna(0)
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate input data for technical indicators."""
        # Ensure required columns
        required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp and symbol
        data = data.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Remove any NaN values in OHLCV data
        data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Ensure positive values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data = data[data[col] > 0]
        
        return data
    
    def create_feature_dataframe(self, index: pd.Index) -> pd.DataFrame:
        """Create empty DataFrame with proper index for features."""
        return pd.DataFrame(index=index)