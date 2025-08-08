"""
Momentum Indicators Calculator

Calculates momentum-based technical indicators that measure the rate of change
and strength of price movements to identify overbought/oversold conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings

from .base_technical import BaseTechnicalCalculator
from ..helpers import (
    create_feature_dataframe, safe_divide, calculate_rolling_mean,
    calculate_rolling_std, normalize_series
)

from main.utils.core import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MomentumIndicatorsCalculator(BaseTechnicalCalculator):
    """
    Calculates momentum and oscillator indicators.
    
    Features include:
    - RSI (Relative Strength Index) with variations
    - MACD (Moving Average Convergence Divergence)
    - Stochastic Oscillators
    - Momentum and Rate of Change
    - CCI (Commodity Channel Index)
    - Williams %R
    - Ultimate Oscillator
    - Money Flow Index
    - Divergence detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize momentum indicators calculator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Momentum-specific parameters
        self.rsi_periods = config.get('rsi_periods', [9, 14, 21])
        self.macd_configs = config.get('macd_configs', [
            (12, 26, 9),  # Standard
            (5, 35, 5),   # Aggressive
            (19, 39, 9)   # Conservative
        ])
        self.stoch_periods = config.get('stoch_periods', [14, 21])
        self.momentum_periods = config.get('momentum_periods', [10, 20, 50])
        
        # Divergence detection
        self.divergence_lookback = config.get('divergence_lookback', 20)
        self.divergence_threshold = config.get('divergence_threshold', 0.02)
        
        logger.info("Initialized MomentumIndicatorsCalculator")
    
    def get_feature_names(self) -> List[str]:
        """Get list of momentum indicator feature names."""
        features = []
        
        # RSI variations
        for period in self.rsi_periods:
            features.extend([
                f'rsi_{period}',
                f'rsi_{period}_sma',
                f'rsi_{period}_oversold',
                f'rsi_{period}_overbought',
                f'rsi_{period}_divergence'
            ])
        
        # MACD variations
        for fast, slow, signal in self.macd_configs:
            prefix = f'macd_{fast}_{slow}_{signal}'
            features.extend([
                f'{prefix}_line',
                f'{prefix}_signal',
                f'{prefix}_histogram',
                f'{prefix}_cross',
                f'{prefix}_divergence'
            ])
        
        # Stochastic oscillators
        for period in self.stoch_periods:
            features.extend([
                f'stoch_k_{period}',
                f'stoch_d_{period}',
                f'stoch_cross_{period}',
                f'stoch_oversold_{period}',
                f'stoch_overbought_{period}'
            ])
        
        # Momentum and ROC
        for period in self.momentum_periods:
            features.extend([
                f'momentum_{period}',
                f'roc_{period}',
                f'momentum_acceleration_{period}'
            ])
        
        # Other oscillators
        features.extend([
            'cci_14',
            'cci_20',
            'williams_r_14',
            'ultimate_oscillator',
            'mfi_14',
            'mfi_oversold',
            'mfi_overbought'
        ])
        
        # Composite momentum
        features.extend([
            'momentum_composite',
            'oscillator_average',
            'momentum_strength',
            'momentum_divergence_score'
        ])
        
        return features
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators from price data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        try:
            # Validate input data
            validated_data = self.validate_ohlcv_data(data)
            if validated_data.empty:
                return self._create_empty_features(data.index)
            
            # Initialize features DataFrame
            features = create_feature_dataframe(validated_data.index)
            
            # Calculate RSI variations
            rsi_features = self._calculate_rsi_features(validated_data)
            features = pd.concat([features, rsi_features], axis=1)
            
            # Calculate MACD variations
            macd_features = self._calculate_macd_features(validated_data)
            features = pd.concat([features, macd_features], axis=1)
            
            # Calculate Stochastic oscillators
            stoch_features = self._calculate_stochastic_features(validated_data)
            features = pd.concat([features, stoch_features], axis=1)
            
            # Calculate momentum and ROC
            momentum_features = self._calculate_momentum_features(validated_data)
            features = pd.concat([features, momentum_features], axis=1)
            
            # Calculate other oscillators
            other_features = self._calculate_other_oscillators(validated_data)
            features = pd.concat([features, other_features], axis=1)
            
            # Calculate composite indicators
            composite_features = self._calculate_composite_indicators(features)
            features = pd.concat([features, composite_features], axis=1)
            
            # Apply postprocessing
            features = self.postprocess_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return self._create_empty_features(data.index)
    
    def _calculate_rsi_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate RSI-based features."""
        features = pd.DataFrame(index=data.index)
        close = data['close']
        
        for period in self.rsi_periods:
            # Calculate RSI
            rsi = self.calculate_rsi(close, period)
            features[f'rsi_{period}'] = rsi
            
            # RSI moving average
            features[f'rsi_{period}_sma'] = rsi.rolling(
                window=period, min_periods=1
            ).mean()
            
            # Oversold/Overbought signals
            features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            
            # RSI divergence
            divergence = self._detect_divergence(close, rsi, self.divergence_lookback)
            features[f'rsi_{period}_divergence'] = divergence
        
        return features
    
    def _calculate_macd_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate MACD-based features."""
        features = pd.DataFrame(index=data.index)
        close = data['close']
        
        for fast, slow, signal in self.macd_configs:
            # Calculate MACD
            macd_line, signal_line, histogram = self.calculate_macd(
                close, fast, slow, signal
            )
            
            prefix = f'macd_{fast}_{slow}_{signal}'
            features[f'{prefix}_line'] = macd_line
            features[f'{prefix}_signal'] = signal_line
            features[f'{prefix}_histogram'] = histogram
            
            # MACD crossover signals
            features[f'{prefix}_cross'] = self._detect_crossover(
                macd_line, signal_line
            )
            
            # MACD divergence
            divergence = self._detect_divergence(
                close, histogram, self.divergence_lookback
            )
            features[f'{prefix}_divergence'] = divergence
        
        return features
    
    def _calculate_stochastic_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Stochastic oscillator features."""
        features = pd.DataFrame(index=data.index)
        
        for period in self.stoch_periods:
            # Calculate Stochastic %K and %D
            k_percent, d_percent = self._calculate_stochastic(
                data['high'], data['low'], data['close'], period
            )
            
            features[f'stoch_k_{period}'] = k_percent
            features[f'stoch_d_{period}'] = d_percent
            
            # Stochastic crossover
            features[f'stoch_cross_{period}'] = self._detect_crossover(
                k_percent, d_percent
            )
            
            # Oversold/Overbought
            features[f'stoch_oversold_{period}'] = (k_percent < 20).astype(int)
            features[f'stoch_overbought_{period}'] = (k_percent > 80).astype(int)
        
        return features
    
    def _calculate_momentum_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate momentum and rate of change features."""
        features = pd.DataFrame(index=data.index)
        close = data['close']
        
        for period in self.momentum_periods:
            # Raw momentum
            momentum = close - close.shift(period)
            features[f'momentum_{period}'] = momentum
            
            # Rate of Change (ROC)
            roc = safe_divide(momentum, close.shift(period)) * 100
            features[f'roc_{period}'] = roc
            
            # Momentum acceleration (second derivative)
            momentum_accel = momentum - momentum.shift(period)
            features[f'momentum_acceleration_{period}'] = momentum_accel
        
        return features
    
    def _calculate_other_oscillators(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate other momentum oscillators."""
        features = pd.DataFrame(index=data.index)
        
        # CCI (Commodity Channel Index)
        features['cci_14'] = self._calculate_cci(data, 14)
        features['cci_20'] = self._calculate_cci(data, 20)
        
        # Williams %R
        features['williams_r_14'] = self._calculate_williams_r(data, 14)
        
        # Ultimate Oscillator
        features['ultimate_oscillator'] = self._calculate_ultimate_oscillator(data)
        
        # Money Flow Index
        mfi = self._calculate_mfi(data, 14)
        features['mfi_14'] = mfi
        features['mfi_oversold'] = (mfi < 20).astype(int)
        features['mfi_overbought'] = (mfi > 80).astype(int)
        
        return features
    
    def _calculate_composite_indicators(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite momentum indicators."""
        composite = pd.DataFrame(index=features.index)
        
        # Momentum composite (average of normalized oscillators)
        oscillators = []
        
        # Normalize RSI (0-100 scale)
        for period in self.rsi_periods:
            if f'rsi_{period}' in features:
                normalized_rsi = (features[f'rsi_{period}'] - 50) / 50
                oscillators.append(normalized_rsi)
        
        # Normalize Stochastic (0-100 scale)
        for period in self.stoch_periods:
            if f'stoch_k_{period}' in features:
                normalized_stoch = (features[f'stoch_k_{period}'] - 50) / 50
                oscillators.append(normalized_stoch)
        
        # Normalize CCI (-100 to 100 typical range)
        if 'cci_14' in features:
            normalized_cci = features['cci_14'] / 100
            oscillators.append(normalized_cci.clip(-1, 1))
        
        if oscillators:
            composite['momentum_composite'] = pd.concat(
                oscillators, axis=1
            ).mean(axis=1)
        else:
            composite['momentum_composite'] = 0
        
        # Oscillator average (simple average)
        osc_cols = [col for col in features.columns if any(
            ind in col for ind in ['rsi', 'stoch_k', 'cci', 'mfi']
        ) and not any(
            suffix in col for suffix in ['oversold', 'overbought', 'cross', 'divergence']
        )]
        
        if osc_cols:
            composite['oscillator_average'] = features[osc_cols].mean(axis=1)
        else:
            composite['oscillator_average'] = 50
        
        # Momentum strength (based on agreement)
        strength_signals = []
        
        # Count bullish/bearish signals
        for col in features.columns:
            if 'oversold' in col:
                strength_signals.append(features[col])
            elif 'overbought' in col:
                strength_signals.append(-features[col])
        
        if strength_signals:
            composite['momentum_strength'] = abs(
                pd.concat(strength_signals, axis=1).sum(axis=1)
            ) / len(strength_signals)
        else:
            composite['momentum_strength'] = 0
        
        # Divergence score (aggregate all divergences)
        div_cols = [col for col in features.columns if 'divergence' in col]
        
        if div_cols:
            composite['momentum_divergence_score'] = features[div_cols].abs().mean(axis=1)
        else:
            composite['momentum_divergence_score'] = 0
        
        return composite
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic %K and %D."""
        # Calculate raw stochastic
        lowest_low = low.rolling(window=period, min_periods=1).min()
        highest_high = high.rolling(window=period, min_periods=1).max()
        
        raw_k = safe_divide(
            close - lowest_low,
            highest_high - lowest_low
        ) * 100
        
        # Smooth %K
        k_percent = raw_k.rolling(window=smooth_k, min_periods=1).mean()
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()
        
        return k_percent, d_percent
    
    def _calculate_cci(
        self,
        data: pd.DataFrame,
        period: int
    ) -> pd.Series:
        """Calculate Commodity Channel Index."""
        # Typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # SMA of typical price
        sma = typical_price.rolling(window=period, min_periods=1).mean()
        
        # Mean deviation
        mean_deviation = typical_price.rolling(
            window=period, min_periods=1
        ).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        # CCI calculation
        cci = safe_divide(typical_price - sma, 0.015 * mean_deviation)
        
        return cci
    
    def _calculate_williams_r(
        self,
        data: pd.DataFrame,
        period: int
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = data['high'].rolling(window=period, min_periods=1).max()
        lowest_low = data['low'].rolling(window=period, min_periods=1).min()
        
        williams_r = safe_divide(
            highest_high - data['close'],
            highest_high - lowest_low
        ) * -100
        
        return williams_r
    
    def _calculate_ultimate_oscillator(
        self,
        data: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """Calculate Ultimate Oscillator."""
        # Calculate buying pressure
        close_prev = data['close'].shift(1)
        low_min = pd.concat([data['low'], close_prev], axis=1).min(axis=1)
        buying_pressure = data['close'] - low_min
        
        # Calculate true range
        high_max = pd.concat([data['high'], close_prev], axis=1).max(axis=1)
        true_range = high_max - low_min
        
        # Calculate averages for each period
        def calculate_ratio(period):
            bp_sum = buying_pressure.rolling(window=period, min_periods=1).sum()
            tr_sum = true_range.rolling(window=period, min_periods=1).sum()
            return safe_divide(bp_sum, tr_sum)
        
        avg1 = calculate_ratio(period1)
        avg2 = calculate_ratio(period2)
        avg3 = calculate_ratio(period3)
        
        # Ultimate Oscillator formula
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        
        return uo
    
    def _calculate_mfi(
        self,
        data: pd.DataFrame,
        period: int
    ) -> pd.Series:
        """Calculate Money Flow Index."""
        # Typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = raw_money_flow.where(price_change > 0, 0)
        negative_flow = raw_money_flow.where(price_change < 0, 0)
        
        # Sum over period
        positive_sum = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_sum = negative_flow.rolling(window=period, min_periods=1).sum()
        
        # Money flow ratio and MFI
        money_ratio = safe_divide(positive_sum, negative_sum)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _detect_divergence(
        self,
        price: pd.Series,
        indicator: pd.Series,
        lookback: int
    ) -> pd.Series:
        """Detect divergence between price and indicator."""
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            # Get windows
            price_window = price.iloc[i-lookback:i+1]
            ind_window = indicator.iloc[i-lookback:i+1]
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(price_window)
            price_troughs = self._find_troughs(price_window)
            ind_peaks = self._find_peaks(ind_window)
            ind_troughs = self._find_troughs(ind_window)
            
            # Bearish divergence: price higher high, indicator lower high
            if len(price_peaks) >= 2 and len(ind_peaks) >= 2:
                if (price_window.iloc[price_peaks[-1]] > price_window.iloc[price_peaks[-2]] and
                    ind_window.iloc[ind_peaks[-1]] < ind_window.iloc[ind_peaks[-2]]):
                    divergence.iloc[i] = -1
            
            # Bullish divergence: price lower low, indicator higher low
            if len(price_troughs) >= 2 and len(ind_troughs) >= 2:
                if (price_window.iloc[price_troughs[-1]] < price_window.iloc[price_troughs[-2]] and
                    ind_window.iloc[ind_troughs[-1]] > ind_window.iloc[ind_troughs[-2]]):
                    divergence.iloc[i] = 1
        
        return divergence
    
    def _detect_crossover(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> pd.Series:
        """Detect crossover between two series."""
        # 1 for bullish cross (series1 crosses above series2)
        # -1 for bearish cross (series1 crosses below series2)
        # 0 for no cross
        
        diff = series1 - series2
        diff_sign = np.sign(diff)
        diff_sign_change = diff_sign.diff()
        
        crossover = pd.Series(0, index=series1.index)
        crossover[diff_sign_change > 0] = 1   # Bullish cross
        crossover[diff_sign_change < 0] = -1  # Bearish cross
        
        return crossover
    
    def _find_peaks(self, series: pd.Series, min_distance: int = 2) -> List[int]:
        """Find peaks in a series."""
        peaks = []
        
        for i in range(min_distance, len(series) - min_distance):
            if (series.iloc[i] > series.iloc[i-min_distance] and 
                series.iloc[i] > series.iloc[i+min_distance]):
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, series: pd.Series, min_distance: int = 2) -> List[int]:
        """Find troughs in a series."""
        troughs = []
        
        for i in range(min_distance, len(series) - min_distance):
            if (series.iloc[i] < series.iloc[i-min_distance] and 
                series.iloc[i] < series.iloc[i+min_distance]):
                troughs.append(i)
        
        return troughs