# A10.1 Technical Indicators Refactoring - Usage Examples

## Overview

This guide provides practical examples for using the modular technical indicators architecture. Examples range from basic usage to advanced implementation patterns.

## Basic Usage Examples

### Example 1: Single Calculator Usage

```python
from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator
import pandas as pd

# Load your data
data = pd.read_csv('price_data.csv', index_col='date', parse_dates=True)

# Initialize calculator
trend_calc = TrendIndicatorsCalculator()

# Calculate trend features
trend_features = trend_calc.calculate(data)

# Access specific features
sma_20 = trend_features['sma_20']
sma_50 = trend_features['sma_50']
adx = trend_features['adx']

print(f"Current SMA 20: {sma_20.iloc[-1]:.2f}")
print(f"Current ADX: {adx.iloc[-1]:.2f}")
```

### Example 2: Multiple Calculator Usage

```python
from ai_trader.feature_pipeline.calculators.technical import (
    TrendIndicatorsCalculator,
    MomentumIndicatorsCalculator,
    VolatilityIndicatorsCalculator
)

# Initialize calculators
trend_calc = TrendIndicatorsCalculator()
momentum_calc = MomentumIndicatorsCalculator()
volatility_calc = VolatilityIndicatorsCalculator()

# Calculate features from each calculator
trend_features = trend_calc.calculate(data)
momentum_features = momentum_calc.calculate(data)
volatility_features = volatility_calc.calculate(data)

# Combine features
all_features = pd.concat([
    trend_features,
    momentum_features,
    volatility_features
], axis=1)

print(f"Total features: {len(all_features.columns)}")
```

### Example 3: Backward Compatibility Usage

```python
from ai_trader.feature_pipeline.calculators.technical import UnifiedTechnicalIndicatorsFacade

# Drop-in replacement for original unified calculator
calculator = UnifiedTechnicalIndicatorsFacade()
all_features = calculator.calculate(data)

# Works exactly like the original unified calculator
print(f"All features calculated: {len(all_features.columns)}")
```

## Configuration Examples

### Example 4: Custom Configuration

```python
from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator

# Custom configuration for trend indicators
custom_config = {
    'sma_periods': [10, 20, 50, 100, 200],  # Additional SMA periods
    'ema_periods': [8, 13, 21, 34, 55],     # Fibonacci EMA periods
    'adx_period': 21,                        # Longer ADX period
    'aroon_period': 14                       # Shorter Aroon period
}

# Initialize with custom config
trend_calc = TrendIndicatorsCalculator(config=custom_config)
features = trend_calc.calculate(data)

# Now you have additional features
print("Available SMA features:", [col for col in features.columns if 'sma_' in col])
print("Available EMA features:", [col for col in features.columns if 'ema_' in col])
```

### Example 5: Configuration for Different Timeframes

```python
# Configuration for different trading timeframes

# Day trading configuration (shorter periods)
day_trading_config = {
    'sma_periods': [5, 10, 20],
    'ema_periods': [3, 8, 13],
    'rsi_period': 7,
    'macd_fast': 6,
    'macd_slow': 13,
    'bollinger_period': 10
}

# Swing trading configuration (medium periods)
swing_trading_config = {
    'sma_periods': [20, 50, 100],
    'ema_periods': [12, 26, 50],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'bollinger_period': 20
}

# Position trading configuration (longer periods)
position_trading_config = {
    'sma_periods': [50, 100, 200],
    'ema_periods': [21, 50, 100],
    'rsi_period': 21,
    'macd_fast': 19,
    'macd_slow': 39,
    'bollinger_period': 30
}

# Use appropriate configuration
from ai_trader.feature_pipeline.calculators.technical import MomentumIndicatorsCalculator

momentum_calc = MomentumIndicatorsCalculator(config=swing_trading_config)
features = momentum_calc.calculate(data)
```

## Trading Strategy Examples

### Example 6: Trend Following Strategy

```python
from ai_trader.feature_pipeline.calculators.technical import (
    TrendIndicatorsCalculator,
    MomentumIndicatorsCalculator
)

# Calculate trend and momentum features
trend_calc = TrendIndicatorsCalculator()
momentum_calc = MomentumIndicatorsCalculator()

trend_features = trend_calc.calculate(data)
momentum_features = momentum_calc.calculate(data)

# Define trend following signals
def trend_following_signals(trend_features, momentum_features):
    signals = pd.DataFrame(index=trend_features.index)

    # Trend conditions
    uptrend = (
        (trend_features['sma_20'] > trend_features['sma_50']) &
        (trend_features['ema_12'] > trend_features['ema_26']) &
        (trend_features['adx'] > 25)
    )

    downtrend = (
        (trend_features['sma_20'] < trend_features['sma_50']) &
        (trend_features['ema_12'] < trend_features['ema_26']) &
        (trend_features['adx'] > 25)
    )

    # Momentum confirmation
    bullish_momentum = (
        (momentum_features['rsi_14'] > 50) &
        (momentum_features['macd'] > momentum_features['macd_signal'])
    )

    bearish_momentum = (
        (momentum_features['rsi_14'] < 50) &
        (momentum_features['macd'] < momentum_features['macd_signal'])
    )

    # Combine signals
    signals['buy_signal'] = uptrend & bullish_momentum
    signals['sell_signal'] = downtrend & bearish_momentum

    return signals

# Generate signals
signals = trend_following_signals(trend_features, momentum_features)
print(f"Buy signals: {signals['buy_signal'].sum()}")
print(f"Sell signals: {signals['sell_signal'].sum()}")
```

### Example 7: Mean Reversion Strategy

```python
from ai_trader.feature_pipeline.calculators.technical import (
    VolatilityIndicatorsCalculator,
    MomentumIndicatorsCalculator
)

# Calculate volatility and momentum features
volatility_calc = VolatilityIndicatorsCalculator()
momentum_calc = MomentumIndicatorsCalculator()

volatility_features = volatility_calc.calculate(data)
momentum_features = momentum_calc.calculate(data)

def mean_reversion_signals(volatility_features, momentum_features, price_data):
    signals = pd.DataFrame(index=volatility_features.index)

    # Price relative to Bollinger Bands
    bb_position = volatility_features['bollinger_position']

    # Oversold conditions
    oversold = (
        (bb_position < 0.1) &  # Near lower Bollinger Band
        (momentum_features['rsi_14'] < 30) &  # RSI oversold
        (momentum_features['williams_r'] < -80)  # Williams %R oversold
    )

    # Overbought conditions
    overbought = (
        (bb_position > 0.9) &  # Near upper Bollinger Band
        (momentum_features['rsi_14'] > 70) &  # RSI overbought
        (momentum_features['williams_r'] > -20)  # Williams %R overbought
    )

    # Mean reversion signals
    signals['buy_signal'] = oversold
    signals['sell_signal'] = overbought

    return signals

# Generate mean reversion signals
signals = mean_reversion_signals(volatility_features, momentum_features, data)
```

### Example 8: Breakout Strategy

```python
from ai_trader.feature_pipeline.calculators.technical import (
    VolatilityIndicatorsCalculator,
    VolumeIndicatorsCalculator,
    TrendIndicatorsCalculator
)

# Calculate required features
volatility_calc = VolatilityIndicatorsCalculator()
volume_calc = VolumeIndicatorsCalculator()
trend_calc = TrendIndicatorsCalculator()

volatility_features = volatility_calc.calculate(data)
volume_features = volume_calc.calculate(data)
trend_features = trend_calc.calculate(data)

def breakout_signals(price_data, volatility_features, volume_features, trend_features):
    signals = pd.DataFrame(index=price_data.index)

    # Bollinger Band breakouts
    bb_upper_breakout = price_data['close'] > volatility_features['bollinger_upper']
    bb_lower_breakout = price_data['close'] < volatility_features['bollinger_lower']

    # Volume confirmation
    volume_spike = volume_features['volume_spike'] == 1
    above_avg_volume = volume_features['volume_ratio'] > 1.5

    # Trend confirmation
    strong_trend = trend_features['adx'] > 30

    # Breakout signals with confirmation
    signals['bullish_breakout'] = (
        bb_upper_breakout &
        (volume_spike | above_avg_volume) &
        strong_trend
    )

    signals['bearish_breakout'] = (
        bb_lower_breakout &
        (volume_spike | above_avg_volume) &
        strong_trend
    )

    return signals

# Generate breakout signals
signals = breakout_signals(data, volatility_features, volume_features, trend_features)
```

## Performance Optimization Examples

### Example 9: Parallel Calculation

```python
from concurrent.futures import ThreadPoolExecutor
from ai_trader.feature_pipeline.calculators.technical import (
    TrendIndicatorsCalculator,
    MomentumIndicatorsCalculator,
    VolatilityIndicatorsCalculator,
    VolumeIndicatorsCalculator
)
import time

def calculate_features_parallel(data):
    """Calculate features using parallel processing."""

    # Define calculators
    calculators = {
        'trend': TrendIndicatorsCalculator(),
        'momentum': MomentumIndicatorsCalculator(),
        'volatility': VolatilityIndicatorsCalculator(),
        'volume': VolumeIndicatorsCalculator()
    }

    def calculate_single(name_calc_pair):
        name, calc = name_calc_pair
        start_time = time.time()
        features = calc.calculate(data)
        end_time = time.time()
        print(f"{name} calculation took {end_time - start_time:.2f} seconds")
        return name, features

    # Parallel execution
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = dict(executor.map(calculate_single, calculators.items()))

    # Combine results
    all_features = pd.concat(results.values(), axis=1)
    return all_features

# Usage
start_time = time.time()
parallel_features = calculate_features_parallel(data)
parallel_time = time.time() - start_time

print(f"Parallel calculation completed in {parallel_time:.2f} seconds")
print(f"Total features: {len(parallel_features.columns)}")
```

### Example 10: Memory-Efficient Streaming

```python
import pandas as pd
from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator

def calculate_features_streaming(data_source, chunk_size=1000):
    """Calculate features in streaming fashion for large datasets."""

    trend_calc = TrendIndicatorsCalculator()
    all_results = []

    # Process data in chunks
    for chunk_start in range(0, len(data_source), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(data_source))

        # Get chunk with some overlap for indicator calculation
        overlap = 200  # Overlap for moving averages
        data_start = max(0, chunk_start - overlap)
        chunk_data = data_source.iloc[data_start:chunk_end]

        # Calculate features
        chunk_features = trend_calc.calculate(chunk_data)

        # Only keep the new part (without overlap)
        if chunk_start > 0:
            new_data_start = overlap
            chunk_features = chunk_features.iloc[new_data_start:]

        all_results.append(chunk_features)

        # Memory management
        if len(all_results) > 10:  # Keep only recent chunks in memory
            # Combine and save older chunks if needed
            older_chunks = pd.concat(all_results[:-5])
            # Save to disk or database
            # older_chunks.to_parquet(f'features_chunk_{chunk_start}.parquet')
            all_results = all_results[-5:]  # Keep only recent chunks

    # Combine final results
    final_features = pd.concat(all_results, ignore_index=False)
    return final_features

# Usage for large datasets
# large_data = pd.read_csv('very_large_dataset.csv')
# features = calculate_features_streaming(large_data, chunk_size=5000)
```

## Error Handling Examples

### Example 11: Robust Error Handling

```python
from ai_trader.feature_pipeline.calculators.technical import (
    TrendIndicatorsCalculator,
    MomentumIndicatorsCalculator
)
import logging

def calculate_features_robust(data):
    """Calculate features with comprehensive error handling."""

    results = {}
    calculators = {
        'trend': TrendIndicatorsCalculator(),
        'momentum': MomentumIndicatorsCalculator()
    }

    for name, calc in calculators.items():
        try:
            # Validate data first
            if data is None or data.empty:
                logging.warning(f"Empty data provided for {name} calculator")
                continue

            if len(data) < 30:  # Minimum data requirement
                logging.warning(f"Insufficient data for {name} calculator: {len(data)} rows")
                continue

            # Calculate features
            features = calc.calculate(data)

            # Validate results
            if features.empty:
                logging.warning(f"{name} calculator returned empty results")
                continue

            # Check for excessive NaN values
            nan_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
            if nan_ratio > 0.5:
                logging.warning(f"{name} calculator has {nan_ratio:.1%} NaN values")

            results[name] = features
            logging.info(f"{name} calculation successful: {len(features.columns)} features")

        except Exception as e:
            logging.error(f"Error calculating {name} features: {str(e)}")
            continue

    # Combine successful results
    if results:
        combined_features = pd.concat(results.values(), axis=1)
        logging.info(f"Total successful features: {len(combined_features.columns)}")
        return combined_features
    else:
        logging.error("All feature calculations failed")
        return pd.DataFrame()

# Usage
features = calculate_features_robust(data)
```

### Example 12: Data Validation and Cleaning

```python
import numpy as np

def validate_and_clean_data(data):
    """Validate and clean price data before feature calculation."""

    cleaned_data = data.copy()

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in cleaned_data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")

    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if (cleaned_data[col] <= 0).any():
            print(f"Warning: Found non-positive values in {col}")
            cleaned_data = cleaned_data[cleaned_data[col] > 0]

    # Check for impossible OHLC relationships
    invalid_ohlc = (
        (cleaned_data['high'] < cleaned_data['low']) |
        (cleaned_data['high'] < cleaned_data['open']) |
        (cleaned_data['high'] < cleaned_data['close']) |
        (cleaned_data['low'] > cleaned_data['open']) |
        (cleaned_data['low'] > cleaned_data['close'])
    )

    if invalid_ohlc.any():
        print(f"Warning: Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
        cleaned_data = cleaned_data[~invalid_ohlc]

    # Handle outliers (prices more than 10 standard deviations from mean)
    for col in price_columns:
        mean_price = cleaned_data[col].mean()
        std_price = cleaned_data[col].std()
        outlier_threshold = 10

        outliers = np.abs(cleaned_data[col] - mean_price) > (outlier_threshold * std_price)
        if outliers.any():
            print(f"Warning: Found {outliers.sum()} outliers in {col}")
            # Cap outliers instead of removing
            upper_bound = mean_price + (outlier_threshold * std_price)
            lower_bound = mean_price - (outlier_threshold * std_price)
            cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)

    # Fill missing values
    if cleaned_data.isnull().any().any():
        print("Filling missing values with forward fill method")
        cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')

    return cleaned_data

# Usage
try:
    clean_data = validate_and_clean_data(data)
    features = calculate_features_robust(clean_data)
except ValueError as e:
    print(f"Data validation error: {e}")
```

## Advanced Integration Examples

### Example 13: Feature Selection Pipeline

```python
from sklearn.feature_selection import SelectKBest, f_regression
from ai_trader.feature_pipeline.calculators.technical import UnifiedTechnicalIndicatorsFacade

def feature_selection_pipeline(price_data, target_returns, k=20):
    """Select best technical features for prediction."""

    # Calculate all technical features
    calc = UnifiedTechnicalIndicatorsFacade()
    all_features = calc.calculate(price_data)

    # Remove features with too many NaN values
    nan_threshold = 0.1
    feature_nan_ratio = all_features.isnull().sum() / len(all_features)
    good_features = feature_nan_ratio[feature_nan_ratio < nan_threshold].index
    features_clean = all_features[good_features].fillna(method='ffill')

    # Align features with target
    common_index = features_clean.index.intersection(target_returns.index)
    X = features_clean.loc[common_index]
    y = target_returns.loc[common_index]

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()]

    print(f"Selected {len(selected_features)} best features:")
    for i, feature in enumerate(selected_features):
        score = selector.scores_[selector.get_support()][i]
        print(f"  {feature}: {score:.2f}")

    return selected_features, X_selected

# Usage
# target = data['close'].pct_change().shift(-1)  # Next period return
# selected_features, feature_matrix = feature_selection_pipeline(data, target)
```

### Example 14: Real-time Feature Updates

```python
import asyncio
from collections import deque

class RealTimeFeatureCalculator:
    """Real-time technical feature calculator."""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.price_buffer = deque(maxlen=window_size)
        self.calculators = {
            'trend': TrendIndicatorsCalculator(),
            'momentum': MomentumIndicatorsCalculator()
        }
        self.latest_features = {}

    def add_price_data(self, price_row):
        """Add new price data and update features."""
        self.price_buffer.append(price_row)

        if len(self.price_buffer) >= 50:  # Minimum data for calculations
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.price_buffer))
            df.index = pd.to_datetime(df.index)

            # Calculate features
            for name, calc in self.calculators.items():
                try:
                    features = calc.calculate(df)
                    if not features.empty:
                        # Store only the latest row
                        self.latest_features[name] = features.iloc[-1].to_dict()
                except Exception as e:
                    print(f"Error updating {name} features: {e}")

    def get_latest_features(self):
        """Get the most recent feature values."""
        combined_features = {}
        for calc_features in self.latest_features.values():
            combined_features.update(calc_features)
        return combined_features

    async def stream_features(self, price_stream):
        """Process streaming price data."""
        async for price_data in price_stream:
            self.add_price_data(price_data)
            yield self.get_latest_features()

# Usage example
# calculator = RealTimeFeatureCalculator()
#
# # Simulate real-time data
# for new_price in price_stream:
#     calculator.add_price_data(new_price)
#     latest_features = calculator.get_latest_features()
#     print(f"Latest RSI: {latest_features.get('rsi_14', 'N/A')}")
```

## Testing Examples

### Example 15: Unit Testing Technical Features

```python
import unittest
import numpy as np

class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        price = 100
        prices = [price]

        for _ in range(99):
            price *= (1 + np.random.normal(0, 0.02))
            prices.append(price)

        self.test_data = pd.DataFrame({
            'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_trend_indicators(self):
        """Test trend indicator calculations."""
        from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator

        calc = TrendIndicatorsCalculator()
        features = calc.calculate(self.test_data)

        # Test feature presence
        self.assertIn('sma_20', features.columns)
        self.assertIn('ema_12', features.columns)
        self.assertIn('adx', features.columns)

        # Test value ranges
        self.assertTrue((features['adx'] >= 0).all())
        self.assertTrue((features['adx'] <= 100).all())

        # Test SMA calculation
        manual_sma = self.test_data['close'].rolling(20).mean()
        np.testing.assert_array_almost_equal(
            features['sma_20'].dropna().values,
            manual_sma.dropna().values,
            decimal=6
        )

    def test_momentum_indicators(self):
        """Test momentum indicator calculations."""
        from ai_trader.feature_pipeline.calculators.technical import MomentumIndicatorsCalculator

        calc = MomentumIndicatorsCalculator()
        features = calc.calculate(self.test_data)

        # Test RSI range
        self.assertTrue((features['rsi_14'] >= 0).all())
        self.assertTrue((features['rsi_14'] <= 100).all())

        # Test Stochastic range
        self.assertTrue((features['stochastic_k'] >= 0).all())
        self.assertTrue((features['stochastic_k'] <= 100).all())

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator

        # Create data with only 5 rows
        small_data = self.test_data.head(5)
        calc = TrendIndicatorsCalculator()
        features = calc.calculate(small_data)

        # Should return empty DataFrame or NaN values
        self.assertTrue(features.empty or features.isnull().all().all())

# Run tests
# unittest.main()
```

This comprehensive examples guide demonstrates the practical usage of the refactored technical indicators system, from basic usage to advanced integration patterns. The modular architecture provides flexibility while maintaining backward compatibility and performance.

---

**Last Updated**: 2025-07-15
**Examples Version**: 2.0
**Coverage**: Complete usage patterns and best practices
