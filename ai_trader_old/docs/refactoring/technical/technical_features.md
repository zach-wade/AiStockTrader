# A10.1 Technical Indicators Refactoring - Feature Reference

## Overview

This document provides a comprehensive reference for all technical indicator features available in the refactored modular architecture. The original 1,463-line monolith has been transformed into 125+ specialized features across 5 focused calculators.

## Feature Summary

| Calculator | Feature Count | Description |
|------------|---------------|-------------|
| TrendIndicatorsCalculator | 28 features | Trend direction and strength analysis |
| MomentumIndicatorsCalculator | 26 features | Price momentum and oscillators |
| VolatilityIndicatorsCalculator | 24 features | Volatility and range analysis |
| VolumeIndicatorsCalculator | 22 features | Volume-based confirmation |
| AdaptiveIndicatorsCalculator | 25 features | Dynamic and adaptive indicators |
| **Total** | **125 features** | **Complete technical analysis suite** |

## TrendIndicatorsCalculator Features (28 features)

### Moving Averages

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `sma_10` | Simple Moving Average (10 periods) | window=10 | Price range |
| `sma_20` | Simple Moving Average (20 periods) | window=20 | Price range |
| `sma_50` | Simple Moving Average (50 periods) | window=50 | Price range |
| `sma_200` | Simple Moving Average (200 periods) | window=200 | Price range |
| `ema_12` | Exponential Moving Average (12 periods) | window=12 | Price range |
| `ema_26` | Exponential Moving Average (26 periods) | window=26 | Price range |
| `ema_50` | Exponential Moving Average (50 periods) | window=50 | Price range |
| `wma_20` | Weighted Moving Average (20 periods) | window=20 | Price range |
| `alma_21` | Arnaud Legoux Moving Average | window=21, offset=0.85, sigma=6 | Price range |

### Trend Strength Indicators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `adx` | Average Directional Index | period=14 | 0-100 |
| `di_plus` | Directional Indicator Plus | period=14 | 0-100 |
| `di_minus` | Directional Indicator Minus | period=14 | 0-100 |
| `aroon_up` | Aroon Up indicator | period=25 | 0-100 |
| `aroon_down` | Aroon Down indicator | period=25 | 0-100 |
| `aroon_oscillator` | Aroon Oscillator (Up - Down) | period=25 | -100 to 100 |

### Trend Channels and Levels

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `parabolic_sar` | Parabolic Stop and Reverse | start=0.02, increment=0.02, maximum=0.2 | Price range |
| `donchian_upper` | Donchian Channel Upper | period=20 | Price range |
| `donchian_lower` | Donchian Channel Lower | period=20 | Price range |
| `donchian_middle` | Donchian Channel Middle | period=20 | Price range |
| `linear_reg` | Linear Regression Line | period=14 | Price range |
| `linear_reg_slope` | Linear Regression Slope | period=14 | Real numbers |
| `linear_reg_angle` | Linear Regression Angle | period=14 | -90 to 90 degrees |

### Support/Resistance

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `pivot_point` | Classic Pivot Point | - | Price range |
| `resistance_1` | First Resistance Level | - | Price range |
| `resistance_2` | Second Resistance Level | - | Price range |
| `support_1` | First Support Level | - | Price range |
| `support_2` | Second Support Level | - | Price range |

### Trend Quality

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `trend_strength` | Overall trend strength score | - | 0-1 |
| `trend_consistency` | Trend consistency measure | window=20 | 0-1 |

## MomentumIndicatorsCalculator Features (26 features)

### Oscillators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `rsi_14` | Relative Strength Index | period=14 | 0-100 |
| `rsi_21` | Relative Strength Index | period=21 | 0-100 |
| `stochastic_k` | Stochastic %K | k_period=14, d_period=3 | 0-100 |
| `stochastic_d` | Stochastic %D | k_period=14, d_period=3 | 0-100 |
| `stochastic_slow_k` | Slow Stochastic %K | k_period=14, d_period=3 | 0-100 |
| `stochastic_slow_d` | Slow Stochastic %D | k_period=14, d_period=3 | 0-100 |
| `williams_r` | Williams %R | period=14 | -100 to 0 |
| `cci` | Commodity Channel Index | period=20 | Unbounded |

### MACD Family

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `macd` | MACD Line | fast=12, slow=26 | Real numbers |
| `macd_signal` | MACD Signal Line | fast=12, slow=26, signal=9 | Real numbers |
| `macd_histogram` | MACD Histogram | fast=12, slow=26, signal=9 | Real numbers |
| `macd_zero_cross` | MACD Zero Line Crossings | - | -1, 0, 1 |
| `macd_signal_cross` | MACD Signal Line Crossings | - | -1, 0, 1 |

### Rate of Change

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `roc_10` | Rate of Change (10 periods) | period=10 | Percentage |
| `roc_20` | Rate of Change (20 periods) | period=20 | Percentage |
| `momentum_10` | Momentum (10 periods) | period=10 | Real numbers |
| `momentum_20` | Momentum (20 periods) | period=20 | Real numbers |

### Momentum Quality

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `momentum_divergence` | Momentum divergence detection | lookback=20 | -1, 0, 1 |
| `overbought_oversold` | Overbought/Oversold signal | rsi_upper=70, rsi_lower=30 | -1, 0, 1 |
| `momentum_strength` | Overall momentum strength | - | 0-1 |
| `momentum_acceleration` | Momentum acceleration | window=5 | Real numbers |

### Advanced Momentum

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `trix` | TRIX Oscillator | period=14 | Percentage |
| `ultimate_oscillator` | Ultimate Oscillator | short=7, medium=14, long=28 | 0-100 |
| `awesome_oscillator` | Awesome Oscillator | short=5, long=34 | Real numbers |
| `ppo` | Percentage Price Oscillator | fast=12, slow=26 | Percentage |
| `ppo_signal` | PPO Signal Line | fast=12, slow=26, signal=9 | Percentage |

## VolatilityIndicatorsCalculator Features (24 features)

### Bollinger Bands

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `bollinger_upper` | Bollinger Band Upper | period=20, std=2 | Price range |
| `bollinger_lower` | Bollinger Band Lower | period=20, std=2 | Price range |
| `bollinger_middle` | Bollinger Band Middle (SMA) | period=20 | Price range |
| `bollinger_width` | Bollinger Band Width | period=20, std=2 | 0+ |
| `bollinger_position` | Price position in bands | period=20, std=2 | 0-1 |
| `bollinger_squeeze` | Bollinger Band Squeeze | period=20 | 0, 1 |

### Average True Range Family

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `atr` | Average True Range | period=14 | 0+ |
| `atr_percent` | ATR as percentage of price | period=14 | 0-100% |
| `true_range` | True Range | - | 0+ |
| `normalized_atr` | Normalized ATR | period=14, lookback=252 | 0-1 |

### Keltner Channels

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `keltner_upper` | Keltner Channel Upper | period=20, multiplier=2 | Price range |
| `keltner_lower` | Keltner Channel Lower | period=20, multiplier=2 | Price range |
| `keltner_middle` | Keltner Channel Middle | period=20 | Price range |
| `keltner_width` | Keltner Channel Width | period=20, multiplier=2 | 0+ |

### Volatility Measures

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `historical_volatility` | Historical Volatility | window=20 | 0+ |
| `volatility_ratio` | High/Low volatility ratio | window=10 | 1+ |
| `price_channel_position` | Position in price channel | period=20 | 0-1 |
| `volatility_breakout` | Volatility breakout signal | lookback=20, threshold=1.5 | 0, 1 |

### Advanced Volatility

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `chaikin_volatility` | Chaikin Volatility | period=10 | Percentage |
| `mass_index` | Mass Index | period=25 | 1+ |
| `volatility_system` | Volatility system signal | - | -1, 0, 1 |
| `volatility_trend` | Volatility trend direction | window=10 | -1, 0, 1 |

## VolumeIndicatorsCalculator Features (22 features)

### Volume Flow

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `obv` | On-Balance Volume | - | Cumulative |
| `ad_line` | Accumulation/Distribution Line | - | Cumulative |
| `cmf` | Chaikin Money Flow | period=20 | -1 to 1 |
| `mfi` | Money Flow Index | period=14 | 0-100 |

### Volume Moving Averages

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `volume_sma_10` | Volume SMA (10 periods) | period=10 | Volume range |
| `volume_sma_20` | Volume SMA (20 periods) | period=20 | Volume range |
| `volume_ema_10` | Volume EMA (10 periods) | period=10 | Volume range |
| `volume_ratio` | Current/Average volume ratio | period=20 | 0+ |

### VWAP Family

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `vwap` | Volume Weighted Average Price | - | Price range |
| `vwap_deviation` | Price deviation from VWAP | - | Real numbers |
| `anchored_vwap` | Anchored VWAP | anchor_period=20 | Price range |

### Volume Oscillators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `volume_oscillator` | Volume Oscillator | short=5, long=10 | Percentage |
| `price_volume_trend` | Price Volume Trend | - | Cumulative |
| `negative_volume_index` | Negative Volume Index | - | Index values |
| `positive_volume_index` | Positive Volume Index | - | Index values |

### Volume Patterns

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `volume_spike` | Volume spike detection | threshold=2.0 | 0, 1 |
| `volume_dry_up` | Volume dry-up detection | threshold=0.5 | 0, 1 |
| `volume_breakout` | Volume breakout confirmation | period=20 | 0, 1 |

### Advanced Volume

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `klinger_oscillator` | Klinger Volume Oscillator | short=34, long=55 | Real numbers |
| `ease_of_movement` | Ease of Movement | period=14 | Real numbers |
| `volume_weighted_macd` | Volume Weighted MACD | fast=12, slow=26 | Real numbers |
| `volume_rsi` | Volume RSI | period=14 | 0-100 |

## AdaptiveIndicatorsCalculator Features (25 features)

### Adaptive Moving Averages

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `adaptive_ma` | Adaptive Moving Average | period=20 | Price range |
| `kaufman_adaptive_ma` | Kaufman's Adaptive MA | period=10 | Price range |
| `variable_ma` | Variable Moving Average | period=20 | Price range |
| `adaptive_rsi` | Adaptive RSI | period=14 | 0-100 |

### Fractal Indicators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `fractal_dimension` | Fractal Dimension | window=20 | 1-2 |
| `hurst_exponent` | Hurst Exponent | window=100 | 0-1 |
| `fractal_adaptive_ma` | Fractal Adaptive MA | period=20 | Price range |

### Efficiency Measures

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `efficiency_ratio` | Efficiency Ratio | period=10 | 0-1 |
| `market_efficiency` | Market Efficiency Index | window=20 | 0-1 |
| `trend_efficiency` | Trend Efficiency | period=20 | 0-1 |

### Dynamic Period Indicators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `dynamic_momentum` | Dynamic Momentum | base_period=14 | Real numbers |
| `adaptive_stochastic` | Adaptive Stochastic | base_period=14 | 0-100 |
| `variable_index_ma` | Variable Index MA | period=20 | Price range |

### Cycle Analysis

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `dominant_cycle` | Dominant Cycle Period | window=50 | Periods |
| `cycle_strength` | Cycle Strength | window=50 | 0-1 |
| `instantaneous_trend` | Instantaneous Trend | window=20 | Real numbers |

### Adaptive Oscillators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `adaptive_cci` | Adaptive CCI | base_period=20 | Unbounded |
| `adaptive_williams_r` | Adaptive Williams %R | base_period=14 | -100 to 0 |
| `rocket_rsi` | Rocket RSI | period=14 | 0-100 |

### Market State Indicators

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `market_state` | Market State Classification | window=20 | 0-3 |
| `regime_filter` | Regime Filter | period=20 | 0, 1 |
| `adaptive_trend_filter` | Adaptive Trend Filter | period=20 | -1, 0, 1 |

### Noise Reduction

| Feature Name | Description | Parameters | Range |
|--------------|-------------|------------|-------|
| `noise_floor` | Noise Floor Level | window=20 | 0+ |
| `signal_to_noise` | Signal to Noise Ratio | window=20 | 0+ |
| `filtered_price` | Noise Filtered Price | alpha=0.07 | Price range |

## Configuration Options

### Common Parameters

All calculators support these common configuration options:

```python
config = {
    'min_periods': 30,           # Minimum data points required
    'max_periods': 1000,         # Maximum lookback period
    'price_columns': ['close'],   # Price columns to use
    'volume_required': False,     # Whether volume data is required
    'handle_missing': 'forward_fill',  # Missing data handling
    'outlier_threshold': 3.0,     # Outlier detection threshold
    'precision': 6                # Decimal precision for outputs
}
```

### Calculator-Specific Configurations

#### TrendIndicatorsCalculator

```python
trend_config = {
    'sma_periods': [10, 20, 50, 200],
    'ema_periods': [12, 26, 50],
    'adx_period': 14,
    'aroon_period': 25,
    'parabolic_sar_start': 0.02,
    'parabolic_sar_increment': 0.02,
    'parabolic_sar_maximum': 0.2
}
```

#### MomentumIndicatorsCalculator

```python
momentum_config = {
    'rsi_periods': [14, 21],
    'stochastic_k_period': 14,
    'stochastic_d_period': 3,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'williams_r_period': 14,
    'cci_period': 20
}
```

#### VolatilityIndicatorsCalculator

```python
volatility_config = {
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14,
    'keltner_period': 20,
    'keltner_multiplier': 2,
    'volatility_window': 20
}
```

#### VolumeIndicatorsCalculator

```python
volume_config = {
    'volume_sma_periods': [10, 20],
    'cmf_period': 20,
    'mfi_period': 14,
    'volume_spike_threshold': 2.0,
    'volume_dry_threshold': 0.5
}
```

#### AdaptiveIndicatorsCalculator

```python
adaptive_config = {
    'adaptive_ma_period': 20,
    'efficiency_ratio_period': 10,
    'fractal_window': 20,
    'hurst_window': 100,
    'cycle_window': 50,
    'adaptive_factor': 2.0
}
```

## Feature Usage Examples

### Basic Feature Access

```python
from ai_trader.feature_pipeline.calculators.technical import TrendIndicatorsCalculator

calc = TrendIndicatorsCalculator()
features = calc.calculate(data)

# Access specific features
sma_20 = features['sma_20']
trend_strength = features['trend_strength']
```

### Feature Combination

```python
# Trend + Momentum combination
trend_calc = TrendIndicatorsCalculator()
momentum_calc = MomentumIndicatorsCalculator()

trend_features = trend_calc.calculate(data)
momentum_features = momentum_calc.calculate(data)

# Create combined signals
bullish_signal = (
    (trend_features['sma_20'] > trend_features['sma_50']) &
    (momentum_features['rsi_14'] > 50) &
    (momentum_features['macd'] > momentum_features['macd_signal'])
)
```

### Feature Validation

```python
# Validate feature ranges
def validate_features(features):
    validation_rules = {
        'rsi_14': (0, 100),
        'stochastic_k': (0, 100),
        'williams_r': (-100, 0),
        'adx': (0, 100)
    }

    for feature, (min_val, max_val) in validation_rules.items():
        if feature in features.columns:
            invalid = (features[feature] < min_val) | (features[feature] > max_val)
            if invalid.any():
                print(f"Warning: {feature} has {invalid.sum()} invalid values")
```

## Performance Considerations

### Memory Usage

- **Individual Calculators**: 60-80% less memory than monolithic approach
- **Feature Subsets**: Calculate only needed indicators
- **Streaming Support**: Process data in chunks for large datasets

### Calculation Speed

- **Parallel Processing**: Run multiple calculators simultaneously
- **Caching**: Intermediate calculations cached for reuse
- **Vectorization**: All calculations use optimized numpy/pandas operations

### Best Practices

1. **Use Specific Calculators**: Only instantiate calculators you need
2. **Configure Appropriately**: Set reasonable periods for your timeframe
3. **Validate Inputs**: Ensure sufficient data for calculations
4. **Handle Errors**: Implement proper error handling for edge cases
5. **Monitor Performance**: Track calculation times and memory usage

---

**Last Updated**: 2025-07-15
**Feature Reference Version**: 2.0
**Total Features**: 125 across 5 specialized calculators
