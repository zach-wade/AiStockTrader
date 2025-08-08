"""
Market Microstructure Calculator

Calculates market microstructure features including:
- Bid-ask spreads and depth analysis
- Order flow imbalance
- Trade size distribution
- Market impact metrics
- Liquidity measures
- Price discovery efficiency
- Volume-weighted metrics
- High-frequency trading indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
from scipy import stats
from collections import deque

from .base_calculator import BaseFeatureCalculator
from .helpers import (
    create_feature_dataframe, safe_divide, calculate_rolling_mean,
    calculate_rolling_std, normalize_series, aggregate_features
)

from main.utils.core import get_logger, RateLimiter

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MicrostructureCalculator(BaseFeatureCalculator):
    """
    Calculates market microstructure features for institutional trading.
    
    Features include:
    - Bid-ask spread analytics
    - Order flow imbalance indicators
    - Trade size clustering
    - Market impact estimation
    - Liquidity resilience metrics
    - Price discovery efficiency
    - Volume profile analysis
    - Microstructure noise measurement
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize microstructure calculator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Microstructure parameters
        self.tick_windows = config.get('tick_windows', [10, 50, 100])
        self.spread_windows = config.get('spread_windows', [20, 60, 120])
        self.impact_windows = config.get('impact_windows', [5, 15, 30])
        
        # Trade size thresholds
        self.small_trade_threshold = config.get('small_trade_threshold', 0.1)
        self.large_trade_threshold = config.get('large_trade_threshold', 2.0)
        
        # Liquidity parameters
        self.depth_levels = config.get('depth_levels', [1, 3, 5])
        self.resilience_window = config.get('resilience_window', 30)
        
        # Price discovery parameters
        self.efficiency_window = config.get('efficiency_window', 60)
        self.noise_estimation_window = config.get('noise_estimation_window', 120)
        
        # Rate limiter for intensive calculations
        self.rate_limiter = RateLimiter(
            max_calls=200,
            time_window=60
        )
        
        logger.info("Initialized MicrostructureCalculator")
    
    def get_feature_names(self) -> List[str]:
        """Get list of microstructure feature names."""
        features = []
        
        # Spread features
        for window in self.spread_windows:
            features.extend([
                f'bid_ask_spread_{window}min',
                f'relative_spread_{window}min',
                f'effective_spread_{window}min',
                f'spread_volatility_{window}min'
            ])
        
        # Order flow features
        features.extend([
            'order_flow_imbalance',
            'trade_direction_persistence',
            'volume_imbalance_ratio',
            'buy_sell_pressure_ratio',
            'aggressive_trade_ratio'
        ])
        
        # Trade size features
        features.extend([
            'avg_trade_size',
            'trade_size_variance',
            'large_trade_frequency',
            'small_trade_frequency',
            'trade_size_skewness',
            'block_trade_indicator'
        ])
        
        # Market impact features
        for window in self.impact_windows:
            features.extend([
                f'price_impact_{window}min',
                f'temporary_impact_{window}min',
                f'permanent_impact_{window}min'
            ])
        
        # Liquidity features
        features.extend([
            'quoted_spread',
            'depth_imbalance',
            'liquidity_ratio',
            'resilience_score',
            'market_depth_change',
            'turnover_velocity'
        ])
        
        # Price discovery features
        features.extend([
            'price_efficiency_ratio',
            'information_share',
            'variance_ratio',
            'autocorrelation_decay',
            'realized_spread'
        ])
        
        # Volume profile features
        features.extend([
            'volume_weighted_price',
            'volume_participation_rate',
            'time_weighted_volume',
            'volume_synchronicity',
            'volume_clustering'
        ])
        
        # Microstructure noise
        features.extend([
            'microstructure_noise',
            'bid_ask_bounce',
            'roll_spread_estimate',
            'hasbrouck_noise',
            'effective_tick_size'
        ])
        
        # High-frequency indicators
        features.extend([
            'tick_frequency',
            'quote_update_frequency',
            'trade_arrival_intensity',
            'quote_slope',
            'market_fragmentation'
        ])
        
        # Composite measures
        features.extend([
            'liquidity_score',
            'market_quality_index',
            'microstructure_alpha',
            'execution_quality_score'
        ])
        
        return features
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate microstructure features.
        
        Args:
            data: DataFrame with OHLCV data and potentially L2 data
            
        Returns:
            DataFrame with microstructure features
        """
        try:
            if data.empty:
                logger.warning("Empty data provided for microstructure analysis")
                return pd.DataFrame()
            
            # Prepare microstructure dataset
            micro_data = self._prepare_microstructure_data(data)
            if micro_data.empty:
                return pd.DataFrame()
            
            # Initialize features
            features = create_feature_dataframe(data.index)
            
            # Calculate spread features
            spread_features = self._calculate_spread_features(micro_data, data.index)
            features = pd.concat([features, spread_features], axis=1)
            
            # Calculate order flow features
            flow_features = self._calculate_order_flow_features(micro_data, data.index)
            features = pd.concat([features, flow_features], axis=1)
            
            # Calculate trade size features
            size_features = self._calculate_trade_size_features(micro_data, data.index)
            features = pd.concat([features, size_features], axis=1)
            
            # Calculate market impact features
            impact_features = self._calculate_market_impact_features(micro_data, data.index)
            features = pd.concat([features, impact_features], axis=1)
            
            # Calculate liquidity features
            liquidity_features = self._calculate_liquidity_features(micro_data, data.index)
            features = pd.concat([features, liquidity_features], axis=1)
            
            # Calculate price discovery features
            discovery_features = self._calculate_price_discovery_features(micro_data, data.index)
            features = pd.concat([features, discovery_features], axis=1)
            
            # Calculate volume profile features
            volume_features = self._calculate_volume_profile_features(micro_data, data.index)
            features = pd.concat([features, volume_features], axis=1)
            
            # Calculate microstructure noise features
            noise_features = self._calculate_noise_features(micro_data, data.index)
            features = pd.concat([features, noise_features], axis=1)
            
            # Calculate high-frequency features
            hf_features = self._calculate_high_frequency_features(micro_data, data.index)
            features = pd.concat([features, hf_features], axis=1)
            
            # Calculate composite measures
            composite_features = self._calculate_composite_measures(features)
            features = pd.concat([features, composite_features], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
            return pd.DataFrame()
    
    def _prepare_microstructure_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare microstructure dataset with inferred metrics."""
        micro_data = data.copy()
        
        # Infer bid-ask spread from OHLC
        if all(col in micro_data.columns for col in ['high', 'low', 'close']):
            # Estimate spread as fraction of HL range
            hl_range = micro_data['high'] - micro_data['low']
            micro_data['estimated_spread'] = hl_range * 0.1  # Conservative estimate
            micro_data['mid_price'] = (micro_data['high'] + micro_data['low']) / 2
        
        # Infer trade direction from price changes
        if 'close' in micro_data.columns:
            price_change = micro_data['close'].diff()
            micro_data['trade_direction'] = np.sign(price_change)
            
            # Estimate aggressive vs passive trades
            volatility = micro_data['close'].rolling(window=20).std()
            large_moves = price_change.abs() > volatility * 0.5
            micro_data['aggressive_trade'] = large_moves.astype(int)
        
        # Volume-based features
        if 'volume' in micro_data.columns:
            # Estimate trade sizes (simplified)
            avg_volume = micro_data['volume'].rolling(window=20).mean()
            micro_data['relative_volume'] = safe_divide(
                micro_data['volume'], avg_volume
            )
            
            # Estimate number of trades
            micro_data['estimated_trades'] = np.sqrt(micro_data['volume']) * 10
            micro_data['avg_trade_size'] = safe_divide(
                micro_data['volume'], micro_data['estimated_trades']
            )
        
        # Time-based features
        if isinstance(micro_data.index, pd.DatetimeIndex):
            micro_data['time_between_trades'] = micro_data.index.to_series().diff().dt.total_seconds()
            micro_data['trade_intensity'] = safe_divide(
                1, micro_data['time_between_trades']
            )
        
        return micro_data.fillna(method='ffill').dropna()
    
    def _calculate_spread_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate bid-ask spread features."""
        features = pd.DataFrame(index=index)
        
        if 'estimated_spread' not in micro_data.columns:
            return features.fillna(0)
        
        spread = micro_data['estimated_spread']
        mid_price = micro_data.get('mid_price', micro_data.get('close', pd.Series()))
        
        for window in self.spread_windows:
            # Basic spread measures
            avg_spread = spread.rolling(window=window).mean()
            features[f'bid_ask_spread_{window}min'] = avg_spread.reindex(
                index, method='ffill'
            )
            
            # Relative spread
            rel_spread = safe_divide(spread, mid_price)
            avg_rel_spread = rel_spread.rolling(window=window).mean()
            features[f'relative_spread_{window}min'] = avg_rel_spread.reindex(
                index, method='ffill'
            )
            
            # Effective spread (using trade direction)
            if 'trade_direction' in micro_data.columns:
                effective_spread = spread * micro_data['trade_direction'].abs()
                avg_eff_spread = effective_spread.rolling(window=window).mean()
                features[f'effective_spread_{window}min'] = avg_eff_spread.reindex(
                    index, method='ffill'
                )
            
            # Spread volatility
            spread_vol = spread.rolling(window=window).std()
            features[f'spread_volatility_{window}min'] = spread_vol.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_order_flow_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate order flow imbalance features."""
        features = pd.DataFrame(index=index)
        
        if 'trade_direction' not in micro_data.columns:
            return features.fillna(0)
        
        direction = micro_data['trade_direction']
        volume = micro_data.get('volume', pd.Series(1, index=micro_data.index))
        
        # Order flow imbalance
        window = 20
        buy_volume = (direction > 0) * volume
        sell_volume = (direction < 0) * volume
        
        rolling_buy = buy_volume.rolling(window=window).sum()
        rolling_sell = sell_volume.rolling(window=window).sum()
        
        order_flow_imb = safe_divide(
            rolling_buy - rolling_sell,
            rolling_buy + rolling_sell
        )
        features['order_flow_imbalance'] = order_flow_imb.reindex(
            index, method='ffill'
        )
        
        # Trade direction persistence
        direction_changes = direction.diff() != 0
        persistence = 1 - direction_changes.rolling(window=window).mean()
        features['trade_direction_persistence'] = persistence.reindex(
            index, method='ffill'
        )
        
        # Volume imbalance ratio
        vol_imbalance = safe_divide(
            buy_volume.rolling(window=window).sum(),
            sell_volume.rolling(window=window).sum()
        )
        features['volume_imbalance_ratio'] = vol_imbalance.reindex(
            index, method='ffill'
        )
        
        # Buy-sell pressure ratio
        buy_pressure = (direction > 0).rolling(window=window).sum()
        sell_pressure = (direction < 0).rolling(window=window).sum()
        pressure_ratio = safe_divide(buy_pressure, sell_pressure)
        features['buy_sell_pressure_ratio'] = pressure_ratio.reindex(
            index, method='ffill'
        )
        
        # Aggressive trade ratio
        if 'aggressive_trade' in micro_data.columns:
            aggressive_ratio = micro_data['aggressive_trade'].rolling(
                window=window
            ).mean()
            features['aggressive_trade_ratio'] = aggressive_ratio.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_trade_size_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate trade size distribution features."""
        features = pd.DataFrame(index=index)
        
        if 'avg_trade_size' not in micro_data.columns:
            return features.fillna(0)
        
        trade_size = micro_data['avg_trade_size']
        window = 60
        
        # Average trade size
        avg_size = trade_size.rolling(window=window).mean()
        features['avg_trade_size'] = avg_size.reindex(index, method='ffill')
        
        # Trade size variance
        size_var = trade_size.rolling(window=window).var()
        features['trade_size_variance'] = size_var.reindex(index, method='ffill')
        
        # Large and small trade frequencies
        median_size = trade_size.rolling(window=window).median()
        
        large_threshold = median_size * self.large_trade_threshold
        small_threshold = median_size * self.small_trade_threshold
        
        large_trades = (trade_size > large_threshold).rolling(window=window).mean()
        small_trades = (trade_size < small_threshold).rolling(window=window).mean()
        
        features['large_trade_frequency'] = large_trades.reindex(
            index, method='ffill'
        )
        features['small_trade_frequency'] = small_trades.reindex(
            index, method='ffill'
        )
        
        # Trade size skewness
        size_skew = trade_size.rolling(window=window).apply(
            lambda x: stats.skew(x) if len(x) > 3 else 0, raw=True
        )
        features['trade_size_skewness'] = size_skew.reindex(
            index, method='ffill'
        )
        
        # Block trade indicator
        percentile_95 = trade_size.rolling(window=window).quantile(0.95)
        block_trades = (trade_size > percentile_95).astype(int)
        features['block_trade_indicator'] = block_trades.reindex(
            index, method='ffill'
        )
        
        return features
    
    def _calculate_market_impact_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate market impact features."""
        features = pd.DataFrame(index=index)
        
        if 'close' not in micro_data.columns:
            return features.fillna(0)
        
        price = micro_data['close']
        volume = micro_data.get('volume', pd.Series(1, index=micro_data.index))
        
        for window in self.impact_windows:
            # Price impact (simplified)
            price_change = price.diff(window)
            volume_window = volume.rolling(window=window).sum()
            
            # Temporary impact (mean reversion component)
            temp_impact = price_change - price_change.shift(window)
            features[f'temporary_impact_{window}min'] = temp_impact.reindex(
                index, method='ffill'
            )
            
            # Permanent impact
            perm_impact = price_change.rolling(window=window*2).mean()
            features[f'permanent_impact_{window}min'] = perm_impact.reindex(
                index, method='ffill'
            )
            
            # Price impact normalized by volume
            impact_per_volume = safe_divide(price_change.abs(), volume_window)
            features[f'price_impact_{window}min'] = impact_per_volume.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_liquidity_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate liquidity and depth features."""
        features = pd.DataFrame(index=index)
        
        # Quoted spread
        if 'estimated_spread' in micro_data.columns:
            quoted_spread = micro_data['estimated_spread']
            features['quoted_spread'] = quoted_spread.reindex(
                index, method='ffill'
            )
        
        # Depth imbalance (simplified)
        if 'volume' in micro_data.columns:
            volume = micro_data['volume']
            avg_volume = volume.rolling(window=20).mean()
            
            depth_imbalance = safe_divide(volume - avg_volume, avg_volume)
            features['depth_imbalance'] = depth_imbalance.reindex(
                index, method='ffill'
            )
        
        # Liquidity ratio
        if all(col in micro_data.columns for col in ['volume', 'close']):
            turnover = micro_data['volume'] * micro_data['close']
            price_change = micro_data['close'].diff().abs()
            
            liquidity_ratio = safe_divide(turnover, price_change)
            features['liquidity_ratio'] = liquidity_ratio.reindex(
                index, method='ffill'
            )
        
        # Resilience score (recovery from shocks)
        if 'close' in micro_data.columns:
            price = micro_data['close']
            returns = price.pct_change()
            
            # Measure recovery from large moves
            large_moves = returns.abs() > returns.rolling(window=60).std() * 2
            recovery_periods = []
            
            for i in range(len(returns)):
                if large_moves.iloc[i]:
                    # Look for recovery in next periods
                    recovery_count = 0
                    for j in range(1, min(11, len(returns) - i)):
                        if abs(returns.iloc[i+j]) < returns.rolling(window=60).std().iloc[i]:
                            recovery_count = j
                            break
                    recovery_periods.append(recovery_count)
                else:
                    recovery_periods.append(0)
            
            resilience = pd.Series(recovery_periods, index=returns.index)
            resilience_score = 10 - resilience.clip(0, 10)  # Higher = more resilient
            features['resilience_score'] = resilience_score.reindex(
                index, method='ffill'
            )
        
        # Market depth change
        if 'volume' in micro_data.columns:
            volume = micro_data['volume']
            depth_change = volume.pct_change().rolling(window=20).std()
            features['market_depth_change'] = depth_change.reindex(
                index, method='ffill'
            )
        
        # Turnover velocity
        if all(col in micro_data.columns for col in ['volume', 'close']):
            turnover = micro_data['volume'] * micro_data['close']
            velocity = turnover / turnover.rolling(window=20).mean()
            features['turnover_velocity'] = velocity.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_price_discovery_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate price discovery efficiency features."""
        features = pd.DataFrame(index=index)
        
        if 'close' not in micro_data.columns:
            return features.fillna(0)
        
        price = micro_data['close']
        returns = price.pct_change()
        
        # Price efficiency ratio
        window = self.efficiency_window
        variance_1 = returns.rolling(window=window).var()
        variance_k = returns.rolling(window=window).apply(
            lambda x: np.var(x[::5]) if len(x) > 5 else 0  # 5-period returns
        )
        
        efficiency_ratio = safe_divide(variance_k * 5, variance_1)
        features['price_efficiency_ratio'] = efficiency_ratio.reindex(
            index, method='ffill'
        )
        
        # Information share (simplified)
        if 'volume' in micro_data.columns:
            volume = micro_data['volume']
            price_impact = returns.abs() / np.sqrt(volume)
            info_share = price_impact.rolling(window=window).mean()
            features['information_share'] = info_share.reindex(
                index, method='ffill'
            )
        
        # Variance ratio test
        variance_ratios = []
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if len(window_returns) >= window:
                var_1 = np.var(window_returns)
                var_2 = np.var(window_returns.rolling(2).sum().dropna()) / 2
                vr = safe_divide(var_2, var_1, default_value=1.0)
                variance_ratios.append(vr)
            else:
                variance_ratios.append(1.0)
        
        vr_series = pd.Series(variance_ratios, index=returns.index[window:])
        features['variance_ratio'] = vr_series.reindex(index, method='ffill')
        
        # Autocorrelation decay
        autocorrs = []
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if len(window_returns) >= 10:
                autocorr = window_returns.autocorr(lag=1)
                autocorrs.append(abs(autocorr) if not np.isnan(autocorr) else 0)
            else:
                autocorrs.append(0)
        
        autocorr_series = pd.Series(autocorrs, index=returns.index[window:])
        features['autocorrelation_decay'] = autocorr_series.reindex(
            index, method='ffill'
        )
        
        # Realized spread
        if 'estimated_spread' in micro_data.columns:
            spread = micro_data['estimated_spread']
            realized_spread = spread * returns.abs()
            features['realized_spread'] = realized_spread.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_volume_profile_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate volume profile features."""
        features = pd.DataFrame(index=index)
        
        if 'volume' not in micro_data.columns:
            return features.fillna(0)
        
        volume = micro_data['volume']
        price = micro_data.get('close', pd.Series(1, index=micro_data.index))
        
        # Volume weighted price
        vwap = (price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        features['volume_weighted_price'] = vwap.reindex(index, method='ffill')
        
        # Volume participation rate
        avg_volume = volume.rolling(window=60).mean()
        participation_rate = safe_divide(volume, avg_volume)
        features['volume_participation_rate'] = participation_rate.reindex(
            index, method='ffill'
        )
        
        # Time-weighted volume
        if isinstance(micro_data.index, pd.DatetimeIndex):
            time_diffs = micro_data.index.to_series().diff().dt.total_seconds().fillna(60)
            time_weights = time_diffs / time_diffs.rolling(window=20).mean()
            tw_volume = volume * time_weights
            features['time_weighted_volume'] = tw_volume.reindex(
                index, method='ffill'
            )
        
        # Volume synchronicity
        volume_changes = volume.pct_change()
        price_changes = price.pct_change()
        
        synchronicity = volume_changes.rolling(window=20).corr(price_changes)
        features['volume_synchronicity'] = synchronicity.reindex(
            index, method='ffill'
        )
        
        # Volume clustering
        volume_volatility = volume.rolling(window=20).std()
        volume_clustering = safe_divide(volume_volatility, volume.rolling(window=20).mean())
        features['volume_clustering'] = volume_clustering.reindex(
            index, method='ffill'
        )
        
        return features
    
    def _calculate_noise_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate microstructure noise features."""
        features = pd.DataFrame(index=index)
        
        if 'close' not in micro_data.columns:
            return features.fillna(0)
        
        price = micro_data['close']
        returns = price.pct_change()
        
        # Microstructure noise (Roll's measure)
        window = self.noise_estimation_window
        autocovariance = returns.rolling(window=window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
        )
        
        noise_variance = -autocovariance
        features['microstructure_noise'] = noise_variance.reindex(
            index, method='ffill'
        )
        
        # Bid-ask bounce
        if 'estimated_spread' in micro_data.columns:
            spread = micro_data['estimated_spread']
            mid_price = micro_data.get('mid_price', price)
            
            bounce = (price - mid_price).abs() / spread
            features['bid_ask_bounce'] = bounce.reindex(index, method='ffill')
        
        # Roll spread estimate
        roll_spread = 2 * np.sqrt(-autocovariance.clip(upper=0))
        features['roll_spread_estimate'] = roll_spread.reindex(
            index, method='ffill'
        )
        
        # Hasbrouck noise estimate
        price_levels = np.log(price)
        price_diffs = price_levels.diff()
        
        hasbrouck_noise = price_diffs.rolling(window=window).var()
        features['hasbrouck_noise'] = hasbrouck_noise.reindex(
            index, method='ffill'
        )
        
        # Effective tick size
        price_changes = price.diff()
        min_tick = price_changes[price_changes > 0].min() if (price_changes > 0).any() else 0.01
        
        effective_tick = price_changes.abs().rolling(window=window).apply(
            lambda x: np.percentile(x[x > 0], 10) if (x > 0).any() else min_tick
        )
        features['effective_tick_size'] = effective_tick.reindex(
            index, method='ffill'
        )
        
        return features
    
    def _calculate_high_frequency_features(
        self,
        micro_data: pd.DataFrame,
        index: pd.Index
    ) -> pd.DataFrame:
        """Calculate high-frequency trading indicators."""
        features = pd.DataFrame(index=index)
        
        # Tick frequency
        if isinstance(micro_data.index, pd.DatetimeIndex):
            time_diffs = micro_data.index.to_series().diff().dt.total_seconds()
            tick_frequency = safe_divide(1, time_diffs)
            features['tick_frequency'] = tick_frequency.reindex(
                index, method='ffill'
            )
        
        # Quote update frequency (simplified)
        if 'estimated_spread' in micro_data.columns:
            spread_changes = micro_data['estimated_spread'].diff() != 0
            quote_updates = spread_changes.rolling(window=20).sum()
            features['quote_update_frequency'] = quote_updates.reindex(
                index, method='ffill'
            )
        
        # Trade arrival intensity
        if 'trade_intensity' in micro_data.columns:
            intensity = micro_data['trade_intensity'].rolling(window=20).mean()
            features['trade_arrival_intensity'] = intensity.reindex(
                index, method='ffill'
            )
        
        # Quote slope (depth vs spread relationship)
        if all(col in micro_data.columns for col in ['estimated_spread', 'volume']):
            spread = micro_data['estimated_spread']
            volume = micro_data['volume']
            
            # Rolling correlation as proxy for quote slope
            quote_slope = spread.rolling(window=20).corr(1/volume)
            features['quote_slope'] = quote_slope.reindex(index, method='ffill')
        
        # Market fragmentation indicator
        if 'volume' in micro_data.columns:
            volume = micro_data['volume']
            volume_variance = volume.rolling(window=20).var()
            volume_mean = volume.rolling(window=20).mean()
            
            fragmentation = safe_divide(volume_variance, volume_mean**2)
            features['market_fragmentation'] = fragmentation.reindex(
                index, method='ffill'
            )
        
        return features
    
    def _calculate_composite_measures(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate composite microstructure measures."""
        composite = pd.DataFrame(index=features.index)
        
        # Liquidity score
        liquidity_components = []
        
        if 'quoted_spread' in features.columns:
            # Lower spread = higher liquidity
            spread_score = 1 / (1 + features['quoted_spread'])
            liquidity_components.append(spread_score)
        
        if 'market_depth_change' in features.columns:
            # Lower depth volatility = higher liquidity
            depth_score = 1 / (1 + features['market_depth_change'])
            liquidity_components.append(depth_score)
        
        if 'resilience_score' in features.columns:
            # Higher resilience = higher liquidity
            liquidity_components.append(features['resilience_score'] / 10)
        
        if liquidity_components:
            composite['liquidity_score'] = np.mean(liquidity_components, axis=0)
        
        # Market quality index
        quality_components = []
        
        if 'price_efficiency_ratio' in features.columns:
            # Closer to 1 = better price discovery
            efficiency_score = 1 - (features['price_efficiency_ratio'] - 1).abs()
            quality_components.append(efficiency_score.clip(0, 1))
        
        if 'microstructure_noise' in features.columns:
            # Lower noise = higher quality
            noise_score = 1 / (1 + features['microstructure_noise'])
            quality_components.append(noise_score)
        
        if quality_components:
            composite['market_quality_index'] = np.mean(quality_components, axis=0)
        
        # Microstructure alpha
        if 'order_flow_imbalance' in features.columns and 'liquidity_score' in composite.columns:
            # Trading signal based on order flow and liquidity
            microstructure_alpha = features['order_flow_imbalance'] * composite['liquidity_score']
            composite['microstructure_alpha'] = microstructure_alpha
        
        # Execution quality score
        execution_components = []
        
        if 'temporary_impact_5min' in features.columns:
            # Lower temporary impact = better execution
            temp_impact_score = 1 / (1 + features['temporary_impact_5min'].abs())
            execution_components.append(temp_impact_score)
        
        if 'realized_spread' in features.columns:
            # Lower realized spread = better execution
            spread_score = 1 / (1 + features['realized_spread'])
            execution_components.append(spread_score)
        
        if execution_components:
            composite['execution_quality_score'] = np.mean(execution_components, axis=0)
        
        return composite