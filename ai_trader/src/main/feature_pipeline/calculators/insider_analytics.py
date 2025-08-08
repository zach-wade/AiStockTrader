"""
Insider Analytics Calculator

Calculates insider trading-based features including:
- Insider transaction analysis
- Buy/Sell ratios and patterns
- Transaction clustering
- Insider sentiment scores
- Executive vs director activity
- Historical insider performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

from .base_calculator import BaseFeatureCalculator
from .helpers import (
    safe_divide, create_feature_dataframe, validate_price_data,
    postprocess_features
)

from main.utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class InsiderConfig:
    """Configuration for insider analytics"""
    # Transaction filters
    min_transaction_value: float = 10000  # Minimum transaction value to consider
    
    # Time windows for analysis
    lookback_windows: List[int] = None  # Days
    cluster_window: int = 7  # Days to consider as clustered activity
    
    # Insider types
    executive_titles: List[str] = None
    director_titles: List[str] = None
    
    # Sentiment thresholds
    bullish_ratio: float = 2.0  # Buy/sell ratio for bullish signal
    bearish_ratio: float = 0.5  # Buy/sell ratio for bearish signal
    
    # Performance tracking
    performance_windows: List[int] = None  # Days to track post-transaction performance
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [30, 90, 180]
        
        if self.executive_titles is None:
            self.executive_titles = ['CEO', 'CFO', 'COO', 'President', 'EVP', 'SVP']
        
        if self.director_titles is None:
            self.director_titles = ['Director', 'Chairman', 'Board Member']
        
        if self.performance_windows is None:
            self.performance_windows = [30, 60, 90]


class InsiderAnalyticsCalculator(BaseFeatureCalculator):
    """Calculator for insider trading analytics and features"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.insider_config = InsiderConfig(
            **config.get('insider', {}) if config else {}
        )
        self.insider_data = None
        self.historical_performance = {}  # Track insider performance history
    
    def set_insider_data(self, insider_data: pd.DataFrame):
        """Set insider transaction data"""
        self.insider_data = insider_data
        
        # Ensure required columns
        required_cols = ['transaction_date', 'insider_name', 'title', 
                        'transaction_type', 'shares', 'price', 'value']
        
        missing_cols = [col for col in required_cols if col not in insider_data.columns]
        if missing_cols:
            logger.warning(f"Missing insider data columns: {missing_cols}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate insider-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with insider features
        """
        if not self.validate_inputs(data):
            logger.error("Invalid input data for insider calculation")
            return pd.DataFrame(index=data.index if not data.empty else [])
        
        if data.empty:
            logger.warning("Empty data provided for insider calculation")
            return pd.DataFrame()
        
        features = create_feature_dataframe(data.index)
        
        try:
            # Preprocess data first
            data = self.preprocess_data(data)
            
            if self.insider_data is not None and not self.insider_data.empty:
                # Validate insider data structure
                if not self._validate_insider_data():
                    logger.error("Insider data validation failed")
                    return self._create_empty_features(data.index)
                
                logger.info(f"Calculating insider features for {len(data)} periods "
                          f"with {len(self.insider_data)} insider transactions")
                
                # Transaction counts and volumes
                features = self._add_transaction_metrics(data, features)
                
                # Buy/Sell ratios
                features = self._add_buysell_ratios(data, features)
                
                # Insider sentiment
                features = self._add_insider_sentiment(data, features)
                
                # Clustering analysis
                features = self._add_clustering_features(data, features)
                
                # Insider types analysis
                features = self._add_insider_type_features(data, features)
                
                # Transaction patterns
                features = self._add_transaction_patterns(data, features)
                
                # Historical performance
                features = self._add_performance_features(data, features)
                
                # Conviction indicators
                features = self._add_conviction_features(data, features)
                
                # Add moving averages of key metrics
                features = self._add_moving_averages(features)
                
                # Add feature interactions
                features = self._add_feature_interactions(features)
                
            else:
                # No insider data available - return features with default values
                logger.warning("No insider data available for feature calculation")
                features = self._create_empty_features(data.index)
            
            # Apply insider-specific postprocessing
            features = self._postprocess_insider_features(features)
            
            # Apply general postprocessing
            features = postprocess_features(features)
            
            logger.info(f"Successfully calculated {len(features.columns)} insider features")
            
        except Exception as e:
            logger.error(f"Error calculating insider features: {e}", exc_info=True)
            # Return empty features rather than failing completely
            features = self._create_empty_features(data.index)
            
        return features
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns"""
        required = ['close']
        missing = [col for col in required if col not in data.columns]
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
            
        return True
    
    def get_required_columns(self) -> List[str]:
        """Return list of required input columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this calculator produces"""
        feature_names = []
        
        # Transaction metrics for each window
        for window in self.insider_config.lookback_windows:
            feature_names.extend([
                f'insider_buy_count_{window}d',
                f'insider_sell_count_{window}d',
                f'insider_buy_value_{window}d',
                f'insider_sell_value_{window}d',
                f'insider_net_count_{window}d',
                f'insider_net_value_{window}d',
                f'insider_bs_ratio_{window}d',
                f'insider_bs_value_ratio_{window}d',
                f'insider_bullish_{window}d',
                f'insider_bearish_{window}d'
            ])
        
        # Sentiment features
        feature_names.extend([
            'insider_sentiment_score',
            'insider_sentiment_strength',
            'insider_sentiment_consistency'
        ])
        
        # Clustering features
        feature_names.extend([
            'insider_cluster_score',
            'insider_cluster_buy',
            'insider_cluster_sell'
        ])
        
        # Insider type features
        for window in [30, 90]:
            feature_names.extend([
                f'executive_net_trades_{window}d',
                f'director_net_trades_{window}d',
                f'executive_confidence_{window}d'
            ])
        
        # Pattern and performance features
        feature_names.extend([
            'insider_avg_transaction_size',
            'insider_size_trend',
            'insider_increasing_positions',
            'insider_quarterly_activity',
            'insider_quarter_end_trading',
            'insider_performance_score',
            'insider_smart_money',
            'ceo_cfo_net_activity',
            'ceo_cfo_buying',
            'ceo_cfo_selling'
        ])
        
        # Conviction features
        feature_names.extend([
            'insider_conviction_ratio',
            'high_conviction_trade',
            'unanimous_insider_action'
        ])
        
        # Moving average features
        feature_names.extend([
            'insider_sentiment_ma_30',
            'insider_sentiment_ma_90',
            'insider_activity_ma_30',
            'insider_conviction_ma_30',
            'insider_sentiment_trend'
        ])
        
        # Interaction features
        feature_names.extend([
            'sentiment_conviction_product',
            'executive_sentiment_alignment',
            'cluster_performance_combo',
            'insider_consistency',
            'smart_conviction_combo'
        ])
        
        return feature_names
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data before calculation"""
        data = super().preprocess_data(data)
        
        # Ensure insider data is properly aligned if available
        if self.insider_data is not None:
            # Convert transaction dates to datetime for proper alignment
            if 'transaction_date' in self.insider_data.columns:
                self.insider_data['transaction_date'] = pd.to_datetime(
                    self.insider_data['transaction_date']
                )
                
                # Filter insider data to relevant time range
                if not data.empty:
                    start_date = data.index.min() - timedelta(days=max(self.insider_config.lookback_windows))
                    end_date = data.index.max()
                    
                    self.insider_data = self.insider_data[
                        (self.insider_data['transaction_date'] >= start_date) &
                        (self.insider_data['transaction_date'] <= end_date)
                    ].copy()
                    
                    logger.info(f"Filtered insider data to {len(self.insider_data)} transactions "
                              f"between {start_date.date()} and {end_date.date()}")
        
        return data
    
    def _postprocess_insider_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply insider-specific postprocessing"""
        # Fill NaN values with 0 for count-based features
        count_columns = [col for col in features.columns if 'count' in col or 'activity' in col]
        features[count_columns] = features[count_columns].fillna(0)
        
        # Fill NaN values with 0 for binary signal features
        binary_columns = [col for col in features.columns if any(keyword in col for keyword in 
                         ['bullish', 'bearish', 'buying', 'selling', 'cluster', 'confidence', 'smart_money'])]
        features[binary_columns] = features[binary_columns].fillna(0).astype(int)
        
        # Clip extreme ratio values to reasonable bounds
        ratio_columns = [col for col in features.columns if 'ratio' in col]
        for col in ratio_columns:
            if col in features.columns:
                features[col] = features[col].clip(-10, 10)
        
        # Clip sentiment scores to [-1, 1] range
        if 'insider_sentiment_score' in features.columns:
            features['insider_sentiment_score'] = features['insider_sentiment_score'].clip(-1, 1)
        
        # Ensure sentiment strength is in [0, 1] range
        if 'insider_sentiment_strength' in features.columns:
            features['insider_sentiment_strength'] = features['insider_sentiment_strength'].clip(0, 1)
        
        return features
    
    def _add_transaction_metrics(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add basic transaction metrics"""
        # Filter significant transactions
        significant_trades = self.insider_data[
            self.insider_data['value'] >= self.insider_config.min_transaction_value
        ].copy()
        
        if significant_trades.empty:
            return features
        
        # Ensure transaction_date is datetime
        significant_trades['transaction_date'] = pd.to_datetime(significant_trades['transaction_date'])
        
        for window in self.insider_config.lookback_windows:
            # Count transactions in window
            buy_counts = []
            sell_counts = []
            total_buy_value = []
            total_sell_value = []
            
            for idx in data.index:
                window_start = idx - timedelta(days=window)
                window_trades = significant_trades[
                    (significant_trades['transaction_date'] >= window_start) &
                    (significant_trades['transaction_date'] <= idx)
                ]
                
                buys = window_trades[window_trades['transaction_type'].str.contains('Buy', case=False, na=False)]
                sells = window_trades[window_trades['transaction_type'].str.contains('Sell', case=False, na=False)]
                
                buy_counts.append(len(buys))
                sell_counts.append(len(sells))
                total_buy_value.append(buys['value'].sum() if not buys.empty else 0)
                total_sell_value.append(sells['value'].sum() if not sells.empty else 0)
            
            features[f'insider_buy_count_{window}d'] = buy_counts
            features[f'insider_sell_count_{window}d'] = sell_counts
            features[f'insider_buy_value_{window}d'] = total_buy_value
            features[f'insider_sell_value_{window}d'] = total_sell_value
            features[f'insider_net_count_{window}d'] = np.array(buy_counts) - np.array(sell_counts)
            features[f'insider_net_value_{window}d'] = np.array(total_buy_value) - np.array(total_sell_value)
        
        return features
    
    def _add_buysell_ratios(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add buy/sell ratio features"""
        for window in self.insider_config.lookback_windows:
            buy_col = f'insider_buy_count_{window}d'
            sell_col = f'insider_sell_count_{window}d'
            
            if buy_col in features.columns and sell_col in features.columns:
                # Count ratio
                features[f'insider_bs_ratio_{window}d'] = safe_divide(
                    features[buy_col], (features[sell_col] + 1)
                )
                
                # Value ratio
                buy_val_col = f'insider_buy_value_{window}d'
                sell_val_col = f'insider_sell_value_{window}d'
                
                if buy_val_col in features.columns and sell_val_col in features.columns:
                    features[f'insider_bs_value_ratio_{window}d'] = safe_divide(
                        features[buy_val_col], (features[sell_val_col] + 1)
                    )
                
                # Bullish/Bearish signals
                features[f'insider_bullish_{window}d'] = (
                    features[f'insider_bs_ratio_{window}d'] > self.insider_config.bullish_ratio
                ).astype(int)
                
                features[f'insider_bearish_{window}d'] = (
                    features[f'insider_bs_ratio_{window}d'] < self.insider_config.bearish_ratio
                ).astype(int)
        
        return features
    
    def _add_insider_sentiment(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add insider sentiment scores"""
        # Composite sentiment based on multiple factors
        sentiment_components = []
        
        for window in self.insider_config.lookback_windows:
            # Net transaction direction
            if f'insider_net_count_{window}d' in features.columns:
                net_direction = np.sign(features[f'insider_net_count_{window}d'])
                sentiment_components.append(net_direction)
            
            # Value-weighted sentiment
            if f'insider_net_value_{window}d' in features.columns:
                value_sentiment = np.sign(features[f'insider_net_value_{window}d'])
                # Weight by log of absolute value
                value_weight = np.log1p(features[f'insider_net_value_{window}d'].abs()) / 10
                sentiment_components.append(value_sentiment * value_weight)
        
        if sentiment_components:
            # Average sentiment score
            features['insider_sentiment_score'] = np.mean(sentiment_components, axis=0)
            
            # Sentiment strength (0 to 1)
            features['insider_sentiment_strength'] = features['insider_sentiment_score'].abs()
            
            # Sentiment consistency
            features['insider_sentiment_consistency'] = (
                np.std(sentiment_components, axis=0) < 0.5
            ).astype(int)
        
        return features
    
    def _add_clustering_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add clustering analysis features"""
        if self.insider_data.empty:
            return features
        
        # Detect clustered buying/selling
        cluster_scores = []
        
        for idx in data.index:
            window_end = idx
            window_start = idx - timedelta(days=self.insider_config.cluster_window)
            
            window_trades = self.insider_data[
                (pd.to_datetime(self.insider_data['transaction_date']) >= window_start) &
                (pd.to_datetime(self.insider_data['transaction_date']) <= window_end)
            ]
            
            if len(window_trades) >= 3:  # Minimum for cluster
                # Check if multiple insiders traded
                unique_insiders = window_trades['insider_name'].nunique()
                
                if unique_insiders >= 2:
                    # Calculate cluster score
                    buys = window_trades[window_trades['transaction_type'].str.contains('Buy', case=False, na=False)]
                    sells = window_trades[window_trades['transaction_type'].str.contains('Sell', case=False, na=False)]
                    
                    if len(buys) > len(sells):
                        cluster_score = unique_insiders * (len(buys) / len(window_trades))
                    elif len(sells) > len(buys):
                        cluster_score = -unique_insiders * (len(sells) / len(window_trades))
                    else:
                        cluster_score = 0
                else:
                    cluster_score = 0
            else:
                cluster_score = 0
            
            cluster_scores.append(cluster_score)
        
        features['insider_cluster_score'] = cluster_scores
        features['insider_cluster_buy'] = (np.array(cluster_scores) > 2).astype(int)
        features['insider_cluster_sell'] = (np.array(cluster_scores) < -2).astype(int)
        
        return features
    
    def _add_insider_type_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add features based on insider types"""
        if self.insider_data.empty or 'title' not in self.insider_data.columns:
            return features
        
        # Classify insiders
        exec_mask = self.insider_data['title'].str.contains(
            '|'.join(self.insider_config.executive_titles), 
            case=False, na=False
        )
        
        director_mask = self.insider_data['title'].str.contains(
            '|'.join(self.insider_config.director_titles), 
            case=False, na=False
        )
        
        exec_trades = self.insider_data[exec_mask]
        director_trades = self.insider_data[director_mask]
        
        # Executive vs Director activity
        for window in [30, 90]:
            exec_buy_count = []
            exec_sell_count = []
            director_buy_count = []
            director_sell_count = []
            
            for idx in data.index:
                window_start = idx - timedelta(days=window)
                
                # Executive trades
                window_exec = exec_trades[
                    (pd.to_datetime(exec_trades['transaction_date']) >= window_start) &
                    (pd.to_datetime(exec_trades['transaction_date']) <= idx)
                ]
                
                exec_buys = window_exec[window_exec['transaction_type'].str.contains('Buy', case=False, na=False)]
                exec_sells = window_exec[window_exec['transaction_type'].str.contains('Sell', case=False, na=False)]
                
                exec_buy_count.append(len(exec_buys))
                exec_sell_count.append(len(exec_sells))
                
                # Director trades
                window_dir = director_trades[
                    (pd.to_datetime(director_trades['transaction_date']) >= window_start) &
                    (pd.to_datetime(director_trades['transaction_date']) <= idx)
                ]
                
                dir_buys = window_dir[window_dir['transaction_type'].str.contains('Buy', case=False, na=False)]
                dir_sells = window_dir[window_dir['transaction_type'].str.contains('Sell', case=False, na=False)]
                
                director_buy_count.append(len(dir_buys))
                director_sell_count.append(len(dir_sells))
            
            features[f'executive_net_trades_{window}d'] = np.array(exec_buy_count) - np.array(exec_sell_count)
            features[f'director_net_trades_{window}d'] = np.array(director_buy_count) - np.array(director_sell_count)
            
            # Executive confidence (executives usually have better info)
            features[f'executive_confidence_{window}d'] = (
                features[f'executive_net_trades_{window}d'] > 0
            ).astype(int)
        
        return features
    
    def _add_transaction_patterns(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add transaction pattern features"""
        if self.insider_data.empty:
            return features
        
        # Transaction size patterns
        avg_transaction_sizes = []
        transaction_size_trends = []
        
        for idx in data.index:
            window_start = idx - timedelta(days=90)
            window_trades = self.insider_data[
                (pd.to_datetime(self.insider_data['transaction_date']) >= window_start) &
                (pd.to_datetime(self.insider_data['transaction_date']) <= idx)
            ]
            
            if not window_trades.empty:
                avg_size = window_trades['value'].mean()
                avg_transaction_sizes.append(avg_size)
                
                # Trend in transaction sizes
                if len(window_trades) >= 3:
                    trade_times = (pd.to_datetime(window_trades['transaction_date']) - window_start).dt.days
                    trend = np.polyfit(trade_times, window_trades['value'], 1)[0]
                    transaction_size_trends.append(trend)
                else:
                    transaction_size_trends.append(0)
            else:
                avg_transaction_sizes.append(0)
                transaction_size_trends.append(0)
        
        features['insider_avg_transaction_size'] = avg_transaction_sizes
        features['insider_size_trend'] = transaction_size_trends
        features['insider_increasing_positions'] = (np.array(transaction_size_trends) > 0).astype(int)
        
        # Timing patterns
        features = self._add_timing_patterns(data, features)
        
        return features
    
    def _add_performance_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add historical insider performance features"""
        if self.insider_data.empty:
            return features
        
        # Track performance after insider trades
        performance_scores = []
        
        for idx in data.index:
            # Look at trades from performance window ago
            perf_window = 90  # Look at 90-day performance
            trade_date = idx - timedelta(days=perf_window)
            
            historical_trades = self.insider_data[
                pd.to_datetime(self.insider_data['transaction_date']) == trade_date.date()
            ]
            
            if not historical_trades.empty and idx - perf_window >= data.index[0]:
                # Calculate return since trade
                entry_price = data.loc[trade_date:, 'close'].iloc[0] if trade_date in data.index else data['close'].iloc[0]
                current_price = data.loc[idx, 'close']
                
                trade_return = (current_price - entry_price) / entry_price
                
                # Weight by trade direction
                buys = historical_trades[historical_trades['transaction_type'].str.contains('Buy', case=False, na=False)]
                sells = historical_trades[historical_trades['transaction_type'].str.contains('Sell', case=False, na=False)]
                
                if len(buys) > len(sells):
                    performance_score = trade_return  # Positive if price went up after buys
                elif len(sells) > len(buys):
                    performance_score = -trade_return  # Positive if price went down after sells
                else:
                    performance_score = 0
            else:
                performance_score = 0
            
            performance_scores.append(performance_score)
        
        features['insider_performance_score'] = performance_scores
        features['insider_smart_money'] = (np.array(performance_scores) > 0.1).astype(int)
        
        # Track specific insider performance
        features = self._add_top_insider_features(data, features)
        
        return features
    
    def _add_conviction_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add conviction indicators"""
        # Large transactions relative to history
        if 'insider_avg_transaction_size' in features.columns:
            historical_avg = features['insider_avg_transaction_size'].rolling(180).mean()
            
            # Recent large transactions
            recent_trades = []
            for idx in data.index:
                window_start = idx - timedelta(days=30)
                window_trades = self.insider_data[
                    (pd.to_datetime(self.insider_data['transaction_date']) >= window_start) &
                    (pd.to_datetime(self.insider_data['transaction_date']) <= idx)
                ]
                
                if not window_trades.empty:
                    max_trade = window_trades['value'].max()
                    recent_trades.append(max_trade)
                else:
                    recent_trades.append(0)
            
            features['insider_conviction_ratio'] = safe_divide(
                np.array(recent_trades), (historical_avg + 1)
            )
            features['high_conviction_trade'] = (features['insider_conviction_ratio'] > 3).astype(int)
        
        # Multiple insiders same direction
        if 'insider_cluster_score' in features.columns:
            features['unanimous_insider_action'] = (
                features['insider_cluster_score'].abs() > 3
            ).astype(int)
        
        return features
    
    def _add_timing_patterns(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add timing pattern features"""
        # Trades before earnings (would need earnings dates)
        # For now, track quarterly patterns
        
        quarterly_patterns = []
        
        for idx in data.index:
            # Check if trades cluster around quarter ends
            quarter_end_months = [3, 6, 9, 12]
            current_month = idx.month
            
            days_to_quarter_end = min(
                abs(current_month - m) * 30 for m in quarter_end_months
            )
            
            if days_to_quarter_end <= 30:  # Within 30 days of quarter end
                window_start = idx - timedelta(days=30)
                window_trades = self.insider_data[
                    (pd.to_datetime(self.insider_data['transaction_date']) >= window_start) &
                    (pd.to_datetime(self.insider_data['transaction_date']) <= idx)
                ]
                pattern_score = len(window_trades)
            else:
                pattern_score = 0
            
            quarterly_patterns.append(pattern_score)
        
        features['insider_quarterly_activity'] = quarterly_patterns
        features['insider_quarter_end_trading'] = (np.array(quarterly_patterns) > 2).astype(int)
        
        return features
    
    def _add_top_insider_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Track top performing insiders"""
        # Identify insiders with best historical performance
        # This would require tracking individual insider performance over time
        # Simplified version: track CEO/CFO activity specifically
        
        ceo_cfo_activity = []
        
        for idx in data.index:
            window_start = idx - timedelta(days=90)
            
            ceo_cfo_trades = self.insider_data[
                (pd.to_datetime(self.insider_data['transaction_date']) >= window_start) &
                (pd.to_datetime(self.insider_data['transaction_date']) <= idx) &
                (self.insider_data['title'].str.contains('CEO|CFO', case=False, na=False))
            ]
            
            if not ceo_cfo_trades.empty:
                buys = ceo_cfo_trades[ceo_cfo_trades['transaction_type'].str.contains('Buy', case=False, na=False)]
                sells = ceo_cfo_trades[ceo_cfo_trades['transaction_type'].str.contains('Sell', case=False, na=False)]
                
                activity_score = len(buys) - len(sells)
            else:
                activity_score = 0
            
            ceo_cfo_activity.append(activity_score)
        
        features['ceo_cfo_net_activity'] = ceo_cfo_activity
        features['ceo_cfo_buying'] = (np.array(ceo_cfo_activity) > 0).astype(int)
        features['ceo_cfo_selling'] = (np.array(ceo_cfo_activity) < 0).astype(int)
        
        return features
    
    def _validate_insider_data(self) -> bool:
        """Validate insider data structure and content"""
        if self.insider_data is None or self.insider_data.empty:
            return False
        
        # Check required columns
        required_cols = ['transaction_date', 'insider_name', 'transaction_type', 'value']
        missing_cols = [col for col in required_cols if col not in self.insider_data.columns]
        
        if missing_cols:
            logger.error(f"Missing required insider data columns: {missing_cols}")
            return False
        
        # Check for valid transaction types
        valid_types = self.insider_data['transaction_type'].str.contains(
            'Buy|Sell|Purchase|Sale', case=False, na=False
        )
        
        if not valid_types.any():
            logger.error("No valid transaction types found in insider data")
            return False
        
        # Check for valid values
        if not (self.insider_data['value'] > 0).any():
            logger.error("No positive transaction values found in insider data")
            return False
        
        return True
    
    def _create_empty_features(self, index) -> pd.DataFrame:
        """Create DataFrame with default feature values"""
        feature_names = self.get_feature_names()
        return create_feature_dataframe(index, feature_names, fill_value=0.0)
    
    def _add_moving_averages(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages of key insider metrics"""
        # Moving averages of sentiment
        if 'insider_sentiment_score' in features.columns:
            features['insider_sentiment_ma_30'] = features['insider_sentiment_score'].rolling(30).mean()
            features['insider_sentiment_ma_90'] = features['insider_sentiment_score'].rolling(90).mean()
        
        # Moving averages of transaction activity
        if 'insider_net_count_30d' in features.columns:
            features['insider_activity_ma_30'] = features['insider_net_count_30d'].rolling(30).mean()
        
        # Moving averages of conviction ratio
        if 'insider_conviction_ratio' in features.columns:
            features['insider_conviction_ma_30'] = features['insider_conviction_ratio'].rolling(30).mean()
        
        # Trend indicators
        if 'insider_sentiment_ma_30' in features.columns and 'insider_sentiment_ma_90' in features.columns:
            features['insider_sentiment_trend'] = (
                features['insider_sentiment_ma_30'] > features['insider_sentiment_ma_90']
            ).astype(int)
        
        return features
    
    def _add_feature_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions and combinations"""
        # Sentiment × Conviction interaction
        if 'insider_sentiment_score' in features.columns and 'insider_conviction_ratio' in features.columns:
            features['sentiment_conviction_product'] = (
                features['insider_sentiment_score'] * 
                np.log1p(features['insider_conviction_ratio'].abs())
            )
        
        # Executive activity × Sentiment
        if 'executive_net_trades_30d' in features.columns and 'insider_sentiment_score' in features.columns:
            features['executive_sentiment_alignment'] = (
                np.sign(features['executive_net_trades_30d']) == 
                np.sign(features['insider_sentiment_score'])
            ).astype(int)
        
        # Cluster activity × Performance
        if 'insider_cluster_score' in features.columns and 'insider_performance_score' in features.columns:
            features['cluster_performance_combo'] = (
                features['insider_cluster_score'] * features['insider_performance_score']
            )
        
        # Activity consistency across windows
        count_30d = features.get('insider_net_count_30d', pd.Series(0, index=features.index))
        count_90d = features.get('insider_net_count_90d', pd.Series(0, index=features.index))
        
        features['insider_consistency'] = (
            (np.sign(count_30d) == np.sign(count_90d)) & 
            (count_30d.abs() > 0) & 
            (count_90d.abs() > 0)
        ).astype(int)
        
        # Smart money confidence
        smart_money = features.get('insider_smart_money', pd.Series(0, index=features.index))
        high_conviction = features.get('high_conviction_trade', pd.Series(0, index=features.index))
        
        features['smart_conviction_combo'] = (smart_money & high_conviction).astype(int)
        
        return features