"""
Target Generator for Machine Learning Training

This module generates forward-looking target variables for supervised learning
from historical price and feature data. It supports multiple target types:
- Forward returns (regression targets)
- Direction classification (binary/multi-class)
- Volatility-adjusted returns (risk-aware targets) 
- Market regime classifications

Author: AI Trading System
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import h5py
from pathlib import Path

logger = logging.getLogger(__name__)


class TargetGenerator:
    """
    Generates forward-looking target variables for machine learning training.
    
    This class takes historical price/feature data and creates various types
    of prediction targets suitable for different ML tasks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TargetGenerator.
        
        Args:
            config: Configuration dictionary with target generation parameters
        """
        self.config = config or {}
        
        # Default horizons for target generation
        self.horizons = self.config.get('target_horizons', [1, 2, 3, 5, 10, 20])
        
        # Classification thresholds
        self.move_thresholds = self.config.get('move_thresholds', [0.01, 0.02, 0.05])
        
        # Volatility lookback for risk adjustment
        self.vol_lookback = self.config.get('volatility_lookback', 20)
        
        logger.info(f"TargetGenerator initialized with horizons: {self.horizons}")
    
    def generate_targets_from_hdf5(
        self, 
        hdf5_path: str, 
        lookback_days: int = 252,
        save_targets: bool = True,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate targets from HDF5 features file.
        
        Args:
            hdf5_path: Path to HDF5 features file
            lookback_days: Number of recent days to process
            save_targets: Whether to save targets to files
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info(f"Loading features from {hdf5_path}")
        
        # Load features from HDF5
        features_df = self._load_features_from_hdf5(hdf5_path, lookback_days)
        
        if features_df.empty:
            logger.error("No features loaded from HDF5 file")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Loaded {len(features_df)} rows of features")
        
        # Generate targets
        targets_df = self.generate_targets(features_df)
        
        # Save if requested
        if save_targets:
            self._save_targets(features_df, targets_df, hdf5_path, output_dir)
        
        return features_df, targets_df
    
    def generate_targets(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive target variables from features DataFrame.
        
        Args:
            features_df: DataFrame with price and feature data
            
        Returns:
            DataFrame with target variables
        """
        logger.info(f"Generating targets for {len(features_df)} samples")
        
        # Ensure we have required price data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        targets = {}
        
        # 1. Forward Return Targets
        targets.update(self._generate_return_targets(features_df))
        
        # 2. Classification Targets
        targets.update(self._generate_classification_targets(features_df))
        
        # 3. Volatility-Adjusted Targets
        targets.update(self._generate_volatility_targets(features_df))
        
        # 4. Market Regime Targets
        targets.update(self._generate_regime_targets(features_df))
        
        # Create targets DataFrame with same index as features
        targets_df = pd.DataFrame(targets, index=features_df.index)
        
        logger.info(f"Generated {len(targets_df.columns)} target variables")
        return targets_df
    
    def _load_features_from_hdf5(self, hdf5_path: str, lookback_days: int) -> pd.DataFrame:
        """Load features from hierarchical HDF5 file structure."""
        try:
            with h5py.File(hdf5_path, 'r') as f:
                features_group = f['features']
                
                # Get all available dates
                dates = sorted([d for d in features_group.keys() if d.isdigit()])
                
                # Take the most recent dates
                if lookback_days > 0:
                    dates = dates[-lookback_days:]
                
                data_list = []
                
                for date_str in dates:
                    date_group = features_group[date_str]
                    
                    # Load all features for this date
                    row_data = {'date': pd.to_datetime(date_str)}
                    
                    for feature_name in date_group.keys():
                        try:
                            value = date_group[feature_name][0]  # Single value array
                            if isinstance(value, bytes):
                                value = value.decode('utf-8')
                            row_data[feature_name] = value
                        except Exception as e:
                            logger.debug(f"Error loading feature {feature_name} for {date_str}: {e}")
                            continue
                    
                    data_list.append(row_data)
                
                # Create DataFrame
                df = pd.DataFrame(data_list)
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Ensure numeric columns are proper types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading HDF5 file {hdf5_path}: {e}")
            return pd.DataFrame()
    
    def _generate_return_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate forward return targets for different horizons."""
        targets = {}
        
        close_prices = df['close']
        
        for horizon in self.horizons:
            # Simple returns
            future_returns = close_prices.pct_change(periods=horizon).shift(-horizon)
            targets[f'next_{horizon}d_return'] = future_returns
            
            # Log returns
            future_log_returns = np.log(close_prices / close_prices.shift(horizon)).shift(-horizon)
            targets[f'next_{horizon}d_log_return'] = future_log_returns
        
        return targets
    
    def _generate_classification_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate classification targets (up/down, strong moves, etc.)."""
        targets = {}
        
        close_prices = df['close']
        
        for horizon in self.horizons:
            # Get future returns for classification
            future_returns = close_prices.pct_change(periods=horizon).shift(-horizon)
            
            # Binary up/down classification
            targets[f'up_down_{horizon}d'] = (future_returns > 0).astype(int)
            
            # Strong move detection for different thresholds
            for threshold in self.move_thresholds:
                threshold_pct = int(threshold * 100)
                
                # Binary strong move (either direction)
                strong_move = (abs(future_returns) > threshold).astype(int)
                targets[f'strong_move_{horizon}d_{threshold_pct}pct'] = strong_move
                
                # Directional classification (-1, 0, 1)
                directional = np.where(
                    future_returns > threshold, 1,
                    np.where(future_returns < -threshold, -1, 0)
                )
                targets[f'directional_{horizon}d_{threshold_pct}pct'] = directional
        
        return targets
    
    def _generate_volatility_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate volatility-adjusted return targets."""
        targets = {}
        
        close_prices = df['close']
        
        # Calculate rolling volatility
        returns = close_prices.pct_change()
        rolling_vol = returns.rolling(window=self.vol_lookback).std()
        
        for horizon in [1, 3, 5, 10]:  # Subset of horizons for vol-adjusted
            if horizon in self.horizons:
                future_returns = close_prices.pct_change(periods=horizon).shift(-horizon)
                
                # Risk-adjusted returns (return / volatility)
                vol_adj_returns = future_returns / rolling_vol
                targets[f'vol_adj_return_{horizon}d'] = vol_adj_returns
                
                # Sharpe-like ratio (assuming 0 risk-free rate)
                sharpe_like = future_returns / (rolling_vol * np.sqrt(horizon))
                targets[f'sharpe_like_{horizon}d'] = sharpe_like
        
        return targets
    
    def _generate_regime_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate market regime classification targets."""
        targets = {}
        
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        
        for horizon in [5, 10, 20]:  # Longer horizons for regime detection
            if horizon in self.horizons:
                # Calculate future price range
                future_highs = high_prices.rolling(window=horizon).max().shift(-horizon)
                future_lows = low_prices.rolling(window=horizon).min().shift(-horizon)
                current_close = close_prices
                
                future_range = (future_highs - future_lows) / current_close
                
                # Trending vs ranging regime
                # Trending: large range with directional bias
                returns = close_prices.pct_change(periods=horizon).shift(-horizon)
                return_to_range_ratio = abs(returns) / future_range
                
                is_trending = (return_to_range_ratio > 0.5).astype(int)
                targets[f'is_trending_{horizon}d'] = is_trending
                
                # Regime strength (continuous)
                targets[f'regime_strength_{horizon}d'] = return_to_range_ratio
                
                # Range size (volatility proxy)
                targets[f'range_size_{horizon}d'] = future_range
        
        return targets
    
    def _save_targets(
        self, 
        features_df: pd.DataFrame, 
        targets_df: pd.DataFrame, 
        source_path: str,
        output_dir: Optional[str] = None
    ):
        """Save targets and combined data to files."""
        try:
            # Determine output directory
            if output_dir is None:
                source_path_obj = Path(source_path)
                output_dir = source_path_obj.parent.parent / 'targets'
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract symbol from filename
            symbol = Path(source_path).stem.split('_')[0]
            
            # Save targets only
            targets_path = output_dir / f'{symbol}_targets.csv'
            targets_df.to_csv(targets_path)
            logger.info(f"Saved targets to {targets_path}")
            
            # Save combined features + targets
            combined_df = pd.concat([features_df, targets_df], axis=1)
            combined_path = output_dir / f'{symbol}_features_with_targets.csv'
            combined_df.to_csv(combined_path)
            logger.info(f"Saved combined data to {combined_path}")
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'generation_timestamp': datetime.now().isoformat(),
                'source_file': str(source_path),
                'n_samples': len(targets_df),
                'n_targets': len(targets_df.columns),
                'target_columns': list(targets_df.columns),
                'horizons': self.horizons,
                'move_thresholds': self.move_thresholds,
                'volatility_lookback': self.vol_lookback,
                'date_range': {
                    'start': str(features_df.index.min()),
                    'end': str(features_df.index.max())
                }
            }
            
            import json
            metadata_path = output_dir / f'{symbol}_targets_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving targets: {e}")


if __name__ == "__main__":
    # Example usage
    generator = TargetGenerator()
    
    # Generate targets for AAPL
    hdf5_path = "/Users/zachwade/StockMonitoring/ai_trader/data_lake/features/features/AAPL_features.h5"
    features_df, targets_df = generator.generate_targets_from_hdf5(hdf5_path, lookback_days=252)
    
    print(f"Generated {len(targets_df.columns)} targets for {len(targets_df)} samples")
    print(f"Target columns: {list(targets_df.columns)}")