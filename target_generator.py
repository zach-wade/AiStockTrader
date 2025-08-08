#!/usr/bin/env python3
"""
Target Generator for AAPL Trading Features
Creates forward-looking target variables for machine learning models
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AAPLTargetGenerator:
    """
    Generates various target variables for AAPL trading predictions
    """
    
    def __init__(self, features_path="/Users/zachwade/StockMonitoring/ai_trader/data_lake/features/features/AAPL_features.h5"):
        self.features_path = Path(features_path)
        self.df = None
        self.targets_df = None
        
    def load_features(self, days_lookback=252):  # 1 year of trading days
        """Load AAPL features from HDF5 file"""
        
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        print(f"Loading AAPL features from: {self.features_path}")
        
        # Load data from HDF5 file
        data_dict = defaultdict(dict)
        
        with h5py.File(self.features_path, 'r') as f:
            features_group = f['features']
            date_keys = sorted(features_group.keys())
            
            print(f"Found {len(date_keys)} dates in dataset")
            print(f"Full date range: {date_keys[0]} to {date_keys[-1]}")
            
            # Load recent data (last N days)
            recent_dates = date_keys[-days_lookback:] if len(date_keys) >= days_lookback else date_keys
            print(f"Loading last {len(recent_dates)} dates for analysis")
            
            for date_key in recent_dates:
                date_group = features_group[date_key]
                
                # Load all features for this date
                for feature_name in date_group.keys():
                    try:
                        value = date_group[feature_name][()]
                        # Handle different data types
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        elif isinstance(value, np.ndarray) and value.size == 1:
                            value = value.item()
                        
                        data_dict[feature_name][date_key] = value
                    except Exception as e:
                        print(f"Warning: Could not load {feature_name} for {date_key}: {e}")
        
        # Convert to DataFrame
        df_data = {}
        for feature_name, date_values in data_dict.items():
            feature_series = pd.Series(date_values, name=feature_name)
            feature_series.index = pd.to_datetime(feature_series.index, format='%Y%m%d')
            df_data[feature_name] = feature_series
        
        self.df = pd.DataFrame(df_data).sort_index()
        
        print(f"Loaded features DataFrame: {self.df.shape}")
        print(f"Date range: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}")
        
        return self.df
    
    def generate_return_targets(self, horizons=[1, 2, 3, 5, 10, 20]):
        """Generate forward return targets for different time horizons"""
        
        if self.df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
        
        if 'close' not in self.df.columns:
            raise ValueError("Close price not found in features")
        
        targets = {}
        close_prices = self.df['close']
        
        print(f"\nGenerating return targets for horizons: {horizons} days")
        
        for horizon in horizons:
            # Forward simple returns
            future_close = close_prices.shift(-horizon)
            simple_return = (future_close / close_prices) - 1
            targets[f'next_{horizon}d_return'] = simple_return
            
            # Forward log returns
            log_return = np.log(future_close / close_prices)
            targets[f'next_{horizon}d_log_return'] = log_return
            
            # Calculate available data points (excluding NaN at the end)
            valid_count = simple_return.dropna().shape[0]
            print(f"  - {horizon}d returns: {valid_count} valid data points")
        
        return targets
    
    def generate_classification_targets(self, horizons=[1, 2, 3, 5, 10], thresholds=[0.01, 0.02, 0.05]):
        """Generate binary classification targets"""
        
        if self.df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
        
        targets = {}
        close_prices = self.df['close']
        
        print(f"\nGenerating classification targets:")
        print(f"  Horizons: {horizons} days")
        print(f"  Thresholds: {[f'{t*100:.0f}%' for t in thresholds]}")
        
        for horizon in horizons:
            future_close = close_prices.shift(-horizon)
            returns = (future_close / close_prices) - 1
            
            # Binary up/down
            targets[f'up_down_{horizon}d'] = (returns > 0).astype(int)
            
            # Multi-class with thresholds
            for threshold in thresholds:
                thresh_pct = int(threshold * 100)
                
                # Strong move (up or down)
                targets[f'strong_move_{horizon}d_{thresh_pct}pct'] = (np.abs(returns) > threshold).astype(int)
                
                # Directional strong move (3 classes: strong_down=-1, neutral=0, strong_up=1)
                directional = np.where(returns > threshold, 1, 
                                     np.where(returns < -threshold, -1, 0))
                targets[f'directional_{horizon}d_{thresh_pct}pct'] = directional
        
        return targets
    
    def generate_volatility_adjusted_targets(self, horizons=[1, 5, 10]):
        """Generate volatility-adjusted return targets"""
        
        if self.df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
        
        targets = {}
        close_prices = self.df['close']
        
        # Use realized volatility if available, otherwise calculate it
        vol_columns = [col for col in self.df.columns if 'volatility' in col.lower()]
        
        if vol_columns:
            volatility = self.df[vol_columns[0]]  # Use first volatility column
            vol_source = vol_columns[0]
        else:
            # Calculate 20-day rolling volatility
            returns = close_prices.pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
            vol_source = "calculated_20d_vol"
        
        print(f"\nGenerating volatility-adjusted targets using: {vol_source}")
        
        for horizon in horizons:
            future_close = close_prices.shift(-horizon)
            returns = (future_close / close_prices) - 1
            
            # Risk-adjusted returns (return / volatility)
            vol_adj_return = returns / volatility
            targets[f'vol_adj_return_{horizon}d'] = vol_adj_return
            
            # Sharpe-like ratio (assuming risk-free rate ≈ 0 for simplicity)
            targets[f'sharpe_like_{horizon}d'] = vol_adj_return * np.sqrt(252)  # Annualized
        
        return targets
    
    def generate_regime_targets(self, horizons=[5, 10]):
        """Generate regime-based targets (trending vs ranging markets)"""
        
        if self.df is None:
            raise ValueError("Features not loaded. Call load_features() first.")
        
        targets = {}
        close_prices = self.df['close']
        
        print(f"\nGenerating regime-based targets for horizons: {horizons} days")
        
        for horizon in horizons:
            # Calculate future high and low over horizon period
            future_highs = self.df['high'].rolling(window=horizon).max().shift(-horizon)
            future_lows = self.df['low'].rolling(window=horizon).min().shift(-horizon)
            current_close = close_prices
            
            # Range size relative to current price
            future_range = (future_highs - future_lows) / current_close
            
            # Trending strength: how much the price moves in one direction
            future_close = close_prices.shift(-horizon)
            net_move = np.abs(future_close - current_close) / current_close
            trending_strength = net_move / future_range
            
            targets[f'regime_trending_{horizon}d'] = trending_strength
            targets[f'regime_range_size_{horizon}d'] = future_range
            
            # Binary trending vs ranging (trending if > 50% of range is directional)
            targets[f'is_trending_{horizon}d'] = (trending_strength > 0.5).astype(int)
        
        return targets
    
    def generate_all_targets(self, return_horizons=[1, 2, 3, 5, 10, 20],
                           classification_horizons=[1, 2, 3, 5, 10],
                           classification_thresholds=[0.01, 0.02, 0.05],
                           vol_adj_horizons=[1, 5, 10],
                           regime_horizons=[5, 10]):
        """Generate all types of targets"""
        
        print("="*80)
        print("🎯 GENERATING ALL TARGET VARIABLES")
        print("="*80)
        
        all_targets = {}
        
        # 1. Return targets
        return_targets = self.generate_return_targets(return_horizons)
        all_targets.update(return_targets)
        
        # 2. Classification targets
        class_targets = self.generate_classification_targets(
            classification_horizons, classification_thresholds)
        all_targets.update(class_targets)
        
        # 3. Volatility-adjusted targets
        vol_targets = self.generate_volatility_adjusted_targets(vol_adj_horizons)
        all_targets.update(vol_targets)
        
        # 4. Regime targets
        regime_targets = self.generate_regime_targets(regime_horizons)
        all_targets.update(regime_targets)
        
        # Create targets DataFrame
        self.targets_df = pd.DataFrame(all_targets, index=self.df.index)
        
        print(f"\n📊 TARGET GENERATION SUMMARY:")
        print(f"   Total target variables: {len(all_targets)}")
        print(f"   Data shape: {self.targets_df.shape}")
        
        # Count valid (non-NaN) targets for each variable
        print(f"\n📈 VALID DATA POINTS BY TARGET TYPE:")
        valid_counts = self.targets_df.count()
        
        target_types = {
            'Return Targets': [col for col in valid_counts.index if 'return' in col and 'vol_adj' not in col],
            'Classification Targets': [col for col in valid_counts.index if any(x in col for x in ['up_down', 'strong_move', 'directional'])],
            'Vol-Adjusted Targets': [col for col in valid_counts.index if 'vol_adj' in col or 'sharpe' in col],
            'Regime Targets': [col for col in valid_counts.index if 'regime' in col or 'trending' in col]
        }
        
        for target_type, cols in target_types.items():
            if cols:
                avg_valid = valid_counts[cols].mean()
                print(f"   {target_type}: {avg_valid:.0f} valid data points (avg)")
        
        return self.targets_df
    
    def get_analysis_summary(self):
        """Get summary analysis of targets"""
        
        if self.targets_df is None:
            print("No targets generated yet. Call generate_all_targets() first.")
            return
        
        print("\n" + "="*80)
        print("📋 TARGET ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic statistics for return targets
        return_cols = [col for col in self.targets_df.columns if 'return' in col and 'vol_adj' not in col]
        if return_cols:
            print(f"\n📈 RETURN TARGETS STATISTICS (sample: {return_cols[:3]}):")
            print(self.targets_df[return_cols[:3]].describe())
        
        # Classification target distribution
        binary_cols = [col for col in self.targets_df.columns if 'up_down' in col]
        if binary_cols:
            print(f"\n🎯 CLASSIFICATION TARGET DISTRIBUTIONS:")
            for col in binary_cols[:3]:  # Show first 3
                value_counts = self.targets_df[col].value_counts()
                total = value_counts.sum()
                print(f"   {col}:")
                for val, count in value_counts.items():
                    pct = (count/total)*100
                    direction = "📈 Up" if val == 1 else "📉 Down" if val == 0 else f"Class {val}"
                    print(f"     {direction}: {count} ({pct:.1f}%)")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("1. Use return targets for regression models")
        print("2. Use classification targets for binary/multi-class prediction")
        print("3. Vol-adjusted targets help with risk-normalized predictions")
        print("4. Regime targets useful for strategy switching models")
        
        return self.targets_df
    
    def save_targets(self, output_dir="/Users/zachwade/StockMonitoring/ai_trader/data_lake/targets/"):
        """Save targets to CSV files"""
        
        if self.targets_df is None:
            print("No targets to save. Generate targets first.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save targets only
        targets_path = output_dir / "AAPL_targets.csv"
        self.targets_df.to_csv(targets_path)
        
        # Also save features with targets combined
        combined_df = pd.concat([self.df, self.targets_df], axis=1)
        combined_path = output_dir / "AAPL_features_with_targets.csv"
        combined_df.to_csv(combined_path)
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'features_shape': self.df.shape,
            'targets_shape': self.targets_df.shape,
            'combined_shape': combined_df.shape,
            'date_range': f"{self.df.index.min()} to {self.df.index.max()}",
            'target_columns': list(self.targets_df.columns),
            'feature_columns': list(self.df.columns)
        }
        
        metadata_path = output_dir / "AAPL_targets_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n💾 SAVED TARGET DATA:")
        print(f"   Targets only: {targets_path}")
        print(f"   Features + Targets: {combined_path}")  
        print(f"   Metadata: {metadata_path}")
        print(f"   Targets shape: {self.targets_df.shape}")
        print(f"   Combined shape: {combined_df.shape}")
        
        return targets_path, combined_path, metadata_path

def main():
    """Main execution function"""
    
    # Initialize target generator
    generator = AAPLTargetGenerator()
    
    # Load features
    df = generator.load_features(days_lookback=252)  # 1 year
    
    # Generate all targets
    targets_df = generator.generate_all_targets()
    
    # Show analysis
    generator.get_analysis_summary()
    
    # Save results
    generator.save_targets()
    
    print(f"\n🎉 TARGET GENERATION COMPLETE!")
    print(f"   Features loaded: {df.shape}")
    print(f"   Targets created: {targets_df.shape}")
    print(f"   Ready for model training!")

if __name__ == "__main__":
    main()