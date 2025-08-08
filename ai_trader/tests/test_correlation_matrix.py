#!/usr/bin/env python3
"""Test the correlation matrix system with synthetic data"""

import sys
from pathlib import Path
from pathlib import Path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); from test_setup import setup_test_path
setup_test_path()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from main.backtesting.analysis.correlation_matrix import CorrelationMatrix

def generate_synthetic_data(n_days=252):
    """Generate synthetic market data for testing"""
    
    # Create date range
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=n_days, freq='D')
    
    # Generate correlated asset returns
    np.random.seed(42)
    
    # Market factor
    market_factor = secure_numpy_normal(0, 0.01, n_days)
    
    # Generate asset returns with different correlations
    assets = {
        # Highly correlated equities
        'SPY': market_factor + secure_numpy_normal(0, 0.005, n_days),
        'QQQ': market_factor * 1.2 + secure_numpy_normal(0, 0.008, n_days),
        'IWM': market_factor * 0.8 + secure_numpy_normal(0, 0.01, n_days),
        
        # Inversely correlated bonds
        'TLT': -market_factor * 0.5 + secure_numpy_normal(0, 0.004, n_days),
        'IEF': -market_factor * 0.3 + secure_numpy_normal(0, 0.003, n_days),
        
        # Commodities with mixed correlation
        'GLD': -market_factor * 0.2 + secure_numpy_normal(0, 0.006, n_days),
        'USO': market_factor * 0.4 + secure_numpy_normal(0, 0.02, n_days),
        
        # High inverse correlation with market
        'VXX': -market_factor * 2 + secure_numpy_normal(0, 0.03, n_days),
        
        # International with high correlation
        'EFA': market_factor * 0.9 + secure_numpy_normal(0, 0.007, n_days),
        'EEM': market_factor * 1.1 + secure_numpy_normal(0, 0.012, n_days),
    }
    
    # Convert returns to prices
    market_data = {}
    for symbol, returns in assets.items():
        prices = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            'open': prices * (1 + np.secure_uniform(-0.005, 0.005, n_days)),
            'high': prices * (1 + np.secure_uniform(0, 0.01, n_days)),
            'low': prices * (1 + np.secure_uniform(-0.01, 0, n_days)),
            'close': prices,
            'volume': np.secure_randint(1000000, 10000000, n_days)
        }, index=dates)
        market_data[symbol] = df
    
    return market_data

def test_correlation_matrix():
    """Test the correlation matrix system"""
    
    print("Generating synthetic market data...")
    market_data = generate_synthetic_data()
    
    print("\nInitializing Correlation Matrix system...")
    config = {
        'lookback_periods': [20, 60],
        'min_correlation': 0.3,
        'z_score_threshold': 2.0
    }
    corr_matrix = CorrelationMatrix(config)
    
    print("\nCalculating correlation matrix...")
    correlation_df = corr_matrix.calculate_correlation_matrix(market_data, lookback_period=60)
    
    print("\nCorrelation Matrix (60-day):")
    print(correlation_df.round(3))
    
    print("\nAnalyzing correlations for signals...")
    signals = corr_matrix.analyze_correlations(market_data)
    
    print(f"\nGenerated {len(signals)} signals:")
    
    # Group signals by type
    signal_types = {}
    for signal in signals:
        if signal.signal_type not in signal_types:
            signal_types[signal.signal_type] = 0
        signal_types[signal.signal_type] += 1
    
    for sig_type, count in signal_types.items():
        print(f"  - {sig_type}: {count} signals")
    
    # Show top signals
    print("\nTop 5 Signals by Strength:")
    sorted_signals = sorted(signals, key=lambda x: x.strength, reverse=True)[:5]
    
    for i, signal in enumerate(sorted_signals, 1):
        print(f"\n{i}. {signal.signal_type.upper()}")
        print(f"   Primary Asset: {signal.primary_asset}")
        print(f"   Related Assets: {', '.join(signal.related_assets[:3])}")
        print(f"   Strength: {signal.strength:.3f}")
        print(f"   Direction: {signal.direction}")
        if 'correlation' in signal.metadata:
            print(f"   Correlation: {signal.metadata['correlation']:.3f}")
    
    # Test correlation pairs
    print("\n\nTesting correlation pairs...")
    returns = corr_matrix._prepare_returns(market_data)
    pairs = corr_matrix.get_correlation_pairs(returns, min_correlation=0.5, lookback_period=60)
    
    print(f"\nFound {len(pairs)} significant correlation pairs:")
    for pair in pairs[:10]:
        print(f"  {pair.asset1} <-> {pair.asset2}: {pair.correlation:.3f} (z-score: {pair.z_score:.2f})")
    
    print("\n✓ Correlation Matrix system test completed successfully!")

if __name__ == "__main__":
    test_correlation_matrix()