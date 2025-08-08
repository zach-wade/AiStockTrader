#!/usr/bin/env python3
"""
Simple Integration Test Script for AI Trading System
Can be run with pytest or directly: python test_integration_simple.py
"""

import asyncio
import sys
from pathlib import Path
from pathlib import Path
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent)); from test_setup import setup_test_path
setup_test_path()

# Test results tracking for standalone mode
test_results = {"passed": 0, "failed": 0, "errors": []}


def test_pass(test_name):
    """Mark test as passed."""
    test_results["passed"] += 1
    print(f"✅ {test_name}")


def test_fail(test_name, error):
    """Mark test as failed."""
    test_results["failed"] += 1
    test_results["errors"].append(f"{test_name}: {error}")
    print(f"❌ {test_name}: {error}")


async def test_data_pipeline():
    """Test 1: Data Pipeline Components"""
    print("\n🔍 Testing Data Pipeline...")
    
    try:
        # Test data standardizer
        from main.data_pipeline.transformers.data_standardizer import DataStandardizer
        
        standardizer = DataStandardizer({})
        
        # Create sample data
        df = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        # Test standardization
        result = standardizer.standardize_market_data(df, 'yahoo', 'AAPL')
        
        if len(result) > 0 and 'open' in result.columns:
            test_pass("Data standardization")
        else:
            test_fail("Data standardization", "Failed to standardize data")
            
        # Test batch operations
        from main.data_pipeline.storage.batch_operations import BatchOperations
        
        # Mock the database adapter
        class MockDBAdapter:
            def __init__(self):
                self.engine = None
                
        batch_ops = BatchOperations(MockDBAdapter(), 'market_data')
        test_pass("Batch operations initialization")
        
    except Exception as e:
        test_fail("Data pipeline", str(e))


async def test_feature_engineering():
    """Test 2: Feature Engineering"""
    print("\n🔍 Testing Feature Engineering...")
    
    try:
        # Test technical indicators
        from main.feature_pipeline.calculators.technical import TechnicalCalculator
        
        calculator = TechnicalCalculator({})
        
        # Create sample data
        df = pd.DataFrame({
            'open': np.secure_uniform(100, 200, 100),
            'high': np.secure_uniform(100, 200, 100),
            'low': np.secure_uniform(100, 200, 100),
            'close': np.secure_uniform(100, 200, 100),
            'volume': np.secure_randint(1000000, 10000000, 100)
        }, index=pd.date_range('2025-01-01', periods=100))
        
        # Calculate features
        result = await calculator.calculate(df)
        
        if 'sma_20' in result.columns:
            test_pass("Technical indicator calculation")
        else:
            test_fail("Technical indicator calculation", "Missing expected indicators")
            
    except Exception as e:
        test_fail("Feature engineering", str(e))


async def test_strategy_system():
    """Test 3: Strategy System"""
    print("\n🔍 Testing Strategy System...")
    
    try:
        # Test base strategy
        from main.models.strategies.base_strategy import BaseStrategy
        
        class TestStrategy(BaseStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.name = "test_strategy"
                
            async def generate_signals(self, market_data):
                return {'AAPL': 0.5}
        
        strategy = TestStrategy({})
        signals = await strategy.generate_signals(pd.DataFrame())
        
        if signals.get('AAPL') == 0.5:
            test_pass("Strategy signal generation")
        else:
            test_fail("Strategy signal generation", "Incorrect signal value")
            
    except Exception as e:
        test_fail("Strategy system", str(e))


async def test_risk_management():
    """Test 4: Risk Management"""
    print("\n🔍 Testing Risk Management...")
    
    try:
        from main.risk_management.real_time.circuit_breaker import CircuitBreaker
        
        config = {
            'risk': {
                'max_daily_loss_pct': 0.02,
                'max_positions': 10,
                'max_leverage': 1.0
            }
        }
        
        breaker = CircuitBreaker(config)
        test_pass("Circuit breaker initialization")
        
        # Test position sizing
        from main.risk_management.position_sizing import PositionSizer
        
        sizer = PositionSizer(config)
        
        # Test equal weight sizing
        portfolio_value = 100000
        num_positions = 5
        size = sizer.calculate_equal_weight_size(portfolio_value, num_positions)
        
        if size == 20000:
            test_pass("Position sizing calculation")
        else:
            test_fail("Position sizing calculation", f"Expected 20000, got {size}")
            
    except Exception as e:
        test_fail("Risk management", str(e))


async def test_database_models():
    """Test 5: Database Models"""
    print("\n🔍 Testing Database Models...")
    
    try:
        from main.data_pipeline.storage.models import MarketData, NewsArticle
        
        # Test market data model
        market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(timezone.utc),
            interval='1day',
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000000
        )
        
        if market_data.symbol == 'AAPL':
            test_pass("MarketData model")
        else:
            test_fail("MarketData model", "Model creation failed")
            
        # Test news model
        news = NewsArticle(
            headline="Test News",
            timestamp=datetime.now(timezone.utc),
            source='test'
        )
        
        if news.headline == "Test News":
            test_pass("NewsArticle model")
        else:
            test_fail("NewsArticle model", "Model creation failed")
            
    except Exception as e:
        test_fail("Database models", str(e))


async def test_ensemble_strategy():
    """Test 6: Ensemble Strategy Attribution"""
    print("\n🔍 Testing Ensemble Strategy...")
    
    try:
        from main.models.strategies.ensemble import EnsembleMetaLearningStrategy
        
        # Create mock strategies
        strategies = {
            'test1': type('MockStrategy', (), {'generate_signals': lambda self, data: {'AAPL': 0.5}})(),
            'test2': type('MockStrategy', (), {'generate_signals': lambda self, data: {'AAPL': -0.3}})()
        }
        
        config = {
            'meta_lookback': 20,
            'reweight_frequency': 5,
            'min_strategy_weight': 0.0,
            'max_strategy_weight': 1.0
        }
        
        ensemble = EnsembleMetaLearningStrategy(config, strategies)
        
        # Test signal combination
        combined, attribution = ensemble._combine_signals({
            'test1': {'AAPL': 0.5},
            'test2': {'AAPL': -0.3}
        })
        
        if 'AAPL' in combined and 'AAPL' in attribution:
            test_pass("Ensemble signal combination")
        else:
            test_fail("Ensemble signal combination", "Missing signal or attribution")
            
    except Exception as e:
        test_fail("Ensemble strategy", str(e))


async def test_config_loading():
    """Test 7: Configuration Loading"""
    print("\n🔍 Testing Configuration Loading...")
    
    try:
        from main.config.config_manager import ModularConfigManager as ConfigManager
        
        # Test with default config
        config = get_config()
        
        if isinstance(config, dict):
            test_pass("Configuration loading")
        else:
            test_fail("Configuration loading", "Config is not a dictionary")
            
    except Exception as e:
        test_fail("Configuration loading", str(e))


async def test_monitoring_components():
    """Test 8: Monitoring Components"""
    print("\n🔍 Testing Monitoring Components...")
    
    try:
        from main.monitoring.logging.performance_logger import PerformanceLogger
        
        logger = PerformanceLogger({})
        test_pass("Performance logger initialization")
        
        from main.monitoring.alerts.unified_alerts import UnifiedAlertSystem
        
        alert_system = UnifiedAlertSystem({})
        test_pass("Alert system initialization")
        
    except Exception as e:
        test_fail("Monitoring components", str(e))


async def main():
    """Run all integration tests."""
    print("🚀 AI Trading System Integration Tests")
    print("=" * 50)
    
    # Run all tests
    await test_data_pipeline()
    await test_feature_engineering()
    await test_strategy_system()
    await test_risk_management()
    await test_database_models()
    await test_ensemble_strategy()
    await test_config_loading()
    await test_monitoring_components()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print("\n🔍 Error Details:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    print("\n" + "=" * 50)
    
    # Return exit code
    return 0 if test_results['failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)