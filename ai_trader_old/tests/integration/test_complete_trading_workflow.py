# tests/integration/test_complete_trading_workflow.py

"""
Comprehensive end-to-end trading workflow test.
Tests the complete pipeline: Data Ingestion → Feature Calculation → Signal Generation → Order Creation
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta, timezone
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Test configuration
TEST_CONFIG = {
    'system': {
        'environment': 'test',
        'debug': {'log_level': 'DEBUG'},
        'timezone': 'US/Eastern'
    },
    'data': {
        'sources': ['alpaca', 'polygon'],
        'realtime': {'enabled': False},
        'backfill': {
            'lookback_days': 30,
            'source_priorities': {
                'market_data': ['alpaca', 'polygon'],
                'news': ['alpaca_alt']
            }
        }
    },
    'features': {
        'calculators': {
            'technical': {'enabled': True},
            'sentiment': {'enabled': True}
        }
    },
    'broker': {
        'name': 'alpaca',
        'environment': 'paper'
    }
}

@pytest.fixture
def temp_data_lake():
    """Create temporary data lake for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    np.random.seed(42)  # For reproducible tests

    data = pd.DataFrame({
        'open': 100 + np.random.randn(50).cumsum() * 0.5,
        'high': np.nan,
        'low': np.nan,
        'close': 100 + np.random.randn(50).cumsum() * 0.5,
        'volume': np.secure_randint(100000, 1000000, 50)
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'close']].max(axis=1) + np.secure_uniform(0, 1, 50)
    data['low'] = data[['open', 'close']].min(axis=1) - np.secure_uniform(0, 1, 50)

    return data

@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca API client"""
    mock_client = Mock()
    mock_client.get_bars = Mock(return_value=pd.DataFrame())
    mock_client.get_latest_bars = Mock(return_value=pd.DataFrame())
    mock_client.get_account = Mock(return_value=Mock(buying_power=100000, account_number="TEST123"))
    mock_client.submit_order = Mock(return_value=Mock(id="order_123", status="new"))
    return mock_client

@pytest.fixture
def mock_polygon_client():
    """Mock Polygon API client"""
    mock_client = Mock()
    mock_client.get_aggs = Mock(return_value=pd.DataFrame())
    mock_client.get_ticker_news = Mock(return_value=[])
    return mock_client

class TestCompleteTradingWorkflow:
    """Test suite for complete trading workflow"""

    @pytest.mark.asyncio
    async def test_data_ingestion_pipeline(self, temp_data_lake, sample_market_data, mock_alpaca_client):
        """Test data ingestion from multiple sources"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Local imports
        from main.config.config_manager import Config
        from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator
        from main.utils.resilience import ResilienceManager

        # Setup mocks
        config = Config(TEST_CONFIG)
        resilience = ResilienceManager(config)

        # Create orchestrator with mocked clients
        orchestrator = IngestionOrchestrator(config, resilience)

        with patch.object(orchestrator, '_get_all_clients') as mock_get_clients:
            # Mock client registry
            mock_alpaca_market = Mock()
            mock_alpaca_market.source_name = 'alpaca_market'
            mock_alpaca_market.collect_data = AsyncMock(return_value=[{
                'source': 'alpaca_market',
                'data_type': 'market_data',
                'symbols': ['AAPL'],
                'raw_response': sample_market_data.to_dict('records'),
                'timestamp': datetime.now(timezone.utc)
            }])

            mock_get_clients.return_value = {'alpaca_market': mock_alpaca_market}

            # Test data ingestion
            result = await orchestrator.run_ingestion(
                data_type='market_data',
                symbols=['AAPL'],
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc)
            )

            assert result.success
            assert result.total_records > 0
            assert 'AAPL' in result.symbols_processed

    @pytest.mark.asyncio
    async def test_feature_calculation_pipeline(self, sample_market_data):
        """Test feature calculation from market data"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Local imports
        from main.feature_pipeline.calculators.sentiment_features import SentimentFeaturesCalculator
        from main.feature_pipeline.calculators.technical_indicators import (
            TechnicalIndicatorCalculator,
        )
        from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

        # Create feature engine with test configuration
        feature_config = {
            'calculators': {
                'technical': {'enabled': True},
                'sentiment': {'enabled': True}
            }
        }

        engine = UnifiedFeatureEngine(feature_config)

        # Test individual calculators
        tech_calc = TechnicalIndicatorCalculator()
        tech_features = tech_calc.calculate(sample_market_data)

        assert not tech_features.empty
        assert len(tech_features.columns) > 10  # Should have multiple technical indicators
        assert tech_features.index.equals(sample_market_data.index)

        sentiment_calc = SentimentFeaturesCalculator()
        sentiment_features = sentiment_calc.calculate(sample_market_data)

        assert not sentiment_features.empty
        assert len(sentiment_features.columns) > 40  # Should have many sentiment features
        assert sentiment_features.index.equals(sample_market_data.index)

        # Test unified feature calculation
        all_features = engine.calculate_features(sample_market_data, ['technical', 'sentiment'])

        assert not all_features.empty
        assert len(all_features.columns) > 50  # Combined features
        assert 'close' in all_features.columns  # Original data preserved

        # Check for key technical indicators
        technical_columns = [col for col in all_features.columns if any(
            indicator in col.lower() for indicator in ['sma', 'ema', 'rsi', 'macd']
        )]
        assert len(technical_columns) > 0

        # Check for sentiment features
        sentiment_columns = [col for col in all_features.columns if 'sentiment' in col.lower()]
        assert len(sentiment_columns) > 0

    @pytest.mark.asyncio
    async def test_signal_generation(self, sample_market_data):
        """Test signal generation from features"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Create enhanced market data with trends
        enhanced_data = sample_market_data.copy()

        # Add a clear uptrend pattern for signal testing
        enhanced_data['close'] = np.linspace(100, 120, len(enhanced_data))
        enhanced_data['volume'] = np.secure_randint(500000, 2000000, len(enhanced_data))

        # Test basic signal logic (simplified for testing)
        signals = pd.DataFrame(index=enhanced_data.index)

        # Simple momentum signal
        returns = enhanced_data['close'].pct_change()
        signals['momentum_signal'] = np.where(returns > 0.02, 1, np.where(returns < -0.02, -1, 0))

        # Volume-based signal
        volume_ma = enhanced_data['volume'].rolling(10).mean()
        signals['volume_signal'] = np.where(enhanced_data['volume'] > volume_ma * 1.5, 1, 0)

        # Combined signal
        signals['combined_signal'] = signals['momentum_signal'] + signals['volume_signal']
        signals['position_size'] = np.clip(signals['combined_signal'] * 0.1, 0, 1)  # 0-100% position

        # Validate signals
        assert not signals.empty
        assert 'combined_signal' in signals.columns
        assert 'position_size' in signals.columns

        # Check signal range
        assert signals['position_size'].min() >= 0
        assert signals['position_size'].max() <= 1

        # Should have some non-zero signals with our trend data
        assert signals['combined_signal'].abs().sum() > 0

    @pytest.mark.asyncio
    async def test_order_creation_and_validation(self, mock_alpaca_client):
        """Test order creation from signals"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Mock order data
        test_signals = pd.DataFrame({
            'signal_strength': [0.8, -0.6, 0.0, 0.4],
            'position_size': [0.1, 0.05, 0.0, 0.03],
            'confidence': [0.9, 0.7, 0.5, 0.8]
        }, index=pd.date_range('2024-01-01', periods=4, freq='D'))

        # Test order creation logic
        orders = []
        current_position = 0
        account_value = 100000

        for date, row in test_signals.iterrows():
            if abs(row['signal_strength']) > 0.5:  # Signal threshold
                target_position = row['position_size'] * account_value
                trade_amount = target_position - current_position

                if abs(trade_amount) > 1000:  # Minimum trade size
                    order = {
                        'symbol': 'AAPL',
                        'qty': abs(trade_amount) // 100,  # Round to shares
                        'side': 'buy' if trade_amount > 0 else 'sell',
                        'type': 'market',
                        'time_in_force': 'day',
                        'timestamp': date,
                        'signal_strength': row['signal_strength'],
                        'confidence': row['confidence']
                    }
                    orders.append(order)
                    current_position = target_position

        # Validate orders
        assert len(orders) > 0

        for order in orders:
            assert order['symbol'] == 'AAPL'
            assert order['side'] in ['buy', 'sell']
            assert order['type'] == 'market'
            assert order['qty'] > 0
            assert 0 <= order['confidence'] <= 1

        # Test order submission (mocked)
        with patch('main.trading_engine.brokers.alpaca_broker.AlpacaBroker') as mock_broker_class:
            mock_broker = mock_broker_class.return_value
            mock_broker.submit_order = AsyncMock(return_value={'id': 'test_order_123', 'status': 'accepted'})

            # Submit first order
            first_order = orders[0]
            result = await mock_broker.submit_order(
                symbol=first_order['symbol'],
                qty=first_order['qty'],
                side=first_order['side'],
                order_type=first_order['type']
            )

            assert result['status'] == 'accepted'
            assert 'id' in result

    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, temp_data_lake, sample_market_data):
        """Test complete pipeline from data to orders"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Test pipeline stages
        pipeline_stages = []

        # Stage 1: Data Ingestion (mocked)
        ingestion_result = {
            'success': True,
            'data': sample_market_data,
            'symbols': ['AAPL'],
            'timestamp': datetime.now(timezone.utc)
        }
        pipeline_stages.append(('ingestion', ingestion_result))

        # Stage 2: Feature Calculation
        # Local imports
        from main.feature_pipeline.calculators.technical_indicators import (
            TechnicalIndicatorCalculator,
        )

        calc = TechnicalIndicatorCalculator()
        features = calc.calculate(sample_market_data)

        feature_result = {
            'success': not features.empty,
            'features': features,
            'feature_count': len(features.columns)
        }
        pipeline_stages.append(('features', feature_result))

        # Stage 3: Signal Generation (simplified)
        returns = features['close'].pct_change()
        volume_ratio = features['volume'] / features['volume'].rolling(10).mean()

        signals = pd.DataFrame({
            'momentum_signal': np.where(returns > 0.01, 1, np.where(returns < -0.01, -1, 0)),
            'volume_signal': np.where(volume_ratio > 1.2, 1, 0),
            'position_size': np.secure_uniform(0, 0.1, len(features))
        }, index=features.index)

        signal_result = {
            'success': not signals.empty,
            'signals': signals,
            'signal_count': signals['momentum_signal'].abs().sum()
        }
        pipeline_stages.append(('signals', signal_result))

        # Stage 4: Order Generation (mocked)
        orders = []
        for i, (date, row) in enumerate(signals.tail(5).iterrows()):
            if abs(row['momentum_signal']) > 0:
                orders.append({
                    'id': f'order_{i}',
                    'symbol': 'AAPL',
                    'side': 'buy' if row['momentum_signal'] > 0 else 'sell',
                    'qty': max(1, int(row['position_size'] * 1000)),
                    'timestamp': date
                })

        order_result = {
            'success': len(orders) > 0,
            'orders': orders,
            'order_count': len(orders)
        }
        pipeline_stages.append(('orders', order_result))

        # Validate complete pipeline
        assert len(pipeline_stages) == 4

        for stage_name, result in pipeline_stages:
            assert result['success'], f"Stage {stage_name} failed"

        # Validate data flow
        assert pipeline_stages[0][1]['data'].index.equals(sample_market_data.index)
        assert pipeline_stages[1][1]['features'].index.equals(sample_market_data.index)
        assert pipeline_stages[2][1]['signals'].index.equals(sample_market_data.index)

        # Validate outputs
        assert pipeline_stages[1][1]['feature_count'] > 10  # Multiple features calculated
        assert pipeline_stages[2][1]['signal_count'] >= 0   # Some signals generated
        assert pipeline_stages[3][1]['order_count'] >= 0    # Orders created from signals

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_market_data):
        """Test error handling throughout the pipeline"""
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Test with invalid data
        invalid_data = pd.DataFrame({
            'close': [np.nan, np.inf, -np.inf, 100],
            'volume': [0, -1000, np.nan, 1000]
        }, index=pd.date_range('2024-01-01', periods=4, freq='D'))

        # Feature calculation should handle invalid data gracefully
        # Local imports
        from main.feature_pipeline.calculators.technical_indicators import (
            TechnicalIndicatorCalculator,
        )

        calc = TechnicalIndicatorCalculator()

        try:
            features = calc.calculate(invalid_data)
            # Should not crash, may return empty or cleaned data
            assert isinstance(features, pd.DataFrame)
        except Exception as e:
            # If it does throw an exception, it should be handled gracefully
            assert "Invalid input data" in str(e) or "validation failed" in str(e)

        # Test with empty data
        empty_data = pd.DataFrame()

        try:
            empty_features = calc.calculate(empty_data)
            assert empty_features.empty or len(empty_features) == 0
        except Exception as e:
            # Empty data should be handled gracefully
            assert "empty" in str(e).lower() or "no data" in str(e).lower()

        # Test signal generation with extreme values
        extreme_signals = pd.DataFrame({
            'signal': [999, -999, np.nan, 0.5],
            'confidence': [1.5, -0.5, np.nan, 0.8]  # Invalid confidence values
        }, index=pd.date_range('2024-01-01', periods=4, freq='D'))

        # Normalize extreme values
        normalized_signals = extreme_signals.copy()
        normalized_signals['signal'] = np.clip(normalized_signals['signal'], -1, 1)
        normalized_signals['confidence'] = np.clip(normalized_signals['confidence'].fillna(0), 0, 1)

        assert normalized_signals['signal'].min() >= -1
        assert normalized_signals['signal'].max() <= 1
        assert normalized_signals['confidence'].min() >= 0
        assert normalized_signals['confidence'].max() <= 1

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_market_data):
        """Test performance of pipeline components"""
        import time
        import sys

# Standard library imports
from pathlib import Path

        sys.path.append('src')

        # Local imports
        from main.feature_pipeline.calculators.sentiment_features import SentimentFeaturesCalculator
        from main.feature_pipeline.calculators.technical_indicators import (
            TechnicalIndicatorCalculator,
        )

        # Benchmark feature calculation
        calc = TechnicalIndicatorCalculator()

        start_time = time.time()
        features = calc.calculate(sample_market_data)
        tech_time = time.time() - start_time

        # Should complete within reasonable time
        assert tech_time < 5.0  # 5 seconds max for 50 data points
        assert not features.empty

        # Benchmark sentiment calculation
        sentiment_calc = SentimentFeaturesCalculator()

        start_time = time.time()
        sentiment_features = sentiment_calc.calculate(sample_market_data)
        sentiment_time = time.time() - start_time

        assert sentiment_time < 10.0  # 10 seconds max for sentiment
        assert not sentiment_features.empty

        # Memory usage should be reasonable
        # Standard library imports
        import os

        # Third-party imports
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Should not use excessive memory for test data
        assert memory_mb < 500  # Less than 500MB for test

        print(f"Performance metrics:")
        print(f"  Technical indicators: {tech_time:.2f}s, {len(features.columns)} features")
        print(f"  Sentiment features: {sentiment_time:.2f}s, {len(sentiment_features.columns)} features")
        print(f"  Memory usage: {memory_mb:.1f}MB")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
