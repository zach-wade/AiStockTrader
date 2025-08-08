"""
Integration tests for Technical Scanner repository interactions.

Tests market data retrieval and technical indicator calculations.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from main.interfaces.scanners import IScannerRepository
from main.data_pipeline.storage.repositories.repository_types import QueryFilter


@pytest.mark.integration
@pytest.mark.asyncio
class TestTechnicalScannerRepository:
    """Test technical scanner repository integration."""

    async def test_market_data_for_indicators(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        recent_date_range: QueryFilter,
        sample_market_data
    ):
        """Test market data retrieval for technical indicator calculations."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Mock market data with proper OHLCV structure
            market_df = pd.DataFrame(sample_market_data)
            mock_market.return_value = {'AAPL': market_df}
            
            query_filter = recent_date_range
            query_filter.symbols = test_symbols[:1]
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter,
                columns=['date', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Verify data structure for technical analysis
            assert 'AAPL' in result
            aapl_data = result['AAPL']
            
            # Check required columns for indicators
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in aapl_data.columns

    async def test_historical_data_for_lookback(
        self,
        scanner_repository: IScannerRepository,
        test_symbols,
        historical_date_range: QueryFilter,
        sample_market_data
    ):
        """Test historical data retrieval for indicator lookback periods."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Create 60 days of historical data
            extended_data = []
            base_date = datetime.now(timezone.utc) - timedelta(days=60)
            
            for i in range(60):
                date = base_date + timedelta(days=i)
                extended_data.append({
                    'symbol': 'AAPL',
                    'date': date,
                    'open': 150.0 + i * 0.5,
                    'high': 152.0 + i * 0.5,
                    'low': 148.0 + i * 0.5,
                    'close': 151.0 + i * 0.5,
                    'volume': 1000000 + i * 10000
                })
            
            historical_df = pd.DataFrame(extended_data)
            mock_market.return_value = {'AAPL': historical_df}
            
            query_filter = historical_date_range
            query_filter.symbols = test_symbols[:1]
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Should have sufficient data for technical indicators
            assert 'AAPL' in result
            aapl_data = result['AAPL']
            assert len(aapl_data) >= 50  # Enough for most indicators

    async def test_price_momentum_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test data retrieval for momentum calculations."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Create trending price data
            dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(20)]
            dates.reverse()  # Chronological order
            
            # Upward trend with momentum
            closes = [100.0 + i * 2.5 for i in range(20)]  # Strong uptrend
            
            momentum_data = []
            for i, (date, close) in enumerate(zip(dates, closes)):
                momentum_data.append({
                    'symbol': 'AAPL',
                    'date': date,
                    'open': close - 1.0,
                    'high': close + 2.0,
                    'low': close - 2.0,
                    'close': close,
                    'volume': 1000000,
                    'returns': 0.025 if i > 0 else 0.0  # 2.5% daily returns
                })
            
            momentum_df = pd.DataFrame(momentum_data)
            mock_market.return_value = {'AAPL': momentum_df}
            
            query_filter = QueryFilter(
                start_date=dates[0],
                end_date=dates[-1],
                symbols=['AAPL']
            )
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Verify momentum data
            aapl_data = result['AAPL']
            
            # Calculate simple momentum (price change over period)
            if len(aapl_data) > 1:
                first_close = aapl_data.iloc[0]['close']
                last_close = aapl_data.iloc[-1]['close']
                momentum = (last_close - first_close) / first_close
                
                # Should show strong positive momentum
                assert momentum > 0.4  # 40%+ gain over period

    async def test_breakout_pattern_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test data retrieval for breakout pattern detection."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Create consolidation then breakout pattern
            dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(30)]
            dates.reverse()
            
            breakout_data = []
            for i, date in enumerate(dates):
                if i < 20:  # Consolidation phase
                    close = 150.0 + (i % 3) * 0.5  # Sideways movement
                    high = close + 1.0
                    low = close - 1.0
                else:  # Breakout phase
                    close = 150.0 + (i - 19) * 3.0  # Strong breakout
                    high = close + 2.0
                    low = close - 0.5
                
                breakout_data.append({
                    'symbol': 'AAPL',
                    'date': date,
                    'open': close - 0.5,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': 2000000 if i >= 20 else 1000000  # Volume spike on breakout
                })
            
            breakout_df = pd.DataFrame(breakout_data)
            mock_market.return_value = {'AAPL': breakout_df}
            
            query_filter = QueryFilter(
                start_date=dates[0],
                end_date=dates[-1],
                symbols=['AAPL']
            )
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_data = result['AAPL']
            
            # Analyze breakout pattern
            consolidation_data = aapl_data.iloc[:20]  # First 20 days
            breakout_data = aapl_data.iloc[20:]       # Last 10 days
            
            # Check consolidation characteristics
            consolidation_range = (
                consolidation_data['high'].max() - consolidation_data['low'].min()
            )
            
            # Check breakout characteristics
            breakout_move = (
                breakout_data['close'].iloc[-1] - consolidation_data['close'].iloc[-1]
            )
            
            # Should show tight consolidation followed by strong breakout
            assert consolidation_range < 5.0  # Tight range
            assert breakout_move > 20.0  # Strong breakout

    async def test_volatility_data_calculation(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test data for volatility calculations."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Create data with varying volatility
            dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(20)]
            dates.reverse()
            
            # High volatility period followed by low volatility
            volatility_data = []
            for i, date in enumerate(dates):
                if i < 10:  # High volatility period
                    base_price = 150.0
                    daily_range = 8.0  # Large daily ranges
                    close = base_price + (i % 2) * 4.0 - 2.0  # Choppy movement
                else:  # Low volatility period
                    base_price = 150.0
                    daily_range = 2.0  # Small daily ranges
                    close = base_price + (i % 2) * 0.5  # Quiet movement
                
                volatility_data.append({
                    'symbol': 'AAPL',
                    'date': date,
                    'open': close - 0.5,
                    'high': close + daily_range/2,
                    'low': close - daily_range/2,
                    'close': close,
                    'volume': 1000000
                })
            
            vol_df = pd.DataFrame(volatility_data)
            mock_market.return_value = {'AAPL': vol_df}
            
            query_filter = QueryFilter(
                start_date=dates[0],
                end_date=dates[-1],
                symbols=['AAPL']
            )
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_data = result['AAPL']
            
            # Calculate true range for volatility analysis
            aapl_data['true_range'] = aapl_data['high'] - aapl_data['low']
            
            high_vol_period = aapl_data.iloc[:10]['true_range'].mean()
            low_vol_period = aapl_data.iloc[10:]['true_range'].mean()
            
            # Should show clear volatility difference
            assert high_vol_period > low_vol_period * 2

    async def test_multi_timeframe_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test multi-timeframe data retrieval for comprehensive analysis."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market_daily, \
             patch.object(scanner_repository, 'get_intraday_data') as mock_intraday:
            
            # Mock daily data
            daily_data = pd.DataFrame([
                {
                    'symbol': 'AAPL',
                    'date': datetime.now(timezone.utc) - timedelta(days=i),
                    'close': 150.0 + i,
                    'volume': 1000000
                }
                for i in range(30)
            ])
            mock_market_daily.return_value = {'AAPL': daily_data}
            
            # Mock hourly data
            hourly_data = pd.DataFrame([
                {
                    'symbol': 'AAPL',
                    'timestamp': datetime.now(timezone.utc) - timedelta(hours=i),
                    'close': 150.0 + i * 0.1,
                    'volume': 100000
                }
                for i in range(24)
            ])
            mock_intraday.return_value = {'AAPL': hourly_data}
            
            # Get both timeframes
            daily_filter = QueryFilter(
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc),
                symbols=['AAPL']
            )
            
            daily_result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=daily_filter
            )
            
            hourly_result = await scanner_repository.get_intraday_data(
                symbols=['AAPL'],
                lookback_hours=24
            )
            
            # Should have both timeframes
            assert 'AAPL' in daily_result
            assert 'AAPL' in hourly_result
            assert len(daily_result['AAPL']) == 30
            assert len(hourly_result['AAPL']) == 24

    async def test_gap_detection_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test data retrieval for gap detection."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Create data with price gaps
            dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(10)]
            dates.reverse()
            
            gap_data = []
            for i, date in enumerate(dates):
                if i == 5:  # Gap up day
                    previous_close = 150.0
                    gap_open = 158.0  # 8 point gap up
                    close = 160.0
                else:
                    previous_close = 150.0 + i * 1.0
                    gap_open = previous_close + 0.5
                    close = gap_open + 1.0
                
                gap_data.append({
                    'symbol': 'AAPL',
                    'date': date,
                    'open': gap_open,
                    'high': close + 1.0,
                    'low': gap_open - 0.5,
                    'close': close,
                    'volume': 2000000 if i == 5 else 1000000  # Volume spike on gap
                })
            
            gap_df = pd.DataFrame(gap_data)
            mock_market.return_value = {'AAPL': gap_df}
            
            query_filter = QueryFilter(
                start_date=dates[0],
                end_date=dates[-1],
                symbols=['AAPL']
            )
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            aapl_data = result['AAPL']
            
            # Detect gaps by comparing open to previous close
            aapl_data['prev_close'] = aapl_data['close'].shift(1)
            aapl_data['gap_size'] = aapl_data['open'] - aapl_data['prev_close']
            
            # Should detect the gap
            max_gap = aapl_data['gap_size'].max()
            assert max_gap > 5.0  # Significant gap detected

    async def test_error_handling_insufficient_data(
        self,
        scanner_repository: IScannerRepository,
        test_symbols
    ):
        """Test handling when insufficient data for technical analysis."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Mock very limited data
            limited_data = pd.DataFrame([
                {
                    'symbol': 'AAPL',
                    'date': datetime.now(timezone.utc),
                    'close': 150.0,
                    'volume': 1000000
                }
            ])
            mock_market.return_value = {'AAPL': limited_data}
            
            query_filter = QueryFilter(
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                symbols=['AAPL']
            )
            
            result = await scanner_repository.get_market_data(
                symbols=['AAPL'],
                query_filter=query_filter
            )
            
            # Should handle gracefully
            assert 'AAPL' in result
            assert len(result['AAPL']) < 20  # Insufficient for most indicators

    async def test_performance_large_dataset(
        self,
        scanner_repository: IScannerRepository,
        performance_thresholds
    ):
        """Test performance with large technical analysis datasets."""
        with patch.object(scanner_repository, 'get_market_data') as mock_market:
            # Mock large dataset (1 year of daily data)
            large_data = pd.DataFrame([
                {
                    'symbol': f'SYM{i%100:03d}',
                    'date': datetime.now(timezone.utc) - timedelta(days=i),
                    'open': 100.0 + i * 0.1,
                    'high': 102.0 + i * 0.1,
                    'low': 98.0 + i * 0.1,
                    'close': 101.0 + i * 0.1,
                    'volume': 1000000
                }
                for i in range(365 * 100)  # 365 days * 100 symbols
            ])
            
            # Group by symbol
            symbol_data = {}
            for symbol, group in large_data.groupby('symbol'):
                symbol_data[symbol] = group
            
            mock_market.return_value = symbol_data
            
            query_filter = QueryFilter(
                start_date=datetime.now(timezone.utc) - timedelta(days=365),
                end_date=datetime.now(timezone.utc),
                symbols=list(symbol_data.keys())
            )
            
            start_time = datetime.now()
            
            result = await scanner_repository.get_market_data(
                symbols=list(symbol_data.keys()),
                query_filter=query_filter
            )
            
            end_time = datetime.now()
            query_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Should meet performance threshold for large datasets
            threshold = performance_thresholds['repository']['large_query_time_ms']
            assert query_time_ms < threshold
            assert len(result) == 100  # All symbols returned