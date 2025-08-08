"""
Integration tests for Position Sizing algorithms.

Tests Kelly criterion, fixed fractional, and dynamic position sizing
with real market data and portfolio constraints.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock

from main.risk_management.position_sizing import (
    PositionSizer, SizingMethod, KellyCriterion,
    FixedFractional, VolatilityBased, DynamicSizing
)
from main.models.common import Signal, Position, MarketData
from main.backtesting.analysis.performance_metrics import PerformanceMetrics


@pytest.fixture
def market_data():
    """Create sample market data for testing."""
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk
    
    return {
        'AAPL': MarketData(
            symbol='AAPL',
            timestamp=dates,
            open=prices - 1,
            high=prices + 1,
            low=prices - 1.5,
            close=prices,
            volume=[1000000 + np.secure_randint(-200000, 200000) for _ in range(100)]
        ),
        'GOOGL': MarketData(
            symbol='GOOGL',
            timestamp=dates,
            open=prices * 25 - 10,
            high=prices * 25 + 10,
            low=prices * 25 - 15,
            close=prices * 25,
            volume=[500000 + np.secure_randint(-100000, 100000) for _ in range(100)]
        )
    }


@pytest.fixture
def portfolio_state():
    """Create mock portfolio state."""
    return {
        'total_equity': 100000.0,
        'cash': 50000.0,
        'positions': {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                avg_entry_price=150.0,
                current_price=155.0,
                market_value=15500.0,
                cost_basis=15000.0,
                unrealized_pnl=500.0,
                unrealized_pnl_pct=3.33,
                realized_pnl=0.0,
                side='long'
            )
        },
        'daily_returns': [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005],
        'win_rate': 0.55,
        'avg_win': 0.02,
        'avg_loss': -0.015
    }


@pytest.fixture
def position_sizer():
    """Create position sizer with test configuration."""
    config = {
        'max_position_size': 0.20,  # 20% max per position
        'max_portfolio_risk': 0.06,  # 6% total portfolio risk
        'default_method': 'kelly',
        'kelly_fraction': 0.25,  # Use 25% Kelly
        'min_position_size': 0.01,  # 1% minimum
        'volatility_target': 0.15,  # 15% annual volatility
        'use_stops': True,
        'stop_loss_pct': 0.02  # 2% stop loss
    }
    
    return PositionSizer(config)


class TestPositionSizingIntegration:
    """Test position sizing algorithms in realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_kelly_criterion_sizing(self, position_sizer, portfolio_state, market_data):
        """Test Kelly criterion position sizing."""
        kelly = KellyCriterion(fraction=0.25)
        
        # Calculate Kelly position size
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='buy',
            strength=0.8,
            confidence=0.7,
            metadata={
                'expected_return': 0.03,
                'win_probability': 0.60,
                'loss_probability': 0.40,
                'avg_win': 0.05,
                'avg_loss': -0.02
            }
        )
        
        position_size = await kelly.calculate_position_size(
            signal=signal,
            portfolio_value=portfolio_state['total_equity'],
            current_price=market_data['AAPL'].close[-1],
            portfolio_state=portfolio_state
        )
        
        # Verify Kelly calculation
        # Kelly % = (p*b - q) / b, where p=win prob, q=loss prob, b=win/loss ratio
        win_prob = 0.60
        loss_prob = 0.40
        win_loss_ratio = 0.05 / 0.02  # 2.5
        
        expected_kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        expected_kelly *= 0.25  # Kelly fraction
        
        actual_percentage = position_size['percentage']
        
        assert abs(actual_percentage - expected_kelly) < 0.01
        assert position_size['shares'] > 0
        assert position_size['dollar_amount'] <= portfolio_state['total_equity'] * 0.20  # Max position size
    
    @pytest.mark.asyncio
    async def test_fixed_fractional_sizing(self, position_sizer, portfolio_state):
        """Test fixed fractional position sizing."""
        fixed = FixedFractional(
            base_fraction=0.02,  # 2% per trade
            scale_by_confidence=True
        )
        
        # Test with different confidence levels
        confidence_levels = [0.5, 0.7, 0.9]
        sizes = []
        
        for confidence in confidence_levels:
            signal = Signal(
                timestamp=datetime.now(),
                symbol='GOOGL',
                direction='buy',
                strength=0.7,
                confidence=confidence
            )
            
            size = await fixed.calculate_position_size(
                signal=signal,
                portfolio_value=portfolio_state['total_equity'],
                current_price=2500.0,
                portfolio_state=portfolio_state
            )
            
            sizes.append(size)
        
        # Verify scaling by confidence
        assert sizes[0]['percentage'] < sizes[1]['percentage'] < sizes[2]['percentage']
        
        # Verify base fraction
        assert sizes[1]['percentage'] == 0.02 * 0.7  # Base * confidence
        
        # Verify risk limits
        for size in sizes:
            assert size['risk_amount'] <= portfolio_state['total_equity'] * 0.02
    
    @pytest.mark.asyncio
    async def test_volatility_based_sizing(self, position_sizer, market_data):
        """Test volatility-based position sizing."""
        vol_sizer = VolatilityBased(
            target_volatility=0.15,  # 15% annual
            lookback_days=20
        )
        
        # Calculate historical volatility
        prices = market_data['AAPL'].close
        returns = np.diff(prices) / prices[:-1]
        historical_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='buy',
            strength=0.8,
            confidence=0.75
        )
        
        size = await vol_sizer.calculate_position_size(
            signal=signal,
            portfolio_value=100000,
            current_price=prices[-1],
            market_data=market_data['AAPL'],
            portfolio_state={'total_equity': 100000}
        )
        
        # Position should be inverse to volatility
        # Higher vol = smaller position
        expected_scaling = min(0.15 / historical_vol, 1.0)
        base_position = 0.10  # 10% base
        
        expected_size = base_position * expected_scaling
        
        assert abs(size['percentage'] - expected_size) < 0.02
        assert size['volatility_adjustment'] == expected_scaling
    
    @pytest.mark.asyncio
    async def test_dynamic_sizing_with_regime(self, position_sizer, portfolio_state, market_data):
        """Test dynamic sizing based on market regime."""
        dynamic = DynamicSizing(
            base_method='kelly',
            regime_adjustments={
                'bull': 1.2,
                'bear': 0.5,
                'neutral': 1.0,
                'high_volatility': 0.3
            }
        )
        
        # Test different market regimes
        regimes = [
            ('bull', {'trend': 'up', 'volatility': 'normal'}),
            ('bear', {'trend': 'down', 'volatility': 'normal'}),
            ('high_volatility', {'trend': 'neutral', 'volatility': 'high'})
        ]
        
        results = []
        
        for regime_name, regime_data in regimes:
            signal = Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='buy',
                strength=0.7,
                confidence=0.7,
                metadata={'market_regime': regime_data}
            )
            
            size = await dynamic.calculate_position_size(
                signal=signal,
                portfolio_value=portfolio_state['total_equity'],
                current_price=155.0,
                portfolio_state=portfolio_state
            )
            
            results.append({
                'regime': regime_name,
                'size': size['percentage'],
                'adjustment': size.get('regime_adjustment', 1.0)
            })
        
        # Verify regime adjustments
        bull_size = results[0]['size']
        bear_size = results[1]['size']
        high_vol_size = results[2]['size']
        
        assert bull_size > bear_size  # Bull market = larger positions
        assert high_vol_size < bear_size  # High vol = smallest positions
        assert results[0]['adjustment'] == 1.2
        assert results[1]['adjustment'] == 0.5
        assert results[2]['adjustment'] == 0.3
    
    @pytest.mark.asyncio
    async def test_position_sizing_with_correlation(self, position_sizer, portfolio_state):
        """Test position sizing considering portfolio correlation."""
        # Add correlated positions to portfolio
        portfolio_state['positions']['SPY'] = Position(
            symbol='SPY',
            quantity=100,
            avg_entry_price=400.0,
            current_price=410.0,
            market_value=41000.0,
            cost_basis=40000.0,
            unrealized_pnl=1000.0,
            unrealized_pnl_pct=2.5,
            realized_pnl=0.0,
            side='long'
        )
        
        portfolio_state['correlations'] = {
            ('AAPL', 'SPY'): 0.8,
            ('AAPL', 'QQQ'): 0.9,
            ('SPY', 'QQQ'): 0.95
        }
        
        # Try to add highly correlated position
        signal = Signal(
            timestamp=datetime.now(),
            symbol='QQQ',
            direction='buy',
            strength=0.8,
            confidence=0.8
        )
        
        size = await position_sizer.calculate_size_with_correlation(
            signal=signal,
            portfolio_value=portfolio_state['total_equity'],
            current_price=300.0,
            portfolio_state=portfolio_state
        )
        
        # Size should be reduced due to high correlation
        assert size['correlation_adjustment'] < 1.0
        assert size['correlation_adjustment'] > 0.3  # Not too aggressive
        
        # Final size should be reduced
        base_size = 0.10  # 10% base
        adjusted_size = base_size * size['correlation_adjustment']
        
        assert abs(size['percentage'] - adjusted_size) < 0.02
    
    @pytest.mark.asyncio
    async def test_risk_parity_sizing(self, position_sizer, portfolio_state, market_data):
        """Test risk parity position sizing."""
        # Calculate risk contributions
        positions = portfolio_state['positions']
        total_risk = 0
        risk_contributions = {}
        
        for symbol, position in positions.items():
            # Simple risk calculation (could use VaR, etc.)
            position_risk = abs(position.market_value) * 0.02  # 2% risk
            risk_contributions[symbol] = position_risk
            total_risk += position_risk
        
        # New position should balance risk
        signal = Signal(
            timestamp=datetime.now(),
            symbol='MSFT',
            direction='buy',
            strength=0.7,
            confidence=0.75
        )
        
        size = await position_sizer.calculate_risk_parity_size(
            signal=signal,
            portfolio_value=portfolio_state['total_equity'],
            current_price=300.0,
            portfolio_state=portfolio_state,
            target_risk_contribution=0.25  # 25% of total risk
        )
        
        # Verify risk parity calculation
        new_position_value = size['dollar_amount']
        new_position_risk = new_position_value * 0.02
        
        # New position should contribute ~25% of total risk
        new_total_risk = total_risk + new_position_risk
        risk_contribution = new_position_risk / new_total_risk
        
        assert abs(risk_contribution - 0.25) < 0.05
    
    @pytest.mark.asyncio
    async def test_position_sizing_with_stops(self, position_sizer, portfolio_state):
        """Test position sizing with stop loss consideration."""
        # Configure stop loss
        stop_loss_pct = 0.02  # 2% stop
        max_risk_per_trade = 0.01  # 1% max risk per trade
        
        signal = Signal(
            timestamp=datetime.now(),
            symbol='TSLA',
            direction='buy',
            strength=0.8,
            confidence=0.8,
            metadata={
                'stop_loss': 0.02,
                'take_profit': 0.06
            }
        )
        
        current_price = 800.0
        
        size = await position_sizer.calculate_size_with_stops(
            signal=signal,
            portfolio_value=portfolio_state['total_equity'],
            current_price=current_price,
            max_risk=max_risk_per_trade
        )
        
        # Verify position sized for max risk
        position_value = size['dollar_amount']
        stop_loss_amount = position_value * stop_loss_pct
        
        # Risk should not exceed max risk per trade
        assert stop_loss_amount <= portfolio_state['total_equity'] * max_risk_per_trade
        
        # Calculate expected position size
        max_risk_dollars = portfolio_state['total_equity'] * max_risk_per_trade
        expected_position_value = max_risk_dollars / stop_loss_pct
        
        assert abs(position_value - expected_position_value) < 100
    
    @pytest.mark.asyncio
    async def test_multi_strategy_position_sizing(self, position_sizer, portfolio_state):
        """Test position sizing across multiple strategies."""
        # Define strategy allocations
        strategy_allocations = {
            'momentum': 0.40,
            'mean_reversion': 0.30,
            'arbitrage': 0.20,
            'events': 0.10
        }
        
        # Signals from different strategies
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='buy',
                strength=0.8,
                confidence=0.8,
                strategy='momentum'
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='GOOGL',
                direction='sell',
                strength=0.7,
                confidence=0.75,
                strategy='mean_reversion'
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='MSFT',
                direction='buy',
                strength=0.9,
                confidence=0.85,
                strategy='momentum'
            )
        ]
        
        # Calculate sizes respecting strategy allocations
        sized_positions = []
        
        for signal in signals:
            strategy = signal.strategy
            strategy_capital = portfolio_state['total_equity'] * strategy_allocations[strategy]
            
            # Count existing positions for this strategy
            strategy_positions = sum(
                1 for s in sized_positions 
                if s['strategy'] == strategy
            )
            
            # Allocate within strategy limit
            if strategy_positions == 0:
                available_capital = strategy_capital
            else:
                available_capital = strategy_capital / (strategy_positions + 1)
            
            size = await position_sizer.calculate_size_with_allocation(
                signal=signal,
                available_capital=available_capital,
                current_price=150.0  # Simplified
            )
            
            sized_positions.append({
                'symbol': signal.symbol,
                'strategy': strategy,
                'size': size['dollar_amount']
            })
        
        # Verify strategy allocations respected
        momentum_total = sum(
            p['size'] for p in sized_positions 
            if p['strategy'] == 'momentum'
        )
        
        mean_rev_total = sum(
            p['size'] for p in sized_positions 
            if p['strategy'] == 'mean_reversion'
        )
        
        total_equity = portfolio_state['total_equity']
        
        assert momentum_total <= total_equity * strategy_allocations['momentum'] * 1.1  # 10% buffer
        assert mean_rev_total <= total_equity * strategy_allocations['mean_reversion'] * 1.1