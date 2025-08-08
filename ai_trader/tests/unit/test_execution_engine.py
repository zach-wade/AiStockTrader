"""
Unit Tests for Trading Engine ExecutionEngine
Tests core execution logic, signal management, and risk integration.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from pathlib import Path
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
# Standardized test setup
sys.path.insert(0, str(Path(__file__).parent.parent)); from test_setup import setup_test_path
setup_test_path()

from main.trading_engine.core.execution_engine import ExecutionEngine, ExecutionMode, TradingSignal, SignalStatus
from main.trading_engine.core.order_manager import OrderManager
from main.trading_engine.core.portfolio_manager import PortfolioManager
from main.trading_engine.core.risk_manager import RiskManager
from main.models.common import Order, Position, OrderStatus, OrderType, OrderSide
from main.config.config_manager import get_config


class TestExecutionEngine:
    """Test ExecutionEngine core functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config_dict = {
            'trading': {
                'execution_mode': 'paper',
                'auto_execution_threshold': 0.8,
                'signal_timeout_minutes': 30,
                'max_concurrent_executions': 5
            },
            'risk': {
                'max_position_size': 0.1,
                'max_portfolio_risk': 0.05,
                'circuit_breaker_enabled': True
            },
            'broker': {
                'api_key': 'test_key',
                'secret_key': 'test_secret',
                'paper_trading': True
            }
        }
        return Config(config_dict)
    
    @pytest.fixture
    def mock_order_manager(self):
        """Create mock OrderManager."""
        order_manager = Mock(spec=OrderManager)
        order_manager.submit_new_order = AsyncMock()
        order_manager.cancel_order = AsyncMock()
        order_manager.get_orders = AsyncMock(return_value=[])
        return order_manager
    
    @pytest.fixture
    def mock_portfolio_manager(self):
        """Create mock PortfolioManager."""
        portfolio_manager = Mock(spec=PortfolioManager)
        portfolio_manager.get_position = Mock(return_value=None)
        portfolio_manager.get_buying_power = Mock(return_value=100000.0)
        portfolio_manager.get_total_value = Mock(return_value=100000.0)
        return portfolio_manager
    
    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock RiskManager."""
        risk_manager = Mock(spec=RiskManager)
        risk_manager.pre_trade_check = AsyncMock(return_value={'approved': True, 'reason': ''})
        risk_manager.is_circuit_breaker_active = Mock(return_value=False)
        return risk_manager
    
    @pytest.fixture
    def execution_engine(self, test_config, mock_order_manager, mock_portfolio_manager, mock_risk_manager):
        """Create ExecutionEngine with mocked dependencies."""
        with patch('trading_engine.core.execution_engine.OrderManager', return_value=mock_order_manager), \
             patch('trading_engine.core.execution_engine.PortfolioManager', return_value=mock_portfolio_manager), \
             patch('trading_engine.core.execution_engine.RiskManager', return_value=mock_risk_manager):
            
            engine = ExecutionEngine(test_config)
            engine.order_manager = mock_order_manager
            engine.portfolio_manager = mock_portfolio_manager
            engine.risk_manager = mock_risk_manager
            return engine
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_valid_parameters(self, execution_engine):
        """Test signal generation with valid parameters."""
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.85,
            'strategy': 'mean_reversion',
            'price_target': 150.0
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        assert signal is not None
        assert signal.symbol == 'AAPL'
        assert signal.side == 'buy'
        assert signal.quantity == 100
        assert signal.confidence == 0.85
        assert signal.status == SignalStatus.PENDING
        assert signal.strategy == 'mean_reversion'
        
        # Should be in pending signals
        assert signal.signal_id in execution_engine.pending_signals
    
    @pytest.mark.asyncio
    async def test_signal_generation_with_risk_manager_block(self, execution_engine, mock_risk_manager):
        """Test signal generation when risk manager blocks the trade."""
        # Configure risk manager to block the trade
        mock_risk_manager.pre_trade_check.return_value = {
            'approved': False,
            'reason': 'Position limit exceeded'
        }
        
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 1000,  # Large quantity to trigger limit
            'confidence': 0.85,
            'strategy': 'mean_reversion'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        assert signal is None
        assert len(execution_engine.pending_signals) == 0
        
        # Verify risk check was called
        mock_risk_manager.pre_trade_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_execution_high_confidence_signal(self, execution_engine, mock_order_manager):
        """Test automatic execution of high confidence signals."""
        execution_engine.execution_mode = ExecutionMode.PAPER
        
        # Mock successful order submission
        mock_order = Mock(spec=Order)
        mock_order.order_id = 'order_123'
        mock_order.status = OrderStatus.SUBMITTED
        mock_order_manager.submit_new_order.return_value = mock_order
        
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.9,  # High confidence for auto-execution
            'strategy': 'ml_momentum'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        # Give a moment for async auto-execution
        await asyncio.sleep(0.1)
        
        assert signal.status == SignalStatus.APPROVED
        mock_order_manager.submit_new_order.assert_called_once()
        
        # Check the order was submitted with correct parameters
        call_args = mock_order_manager.submit_new_order.call_args[1]
        assert call_args['symbol'] == 'AAPL'
        assert call_args['side'] == Side.BUY
        assert call_args['quantity'] == 100
    
    @pytest.mark.asyncio
    async def test_manual_execution_mode_requires_approval(self, execution_engine):
        """Test that manual mode requires explicit approval."""
        execution_engine.execution_mode = ExecutionMode.MANUAL
        
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.9,  # High confidence but in manual mode
            'strategy': 'mean_reversion'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        # Give a moment for any auto-execution attempt
        await asyncio.sleep(0.1)
        
        # Should remain pending in manual mode
        assert signal.status == SignalStatus.PENDING
        assert signal.signal_id in execution_engine.pending_signals
    
    @pytest.mark.asyncio
    async def test_signal_expiry_handling(self, execution_engine):
        """Test that signals expire after timeout."""
        execution_engine.signal_timeout = timedelta(seconds=1)  # Short timeout for testing
        
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.5,  # Low confidence to avoid auto-execution
            'strategy': 'mean_reversion'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        # Signal should initially be pending
        assert signal.status == SignalStatus.PENDING
        
        # Wait for expiry
        await asyncio.sleep(1.5)
        
        # Manually trigger expiry check (normally done by background task)
        await execution_engine._check_signal_expiry()
        
        # Signal should be expired and removed
        assert signal.status == SignalStatus.EXPIRED
        assert signal.signal_id not in execution_engine.pending_signals
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation_all_positions(self, execution_engine, mock_portfolio_manager, mock_order_manager):
        """Test emergency liquidation of all positions."""
        # Mock current positions
        positions = [
            Mock(spec=Position, symbol='AAPL', quantity=100, side='long'),
            Mock(spec=Position, symbol='GOOGL', quantity=50, side='long'),
            Mock(spec=Position, symbol='MSFT', quantity=-75, side='short')
        ]
        mock_portfolio_manager.get_all_positions.return_value = positions
        
        # Mock order submissions
        mock_order_manager.submit_new_order.return_value = Mock(spec=Order, order_id='emergency_123')
        
        result = await execution_engine.emergency_liquidate_all()
        
        assert result['success'] is True
        assert len(result['orders']) == 3  # One order per position
        
        # Verify orders were submitted to close each position
        assert mock_order_manager.submit_new_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execution_mode_switching(self, execution_engine):
        """Test switching between execution modes."""
        # Start in paper mode
        assert execution_engine.execution_mode == ExecutionMode.PAPER
        
        # Switch to manual
        execution_engine.set_execution_mode(ExecutionMode.MANUAL)
        assert execution_engine.execution_mode == ExecutionMode.MANUAL
        
        # Switch to live (with confirmation)
        with patch('builtins.input', return_value='yes'):
            execution_engine.set_execution_mode(ExecutionMode.LIVE)
            assert execution_engine.execution_mode == ExecutionMode.LIVE
    
    @pytest.mark.asyncio
    async def test_signal_callback_registration(self, execution_engine):
        """Test signal status callback registration and triggering."""
        callback_calls = []
        
        def test_callback(signal: Signal):
            callback_calls.append((signal.signal_id, signal.status))
        
        execution_engine.register_signal_callback(test_callback)
        
        # Generate a signal
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.5,
            'strategy': 'test'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        # Callback should have been triggered for signal creation
        assert len(callback_calls) > 0
        assert callback_calls[0][0] == signal.signal_id
        assert callback_calls[0][1] == SignalStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_background_task_lifecycle(self, execution_engine):
        """Test background task startup and shutdown."""
        # Start background tasks
        await execution_engine.start_background_tasks()
        
        # Verify tasks are running
        assert execution_engine._signal_expiry_task is not None
        assert not execution_engine._signal_expiry_task.done()
        
        # Stop background tasks
        await execution_engine.stop_background_tasks()
        
        # Verify tasks are stopped
        assert execution_engine._signal_expiry_task.done()
    
    @pytest.mark.asyncio
    async def test_execution_with_circuit_breaker_active(self, execution_engine, mock_risk_manager):
        """Test that execution is blocked when circuit breaker is active."""
        # Activate circuit breaker
        mock_risk_manager.is_circuit_breaker_active.return_value = True
        
        signal_data = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'confidence': 0.95,  # Very high confidence
            'strategy': 'mean_reversion'
        }
        
        signal = await execution_engine.generate_signal(**signal_data)
        
        # Signal should be blocked due to circuit breaker
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_get_system_health(self, execution_engine, mock_portfolio_manager, mock_risk_manager):
        """Test system health reporting."""
        # Mock system state
        mock_portfolio_manager.get_total_value.return_value = 105000.0
        mock_risk_manager.get_risk_metrics.return_value = {
            'var_1d': 1500.0,
            'max_drawdown': 0.02
        }
        
        health = await execution_engine.get_system_health()
        
        assert 'execution_engine' in health
        assert 'pending_signals' in health['execution_engine']
        assert 'execution_mode' in health['execution_engine']
        assert 'portfolio_value' in health
        assert 'risk_metrics' in health
        
        # Verify values
        assert health['portfolio_value'] == 105000.0
        assert health['risk_metrics']['var_1d'] == 1500.0


class TestSignalLifecycle:
    """Test Signal object lifecycle and state transitions."""
    
    def test_signal_creation(self):
        """Test Signal object creation and initial state."""
        signal = Signal(
            symbol='AAPL',
            side='buy',
            quantity=100,
            confidence=0.8,
            strategy='test',
            metadata={'price_target': 150.0}
        )
        
        assert signal.symbol == 'AAPL'
        assert signal.side == 'buy'
        assert signal.quantity == 100
        assert signal.confidence == 0.8
        assert signal.status == SignalStatus.PENDING
        assert signal.strategy == 'test'
        assert signal.metadata['price_target'] == 150.0
        assert signal.created_at is not None
        assert signal.signal_id is not None
    
    def test_signal_state_transitions(self):
        """Test valid signal state transitions."""
        signal = Signal('AAPL', 'buy', 100, 0.8, 'test')
        
        # PENDING -> APPROVED
        signal.approve()
        assert signal.status == SignalStatus.APPROVED
        
        # APPROVED -> EXECUTED
        signal.execute('order_123')
        assert signal.status == SignalStatus.EXECUTED
        assert signal.order_id == 'order_123'
    
    def test_signal_invalid_transitions(self):
        """Test invalid signal state transitions are blocked."""
        signal = Signal('AAPL', 'buy', 100, 0.8, 'test')
        
        # Cannot go directly from main.PENDING to EXECUTED
        with pytest.raises(ValueError):
            signal.execute('order_123')
        
        # Cannot approve an expired signal
        signal.expire()
        with pytest.raises(ValueError):
            signal.approve()
    
    def test_signal_expiry(self):
        """Test signal expiry functionality."""
        signal = Signal('AAPL', 'buy', 100, 0.8, 'test')
        
        # Check if signal is expired (should not be initially)
        assert not signal.is_expired(timedelta(minutes=30))
        
        # Mock an old creation time
        signal.created_at = datetime.now() - timedelta(minutes=31)
        assert signal.is_expired(timedelta(minutes=30))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])