"""
Basic Trading Engine Tests
Tests core trading engine functionality with simplified interface.
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

from main.config.config_manager import get_config


class TestTradingEngineBasic:
    """Test basic trading engine components with mocked dependencies."""
    
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
    
    @pytest.mark.asyncio
    async def test_config_initialization(self, test_config):
        """Test that configuration is properly loaded."""
        assert test_config.get('trading.execution_mode') == 'paper'
        assert test_config.get('trading.auto_execution_threshold') == 0.8
        assert test_config.get('risk.max_position_size') == 0.1
        assert test_config.get('broker.paper_trading') is True
    
    @pytest.mark.asyncio
    async def test_mock_broker_basic_functionality(self):
        """Test basic mock broker functionality."""
        from main.tests.fixtures.mock_broker import MockBroker
        
        config = {'test': True}
        broker = MockBroker(config)
        
        # Test connection
        assert await broker.connect()
        assert broker.is_connected()
        
        # Test order submission
        order_params = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'market'
        }
        
        try:
            result = await broker.submit_order(**order_params)
            assert 'order_id' in result
            assert result['status'] == 'submitted'
        except ImportError:
            # Skip if dependencies are missing
            pytest.skip("Mock broker dependencies not available")
        
        # Test disconnection
        assert await broker.disconnect()
        assert not broker.is_connected()
    
    @pytest.mark.asyncio
    async def test_order_validation_logic(self):
        """Test order validation without full broker integration."""
        # Test basic validation rules
        
        def validate_order_params(**params):
            """Simple order validation function."""
            errors = []
            
            if 'symbol' not in params or not params['symbol']:
                errors.append("Symbol is required")
            
            if 'quantity' not in params or params['quantity'] <= 0:
                errors.append("Quantity must be positive")
            
            if 'side' not in params or params['side'] not in ['buy', 'sell']:
                errors.append("Side must be 'buy' or 'sell'")
            
            return errors
        
        # Valid order
        valid_order = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'market'
        }
        
        errors = validate_order_params(**valid_order)
        assert len(errors) == 0
        
        # Invalid orders
        invalid_cases = [
            {'symbol': '', 'side': 'buy', 'quantity': 100},  # Empty symbol
            {'symbol': 'AAPL', 'side': 'buy', 'quantity': -10},  # Negative quantity
            {'symbol': 'AAPL', 'side': 'invalid', 'quantity': 100},  # Invalid side
            {'symbol': 'AAPL', 'side': 'buy'},  # Missing quantity
        ]
        
        for invalid_params in invalid_cases:
            errors = validate_order_params(**invalid_params)
            assert len(errors) > 0
    
    @pytest.mark.asyncio
    async def test_position_calculation_logic(self):
        """Test position calculation logic."""
        
        class SimplePosition:
            def __init__(self):
                self.quantity = 0
                self.average_price = 0.0
                self.market_value = 0.0
                self.unrealized_pnl = 0.0
            
            def add_trade(self, quantity: int, price: float):
                """Add a trade to the position."""
                if self.quantity == 0:
                    self.quantity = quantity
                    self.average_price = price
                else:
                    total_cost = self.quantity * self.average_price
                    new_cost = quantity * price
                    new_quantity = self.quantity + quantity
                    
                    if new_quantity != 0:
                        self.average_price = (total_cost + new_cost) / new_quantity
                    self.quantity = new_quantity
            
            def update_market_value(self, current_price: float):
                """Update market value and unrealized P&L."""
                self.market_value = abs(self.quantity) * current_price
                if self.quantity != 0:
                    self.unrealized_pnl = self.quantity * (current_price - self.average_price)
        
        # Test position building
        position = SimplePosition()
        
        # First trade: Buy 100 shares at $150
        position.add_trade(100, 150.0)
        assert position.quantity == 100
        assert position.average_price == 150.0
        
        # Second trade: Buy 50 more shares at $160
        position.add_trade(50, 160.0)
        assert position.quantity == 150
        # Average price should be weighted: (100*150 + 50*160) / 150 = 153.33
        assert abs(position.average_price - 153.33) < 0.01
        
        # Update market value
        current_price = 155.0
        position.update_market_value(current_price)
        assert position.market_value == 150 * 155.0
        # Unrealized P&L: 150 * (155 - 153.33) = 250
        assert abs(position.unrealized_pnl - 250.0) < 1.0
        
        # Sell some shares
        position.add_trade(-50, 158.0)
        assert position.quantity == 100
        # Average price should remain the same for partial sales
        assert abs(position.average_price - 153.33) < 0.01
    
    @pytest.mark.asyncio
    async def test_risk_check_logic(self):
        """Test basic risk check logic."""
        
        def check_position_risk(new_quantity: int, current_positions: Dict[str, int], 
                              max_position_size: int) -> Dict[str, Any]:
            """Simple position risk check."""
            result = {'approved': True, 'reason': ''}
            
            if abs(new_quantity) > max_position_size:
                result['approved'] = False
                result['reason'] = f"Position size {abs(new_quantity)} exceeds limit {max_position_size}"
            
            total_exposure = sum(abs(pos) for pos in current_positions.values()) + abs(new_quantity)
            max_total_exposure = max_position_size * 5  # Allow 5x max individual position
            
            if total_exposure > max_total_exposure:
                result['approved'] = False
                result['reason'] = f"Total exposure {total_exposure} exceeds limit {max_total_exposure}"
            
            return result
        
        # Test valid position
        current_positions = {'AAPL': 100, 'GOOGL': 50}
        max_position_size = 200
        
        result = check_position_risk(150, current_positions, max_position_size)
        assert result['approved'] is True
        
        # Test position size limit
        result = check_position_risk(250, current_positions, max_position_size)
        assert result['approved'] is False
        assert 'exceeds limit' in result['reason']
        
        # Test total exposure limit
        large_positions = {'AAPL': 200, 'GOOGL': 200, 'MSFT': 200, 'TSLA': 200}
        result = check_position_risk(150, large_positions, max_position_size)
        assert result['approved'] is False
        assert 'Total exposure' in result['reason']
    
    @pytest.mark.asyncio
    async def test_signal_lifecycle(self):
        """Test trading signal lifecycle management."""
        
        class TradingSignal:
            def __init__(self, symbol: str, side: str, quantity: int, confidence: float):
                self.symbol = symbol
                self.side = side
                self.quantity = quantity
                self.confidence = confidence
                self.status = 'pending'
                self.created_at = datetime.now()
                self.signal_id = f"signal_{hash((symbol, side, quantity, confidence))}"
            
            def approve(self):
                if self.status == 'pending':
                    self.status = 'approved'
                else:
                    raise ValueError(f"Cannot approve signal in status {self.status}")
            
            def execute(self, order_id: str):
                if self.status == 'approved':
                    self.status = 'executed'
                    self.order_id = order_id
                else:
                    raise ValueError(f"Cannot execute signal in status {self.status}")
            
            def reject(self, reason: str):
                if self.status in ['pending', 'approved']:
                    self.status = 'rejected'
                    self.rejection_reason = reason
                else:
                    raise ValueError(f"Cannot reject signal in status {self.status}")
            
            def is_expired(self, timeout: timedelta) -> bool:
                return datetime.now() - self.created_at > timeout
        
        # Test signal creation
        signal = TradingSignal('AAPL', 'buy', 100, 0.85)
        assert signal.status == 'pending'
        assert signal.confidence == 0.85
        
        # Test approval
        signal.approve()
        assert signal.status == 'approved'
        
        # Test execution
        signal.execute('order_123')
        assert signal.status == 'executed'
        assert signal.order_id == 'order_123'
        
        # Test invalid state transitions
        new_signal = TradingSignal('GOOGL', 'sell', 50, 0.7)
        
        # Cannot execute without approval
        with pytest.raises(ValueError):
            new_signal.execute('order_456')
        
        # Test rejection
        another_signal = TradingSignal('MSFT', 'buy', 75, 0.6)
        another_signal.reject('Risk limit exceeded')
        assert another_signal.status == 'rejected'
        assert another_signal.rejection_reason == 'Risk limit exceeded'
        
        # Test expiry
        old_signal = TradingSignal('TSLA', 'sell', 25, 0.9)
        old_signal.created_at = datetime.now() - timedelta(minutes=31)
        assert old_signal.is_expired(timedelta(minutes=30))
    
    @pytest.mark.asyncio
    async def test_execution_mode_behavior(self):
        """Test different execution mode behaviors."""
        
        class ExecutionController:
            def __init__(self, mode: str, auto_threshold: float = 0.8):
                self.mode = mode
                self.auto_threshold = auto_threshold
            
            def should_auto_execute(self, signal) -> bool:
                """Determine if signal should be auto-executed."""
                if self.mode == 'manual':
                    return False
                elif self.mode == 'paper':
                    return signal.confidence >= self.auto_threshold
                elif self.mode == 'live':
                    return signal.confidence >= self.auto_threshold
                else:
                    return False
        
        # Test manual mode
        manual_controller = ExecutionController('manual')
        
        high_confidence_signal = Mock()
        high_confidence_signal.confidence = 0.95
        
        assert not manual_controller.should_auto_execute(high_confidence_signal)
        
        # Test paper mode with auto-execution
        paper_controller = ExecutionController('paper', auto_threshold=0.8)
        
        assert paper_controller.should_auto_execute(high_confidence_signal)  # 0.95 > 0.8
        
        low_confidence_signal = Mock()
        low_confidence_signal.confidence = 0.6
        
        assert not paper_controller.should_auto_execute(low_confidence_signal)  # 0.6 < 0.8
        
        # Test live mode
        live_controller = ExecutionController('live', auto_threshold=0.9)
        
        assert live_controller.should_auto_execute(high_confidence_signal)  # 0.95 > 0.9
        assert not live_controller.should_auto_execute(low_confidence_signal)  # 0.6 < 0.9
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self):
        """Test basic performance metrics calculation."""
        
        def calculate_execution_metrics(fills: List[Dict], benchmark_price: float) -> Dict[str, float]:
            """Calculate basic execution performance metrics."""
            if not fills:
                return {}
            
            total_quantity = sum(fill['quantity'] for fill in fills)
            weighted_price = sum(fill['quantity'] * fill['price'] for fill in fills) / total_quantity
            
            # Calculate slippage
            slippage_bps = ((weighted_price - benchmark_price) / benchmark_price) * 10000
            
            # Calculate other metrics
            fill_rate = len(fills) / len(fills)  # Simplified - would need total attempts
            total_commission = sum(fill.get('commission', 0) for fill in fills)
            
            return {
                'volume_weighted_price': weighted_price,
                'slippage_bps': slippage_bps,
                'fill_rate': fill_rate,
                'total_commission': total_commission,
                'total_quantity': total_quantity
            }
        
        # Test metrics calculation
        fills = [
            {'quantity': 60, 'price': 149.50, 'commission': 0.60},
            {'quantity': 40, 'price': 150.25, 'commission': 0.40}
        ]
        benchmark_price = 150.00
        
        metrics = calculate_execution_metrics(fills, benchmark_price)
        
        # Expected VWAP: (60 * 149.50 + 40 * 150.25) / 100 = 149.8
        expected_vwap = 149.8
        assert abs(metrics['volume_weighted_price'] - expected_vwap) < 0.1
        
        # Expected slippage: (149.83 - 150.00) / 150.00 * 10000 = -11.33 bps
        assert abs(metrics['slippage_bps'] + 11.33) < 0.1
        
        assert metrics['total_quantity'] == 100
        assert metrics['total_commission'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])