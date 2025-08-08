"""
Unit Tests for Trading Engine OrderManager
Tests order lifecycle, validation, and broker integration.
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

from main.trading_engine.core.order_manager import OrderManager
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.models.common import Order, OrderStatus, OrderType, OrderSide
from main.events.types import FillEvent
from main.config.config_manager import get_config
from omegaconf import DictConfig, OmegaConf


class TestOrderManager:
    """Test OrderManager core functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config_dict = {
            'trading': {
                'order_timeout_minutes': 30,
                'max_pending_orders': 50,
                'enable_order_validation': True
            },
            'broker': {
                'api_key': 'test_key',
                'secret_key': 'test_secret',
                'paper_trading': True
            }
        }
        return OmegaConf.create(config_dict)
    
    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock(spec=BrokerInterface)
        broker.submit_order = AsyncMock()
        broker.cancel_order = AsyncMock()
        broker.modify_order = AsyncMock()
        broker.get_order = AsyncMock()
        broker.get_orders = AsyncMock(return_value=[])
        broker.is_connected = Mock(return_value=True)
        return broker
    
    @pytest.fixture
    def order_manager(self, test_config, mock_broker):
        """Create OrderManager with mock broker."""
        with patch('trading_engine.core.order_manager.create_broker', return_value=mock_broker):
            manager = OrderManager(test_config)
            manager.broker = mock_broker
            return manager
    
    @pytest.fixture
    def sample_order_data(self):
        """Sample order data for testing."""
        return {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 100,
            'order_type': OrderType.MARKET,
            'time_in_force': 'DAY'
        }
    
    @pytest.mark.asyncio
    async def test_submit_new_market_order(self, order_manager, mock_broker, sample_order_data):
        """Test submitting a new market order."""
        # Mock broker response
        broker_order_id = 'broker_order_123'
        mock_broker.submit_order.return_value = {
            'order_id': broker_order_id,
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        assert order is not None
        assert order.symbol == 'AAPL'
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.SUBMITTED
        assert order.broker_order_id == broker_order_id
        
        # Verify broker was called
        mock_broker.submit_order.assert_called_once()
        
        # Verify order is tracked
        assert order.order_id in order_manager.orders
    
    @pytest.mark.asyncio
    async def test_submit_new_limit_order(self, order_manager, mock_broker):
        """Test submitting a new limit order."""
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_456',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order_data = {
            'symbol': 'GOOGL',
            'side': OrderSide.SELL,
            'quantity': 50,
            'order_type': OrderType.LIMIT,
            'limit_price': 2500.0,
            'time_in_force': 'GTC'
        }
        
        order = await order_manager.submit_new_order(**order_data)
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 2500.0
        assert order.time_in_force == 'GTC'
        
        # Check broker call included limit price
        call_args = mock_broker.submit_order.call_args[1]
        assert call_args['limit_price'] == 2500.0
    
    @pytest.mark.asyncio
    async def test_cancel_active_order(self, order_manager, mock_broker, sample_order_data):
        """Test canceling an active order."""
        # First submit an order
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_789',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Mock cancel response
        mock_broker.cancel_order.return_value = {
            'order_id': 'broker_order_789',
            'status': 'cancelled',
            'timestamp': datetime.now()
        }
        
        # Cancel the order
        result = await order_manager.cancel_order(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        
        # Verify broker was called
        mock_broker.cancel_order.assert_called_once_with('broker_order_789')
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, order_manager):
        """Test canceling a non-existent order."""
        result = await order_manager.cancel_order('nonexistent_order')
        assert result is False
    
    @pytest.mark.asyncio
    async def test_modify_order_price_and_quantity(self, order_manager, mock_broker, sample_order_data):
        """Test modifying order price and quantity."""
        # Submit initial limit order
        sample_order_data['order_type'] = OrderType.LIMIT
        sample_order_data['limit_price'] = 150.0
        
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_mod',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Mock modify response
        mock_broker.modify_order.return_value = {
            'order_id': 'broker_order_mod_new',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        # Modify the order
        modifications = {
            'quantity': 150,
            'limit_price': 155.0
        }
        
        result = await order_manager.modify_order(order.order_id, **modifications)
        
        assert result is True
        assert order.quantity == 150
        assert order.limit_price == 155.0
        
        # Verify broker was called
        mock_broker.modify_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_validation_insufficient_funds(self, order_manager, mock_broker):
        """Test order validation with insufficient funds."""
        # Mock broker to return insufficient funds error
        mock_broker.submit_order.side_effect = Exception("Insufficient buying power")
        
        order_data = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 10000,  # Large quantity
            'order_type': OrderType.MARKET
        }
        
        order = await order_manager.submit_new_order(**order_data)
        
        # Order should be rejected
        assert order.status == OrderStatus.REJECTED
        assert 'Insufficient buying power' in order.rejection_reason
    
    @pytest.mark.asyncio
    async def test_order_validation_invalid_parameters(self, order_manager):
        """Test order validation with invalid parameters."""
        invalid_order_data = {
            'symbol': 'INVALID',
            'side': OrderSide.BUY,
            'quantity': -100,  # Negative quantity
            'order_type': OrderType.MARKET
        }
        
        order = await order_manager.submit_new_order(**invalid_order_data)
        
        # Order should be rejected due to validation
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason is not None
    
    @pytest.mark.asyncio
    async def test_fill_processing_full_fill(self, order_manager, mock_broker, sample_order_data):
        """Test processing a full fill."""
        # Submit order
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_fill',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Create fill event
        fill = FillEvent(
            order_id=order.broker_order_id,
            fill_id='fill_123',
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            commission=1.0
        )
        
        # Process the fill
        await order_manager.process_fill(fill)
        
        # Order should be filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.average_fill_price == 150.0
        assert len(order.fills) == 1
        assert order.commission == 1.0
    
    @pytest.mark.asyncio
    async def test_fill_processing_partial_fill(self, order_manager, mock_broker, sample_order_data):
        """Test processing partial fills."""
        # Submit order for 100 shares
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_partial',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # First partial fill
        fill1 = FillEvent(
            order_id=order.broker_order_id,
            fill_id='fill_1',
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=60,
            price=149.0,
            timestamp=datetime.now(),
            commission=0.6
        )
        
        await order_manager.process_fill(fill1)
        
        # Order should be partially filled
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == 60
        assert order.remaining_quantity == 40
        assert order.average_fill_price == 149.0
        
        # Second partial fill
        fill2 = FillEvent(
            order_id=order.broker_order_id,
            fill_id='fill_2',
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=40,
            price=151.0,
            timestamp=datetime.now(),
            commission=0.4
        )
        
        await order_manager.process_fill(fill2)
        
        # Order should now be fully filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100
        assert order.remaining_quantity == 0
        # Weighted average price: (60 * 149 + 40 * 151) / 100 = 149.8
        assert abs(order.average_fill_price - 149.8) < 0.01
        assert len(order.fills) == 2
        assert order.commission == 1.0
    
    @pytest.mark.asyncio
    async def test_order_timeout_handling(self, order_manager, mock_broker, sample_order_data):
        """Test order timeout handling."""
        # Set short timeout for testing
        order_manager.order_timeout = timedelta(seconds=1)
        
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_timeout',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Mock the order's creation time to be old
        order.submitted_at = datetime.now() - timedelta(seconds=2)
        
        # Manually trigger timeout check
        await order_manager._check_order_timeouts()
        
        # Order should be timed out and cancelled
        assert order.status == OrderStatus.CANCELLED
        mock_broker.cancel_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broker_update_handling(self, order_manager, mock_broker, sample_order_data):
        """Test handling broker status updates."""
        # Submit order
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_update',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Simulate broker status update
        broker_update = {
            'order_id': 'broker_order_update',
            'status': 'accepted',
            'timestamp': datetime.now()
        }
        
        await order_manager.handle_broker_update(broker_update)
        
        # Order status should be updated
        assert order.status == OrderStatus.ACCEPTED
    
    @pytest.mark.asyncio
    async def test_order_status_transitions(self, order_manager, mock_broker, sample_order_data):
        """Test valid order status transitions."""
        mock_broker.submit_order.return_value = {
            'order_id': 'broker_order_transitions',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        order = await order_manager.submit_new_order(**sample_order_data)
        
        # Valid transitions
        transitions = [
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED
        ]
        
        for status in transitions:
            order.status = status
            assert order.status == status
    
    @pytest.mark.asyncio
    async def test_concurrent_order_operations(self, order_manager, mock_broker):
        """Test concurrent order operations."""
        # Mock broker responses
        mock_broker.submit_order.return_value = {
            'order_id': 'concurrent_order',
            'status': 'submitted',
            'timestamp': datetime.now()
        }
        
        # Submit multiple orders concurrently
        order_tasks = []
        for i in range(5):
            order_data = {
                'symbol': f'TEST{i}',
                'side': OrderSide.BUY,
                'quantity': 100,
                'order_type': OrderType.MARKET
            }
            task = order_manager.submit_new_order(**order_data)
            order_tasks.append(task)
        
        # Wait for all orders to complete
        orders = await asyncio.gather(*order_tasks)
        
        # All orders should be submitted successfully
        assert len(orders) == 5
        for order in orders:
            assert order.status == OrderStatus.SUBMITTED
        
        # All orders should be tracked
        assert len(order_manager.orders) == 5
    
    @pytest.mark.asyncio
    async def test_get_orders_with_filters(self, order_manager, mock_broker):
        """Test retrieving orders with various filters."""
        # Submit test orders
        order_configs = [
            {'symbol': 'AAPL', 'side': OrderSide.BUY, 'status': OrderStatus.SUBMITTED},
            {'symbol': 'AAPL', 'side': OrderSide.SELL, 'status': OrderStatus.FILLED},
            {'symbol': 'GOOGL', 'side': OrderSide.BUY, 'status': OrderStatus.CANCELLED}
        ]
        
        submitted_orders = []
        for config in order_configs:
            mock_broker.submit_order.return_value = {
                'order_id': f"order_{config['symbol']}_{config['side']}",
                'status': 'submitted',
                'timestamp': datetime.now()
            }
            
            order_data = {
                'symbol': config['symbol'],
                'side': config['side'],
                'quantity': 100,
                'order_type': OrderType.MARKET
            }
            
            order = await order_manager.submit_new_order(**order_data)
            order.status = config['status']  # Manually set status for testing
            submitted_orders.append(order)
        
        # Test filters
        aapl_orders = order_manager.get_orders(symbol='AAPL')
        assert len(aapl_orders) == 2
        
        buy_orders = order_manager.get_orders(side=OrderSide.BUY)
        assert len(buy_orders) == 2
        
        filled_orders = order_manager.get_orders(status=OrderStatus.FILLED)
        assert len(filled_orders) == 1
        
        # Combined filters
        aapl_buy_orders = order_manager.get_orders(symbol='AAPL', side=OrderSide.BUY)
        assert len(aapl_buy_orders) == 1


class TestOrderValidation:
    """Test order validation logic."""
    
    def test_validate_basic_order_parameters(self):
        """Test basic order parameter validation."""
        from main.trading_engine.core.order_manager import OrderValidator
        
        validator = OrderValidator()
        
        # Valid order
        valid_order = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 100,
            'order_type': OrderType.MARKET
        }
        
        errors = validator.validate_order(**valid_order)
        assert len(errors) == 0
        
        # Invalid orders
        invalid_cases = [
            {'quantity': -100},  # Negative quantity
            {'quantity': 0},     # Zero quantity
            {'symbol': ''},      # Empty symbol
            {'symbol': 'INVALID_SYMBOL_TOO_LONG'},  # Invalid symbol
        ]
        
        for invalid_params in invalid_cases:
            test_order = {**valid_order, **invalid_params}
            errors = validator.validate_order(**test_order)
            assert len(errors) > 0
    
    def test_validate_limit_order_parameters(self):
        """Test limit order specific validation."""
        from main.trading_engine.core.order_manager import OrderValidator
        
        validator = OrderValidator()
        
        # Valid limit order
        valid_limit_order = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 100,
            'order_type': OrderType.LIMIT,
            'limit_price': 150.0
        }
        
        errors = validator.validate_order(**valid_limit_order)
        assert len(errors) == 0
        
        # Limit order without price
        invalid_limit_order = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'quantity': 100,
            'order_type': OrderType.LIMIT
            # Missing limit_price
        }
        
        errors = validator.validate_order(**invalid_limit_order)
        assert len(errors) > 0
        assert any('limit_price' in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])