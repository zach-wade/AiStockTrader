"""
Integration tests for trading order lifecycle.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, List
from decimal import Decimal

from main.trading.trading_engine import TradingEngine
from main.trading.order_manager import OrderManager
from main.trading.portfolio_tracker import PortfolioTracker
from main.risk_management.pre_trade.risk_checker import PreTradeRiskChecker
from main.utils.exceptions import OrderError, RiskLimitError


@pytest.mark.integration
@pytest.mark.asyncio
class TestOrderLifecycle:
    """Test complete order lifecycle from creation to execution."""
    
    @pytest_asyncio.fixture
    async def trading_engine(self, test_config, test_db_pool):
        """Create trading engine instance."""
        engine = TradingEngine(test_config)
        engine.db_pool = test_db_pool
        
        # Initialize in paper trading mode
        test_config.trading.paper_trading = True
        
        await engine.initialize()
        yield engine
        await engine.shutdown()
    
    @pytest_asyncio.fixture
    async def portfolio_tracker(self, test_config, test_db_pool):
        """Create portfolio tracker instance."""
        tracker = PortfolioTracker(test_config)
        tracker.db_pool = test_db_pool
        
        # Set initial capital
        await tracker.initialize(initial_capital=100000.0)
        yield tracker
    
    async def test_order_creation_and_validation(
        self,
        trading_engine,
        portfolio_tracker
    ):
        """Test order creation with validation."""
        # Create buy order
        order_request = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'limit',
            'limit_price': 150.00,
            'time_in_force': 'day',
        }
        
        # Submit order
        order = await trading_engine.submit_order(order_request)
        
        # Verify order created
        assert order is not None
        assert order['order_id'] is not None
        assert order['status'] == 'pending'
        assert order['symbol'] == 'AAPL'
        assert order['quantity'] == 100
        assert order['limit_price'] == 150.00
        
        # Verify risk checks passed
        assert 'risk_checks' in order
        assert order['risk_checks']['passed'] == True
        assert 'position_size_check' in order['risk_checks']
        assert 'buying_power_check' in order['risk_checks']
    
    async def test_order_execution_flow(
        self,
        trading_engine,
        portfolio_tracker
    ):
        """Test complete order execution flow."""
        # Submit market order
        order_request = {
            'symbol': 'MSFT',
            'side': 'buy',
            'quantity': 50,
            'order_type': 'market',
        }
        
        order = await trading_engine.submit_order(order_request)
        order_id = order['order_id']
        
        # Simulate order fill
        fill_event = {
            'order_id': order_id,
            'fill_price': 300.50,
            'fill_quantity': 50,
            'fill_time': datetime.now(),
            'commission': 1.00,
        }
        
        await trading_engine.process_fill(fill_event)
        
        # Verify order status
        updated_order = await trading_engine.get_order(order_id)
        assert updated_order['status'] == 'filled'
        assert updated_order['fill_price'] == 300.50
        assert updated_order['fill_quantity'] == 50
        
        # Verify portfolio updated
        position = await portfolio_tracker.get_position('MSFT')
        assert position is not None
        assert position['quantity'] == 50
        assert position['avg_price'] == 300.50
        
        # Verify buying power reduced
        account = await portfolio_tracker.get_account_summary()
        expected_cost = (50 * 300.50) + 1.00  # quantity * price + commission
        assert account['buying_power'] == 100000.0 - expected_cost
    
    async def test_risk_limit_rejection(
        self,
        trading_engine,
        portfolio_tracker
    ):
        """Test order rejection due to risk limits."""
        # Try to buy too large position
        order_request = {
            'symbol': 'TSLA',
            'side': 'buy',
            'quantity': 1000,  # Too large
            'order_type': 'market',
        }
        
        # Should raise risk limit error
        with pytest.raises(RiskLimitError) as exc_info:
            await trading_engine.submit_order(order_request)
        
        assert 'position_size' in str(exc_info.value)
        
        # Verify no order created
        orders = await trading_engine.get_active_orders()
        tsla_orders = [o for o in orders if o['symbol'] == 'TSLA']
        assert len(tsla_orders) == 0
    
    async def test_order_cancellation(
        self,
        trading_engine
    ):
        """Test order cancellation flow."""
        # Submit limit order
        order_request = {
            'symbol': 'GOOGL',
            'side': 'buy',
            'quantity': 10,
            'order_type': 'limit',
            'limit_price': 2500.00,
        }
        
        order = await trading_engine.submit_order(order_request)
        order_id = order['order_id']
        
        # Cancel order
        cancel_result = await trading_engine.cancel_order(order_id)
        
        # Verify cancellation
        assert cancel_result['success'] == True
        assert cancel_result['order_id'] == order_id
        
        # Verify order status
        cancelled_order = await trading_engine.get_order(order_id)
        assert cancelled_order['status'] == 'cancelled'
        assert cancelled_order['cancelled_at'] is not None
    
    async def test_partial_fill_handling(
        self,
        trading_engine,
        portfolio_tracker
    ):
        """Test handling of partial order fills."""
        # Submit large limit order
        order_request = {
            'symbol': 'AMZN',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'limit',
            'limit_price': 3000.00,
        }
        
        order = await trading_engine.submit_order(order_request)
        order_id = order['order_id']
        
        # Simulate partial fill
        partial_fill = {
            'order_id': order_id,
            'fill_price': 3000.00,
            'fill_quantity': 30,  # Partial
            'fill_time': datetime.now(),
            'commission': 0.30,
        }
        
        await trading_engine.process_fill(partial_fill)
        
        # Verify order still open
        order_status = await trading_engine.get_order(order_id)
        assert order_status['status'] == 'partially_filled'
        assert order_status['filled_quantity'] == 30
        assert order_status['remaining_quantity'] == 70
        
        # Verify position
        position = await portfolio_tracker.get_position('AMZN')
        assert position['quantity'] == 30
        
        # Process remaining fill
        remaining_fill = {
            'order_id': order_id,
            'fill_price': 2999.50,
            'fill_quantity': 70,
            'fill_time': datetime.now(),
            'commission': 0.70,
        }
        
        await trading_engine.process_fill(remaining_fill)
        
        # Verify order fully filled
        final_order = await trading_engine.get_order(order_id)
        assert final_order['status'] == 'filled'
        assert final_order['filled_quantity'] == 100
        
        # Verify position updated
        final_position = await portfolio_tracker.get_position('AMZN')
        assert final_position['quantity'] == 100
        # Average price should be weighted
        expected_avg = (30 * 3000.00 + 70 * 2999.50) / 100
        assert abs(final_position['avg_price'] - expected_avg) < 0.01
    
    async def test_stop_loss_order_execution(
        self,
        trading_engine,
        portfolio_tracker
    ):
        """Test stop loss order execution."""
        # First, establish a position
        buy_order = {
            'symbol': 'META',
            'side': 'buy',
            'quantity': 100,
            'order_type': 'market',
        }
        
        initial_order = await trading_engine.submit_order(buy_order)
        
        # Simulate fill
        await trading_engine.process_fill({
            'order_id': initial_order['order_id'],
            'fill_price': 250.00,
            'fill_quantity': 100,
            'fill_time': datetime.now(),
            'commission': 1.00,
        })
        
        # Place stop loss order
        stop_loss_order = {
            'symbol': 'META',
            'side': 'sell',
            'quantity': 100,
            'order_type': 'stop',
            'stop_price': 240.00,  # 4% stop loss
        }
        
        sl_order = await trading_engine.submit_order(stop_loss_order)
        
        # Simulate price drop triggering stop loss
        trigger_event = {
            'symbol': 'META',
            'price': 239.50,
            'timestamp': datetime.now(),
        }
        
        await trading_engine.check_stop_orders(trigger_event)
        
        # Verify stop loss triggered
        sl_status = await trading_engine.get_order(sl_order['order_id'])
        assert sl_status['status'] in ['pending_execution', 'filled']
        
        # Simulate execution
        await trading_engine.process_fill({
            'order_id': sl_order['order_id'],
            'fill_price': 239.50,
            'fill_quantity': 100,
            'fill_time': datetime.now(),
            'commission': 1.00,
        })
        
        # Verify position closed
        position = await portfolio_tracker.get_position('META')
        assert position is None or position['quantity'] == 0
        
        # Calculate P&L
        pnl = await portfolio_tracker.calculate_realized_pnl('META')
        expected_pnl = (239.50 - 250.00) * 100 - 2.00  # price diff * qty - commissions
        assert abs(pnl - expected_pnl) < 0.01
    
    async def test_concurrent_order_processing(
        self,
        trading_engine,
        performance_threshold
    ):
        """Test concurrent order processing performance."""
        import asyncio
        import time
        
        # Create multiple orders
        order_requests = [
            {
                'symbol': symbol,
                'side': 'buy',
                'quantity': 10,
                'order_type': 'limit',
                'limit_price': price,
            }
            for symbol, price in [
                ('AAPL', 150.00),
                ('MSFT', 300.00),
                ('GOOGL', 2500.00),
                ('AMZN', 3000.00),
                ('META', 250.00),
            ]
        ]
        
        start_time = time.time()
        
        # Submit orders concurrently
        order_tasks = [
            trading_engine.submit_order(req)
            for req in order_requests
        ]
        
        orders = await asyncio.gather(*order_tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Verify all orders processed
        successful_orders = [o for o in orders if isinstance(o, dict)]
        assert len(successful_orders) == len(order_requests)
        
        # Check performance
        orders_per_second = len(order_requests) / (processing_time_ms / 1000)
        print(f"Processed {orders_per_second:.2f} orders/second")
        
        # Verify no duplicate order IDs
        order_ids = [o['order_id'] for o in successful_orders]
        assert len(order_ids) == len(set(order_ids))