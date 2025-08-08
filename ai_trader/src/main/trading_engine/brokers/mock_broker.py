"""
Mock Broker for Unit Testing

This module provides a mock broker implementation that allows fine-grained
control over broker behavior for testing purposes.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import MagicMock
import uuid

from main.models.common import (
    AccountInfo, Position, Order, OrderStatus, OrderType, 
    OrderSide, TimeInForce, MarketData
)
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.utils.exceptions import OrderExecutionError
from main.utils.core import get_logger

logger = get_logger(__name__)


class MockBroker(BrokerInterface):
    """
    Mock broker for unit testing trading engine components.
    
    Features:
    - Configurable responses for all broker methods
    - Error injection for testing error handling
    - Call tracking for verification
    - Deterministic behavior
    """
    
    def __init__(self, config=None):
        """Initialize mock broker with test configuration."""
        # Create minimal config if not provided
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace()
            config._raw_config = {'trading': {'starting_cash': 100000}}
        
        super().__init__(config)
        
        # Mock state
        self._connected = False
        self._call_history = []
        self._response_overrides = {}
        self._error_injections = {}
        self._delay_config = {}
        
        # Default responses
        self._default_account_info = AccountInfo(
            account_id="mock_account_001",
            buying_power=100000.0,
            cash=100000.0,
            portfolio_value=100000.0,
            equity=100000.0,
            last_equity=100000.0,
            long_market_value=0.0,
            short_market_value=0.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            sma=100000.0,
            daytrade_count=0,
            balance_asof=datetime.now(timezone.utc),
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            currency="USD"
        )
        
        self._positions = {}
        self._orders = {}
        self._order_counter = 0
        
        # Callbacks for custom behavior
        self._order_fill_callback: Optional[Callable] = None
        self._price_callback: Optional[Callable] = None
        
        logger.info("Mock broker initialized")
    
    # Configuration methods for tests
    
    def set_response(self, method_name: str, response: Any):
        """Set a specific response for a method."""
        self._response_overrides[method_name] = response
    
    def inject_error(self, method_name: str, error: Exception):
        """Inject an error for a specific method."""
        self._error_injections[method_name] = error
    
    def set_delay(self, method_name: str, delay_seconds: float):
        """Set a delay for a specific method."""
        self._delay_config[method_name] = delay_seconds
    
    def set_order_fill_callback(self, callback: Callable):
        """Set callback for order fill behavior."""
        self._order_fill_callback = callback
    
    def set_price_callback(self, callback: Callable):
        """Set callback for price generation."""
        self._price_callback = callback
    
    def get_call_history(self, method_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get call history, optionally filtered by method name."""
        if method_name:
            return [call for call in self._call_history if call['method'] == method_name]
        return self._call_history.copy()
    
    def reset(self):
        """Reset mock broker state."""
        self._call_history.clear()
        self._response_overrides.clear()
        self._error_injections.clear()
        self._delay_config.clear()
        self._positions.clear()
        self._orders.clear()
        self._order_counter = 0
        self._connected = False
    
    # Helper methods
    
    async def _track_call(self, method_name: str, **kwargs):
        """Track method calls for verification."""
        self._call_history.append({
            'method': method_name,
            'timestamp': datetime.now(timezone.utc),
            'args': kwargs
        })
        
        # Apply delay if configured
        if method_name in self._delay_config:
            await asyncio.sleep(self._delay_config[method_name])
        
        # Inject error if configured
        if method_name in self._error_injections:
            raise self._error_injections[method_name]
        
        # Return override if configured
        if method_name in self._response_overrides:
            return self._response_overrides[method_name]
    
    # BrokerInterface implementation
    
    async def connect(self) -> bool:
        """Connect to mock broker."""
        result = await self._track_call('connect')
        if result is not None:
            return result
        
        self._connected = True
        logger.info("Mock broker connected")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from mock broker."""
        await self._track_call('disconnect')
        self._connected = False
        logger.info("Mock broker disconnected")
    
    async def is_connected(self) -> bool:
        """Check connection status."""
        result = await self._track_call('is_connected')
        return result if result is not None else self._connected
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        result = await self._track_call('get_account_info')
        return result if result is not None else self._default_account_info
    
    async def submit_order(self, order: Order) -> Order:
        """Submit an order."""
        result = await self._track_call('submit_order', order=order)
        if result is not None:
            return result
        
        # Generate order ID
        self._order_counter += 1
        order_id = f"mock_order_{self._order_counter:06d}"
        
        # Create order copy with ID and status
        submitted_order = Order(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            time_in_force=order.time_in_force,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc)
        )
        
        self._orders[order_id] = submitted_order
        
        # Apply fill callback if configured
        if self._order_fill_callback:
            fill_result = self._order_fill_callback(submitted_order)
            if fill_result:
                submitted_order.status = OrderStatus.FILLED
                submitted_order.filled_quantity = submitted_order.quantity
                submitted_order.fill_price = fill_result.get('fill_price', 100.0)
                submitted_order.filled_at = datetime.now(timezone.utc)
        
        logger.info(f"Mock order submitted: {order_id}")
        return submitted_order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        result = await self._track_call('cancel_order', order_id=order_id)
        if result is not None:
            return result
        
        if order_id in self._orders:
            order = self._orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Mock order cancelled: {order_id}")
                return True
        
        return False
    
    async def modify_order(self, order_id: str, 
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          quantity: Optional[float] = None) -> Order:
        """Modify an order."""
        result = await self._track_call(
            'modify_order',
            order_id=order_id,
            limit_price=limit_price,
            stop_price=stop_price,
            quantity=quantity
        )
        if result is not None:
            return result
        
        if order_id not in self._orders:
            raise OrderExecutionError(f"Order {order_id} not found")
        
        order = self._orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            raise OrderExecutionError(f"Cannot modify order with status {order.status}")
        
        # Apply modifications
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        if quantity is not None:
            order.quantity = quantity
        
        logger.info(f"Mock order modified: {order_id}")
        return order
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        result = await self._track_call('get_positions')
        return result if result is not None else self._positions.copy()
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        result = await self._track_call('get_position', symbol=symbol)
        if result is not None:
            return result
        
        return self._positions.get(symbol)
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders."""
        result = await self._track_call('get_orders', status=status)
        if result is not None:
            return result
        
        orders = list(self._orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return orders
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get specific order."""
        result = await self._track_call('get_order', order_id=order_id)
        if result is not None:
            return result
        
        return self._orders.get(order_id)
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get market data."""
        result = await self._track_call('get_market_data', symbol=symbol)
        if result is not None:
            return result
        
        # Generate price using callback or default
        if self._price_callback:
            price = self._price_callback(symbol)
        else:
            price = 100.0
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            last=price,
            bid=price * 0.999,
            ask=price * 1.001,
            volume=1000000
        )
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get quote."""
        result = await self._track_call('get_quote', symbol=symbol)
        if result is not None:
            return result
        
        # Generate price using callback or default
        if self._price_callback:
            price = self._price_callback(symbol)
        else:
            price = 100.0
        
        return {
            'symbol': symbol,
            'bid_price': price * 0.999,
            'ask_price': price * 1.001,
            'bid_size': 100,
            'ask_size': 100,
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to market data."""
        await self._track_call('subscribe_market_data', symbols=symbols)
        logger.info(f"Mock subscription to {symbols}")
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        await self._track_call('unsubscribe_market_data', symbols=symbols)
        logger.info(f"Mock unsubscription from {symbols}")
    
    async def get_historical_data(self, 
                                 symbol: str,
                                 start: datetime,
                                 end: datetime,
                                 timeframe: str = "1Day") -> List[Dict[str, Any]]:
        """Get historical data."""
        result = await self._track_call(
            'get_historical_data',
            symbol=symbol,
            start=start,
            end=end,
            timeframe=timeframe
        )
        if result is not None:
            return result
        
        # Generate simple historical data
        data = []
        current = start
        base_price = 100.0
        
        while current <= end:
            data.append({
                'timestamp': current,
                'symbol': symbol,
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price * 1.01,
                'volume': 1000000
            })
            current = current.replace(hour=current.hour + 1)
            base_price *= 1.001
        
        return data
    
    async def get_tradable_symbols(self) -> List[str]:
        """Get tradable symbols."""
        result = await self._track_call('get_tradable_symbols')
        if result is not None:
            return result
        
        return ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    async def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        result = await self._track_call('is_tradable', symbol=symbol)
        if result is not None:
            return result
        
        tradable = await self.get_tradable_symbols()
        return symbol in tradable
    
    async def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours."""
        result = await self._track_call('get_market_hours')
        if result is not None:
            return result
        
        return {
            'is_open': True,
            'next_open': None,
            'next_close': None
        }
    
    # Test helper methods
    
    def add_test_position(self, symbol: str, quantity: float, entry_price: float):
        """Add a test position."""
        self._positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price=entry_price,
            current_price=entry_price,
            market_value=quantity * entry_price,
            cost_basis=quantity * entry_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            realized_pnl=0.0,
            side='long' if quantity > 0 else 'short',
            timestamp=datetime.now(timezone.utc)
        )
    
    def add_test_order(self, order: Order):
        """Add a test order."""
        if not order.order_id:
            self._order_counter += 1
            order.order_id = f"mock_order_{self._order_counter:06d}"
        self._orders[order.order_id] = order
    
    def verify_calls(self, expected_calls: List[str]) -> bool:
        """Verify that expected method calls were made."""
        actual_calls = [call['method'] for call in self._call_history]
        return actual_calls == expected_calls
    
    def get_call_count(self, method_name: str) -> int:
        """Get number of times a method was called."""
        return len([c for c in self._call_history if c['method'] == method_name])