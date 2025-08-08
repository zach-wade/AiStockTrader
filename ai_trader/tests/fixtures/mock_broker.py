"""
Mock Broker Implementation for Testing
Provides deterministic behavior for trading engine tests.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from unittest.mock import Mock
import uuid
import random  # DEPRECATED - use secure_random
from main.utils.core import secure_uniform, secure_randint, secure_choice, secure_sample, secure_shuffle

from trading_engine.brokers.broker_interface import BrokerInterface
from models.common import Order, Position, OrderStatus, OrderType, OrderSide, AccountInfo


class MockBroker(BrokerInterface):
    """
    Mock broker implementation that simulates realistic broker behavior
    for testing purposes without requiring actual broker connections.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize mock broker with test configuration."""
        super().__init__(config)
        
        # Mock broker state
        self.orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Position] = {}
        self.account_info = AccountInfo(
            account_id='test_account_123',
            buying_power=100000.0,
            cash=100000.0,
            portfolio_value=100000.0,
            day_trade_count=0,
            pattern_day_trader=False
        )
        
        # Market data simulation
        self.market_prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 800.0,
            'AMZN': 3200.0
        }
        
        # Behavior configuration
        self.fill_delay_seconds = 0.1  # Simulated fill delay
        self.fill_probability = 0.95   # Probability of order being filled
        self.slippage_bps = 5          # Slippage in basis points
        self.reject_probability = 0.02 # Probability of order rejection
        
        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.fill_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []
        
        # Connection state
        self._connected = True
        
        # Background tasks
        self._fill_simulation_task: Optional[asyncio.Task] = None
        self._market_data_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """Simulate broker connection."""
        self._connected = True
        
        # Start background simulation tasks
        self._fill_simulation_task = asyncio.create_task(self._simulate_fills())
        self._market_data_task = asyncio.create_task(self._simulate_market_data())
        
        return True
    
    async def disconnect(self) -> bool:
        """Simulate broker disconnection."""
        self._connected = False
        
        # Stop background tasks
        if self._fill_simulation_task:
            self._fill_simulation_task.cancel()
        if self._market_data_task:
            self._market_data_task.cancel()
        
        return True
    
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected
    
    async def submit_order(self, **order_params) -> Dict[str, Any]:
        """
        Submit order to mock broker.
        Simulates real broker order submission with validation and responses.
        """
        if not self._connected:
            raise Exception("Broker not connected")
        
        # Generate broker order ID
        broker_order_id = f"mock_order_{uuid.uuid4().hex[:8]}"
        
        # Simulate order validation
        validation_errors = self._validate_order(**order_params)
        if validation_errors:
            raise Exception(f"Order validation failed: {', '.join(validation_errors)}")
        
        # Simulate random rejection
        if random.random() < self.reject_probability:
            raise Exception("Order rejected by broker")
        
        # Create order record
        order_record = {
            'broker_order_id': broker_order_id,
            'symbol': order_params['symbol'],
            'side': order_params['side'],
            'quantity': order_params['quantity'],
            'order_type': order_params['order_type'],
            'limit_price': order_params.get('limit_price'),
            'stop_price': order_params.get('stop_price'),
            'time_in_force': order_params.get('time_in_force', 'DAY'),
            'status': 'submitted',
            'submitted_at': datetime.now(),
            'filled_quantity': 0,
            'remaining_quantity': order_params['quantity'],
            'fills': []
        }
        
        self.orders[broker_order_id] = order_record
        
        # Trigger order callback
        for callback in self.order_callbacks:
            callback({
                'order_id': broker_order_id,
                'status': 'submitted',
                'timestamp': datetime.now()
            })
        
        return {
            'order_id': broker_order_id,
            'status': 'submitted',
            'timestamp': datetime.now()
        }
    
    async def cancel_order(self, broker_order_id: str) -> Dict[str, Any]:
        """Cancel order in mock broker."""
        if broker_order_id not in self.orders:
            raise Exception(f"Order {broker_order_id} not found")
        
        order = self.orders[broker_order_id]
        if order['status'] in ['filled', 'cancelled']:
            raise Exception(f"Cannot cancel order in status {order['status']}")
        
        order['status'] = 'cancelled'
        order['cancelled_at'] = datetime.now()
        
        # Trigger order callback
        for callback in self.order_callbacks:
            callback({
                'order_id': broker_order_id,
                'status': 'cancelled',
                'timestamp': datetime.now()
            })
        
        return {
            'order_id': broker_order_id,
            'status': 'cancelled',
            'timestamp': datetime.now()
        }
    
    async def modify_order(self, broker_order_id: str, **modifications) -> Dict[str, Any]:
        """Modify order in mock broker (cancel/replace)."""
        if broker_order_id not in self.orders:
            raise Exception(f"Order {broker_order_id} not found")
        
        old_order = self.orders[broker_order_id]
        if old_order['status'] not in ['submitted', 'accepted', 'partially_filled']:
            raise Exception(f"Cannot modify order in status {old_order['status']}")
        
        # Cancel old order
        await self.cancel_order(broker_order_id)
        
        # Create new order with modifications
        new_order_params = {
            'symbol': old_order['symbol'],
            'side': old_order['side'],
            'quantity': old_order['quantity'],
            'order_type': old_order['order_type'],
            'limit_price': old_order['limit_price'],
            'time_in_force': old_order['time_in_force']
        }
        new_order_params.update(modifications)
        
        return await self.submit_order(**new_order_params)
    
    async def get_order(self, broker_order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details from mock broker."""
        return self.orders.get(broker_order_id)
    
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered list of orders."""
        orders = list(self.orders.values())
        
        if symbol:
            orders = [o for o in orders if o['symbol'] == symbol]
        if status:
            orders = [o for o in orders if o['status'] == status]
        
        return orders
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return self.positions.get(symbol)
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        return self.account_info
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol."""
        if symbol not in self.market_prices:
            raise Exception(f"No market data for symbol {symbol}")
        
        price = self.market_prices[symbol]
        spread = price * 0.001  # 0.1% spread
        
        return {
            'symbol': symbol,
            'bid': price - spread / 2,
            'ask': price + spread / 2,
            'last': price,
            'volume': secure_randint(100000, 1000000),
            'timestamp': datetime.now()
        }
    
    async def get_historical_data(self, symbol: str, start: datetime, 
                                 end: datetime, timeframe: str = '1min') -> List[Dict[str, Any]]:
        """Get historical market data."""
        # Simple simulation - return daily bars
        data = []
        current = start
        base_price = self.market_prices.get(symbol, 100.0)
        
        while current <= end:
            # Random walk for price simulation
            price_change = random.gauss(0, base_price * 0.02)
            base_price = max(base_price + price_change, base_price * 0.5)  # Prevent negative prices
            
            data.append({
                'timestamp': current,
                'open': base_price,
                'high': base_price * (1 + abs(random.gauss(0, 0.01))),
                'low': base_price * (1 - abs(random.gauss(0, 0.01))),
                'close': base_price,
                'volume': secure_randint(100000, 1000000)
            })
            
            current += timedelta(days=1)
        
        return data
    
    def register_order_callback(self, callback: Callable) -> None:
        """Register callback for order updates."""
        self.order_callbacks.append(callback)
    
    def register_fill_callback(self, callback: Callable) -> None:
        """Register callback for fill updates."""
        self.fill_callbacks.append(callback)
    
    def register_position_callback(self, callback: Callable) -> None:
        """Register callback for position updates."""
        self.position_callbacks.append(callback)
    
    # Internal simulation methods
    
    def _validate_order(self, **order_params) -> List[str]:
        """Validate order parameters."""
        errors = []
        
        # Check required fields
        required_fields = ['symbol', 'side', 'quantity', 'order_type']
        for field in required_fields:
            if field not in order_params:
                errors.append(f"Missing required field: {field}")
        
        # Check quantity
        quantity = order_params.get('quantity', 0)
        if quantity <= 0:
            errors.append("Quantity must be positive")
        
        # Check limit price for limit orders
        if order_params.get('order_type') == OrderType.LIMIT:
            if 'limit_price' not in order_params or order_params['limit_price'] <= 0:
                errors.append("Limit orders require a positive limit_price")
        
        # Check buying power for buy orders
        if order_params.get('side') == Side.BUY:
            estimated_cost = quantity * self.market_prices.get(order_params.get('symbol', ''), 100.0)
            if estimated_cost > self.account_info.buying_power:
                errors.append("Insufficient buying power")
        
        return errors
    
    async def _simulate_fills(self):
        """Background task to simulate order fills."""
        while self._connected:
            try:
                await asyncio.sleep(self.fill_delay_seconds)
                
                # Check for orders to fill
                for order_id, order in self.orders.items():
                    if order['status'] in ['submitted', 'accepted', 'partially_filled']:
                        if random.random() < self.fill_probability:
                            await self._fill_order(order_id, order)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in fill simulation: {e}")
    
    async def _fill_order(self, order_id: str, order: Dict[str, Any]):
        """Simulate filling an order."""
        symbol = order['symbol']
        remaining_qty = order['remaining_quantity']
        
        if remaining_qty <= 0:
            return
        
        # Determine fill quantity (partial or full)
        if order['order_type'] == OrderType.MARKET:
            fill_qty = remaining_qty  # Market orders fill completely
        else:
            # Limit orders might fill partially
            fill_qty = min(remaining_qty, secure_randint(1, remaining_qty))
        
        # Determine fill price with slippage
        market_price = self.market_prices.get(symbol, 100.0)
        slippage = market_price * (self.slippage_bps / 10000)
        
        if order['side'] == Side.BUY:
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage
        
        # Apply limit price constraints
        if order['order_type'] == OrderType.LIMIT:
            limit_price = order['limit_price']
            if order['side'] == Side.BUY and fill_price > limit_price:
                return  # Don't fill if price is above limit
            elif order['side'] == Side.SELL and fill_price < limit_price:
                return  # Don't fill if price is below limit
        
        # Create fill
        fill = Fill(
            order_id=order_id,
            fill_id=f"fill_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=order['side'],
            quantity=fill_qty,
            price=fill_price,
            timestamp=datetime.now(),
            commission=max(0.01 * fill_qty, 1.0)  # Simple commission model
        )
        
        # Update order
        order['filled_quantity'] += fill_qty
        order['remaining_quantity'] -= fill_qty
        order['fills'].append(fill)
        
        # Update order status
        if order['remaining_quantity'] == 0:
            order['status'] = 'filled'
        else:
            order['status'] = 'partially_filled'
        
        # Update positions
        self._update_position(symbol, order['side'], fill_qty, fill_price)
        
        # Trigger callbacks
        for callback in self.fill_callbacks:
            callback(fill)
        
        for callback in self.order_callbacks:
            callback({
                'order_id': order_id,
                'status': order['status'],
                'timestamp': datetime.now()
            })
    
    def _update_position(self, symbol: str, side: Side, quantity: int, price: float):
        """Update position for a fill."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[symbol]
        
        if side == Side.BUY:
            # Increase long position or reduce short position
            new_quantity = position.quantity + quantity
            if position.quantity != 0:
                new_avg_price = ((position.quantity * position.average_price) + (quantity * price)) / new_quantity
            else:
                new_avg_price = price
            
            position.quantity = new_quantity
            position.average_price = new_avg_price
        else:
            # Decrease long position or increase short position
            new_quantity = position.quantity - quantity
            position.quantity = new_quantity
            
            # For sales, calculate realized P&L
            if position.quantity >= 0:  # Closing/reducing long position
                realized_pnl = quantity * (price - position.average_price)
                position.realized_pnl += realized_pnl
        
        # Update market value and unrealized P&L
        current_price = self.market_prices.get(symbol, price)
        position.market_value = abs(position.quantity) * current_price
        position.unrealized_pnl = position.quantity * (current_price - position.average_price)
        
        # Remove position if quantity is zero
        if position.quantity == 0:
            del self.positions[symbol]
        
        # Trigger position callback
        for callback in self.position_callbacks:
            callback(position)
    
    async def _simulate_market_data(self):
        """Background task to simulate market data updates."""
        while self._connected:
            try:
                await asyncio.sleep(1.0)  # Update every second
                
                # Random walk for each symbol
                for symbol in self.market_prices:
                    current_price = self.market_prices[symbol]
                    change_pct = random.gauss(0, 0.001)  # 0.1% volatility
                    new_price = current_price * (1 + change_pct)
                    self.market_prices[symbol] = max(new_price, current_price * 0.01)  # Prevent negative prices
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in market data simulation: {e}")


class MockBrokerFactory:
    """Factory for creating configured mock brokers."""
    
    @staticmethod
    def create_realistic_broker(config: Dict[str, Any]) -> MockBroker:
        """Create a mock broker with realistic behavior."""
        broker = MockBroker(config)
        
        # Configure realistic behavior
        broker.fill_delay_seconds = 0.5
        broker.fill_probability = 0.98
        broker.slippage_bps = 3
        broker.reject_probability = 0.01
        
        return broker
    
    @staticmethod
    def create_fast_broker(config: Dict[str, Any]) -> MockBroker:
        """Create a mock broker optimized for fast testing."""
        broker = MockBroker(config)
        
        # Configure for fast testing
        broker.fill_delay_seconds = 0.01
        broker.fill_probability = 1.0
        broker.slippage_bps = 1
        broker.reject_probability = 0.0
        
        return broker
    
    @staticmethod
    def create_problematic_broker(config: Dict[str, Any]) -> MockBroker:
        """Create a mock broker that simulates various problems."""
        broker = MockBroker(config)
        
        # Configure for error testing
        broker.fill_delay_seconds = 2.0
        broker.fill_probability = 0.7
        broker.slippage_bps = 20
        broker.reject_probability = 0.1
        
        return broker