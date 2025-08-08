"""
Abstract base class for broker integrations.

This module defines the interface that all broker implementations must follow
to integrate with the trading engine.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from decimal import Decimal

from main.models.common import (
    Order, Position, OrderStatus, OrderType, OrderSide,
    TimeInForce, MarketData, AccountInfo, PositionSide
)
from main.utils.core import AITraderException, create_event_tracker
from main.utils.resilience import retry
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class BrokerException(AITraderException):
    """Base exception for broker-related errors."""
    pass


class BrokerConnectionError(BrokerException):
    """Raised when broker connection fails."""
    pass


class OrderSubmissionError(BrokerException):
    """Raised when order submission fails."""
    pass


class InsufficientFundsError(BrokerException):
    """Raised when account has insufficient funds."""
    pass


class BrokerInterface(ABC):
    """
    Abstract interface for broker integrations.
    
    All broker implementations must inherit from this class and implement
    the required methods for order management, position tracking, and
    market data access.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize broker interface.
        
        Args:
            config: Broker-specific configuration
            metrics_collector: Optional metrics collector for monitoring
        """
        self.config = config
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("broker")
        self._connected = False
        
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the broker.
        
        Raises:
            BrokerConnectionError: If connection fails
        """
        pass
        
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass
        
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if broker is connected."""
        pass
        
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID from broker
            
        Raises:
            OrderSubmissionError: If order submission fails
            InsufficientFundsError: If insufficient funds
        """
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
        
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_orders(
        self, 
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get orders with optional filters.
        
        Args:
            status: Filter by order status
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of orders to return
            
        Returns:
            List of orders matching filters
        """
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        pass
        
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position if exists, None otherwise
        """
        pass
        
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            Account information including balances and buying power
        """
        pass
        
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Current market data
        """
        pass
        
    @abstractmethod
    async def stream_market_data(self, symbols: List[str]) -> AsyncIterator[MarketData]:
        """
        Stream real-time market data for symbols.
        
        Args:
            symbols: List of symbols to stream
            
        Yields:
            Market data updates
        """
        pass
        
    @abstractmethod
    async def stream_order_updates(self) -> AsyncIterator[Order]:
        """
        Stream real-time order updates.
        
        Yields:
            Order updates
        """
        pass
        
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None
    ) -> bool:
        """
        Modify an existing order.
        
        Default implementation cancels and resubmits.
        Brokers can override for native modify support.
        
        Args:
            order_id: ID of order to modify
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price
            
        Returns:
            True if modification successful
        """
        # Get existing order
        order = await self.get_order(order_id)
        if not order or order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            return False
            
        # Cancel existing order
        if not await self.cancel_order(order_id):
            return False
            
        # Create modified order
        modified_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=quantity or order.quantity,
            limit_price=limit_price or order.limit_price,
            stop_price=stop_price or order.stop_price,
            time_in_force=order.time_in_force,
            client_order_id=f"{order.client_order_id}_mod"
        )
        
        # Submit modified order
        try:
            await self.submit_order(modified_order)
            return True
        except BrokerException:
            logger.exception(f"Failed to submit modified order for {order_id}")
            return False
            
    async def close_position(self, symbol: str) -> Optional[str]:
        """
        Close a position by submitting a market order.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            Order ID if position closed, None if no position
        """
        position = await self.get_position(symbol)
        if not position:
            return None
            
        # Determine order side to close position
        close_side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY
        
        # Create market order to close
        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            time_in_force=TimeInForce.IOC
        )
        
        return await self.submit_order(close_order)
        
    async def close_all_positions(self) -> List[str]:
        """
        Close all open positions.
        
        Returns:
            List of order IDs for closing orders
        """
        positions = await self.get_positions()
        order_ids = []
        
        for position in positions:
            order_id = await self.close_position(position.symbol)
            if order_id:
                order_ids.append(order_id)
                
        return order_ids
        
    def _track_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Track broker events for monitoring."""
        if self.event_tracker:
            self.event_tracker.track(event_type, data)
            
        if self.metrics:
            self.metrics.increment(f"broker.{event_type}", tags={"broker": self.__class__.__name__})