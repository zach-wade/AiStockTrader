"""
Broker Interface - Defines the contract for broker implementations
"""

# Standard library imports
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol
from uuid import UUID

# Local imports
from src.domain.entities.order import Order, OrderStatus
from src.domain.entities.position import Position


@dataclass
class AccountInfo:
    """Account information from the broker"""

    # Account identifiers
    account_id: str
    account_type: str  # "paper", "live", "backtest"

    # Capital information
    equity: Decimal
    cash: Decimal
    buying_power: Decimal

    # Position information
    positions_value: Decimal

    # P&L tracking
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    # Risk metrics
    margin_used: Decimal | None = None
    margin_available: Decimal | None = None

    # Trading restrictions
    pattern_day_trader: bool = False
    trades_today: int = 0
    trades_remaining: int | None = None

    # Timestamps
    last_updated: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "account_id": self.account_id,
            "account_type": self.account_type,
            "equity": float(self.equity),
            "cash": float(self.cash),
            "buying_power": float(self.buying_power),
            "positions_value": float(self.positions_value),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "margin_used": float(self.margin_used) if self.margin_used else None,
            "margin_available": float(self.margin_available) if self.margin_available else None,
            "pattern_day_trader": self.pattern_day_trader,
            "trades_today": self.trades_today,
            "trades_remaining": self.trades_remaining,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class MarketHours:
    """Market trading hours information"""

    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None

    def __str__(self) -> str:
        if self.is_open:
            close_str = f", closes at {self.next_close}" if self.next_close else ""
            return f"Market is OPEN{close_str}"
        else:
            open_str = f", opens at {self.next_open}" if self.next_open else ""
            return f"Market is CLOSED{open_str}"


class IBroker(Protocol):
    """
    Broker interface for order execution and account management.

    All broker implementations must conform to this protocol.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.

        Args:
            order: Order to submit

        Returns:
            Order with updated broker_order_id and status

        Raises:
            BrokerConnectionError: If broker connection fails
            InsufficientFundsError: If insufficient buying power
            InvalidOrderError: If order validation fails
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: UUID) -> bool:
        """
        Cancel an order.

        Args:
            order_id: UUID of the order to cancel

        Returns:
            True if cancellation was successful, False otherwise

        Raises:
            BrokerConnectionError: If broker connection fails
            OrderNotFoundError: If order doesn't exist
        """
        ...

    @abstractmethod
    def get_order_status(self, order_id: UUID) -> OrderStatus:
        """
        Get the current status of an order.

        Args:
            order_id: UUID of the order

        Returns:
            Current order status

        Raises:
            BrokerConnectionError: If broker connection fails
            OrderNotFoundError: If order doesn't exist
        """
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """
        Get all current positions.

        Returns:
            List of current positions

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        ...

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information.

        Returns:
            Account information including balances and buying power

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open for trading.

        Returns:
            True if market is open, False otherwise

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        ...

    @abstractmethod
    def get_market_hours(self) -> MarketHours:
        """
        Get detailed market hours information.

        Returns:
            MarketHours object with current status and next open/close times

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        ...

    @abstractmethod
    def update_order(self, order: Order) -> Order:
        """
        Update an existing order with latest status from broker.

        Args:
            order: Order to update

        Returns:
            Updated order with latest status and fills

        Raises:
            BrokerConnectionError: If broker connection fails
            OrderNotFoundError: If order doesn't exist at broker
        """
        ...

    @abstractmethod
    def get_recent_orders(self, limit: int = 100) -> list[Order]:
        """
        Get recent orders from the broker.

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of recent orders

        Raises:
            BrokerConnectionError: If broker connection fails
        """
        ...

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the broker.

        Raises:
            BrokerConnectionError: If connection fails
            InvalidCredentialsError: If authentication fails
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to the broker.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if broker connection is active.

        Returns:
            True if connected, False otherwise
        """
        ...


# Broker-specific exceptions
class BrokerError(Exception):
    """Base exception for broker-related errors"""

    pass


class BrokerConnectionError(BrokerError):
    """Raised when broker connection fails"""

    pass


class InsufficientFundsError(BrokerError):
    """Raised when account has insufficient funds for order"""

    pass


class InvalidOrderError(BrokerError):
    """Raised when order validation fails"""

    pass


class OrderNotFoundError(BrokerError):
    """Raised when order is not found at broker"""

    pass


class InvalidCredentialsError(BrokerError):
    """Raised when broker authentication fails"""

    pass


class RateLimitError(BrokerError):
    """Raised when broker API rate limit is exceeded"""

    pass


class MarketClosedError(BrokerError):
    """Raised when attempting to trade while market is closed"""

    pass
