"""
Order Entity - Core trading order with business logic
"""

from __future__ import annotations

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from ..value_objects import Money, Price, Quantity


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force enumeration"""

    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderRequest:
    """Request parameters for creating an order."""

    symbol: str
    quantity: Quantity
    side: OrderSide
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    reason: str | None = None


@dataclass
class Order:
    """
    Order entity representing a trading order.

    This is a domain entity with business logic.
    All financial values use Decimal for precision.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)

    # Core attributes
    symbol: str = ""
    quantity: Quantity = Quantity(Decimal("0"))
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET

    # Pricing
    limit_price: Price | None = None
    stop_price: Price | None = None

    # Execution
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.DAY

    # Tracking
    broker_order_id: str | None = None
    filled_quantity: Quantity = Quantity(Decimal("0"))
    average_fill_price: Price | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None

    # Metadata
    reason: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate order after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate order attributes"""
        if not self.symbol:
            raise ValueError("Order symbol cannot be empty")

        # Check if quantity is positive
        if self.quantity.value <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity.value}")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit order requires limit price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop order requires stop price")

        if self.order_type == OrderType.STOP_LIMIT and (
            self.stop_price is None or self.limit_price is None
        ):
            raise ValueError("Stop-limit order requires both stop and limit prices")

        if self.filled_quantity.value < 0:
            raise ValueError("Filled quantity cannot be negative")

        # Compare filled_quantity with quantity
        if self.filled_quantity.value > self.quantity.value:
            raise ValueError("Filled quantity cannot exceed order quantity")

    @classmethod
    def create_market_order(cls, request: OrderRequest) -> Order:
        """Factory method to create a market order"""
        return cls(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=OrderType.MARKET,
            reason=request.reason,
        )

    @classmethod
    def create_limit_order(
        cls,
        request: OrderRequest,
    ) -> Order:
        """Factory method to create a limit order.

        Args:
            request: Order request with parameters

        Returns:
            New limit order

        Raises:
            ValueError: If limit_price is not provided
        """
        if request.limit_price is None:
            raise ValueError("Limit price is required for limit orders")

        return cls(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=OrderType.LIMIT,
            limit_price=request.limit_price,
            time_in_force=request.time_in_force,
            reason=request.reason,
        )

    @classmethod
    def create_stop_order(
        cls,
        request: OrderRequest,
    ) -> Order:
        """Factory method to create a stop order.

        Args:
            request: Order request with parameters

        Returns:
            New stop order

        Raises:
            ValueError: If stop_price is not provided
        """
        if request.stop_price is None:
            raise ValueError("Stop price is required for stop orders")

        return cls(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=OrderType.STOP,
            stop_price=request.stop_price,
            reason=request.reason,
        )

    @classmethod
    def create_stop_limit_order(
        cls,
        request: OrderRequest,
    ) -> Order:
        """Factory method to create a stop-limit order.

        Args:
            request: Order request with parameters

        Returns:
            New stop-limit order

        Raises:
            ValueError: If stop_price or limit_price is not provided
        """
        if request.stop_price is None or request.limit_price is None:
            raise ValueError("Both stop price and limit price are required for stop-limit orders")

        return cls(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=OrderType.STOP_LIMIT,
            stop_price=request.stop_price,
            limit_price=request.limit_price,
            time_in_force=request.time_in_force,
            reason=request.reason,
        )

    def submit(self, broker_order_id: str) -> None:
        """Mark order as submitted to broker"""
        if self.status != OrderStatus.PENDING:
            raise ValueError(f"Cannot submit order in {self.status} status")

        self.broker_order_id = broker_order_id
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.now(UTC)

    def fill(
        self, filled_quantity: Quantity, fill_price: Price, timestamp: datetime | None = None
    ) -> None:
        """Record a fill or partial fill"""
        if self.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot fill order in {self.status} status")

        if filled_quantity.value <= 0:
            raise ValueError("Fill quantity must be positive")

        if fill_price.value <= 0:
            raise ValueError("Fill price must be positive")

        new_filled_quantity_value = self.filled_quantity.value + filled_quantity.value
        if new_filled_quantity_value > self.quantity.value:
            raise ValueError(
                f"Total filled quantity {new_filled_quantity_value} exceeds order quantity {self.quantity.value}"
            )

        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            total_value = (self.filled_quantity.value * self.average_fill_price.value) + (
                filled_quantity.value * fill_price.value
            )
            self.average_fill_price = Price(total_value / new_filled_quantity_value)

        self.filled_quantity = Quantity(new_filled_quantity_value)

        # Update status
        if self.filled_quantity.value >= self.quantity.value:
            self.status = OrderStatus.FILLED
            self.filled_at = timestamp or datetime.now(UTC)
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self, reason: str | None = None) -> None:
        """Cancel the order"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in {self.status} status")

        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.now(UTC)
        if reason:
            self.tags["cancel_reason"] = reason

    def reject(self, reason: str) -> None:
        """Reject the order"""
        if self.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            raise ValueError(f"Cannot reject order in {self.status} status")

        self.status = OrderStatus.REJECTED
        self.tags["reject_reason"] = reason

    def is_active(self) -> bool:
        """Check if order is in active state"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    def is_complete(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    def get_remaining_quantity(self) -> Quantity:
        """Get quantity still to be filled"""
        return Quantity(self.quantity.value - self.filled_quantity.value)

    def get_fill_ratio(self) -> Decimal:
        """Get ratio of filled quantity to total quantity"""
        if self.quantity.value == 0:
            return Decimal("0")
        return self.filled_quantity.value / self.quantity.value

    def get_notional_value(self) -> Money | None:
        """Get total value of the order"""
        if self.order_type == OrderType.MARKET:
            if self.average_fill_price:
                return Money(self.filled_quantity.value * self.average_fill_price.value)
            return None
        elif self.order_type == OrderType.LIMIT and self.limit_price:
            return Money(self.quantity.value * self.limit_price.value)
        return None

    def __str__(self) -> str:
        """String representation"""
        price_str = ""
        if self.order_type == OrderType.LIMIT and self.limit_price:
            price_str = f" @ {self.limit_price}"
        elif self.order_type == OrderType.STOP and self.stop_price:
            price_str = f" stop @ {self.stop_price}"

        return (
            f"Order({self.id}: {self.side.value.upper()} {self.quantity} {self.symbol}"
            f"{price_str} - {self.status.value})"
        )
