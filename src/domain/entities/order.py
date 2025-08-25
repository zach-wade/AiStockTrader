"""
Order Entity - Core trading order with business logic
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


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
    quantity: Decimal
    side: OrderSide
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
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
    quantity: Decimal = Decimal("0")
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET

    # Pricing
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None

    # Execution
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.DAY

    # Tracking
    broker_order_id: str | None = None
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None

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

        # Check if quantity is positive (handle both Decimal and Quantity types)
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        if qty_value <= 0:
            raise ValueError(f"Order quantity must be positive, got {qty_value}")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit order requires limit price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop order requires stop price")

        if self.order_type == OrderType.STOP_LIMIT and (
            self.stop_price is None or self.limit_price is None
        ):
            raise ValueError("Stop-limit order requires both stop and limit prices")

        if self.filled_quantity < 0:
            raise ValueError("Filled quantity cannot be negative")

        # Compare filled_quantity with quantity (handle both Decimal and Quantity types)
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        if self.filled_quantity > qty_value:
            raise ValueError("Filled quantity cannot exceed order quantity")

    @classmethod
    def create_market_order(cls, request: OrderRequest) -> "Order":
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
    ) -> "Order":
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
    ) -> "Order":
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
    ) -> "Order":
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
        self, filled_quantity: Decimal, fill_price: Decimal, timestamp: datetime | None = None
    ) -> None:
        """Record a fill or partial fill"""
        if self.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot fill order in {self.status} status")

        if filled_quantity <= 0:
            raise ValueError("Fill quantity must be positive")

        if fill_price <= 0:
            raise ValueError("Fill price must be positive")

        new_filled_quantity = self.filled_quantity + filled_quantity
        # Handle both Decimal and Quantity types for comparison
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        if new_filled_quantity > qty_value:
            raise ValueError(
                f"Total filled quantity {new_filled_quantity} exceeds order quantity {qty_value}"
            )

        # Update average fill price
        if self.average_fill_price is None:
            self.average_fill_price = fill_price
        else:
            total_value = (self.filled_quantity * self.average_fill_price) + (
                filled_quantity * fill_price
            )
            self.average_fill_price = total_value / new_filled_quantity

        self.filled_quantity = new_filled_quantity

        # Update status (handle both Decimal and Quantity types)
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        if self.filled_quantity >= qty_value:
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
        if self.status != OrderStatus.PENDING:
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

    def get_remaining_quantity(self) -> Decimal:
        """Get quantity still to be filled"""
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        return Decimal(qty_value) - self.filled_quantity

    def get_fill_ratio(self) -> Decimal:
        """Get ratio of filled quantity to total quantity"""
        qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
        qty_decimal = Decimal(qty_value)
        if qty_decimal == 0:
            return Decimal("0")
        return self.filled_quantity / qty_decimal

    def get_notional_value(self) -> Decimal | None:
        """Get total value of the order"""
        if self.order_type == OrderType.MARKET:
            if self.average_fill_price:
                return self.filled_quantity * self.average_fill_price
            return None
        elif self.order_type == OrderType.LIMIT and self.limit_price:
            qty_value = self.quantity.value if hasattr(self.quantity, "value") else self.quantity
            limit_value = (
                self.limit_price.value if hasattr(self.limit_price, "value") else self.limit_price
            )
            return Decimal(qty_value) * Decimal(limit_value)
        return None

    def __str__(self) -> str:
        """String representation"""
        price_str = ""
        if self.order_type == OrderType.LIMIT and self.limit_price:
            price_str = f" @ ${self.limit_price}"
        elif self.order_type == OrderType.STOP and self.stop_price:
            price_str = f" stop @ ${self.stop_price}"

        return (
            f"Order({self.id}: {self.side.value.upper()} {self.quantity} {self.symbol}"
            f"{price_str} - {self.status.value})"
        )
