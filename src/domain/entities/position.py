"""
Position Entity - Represents a trading position with P&L tracking
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from ..value_objects import Money, Price, Quantity


class PositionSide(Enum):
    """Position side enumeration"""

    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """
    Position entity representing an open trading position.

    Tracks quantity, average entry price, and P&L calculations.
    All financial values use Decimal for precision.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    symbol: str = ""

    # Position details
    quantity: Quantity = Quantity(Decimal("0"))  # Positive for long, negative for short
    average_entry_price: Price = Price(Decimal("0"))

    # Market data
    current_price: Price | None = None
    last_updated: datetime | None = None

    # P&L tracking
    realized_pnl: Money = Money(Decimal("0"))
    commission_paid: Money = Money(Decimal("0"))

    # Risk management
    stop_loss_price: Price | None = None
    take_profit_price: Price | None = None
    max_position_value: Money | None = None

    # Timestamps
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None

    # Metadata
    strategy: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate position after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate position attributes"""
        if not self.symbol:
            raise ValueError("Position symbol cannot be empty")

        if self.average_entry_price.value < 0:
            raise ValueError("Average entry price cannot be negative")

        if self.quantity.value == 0 and self.closed_at is None:
            raise ValueError("Open position cannot have zero quantity")

    @classmethod
    def open_position(
        cls,
        symbol: str,
        quantity: Quantity,
        entry_price: Price,
        commission: Money = Money(Decimal("0")),
        strategy: str | None = None,
    ) -> "Position":
        """Factory method to open a new position.

        Note: Updated to use value objects.
        """
        if quantity.value == 0:
            raise ValueError("Cannot open position with zero quantity")
        if entry_price.value <= 0:
            raise ValueError("Entry price must be positive")

        return cls(
            symbol=symbol,
            quantity=quantity,
            average_entry_price=entry_price,
            commission_paid=commission,
            strategy=strategy,
        )

    def add_to_position(
        self, quantity: Quantity, price: Price, commission: Money = Money(Decimal("0"))
    ) -> None:
        """Add to existing position (same direction)"""
        if quantity.value == 0:
            return

        if self.is_long() and quantity.value < 0:
            raise ValueError("Cannot add short quantity to long position")
        if self.is_short() and quantity.value > 0:
            raise ValueError("Cannot add long quantity to short position")

        # Calculate new average entry price
        total_cost = (abs(self.quantity.value) * self.average_entry_price.value) + (
            abs(quantity.value) * price.value
        )
        new_quantity = self.quantity.value + quantity.value

        if new_quantity != 0:
            self.average_entry_price = Price(total_cost / abs(new_quantity))

        self.quantity = Quantity(new_quantity)
        self.commission_paid = self.commission_paid + commission

    def reduce_position(
        self, quantity: Quantity, exit_price: Price, commission: Money = Money(Decimal("0"))
    ) -> Money:
        """
        Reduce position size and calculate realized P&L.

        Returns:
            Realized P&L from this reduction
        """
        if quantity.value == 0:
            return Money(Decimal("0"))

        if abs(quantity.value) > abs(self.quantity.value):
            raise ValueError(
                f"Cannot reduce position by {quantity.value}, current quantity is {self.quantity.value}"
            )

        # Calculate P&L for the reduced portion
        if self.is_long():
            pnl = Money(quantity.value * (exit_price.value - self.average_entry_price.value))
        else:
            pnl = Money(abs(quantity.value) * (self.average_entry_price.value - exit_price.value))

        pnl = pnl - commission

        # Update position
        reduction_quantity = quantity.value if self.is_long() else -quantity.value
        self.quantity = Quantity(self.quantity.value - reduction_quantity)
        self.realized_pnl = self.realized_pnl + pnl
        self.commission_paid = self.commission_paid + commission

        # Mark as closed if fully exited
        if self.quantity.value == 0:
            self.closed_at = datetime.now(UTC)

        return pnl

    def close_position(self, exit_price: Price, commission: Money = Money(Decimal("0"))) -> Money:
        """
        Close entire position and calculate final P&L.

        Returns:
            Total realized P&L
        """
        if self.is_closed():
            raise ValueError("Position is already closed")

        # Reduce the full position
        close_quantity = Quantity(self.quantity.value if self.is_long() else -self.quantity.value)
        self.reduce_position(close_quantity, exit_price, commission)

        return self.realized_pnl

    def update_market_price(self, price: Price) -> None:
        """Update current market price"""
        if price.value <= 0:
            raise ValueError("Market price must be positive")

        self.current_price = price
        self.last_updated = datetime.now(UTC)

    def set_stop_loss(self, price: Price) -> None:
        """Set stop loss price"""
        if price.value <= 0:
            raise ValueError("Stop loss price must be positive")

        # Validate stop loss makes sense for position direction
        if self.is_long() and self.current_price and price.value > self.current_price.value:
            raise ValueError("Stop loss for long position must be below current price")
        if self.is_short() and self.current_price and price.value < self.current_price.value:
            raise ValueError("Stop loss for short position must be above current price")

        self.stop_loss_price = price

    def set_take_profit(self, price: Price) -> None:
        """Set take profit price"""
        if price.value <= 0:
            raise ValueError("Take profit price must be positive")

        # Validate take profit makes sense for position direction
        if self.is_long() and self.current_price and price.value < self.current_price.value:
            raise ValueError("Take profit for long position must be above current price")
        if self.is_short() and self.current_price and price.value > self.current_price.value:
            raise ValueError("Take profit for short position must be below current price")

        self.take_profit_price = price

    def is_long(self) -> bool:
        """Check if this is a long position"""
        return self.quantity.value > 0

    def is_short(self) -> bool:
        """Check if this is a short position"""
        return self.quantity.value < 0

    def is_closed(self) -> bool:
        """Check if position is closed"""
        return self.quantity.value == 0 or self.closed_at is not None

    def get_unrealized_pnl(self) -> Money | None:
        """Calculate unrealized P&L based on current price"""
        if self.is_closed() or self.current_price is None:
            return None

        if self.is_long():
            return Money(
                self.quantity.value * (self.current_price.value - self.average_entry_price.value)
            )
        else:
            return Money(
                abs(self.quantity.value)
                * (self.average_entry_price.value - self.current_price.value)
            )

    def get_total_pnl(self) -> Money | None:
        """Calculate total P&L (realized + unrealized)"""
        unrealized = self.get_unrealized_pnl()
        if unrealized is None:
            return self.realized_pnl

        return self.realized_pnl + unrealized - self.commission_paid

    def get_position_value(self) -> Money | None:
        """Get current market value of position"""
        if self.current_price is None:
            return None

        return Money(abs(self.quantity.value) * self.current_price.value)

    def get_return_percentage(self) -> Decimal | None:
        """Calculate return percentage"""
        if self.average_entry_price.value == 0:
            return None

        total_pnl = self.get_total_pnl()
        if total_pnl is None:
            return None

        initial_value = abs(self.quantity.value) * self.average_entry_price.value
        if initial_value == 0:
            return None

        return (total_pnl.amount / initial_value) * Decimal("100")

    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss_price is None or self.current_price is None:
            return False

        if self.is_long():
            return self.current_price.value <= self.stop_loss_price.value
        else:
            return self.current_price.value >= self.stop_loss_price.value

    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit_price is None or self.current_price is None:
            return False

        if self.is_long():
            return self.current_price.value >= self.take_profit_price.value
        else:
            return self.current_price.value <= self.take_profit_price.value

    def close(self, final_price: Price, close_time: datetime) -> None:
        """
        Close the position at the specified price and time.

        Args:
            final_price: The final price at which the position was closed
            close_time: The time when the position was closed
        """
        if self.is_closed():
            raise ValueError("Position is already closed")

        # Calculate final P&L
        if self.is_long():
            final_pnl = Money(
                self.quantity.value * (final_price.value - self.average_entry_price.value)
            )
        else:
            final_pnl = Money(
                abs(self.quantity.value) * (self.average_entry_price.value - final_price.value)
            )

        # Update position state
        self.realized_pnl = final_pnl
        self.current_price = final_price
        self.closed_at = close_time
        self.last_updated = close_time
        self.quantity = Quantity(Decimal("0"))  # Mark as fully closed

    def __str__(self) -> str:
        """String representation"""
        direction = "LONG" if self.is_long() else "SHORT"
        status = "CLOSED" if self.is_closed() else "OPEN"

        pnl_str = ""
        if not self.is_closed():
            unrealized = self.get_unrealized_pnl()
            if unrealized is not None:
                pnl_str = f", Unrealized P&L: {unrealized}"
        else:
            pnl_str = f", Realized P&L: {self.realized_pnl}"

        return (
            f"Position({self.symbol}: {direction} {abs(self.quantity.value)} @ {self.average_entry_price}"
            f" - {status}{pnl_str})"
        )
