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
    quantity: Decimal = Decimal("0")  # Positive for long, negative for short
    average_entry_price: Decimal = Decimal("0")

    # Market data
    current_price: Decimal | None = None
    last_updated: datetime | None = None

    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    commission_paid: Decimal = Decimal("0")

    # Risk management
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    max_position_value: Decimal | None = None

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

        if self.average_entry_price < 0:
            raise ValueError("Average entry price cannot be negative")

        if self.quantity == 0 and not self.is_closed():
            raise ValueError("Open position cannot have zero quantity")

    @classmethod
    def open_position(
        cls,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        commission: Decimal = Decimal("0"),
        strategy: str | None = None,
    ) -> "Position":
        """Factory method to open a new position.

        Note: Kept original signature for backward compatibility.
        Use PositionRequest in Portfolio.open_position for cleaner API.
        """
        if quantity == 0:
            raise ValueError("Cannot open position with zero quantity")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")

        return cls(
            symbol=symbol,
            quantity=quantity,
            average_entry_price=entry_price,
            commission_paid=commission,
            strategy=strategy,
        )

    def add_to_position(
        self, quantity: Decimal, price: Decimal, commission: Decimal = Decimal("0")
    ) -> None:
        """Add to existing position (same direction)"""
        if quantity == 0:
            return

        if self.is_long() and quantity < 0:
            raise ValueError("Cannot add short quantity to long position")
        if self.is_short() and quantity > 0:
            raise ValueError("Cannot add long quantity to short position")

        # Calculate new average entry price
        total_cost = (abs(self.quantity) * self.average_entry_price) + (abs(quantity) * price)
        new_quantity = self.quantity + quantity

        if new_quantity != 0:
            self.average_entry_price = total_cost / abs(new_quantity)

        self.quantity = new_quantity
        self.commission_paid += commission

    def reduce_position(
        self, quantity: Decimal, exit_price: Decimal, commission: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Reduce position size and calculate realized P&L.

        Returns:
            Realized P&L from this reduction
        """
        if quantity == 0:
            return Decimal("0")

        if abs(quantity) > abs(self.quantity):
            raise ValueError(
                f"Cannot reduce position by {quantity}, current quantity is {self.quantity}"
            )

        # Calculate P&L for the reduced portion
        if self.is_long():
            pnl = quantity * (exit_price - self.average_entry_price)
        else:
            pnl = abs(quantity) * (self.average_entry_price - exit_price)

        pnl -= commission

        # Update position
        self.quantity -= quantity if self.is_long() else -quantity
        self.realized_pnl += pnl
        self.commission_paid += commission

        # Mark as closed if fully exited
        if self.quantity == 0:
            self.closed_at = datetime.now(UTC)

        return pnl

    def close_position(self, exit_price: Decimal, commission: Decimal = Decimal("0")) -> Decimal:
        """
        Close entire position and calculate final P&L.

        Returns:
            Total realized P&L
        """
        if self.is_closed():
            raise ValueError("Position is already closed")

        self.reduce_position(
            self.quantity if self.is_long() else -self.quantity, exit_price, commission
        )

        return self.realized_pnl

    def update_market_price(self, price: Decimal) -> None:
        """Update current market price"""
        if price <= 0:
            raise ValueError("Market price must be positive")

        self.current_price = price
        self.last_updated = datetime.now(UTC)

    def set_stop_loss(self, price: Decimal) -> None:
        """Set stop loss price"""
        if price <= 0:
            raise ValueError("Stop loss price must be positive")

        # Validate stop loss makes sense for position direction
        if self.is_long() and self.current_price and price > self.current_price:
            raise ValueError("Stop loss for long position must be below current price")
        if self.is_short() and self.current_price and price < self.current_price:
            raise ValueError("Stop loss for short position must be above current price")

        self.stop_loss_price = price

    def set_take_profit(self, price: Decimal) -> None:
        """Set take profit price"""
        if price <= 0:
            raise ValueError("Take profit price must be positive")

        # Validate take profit makes sense for position direction
        if self.is_long() and self.current_price and price < self.current_price:
            raise ValueError("Take profit for long position must be above current price")
        if self.is_short() and self.current_price and price > self.current_price:
            raise ValueError("Take profit for short position must be below current price")

        self.take_profit_price = price

    def is_long(self) -> bool:
        """Check if this is a long position"""
        return self.quantity > 0

    def is_short(self) -> bool:
        """Check if this is a short position"""
        return self.quantity < 0

    def is_closed(self) -> bool:
        """Check if position is closed"""
        return self.quantity == 0 or self.closed_at is not None

    def get_unrealized_pnl(self) -> Decimal | None:
        """Calculate unrealized P&L based on current price"""
        if self.is_closed() or self.current_price is None:
            return None

        if self.is_long():
            return self.quantity * (self.current_price - self.average_entry_price)
        else:
            return abs(self.quantity) * (self.average_entry_price - self.current_price)

    def get_total_pnl(self) -> Decimal | None:
        """Calculate total P&L (realized + unrealized)"""
        unrealized = self.get_unrealized_pnl()
        if unrealized is None:
            return self.realized_pnl

        return self.realized_pnl + unrealized - self.commission_paid

    def get_position_value(self) -> Decimal | None:
        """Get current market value of position"""
        if self.current_price is None:
            return None

        return abs(self.quantity) * self.current_price

    def get_return_percentage(self) -> Decimal | None:
        """Calculate return percentage"""
        if self.average_entry_price == 0:
            return None

        total_pnl = self.get_total_pnl()
        if total_pnl is None:
            return None

        initial_value = abs(self.quantity) * self.average_entry_price
        if initial_value == 0:
            return None

        return (total_pnl / initial_value) * Decimal("100")

    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss_price is None or self.current_price is None:
            return False

        if self.is_long():
            return self.current_price <= self.stop_loss_price
        else:
            return self.current_price >= self.stop_loss_price

    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit_price is None or self.current_price is None:
            return False

        if self.is_long():
            return self.current_price >= self.take_profit_price
        else:
            return self.current_price <= self.take_profit_price

    def __str__(self) -> str:
        """String representation"""
        direction = "LONG" if self.is_long() else "SHORT"
        status = "CLOSED" if self.is_closed() else "OPEN"

        pnl_str = ""
        if not self.is_closed():
            unrealized = self.get_unrealized_pnl()
            if unrealized is not None:
                pnl_str = f", Unrealized P&L: ${unrealized:.2f}"
        else:
            pnl_str = f", Realized P&L: ${self.realized_pnl:.2f}"

        return (
            f"Position({self.symbol}: {direction} {abs(self.quantity)} @ ${self.average_entry_price:.2f}"
            f" - {status}{pnl_str})"
        )
