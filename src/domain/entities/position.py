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

# Internal imports
from src.domain.value_objects.converter import ValueObjectConverter

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

    # Required fields (must come first for dataclass)
    average_entry_price: Price  # No default - must be set explicitly

    # Identity
    id: UUID = field(default_factory=uuid4)
    symbol: str = ""

    # Position details
    quantity: Quantity = field(
        default_factory=lambda: Quantity(Decimal("0"))
    )  # Positive for long, negative for short
    original_quantity: Quantity | None = None  # Store original quantity for closed positions

    # Market data
    current_price: Price | None = None
    last_updated: datetime | None = None

    # P&L tracking
    realized_pnl: Money = field(default_factory=lambda: Money(Decimal("0")))
    commission_paid: Money = field(default_factory=lambda: Money(Decimal("0")))

    # Risk management
    stop_loss_price: Price | None = None
    take_profit_price: Price | None = None
    max_position_value: Money | None = None

    # Timestamps
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    exit_time: datetime | None = None
    exit_price: Price | None = None

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

        # Handle both Price objects and raw Decimal values
        entry_price_value = ValueObjectConverter.extract_value(self.average_entry_price)
        if entry_price_value < 0:
            raise ValueError("Average entry price cannot be negative")

        # Handle both Quantity objects and raw Decimal values
        quantity_value = ValueObjectConverter.extract_value(self.quantity)
        # Allow zero quantity only if position is marked as closed
        if quantity_value == 0 and self.closed_at is None:
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
            original_quantity=quantity,  # Store original quantity
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

        # Store original quantity if not already stored
        if self.original_quantity is None:
            self.original_quantity = self.quantity

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
        # Handle both Quantity objects and raw Decimal values
        quantity_value = ValueObjectConverter.extract_value(self.quantity)
        return quantity_value > 0

    def is_short(self) -> bool:
        """Check if this is a short position"""
        # Handle both Quantity objects and raw Decimal values
        quantity_value = ValueObjectConverter.extract_value(self.quantity)
        return quantity_value < 0

    def is_closed(self) -> bool:
        """Check if position is closed"""
        # Handle both Quantity objects and raw Decimal values
        quantity_value = ValueObjectConverter.extract_value(self.quantity)
        return quantity_value == 0 or self.closed_at is not None

    def is_open(self) -> bool:
        """Check if position is open"""
        return not self.is_closed()

    @property
    def side(self) -> PositionSide:
        """Get the position side (LONG or SHORT)"""
        if self.is_long():
            return PositionSide.LONG
        elif self.is_short():
            return PositionSide.SHORT
        else:
            # Closed position - return LONG by default for compatibility
            return PositionSide.LONG

    def get_unrealized_pnl(self) -> Money | None:
        """Calculate unrealized P&L based on current price"""
        if self.is_closed() or self.current_price is None:
            return None

        # Extract decimal values from domain objects
        current_value = self.current_price.value
        entry_value = self.average_entry_price.value
        quantity_value = self.quantity.value

        if self.is_long():
            return Money(quantity_value * (current_value - entry_value))
        else:
            return Money(abs(quantity_value) * (entry_value - current_value))

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

        # Handle both value objects and raw Decimal values
        quantity_value = ValueObjectConverter.extract_value(self.quantity)
        price_value = ValueObjectConverter.extract_value(self.current_price)
        return Money(abs(quantity_value) * price_value)

    def get_return_percentage(self) -> Decimal | None:
        """Calculate return percentage"""
        if self.average_entry_price.value == 0:
            return None

        total_pnl = self.get_total_pnl()
        if total_pnl is None:
            return None

        # Use original quantity for closed positions, current quantity for open
        quantity_for_calc = (
            self.original_quantity if self.is_closed() and self.original_quantity else self.quantity
        )
        initial_value = abs(quantity_for_calc.value) * self.average_entry_price.value
        if initial_value == 0:
            return None

        return (total_pnl.amount / initial_value) * Decimal("100")

    def should_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss_price is None or self.current_price is None:
            return False

        # Extract decimal values from Price objects
        current_value = self.current_price.value
        stop_loss_value = self.stop_loss_price.value

        if self.is_long():
            return current_value <= stop_loss_value
        else:
            return current_value >= stop_loss_value

    def should_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit_price is None or self.current_price is None:
            return False

        # Extract decimal values from Price objects
        current_value = self.current_price.value
        take_profit_value = self.take_profit_price.value

        if self.is_long():
            return current_value >= take_profit_value
        else:
            return current_value <= take_profit_value

    def close(self, final_price: Price, close_time: datetime) -> None:
        """
        Close the position at the specified price and time.

        Args:
            final_price: The final price at which the position was closed
            close_time: The time when the position was closed
        """
        if self.is_closed():
            raise ValueError("Position is already closed")

        # Store original quantity if not already stored
        if self.original_quantity is None:
            self.original_quantity = self.quantity

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
        # Use original quantity to determine direction for closed positions
        if self.is_closed() and self.original_quantity:
            direction = "LONG" if self.original_quantity.value > 0 else "SHORT"
        else:
            direction = "LONG" if self.is_long() else "SHORT"

        status = "CLOSED" if self.is_closed() else "OPEN"

        pnl_str = ""
        if not self.is_closed():
            unrealized = self.get_unrealized_pnl()
            if unrealized is not None:
                pnl_str = f", Unrealized P&L: {unrealized}"
        else:
            pnl_str = f", Realized P&L: {self.realized_pnl}"

        # Display original quantity for closed positions, current quantity for open
        display_quantity = (
            abs(self.original_quantity.value)
            if self.is_closed() and self.original_quantity
            else abs(self.quantity.value)
        )

        return (
            f"Position({self.symbol}: {direction} {display_quantity} @ {self.average_entry_price}"
            f" - {status}{pnl_str})"
        )

    def mark_as_closed(self, exit_price: Price) -> None:
        """Mark position as closed with exit details.

        Args:
            exit_price: The exit price for closing the position
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now(UTC)
        self.closed_at = datetime.now(UTC)
