"""Position Manager domain service for managing trading positions."""

# Standard library imports
from decimal import Decimal

from ..entities import Order, OrderSide, OrderStatus, Position
from ..value_objects import Money, Price, Quantity


class PositionManager:
    """Domain service for managing position lifecycle and calculations."""

    def open_position(self, order: Order, fill_price: Price | None = None) -> Position:
        """Open a new position from a filled order.

        Args:
            order: The filled order to create position from
            fill_price: Optional override price (uses order's average fill price if not provided)

        Returns:
            New Position instance

        Raises:
            ValueError: If order is not filled or invalid
        """
        if order.status != OrderStatus.FILLED:
            raise ValueError(f"Cannot open position from {order.status} order")

        if order.filled_quantity <= 0:
            raise ValueError("Cannot open position with zero or negative quantity")

        # Determine entry price
        if fill_price:
            entry_price = fill_price.value
        elif order.average_fill_price:
            entry_price = order.average_fill_price
        else:
            raise ValueError("No fill price available for position")

        # Determine position quantity (negative for short)
        quantity = -order.filled_quantity if order.side == OrderSide.SELL else order.filled_quantity

        return Position.open_position(
            symbol=order.symbol,
            quantity=quantity,
            entry_price=entry_price,
            commission=Decimal("0"),  # Commission tracked separately
            strategy=order.tags.get("strategy"),
        )

    def update_position(
        self, position: Position, order: Order, fill_price: Price | None = None
    ) -> None:
        """Update existing position with a new fill.

        Args:
            position: Position to update
            order: The filled order
            fill_price: Optional override price

        Raises:
            ValueError: If order is not filled or incompatible
        """
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            raise ValueError(f"Cannot update position with {order.status} order")

        if order.symbol != position.symbol:
            raise ValueError(f"Symbol mismatch: position {position.symbol} vs order {order.symbol}")

        # Determine fill price
        if fill_price:
            price = fill_price.value
        elif order.average_fill_price:
            price = order.average_fill_price
        else:
            raise ValueError("No fill price available")

        # Determine quantity change
        if order.side == OrderSide.BUY:
            quantity_change = order.filled_quantity
        else:
            quantity_change = -order.filled_quantity

        # Check if adding or reducing
        if position.is_long() and quantity_change > 0:
            # Adding to long
            position.add_to_position(quantity_change, price)
        elif position.is_short() and quantity_change < 0:
            # Adding to short
            position.add_to_position(quantity_change, price)
        else:
            # Reducing position
            position.reduce_position(abs(quantity_change), price)

    def close_position(
        self, position: Position, order: Order, exit_price: Price | None = None
    ) -> Decimal:
        """Close a position with an order.

        Args:
            position: Position to close
            order: The closing order
            exit_price: Optional override price

        Returns:
            Realized P&L from closing

        Raises:
            ValueError: If order is incompatible
        """
        if position.is_closed():
            raise ValueError("Position is already closed")

        if order.symbol != position.symbol:
            raise ValueError(f"Symbol mismatch: position {position.symbol} vs order {order.symbol}")

        # Determine exit price
        if exit_price:
            price = exit_price.value
        elif order.average_fill_price:
            price = order.average_fill_price
        else:
            raise ValueError("No exit price available")

        return position.close_position(price)

    def calculate_pnl(self, position: Position, current_price: Price) -> Money:
        """Calculate P&L for a position.

        Args:
            position: Position to calculate P&L for
            current_price: Current market price

        Returns:
            Money object with P&L amount
        """
        position.update_market_price(current_price.value)

        if position.is_closed():
            pnl = position.realized_pnl
        else:
            pnl = position.get_total_pnl()
            if pnl is None:
                pnl = Decimal("0")

        return Money(pnl, "USD")

    def merge_positions(self, positions: list[Position]) -> Position | None:
        """Merge multiple positions of the same symbol.

        Args:
            positions: List of positions to merge

        Returns:
            Merged position or None if empty list

        Raises:
            ValueError: If positions have different symbols
        """
        if not positions:
            return None

        if len(positions) == 1:
            return positions[0]

        # Validate all same symbol
        symbol = positions[0].symbol
        if not all(p.symbol == symbol for p in positions):
            raise ValueError("Cannot merge positions with different symbols")

        # Calculate weighted average entry
        total_quantity = Decimal("0")
        total_cost = Decimal("0")
        total_pnl = Decimal("0")
        total_commission = Decimal("0")

        for pos in positions:
            qty = abs(pos.quantity)
            total_quantity += pos.quantity
            total_cost += qty * pos.average_entry_price
            total_pnl += pos.realized_pnl
            total_commission += pos.commission_paid

        if total_quantity == 0:
            # All positions cancelled out
            merged = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                average_entry_price=Decimal("0"),
                realized_pnl=total_pnl,
                commission_paid=total_commission,
            )
            merged.closed_at = positions[-1].closed_at
        else:
            avg_entry = total_cost / abs(total_quantity)
            merged = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_entry_price=avg_entry,
                realized_pnl=total_pnl,
                commission_paid=total_commission,
            )

        return merged

    def should_close_position(
        self,
        position: Position,
        current_price: Price,
        max_loss: Money | None = None,
        target_profit: Money | None = None,
    ) -> tuple[bool, str]:
        """Determine if a position should be closed.

        Args:
            position: Position to evaluate
            current_price: Current market price
            max_loss: Optional maximum loss threshold
            target_profit: Optional profit target

        Returns:
            Tuple of (should_close, reason)
        """
        position.update_market_price(current_price.value)

        # Check stop loss
        if position.should_stop_loss():
            return True, "Stop loss triggered"

        # Check take profit
        if position.should_take_profit():
            return True, "Take profit triggered"

        # Check max loss
        if max_loss:
            pnl = self.calculate_pnl(position, current_price)
            if pnl.amount < -abs(max_loss.amount):
                return True, f"Max loss exceeded: {pnl.format()}"

        # Check target profit
        if target_profit:
            pnl = self.calculate_pnl(position, current_price)
            if pnl.amount >= target_profit.amount:
                return True, f"Target profit reached: {pnl.format()}"

        return False, ""

    def calculate_position_size(
        self,
        account_balance: Money,
        risk_per_trade: Decimal,
        entry_price: Price,
        stop_loss_price: Price,
    ) -> Quantity:
        """Calculate optimal position size based on risk.

        Args:
            account_balance: Total account balance
            risk_per_trade: Risk percentage (e.g., 0.02 for 2%)
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price

        Returns:
            Optimal position size as Quantity

        Raises:
            ValueError: If parameters are invalid
        """
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("Risk per trade must be between 0 and 1")

        if entry_price.value <= 0 or stop_loss_price.value <= 0:
            raise ValueError("Prices must be positive")

        # Calculate risk amount
        risk_amount = account_balance.amount * risk_per_trade

        # Calculate price difference
        price_diff = abs(entry_price.value - stop_loss_price.value)

        if price_diff == 0:
            raise ValueError("Entry and stop loss prices cannot be the same")

        # Calculate position size
        position_size = risk_amount / price_diff

        return Quantity(position_size.quantize(Decimal("1")))
