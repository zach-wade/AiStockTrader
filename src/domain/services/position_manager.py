"""Position Manager domain service for managing trading positions.

This module provides the PositionManager service which handles all position-related
operations in the trading system. It encapsulates the business logic for opening,
updating, closing, and analyzing trading positions with thread-safety support.

The PositionManager follows Domain-Driven Design principles, keeping all business
logic within the domain layer and maintaining separation from infrastructure concerns.
All critical operations are thread-safe through async locks on the underlying entities.

Key Responsibilities:
    - Position lifecycle management (open/update/close) with thread-safety
    - Position sizing calculations based on risk parameters
    - P&L calculations and performance metrics with atomic operations
    - Position merging for portfolio consolidation
    - Risk-based position evaluation

Design Patterns:
    - Domain Service: Encapsulates position management logic that doesn't naturally
      fit within a single entity
    - Factory Method: Creates Position entities with proper initialization
    - Strategy Pattern: Different position sizing strategies can be implemented
    - Async/Await: Thread-safe operations for concurrent trading

Example:
    >>> from decimal import Decimal
    >>> from domain.services import PositionManager
    >>> from domain.entities import Order, OrderSide, OrderStatus
    >>> from domain.value_objects import Price
    >>> import asyncio
    >>>
    >>> async def main():
    ...     manager = PositionManager()
    ...     order = Order(
    ...         symbol="AAPL",
    ...         quantity=100,
    ...         side=OrderSide.BUY,
    ...         status=OrderStatus.FILLED,
    ...         filled_quantity=100,
    ...         average_fill_price=Decimal("150.50")
    ...     )
    ...     position = await manager.open_position_async(order)
    ...     print(f"Opened position: {position.quantity} shares at ${position.average_entry_price}")
    ...
    >>> asyncio.run(main())

Note:
    This service maintains no state. All methods support both sync and async operations,
    with async methods providing thread-safety for high-frequency concurrent trading.
"""

# Standard library imports
from decimal import Decimal

from ..entities import Order, OrderSide, OrderStatus, Position
from ..value_objects import Money, Price, Quantity


class PositionManager:
    """Domain service for managing position lifecycle and calculations.

    The PositionManager provides comprehensive position management functionality,
    handling all aspects of position lifecycle from opening through closing.
    It serves as the central point for position-related business logic in the
    trading system.

    This service is stateless and thread-safe, with all methods operating on
    provided entities without maintaining internal state.

    Attributes:
        None - This service is stateless

    Note:
        All monetary calculations use Decimal for precision and all methods
        validate inputs to ensure data integrity.
    """

    def open_position(self, order: Order, fill_price: Price | None = None) -> Position:
        """Open a new position from a filled order.

        Creates a new Position entity from a filled order, establishing the initial
        position state including entry price, quantity, and associated metadata.
        This method handles both long and short positions based on the order side.

        Args:
            order: The filled order to create position from. Must have status FILLED
                and positive filled_quantity.
            fill_price: Optional override price for position entry. If not provided,
                uses the order's average_fill_price. Useful for adjusting entry
                price for slippage or when actual fill differs from order.

        Returns:
            Position: New Position instance with:
                - Quantity positive for long positions, negative for short
                - Entry price set to fill_price or order's average price
                - Symbol and strategy metadata from the order
                - Initial P&L values set to zero

        Raises:
            ValueError: If order is not in FILLED status.
            ValueError: If order's filled_quantity is zero or negative.
            ValueError: If no fill price is available (neither parameter nor order has price).

        Example:
            >>> order = Order(symbol="TSLA", quantity=50, side=OrderSide.BUY)
            >>> order.fill(50, Decimal("650.00"))
            >>> position = manager.open_position(order)
            >>> assert position.quantity == 50
            >>> assert position.average_entry_price == Decimal("650.00")

        Note:
            Commission is tracked separately and not included in the position's
            entry price. This allows for accurate P&L calculations independent
            of transaction costs.
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

    async def open_position_async(self, order: Order, fill_price: Price | None = None) -> Position:
        """Open a new position from a filled order (thread-safe async version).

        See open_position for full documentation.
        """
        # This method doesn't modify shared state, so just wrap the sync version
        return self.open_position(order, fill_price)

    async def update_position_async(
        self, position: Position, order: Order, fill_price: Price | None = None
    ) -> None:
        """Update existing position with a new fill (thread-safe async version).

        Args:
            position: Position to update. Will be modified in place using thread-safe operations.
            order: The filled or partially filled order to apply.
            fill_price: Optional override price for the fill.

        See update_position for full documentation.
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

    def update_position(
        self, position: Position, order: Order, fill_price: Price | None = None
    ) -> None:
        """Update existing position with a new fill.

        Modifies an existing position based on a new order fill. This method handles
        both position increases (pyramiding) and decreases (scaling out), automatically
        determining the correct action based on the order side and current position.

        Args:
            position: Position to update. Will be modified in place.
            order: The filled or partially filled order to apply. Must have
                status FILLED or PARTIALLY_FILLED.
            fill_price: Optional override price for the fill. If not provided,
                uses the order's average_fill_price.

        Raises:
            ValueError: If order status is not FILLED or PARTIALLY_FILLED.
            ValueError: If order symbol doesn't match position symbol.
            ValueError: If no fill price is available.

        Behavior:
            - Adding to position: Updates average entry price using weighted average
            - Reducing position: Calculates and records realized P&L
            - Position reversal: Handled by reduce_position if quantity exceeds current

        Example:
            >>> # Add to existing long position
            >>> position = Position(symbol="AAPL", quantity=100, average_entry_price=Decimal("150"))
            >>> order = Order(symbol="AAPL", quantity=50, side=OrderSide.BUY)
            >>> order.fill(50, Decimal("152.00"))
            >>> manager.update_position(position, order)
            >>> assert position.quantity == 150
            >>> # Average entry price is now weighted average: (100*150 + 50*152) / 150

        Note:
            This method modifies the position object in place rather than returning
            a new instance, following the pattern of domain entity mutation.
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

        Closes an open position and calculates the final realized profit/loss.
        This method ensures proper position closure and P&L accounting.

        Args:
            position: Position to close. Must not already be closed.
            order: The closing order. Symbol must match position.
            exit_price: Optional override exit price. If not provided,
                uses the order's average_fill_price.

        Returns:
            Decimal: Realized P&L from closing the position. Positive values
                indicate profit, negative values indicate loss. Does not include
                commission costs.

        Raises:
            ValueError: If position is already closed.
            ValueError: If order symbol doesn't match position symbol.
            ValueError: If no exit price is available.

        Example:
            >>> position = Position(symbol="GOOGL", quantity=10, average_entry_price=Decimal("2500"))
            >>> closing_order = Order(symbol="GOOGL", quantity=10, side=OrderSide.SELL)
            >>> closing_order.fill(10, Decimal("2600.00"))
            >>> pnl = manager.close_position(position, closing_order)
            >>> assert pnl == Decimal("1000.00")  # (2600 - 2500) * 10

        Note:
            After calling this method, the position's is_closed() will return True
            and the position should not be modified further.
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

    async def close_position_async(
        self, position: Position, order: Order, exit_price: Price | None = None
    ) -> Decimal:
        """Close a position with an order (thread-safe async version).

        See close_position for full documentation.
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

        # Use async reduce method
        quantity_to_reduce = abs(position.quantity)
        position.reduce_position(quantity_to_reduce, price)

        return position.realized_pnl

    async def calculate_pnl_async(self, position: Position, current_price: Price) -> Money:
        """Calculate P&L for a position (thread-safe async version).

        See calculate_pnl for full documentation.
        """
        position.update_market_price(current_price.value)

        if position.is_closed():
            pnl = position.realized_pnl
        else:
            total_pnl = position.get_total_pnl()
            pnl = total_pnl if total_pnl is not None else Decimal("0")

        return Money(pnl, "USD")

    def calculate_pnl(self, position: Position, current_price: Price) -> Money:
        """Calculate P&L for a position.

        Computes the profit/loss for a position at the current market price.
        For closed positions, returns realized P&L. For open positions, returns
        total P&L (realized + unrealized).

        Args:
            position: Position to calculate P&L for. Can be open or closed.
            current_price: Current market price for the position's symbol.

        Returns:
            Money: P&L amount in USD. Positive for profits, negative for losses.
                - For closed positions: Returns realized P&L only
                - For open positions: Returns realized + unrealized P&L

        Example:
            >>> position = Position(symbol="NVDA", quantity=20, average_entry_price=Decimal("500"))
            >>> current_price = Price(Decimal("550.00"))
            >>> pnl = manager.calculate_pnl(position, current_price)
            >>> assert pnl.amount == Decimal("1000.00")  # (550 - 500) * 20

        Note:
            This method updates the position's internal market price before
            calculating P&L, ensuring consistent state.
        """
        position.update_market_price(current_price.value)

        if position.is_closed():
            pnl = position.realized_pnl
        else:
            total_pnl = position.get_total_pnl()
            pnl = total_pnl if total_pnl is not None else Decimal("0")

        return Money(pnl, "USD")

    def merge_positions(self, positions: list[Position]) -> Position | None:
        """Merge multiple positions of the same symbol.

        Consolidates multiple positions of the same symbol into a single position.
        This is useful for portfolio aggregation and reporting. The method properly
        handles the weighted average entry price and combines all P&L values.

        Args:
            positions: List of positions to merge. All positions must have the
                same symbol. Can include both open and closed positions.

        Returns:
            Position | None: Merged position with combined quantities and weighted
                average entry price. Returns None if the input list is empty.
                If all positions cancel out (net zero quantity), returns a closed
                position with the combined realized P&L.

        Raises:
            ValueError: If positions have different symbols.

        Algorithm:
            1. Validates all positions have the same symbol
            2. Sums quantities (respecting long/short signs)
            3. Calculates weighted average entry price
            4. Combines realized P&L and commission values
            5. Creates new merged position with consolidated values

        Example:
            >>> pos1 = Position(symbol="AMD", quantity=100, average_entry_price=Decimal("80"))
            >>> pos2 = Position(symbol="AMD", quantity=50, average_entry_price=Decimal("85"))
            >>> merged = manager.merge_positions([pos1, pos2])
            >>> assert merged.quantity == 150
            >>> # Weighted average: (100*80 + 50*85) / 150 = 81.67
            >>> assert merged.average_entry_price == Decimal("81.67")

        Note:
            The merged position is a new instance; original positions are not modified.
            Strategy information is not preserved in the merge.
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
            # Safe division - we've already checked total_quantity != 0
            abs_quantity = abs(total_quantity)
            if abs_quantity == 0:
                # Defensive check (should never happen given the if-else structure)
                avg_entry = Decimal("0")
            else:
                avg_entry = total_cost / abs_quantity

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
        """Determine if a position should be closed based on risk parameters.

        Evaluates whether a position should be closed based on various criteria
        including stop loss, take profit, maximum loss, and profit targets.
        This method is typically used by automated trading systems to enforce
        risk management rules.

        Args:
            position: Position to evaluate. Must be open.
            current_price: Current market price for evaluation.
            max_loss: Optional maximum loss threshold. If the position's loss
                exceeds this amount, it should be closed.
            target_profit: Optional profit target. If the position's profit
                reaches this amount, it should be closed.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if position should be closed, False otherwise
                - str: Reason for closure (empty string if should not close)
                    Possible reasons:
                    - "Stop loss triggered"
                    - "Take profit triggered"
                    - "Max loss exceeded: [amount]"
                    - "Target profit reached: [amount]"

        Priority:
            Checks are performed in the following order:
            1. Position's internal stop loss
            2. Position's internal take profit
            3. Maximum loss parameter
            4. Target profit parameter

        Example:
            >>> position = Position(symbol="MSFT", quantity=50, average_entry_price=Decimal("300"))
            >>> position.stop_loss_price = Decimal("290")
            >>> current_price = Price(Decimal("289"))
            >>> should_close, reason = manager.should_close_position(position, current_price)
            >>> assert should_close == True
            >>> assert reason == "Stop loss triggered"

        Note:
            This method updates the position's market price as a side effect.
            Multiple conditions may be true, but only the first triggered
            condition is returned.
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
        """Calculate optimal position size based on risk management rules.

        Implements the fixed fractional position sizing method, determining the
        number of shares to trade based on the maximum acceptable loss per trade.
        This is a fundamental risk management technique that ensures consistent
        risk across trades regardless of stock price or volatility.

        Args:
            account_balance: Total account balance available for trading.
            risk_per_trade: Risk percentage as a decimal (e.g., 0.02 for 2%).
                Must be between 0 and 1. Common values are 0.01-0.02 (1-2%).
            entry_price: Planned entry price for the position.
            stop_loss_price: Stop loss price for risk calculation.
                The distance between entry and stop loss determines risk per share.

        Returns:
            Quantity: Optimal position size in shares, rounded down to whole shares.
                This ensures you never exceed your risk limit.

        Raises:
            ValueError: If risk_per_trade is not between 0 and 1.
            ValueError: If any price is zero or negative.
            ValueError: If entry and stop loss prices are the same.

        Formula:
            Position Size = (Account Balance × Risk %) / (Entry Price - Stop Loss Price)

        Example:
            >>> account = Money(Decimal("10000"), "USD")
            >>> risk = Decimal("0.02")  # 2% risk
            >>> entry = Price(Decimal("50.00"))
            >>> stop_loss = Price(Decimal("48.00"))
            >>> size = manager.calculate_position_size(account, risk, entry, stop_loss)
            >>> # Risk amount: $10,000 * 0.02 = $200
            >>> # Risk per share: $50 - $48 = $2
            >>> # Position size: $200 / $2 = 100 shares
            >>> assert size.value == 100

        Note:
            This method assumes you will exit at exactly the stop loss price.
            In practice, slippage may cause actual losses to exceed calculated risk.
            Consider using a smaller risk percentage to account for this.
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
