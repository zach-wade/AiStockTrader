"""
Portfolio Position Manager Service

Handles all position lifecycle management operations.
Extracted from Portfolio entity to follow Single Responsibility Principle.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from ..entities.position import Position
from ..value_objects import Money, Price

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio, PositionRequest


class PortfolioPositionManager:
    """
    Service for advanced position management operations.

    Handles complex position operations that require external coordination,
    batch processing, or sophisticated trading logic. Basic position management
    is now handled directly by the Portfolio entity.
    """

    @staticmethod
    def execute_complex_position_strategy(
        portfolio: "Portfolio",
        requests: list["PositionRequest"],
        execution_strategy: str = "all_or_none",
    ) -> list[Position]:
        """Execute complex position opening strategies.

        Handles batch position opening with sophisticated execution logic.
        Basic single position opening is handled by Portfolio entity.

        Args:
            portfolio: The portfolio to add positions to
            requests: List of position requests to execute
            execution_strategy: Strategy for batch execution

        Returns:
            List of successfully opened positions

        Raises:
            ValueError: If execution strategy fails
        """
        opened_positions = []

        if execution_strategy == "all_or_none":
            # Validate all positions can be opened before opening any
            for request in requests:
                can_open, reason = portfolio.can_open_position(
                    request.symbol, request.quantity, request.entry_price
                )
                if not can_open:
                    raise ValueError(f"Cannot execute batch: {reason} for {request.symbol}")

            # Open all positions
            for request in requests:
                position = portfolio.open_position(request)
                opened_positions.append(position)

        elif execution_strategy == "best_effort":
            # Open as many positions as possible
            for request in requests:
                try:
                    position = portfolio.open_position(request)
                    opened_positions.append(position)
                except ValueError:
                    # Skip positions that cannot be opened
                    continue
        else:
            raise ValueError(f"Unknown execution strategy: {execution_strategy}")

        return opened_positions

    @staticmethod
    def execute_stop_loss_and_take_profit(
        portfolio: "Portfolio", positions_config: dict[str, dict[str, Price]]
    ) -> dict[str, Money]:
        """
        Execute stop-loss and take-profit orders for multiple positions.

        Handles complex position closing logic with risk management rules.
        Basic single position closing is handled by Portfolio entity.

        Args:
            portfolio: The portfolio containing positions
            positions_config: Dict mapping symbol to {stop_loss: Price, take_profit: Price, current_price: Price}

        Returns:
            Dict mapping symbols to realized P&L for closed positions

        Raises:
            ValueError: If positions cannot be processed
        """
        results = {}

        for symbol, config in positions_config.items():
            if not portfolio.has_position(symbol):
                continue

            position = portfolio.get_position(symbol)
            if position is None or position.is_closed():
                continue

            current_price = config.get("current_price")
            stop_loss = config.get("stop_loss")
            take_profit = config.get("take_profit")

            if current_price is None:
                continue

            should_close = False
            exit_price = current_price

            # Check stop-loss condition
            if stop_loss and current_price.value <= stop_loss.value:
                should_close = True
                exit_price = stop_loss

            # Check take-profit condition
            elif take_profit and current_price.value >= take_profit.value:
                should_close = True
                exit_price = take_profit

            if should_close:
                try:
                    pnl = portfolio.close_position(symbol, exit_price)
                    results[symbol] = pnl
                except ValueError:
                    # Log error but continue with other positions
                    continue

        return results

    @staticmethod
    def rebalance_portfolio(
        portfolio: "Portfolio",
        target_allocations: dict[str, Decimal],
        current_prices: dict[str, Price],
        tolerance: Decimal = Decimal("0.05"),
    ) -> dict[str, str]:
        """Rebalance portfolio to target allocations.

        Complex portfolio rebalancing operation that may require opening
        and closing multiple positions.

        Args:
            portfolio: The portfolio to rebalance
            target_allocations: Dict mapping symbols to target allocation percentages
            current_prices: Current market prices for all symbols
            tolerance: Rebalancing tolerance (5% by default)

        Returns:
            Dict mapping symbols to rebalancing actions taken
        """
        actions: dict[str, str] = {}
        total_value = portfolio.get_total_value()

        if total_value.amount == 0:
            return actions

        for symbol, target_pct in target_allocations.items():
            if symbol not in current_prices:
                continue

            target_value = total_value * target_pct
            current_position = portfolio.get_position(symbol)
            current_value = Money(Decimal("0"))

            if current_position and not current_position.is_closed():
                current_value = current_position.get_position_value() or Money(Decimal("0"))

            difference = target_value - current_value
            difference_pct = abs(difference.amount) / total_value.amount

            if difference_pct > tolerance:
                if difference.amount > 0:
                    actions[symbol] = f"BUY ${difference.amount}"
                else:
                    actions[symbol] = f"SELL ${abs(difference.amount)}"

        return actions
