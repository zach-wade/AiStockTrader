"""
Portfolio Transaction Service

Handles position lifecycle operations (open, close, update) for portfolios.
Extracted from Portfolio entity to follow Single Responsibility Principle.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from ..entities.position import Position
from ..value_objects import Money, Price, Quantity
from ..value_objects.converter import ValueObjectConverter

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio, PositionRequest


class PortfolioTransactionService:
    """
    Service for managing portfolio transactions and position lifecycle.

    Handles the complex logic of opening, closing, and updating positions
    while maintaining portfolio state consistency.
    """

    @staticmethod
    def can_open_position(
        portfolio: "Portfolio", symbol: str, quantity: Quantity, entry_price: Price
    ) -> tuple[bool, str | None]:
        """Check if a position can be opened.

        Args:
            portfolio: The portfolio to check
            symbol: Symbol to open position for
            quantity: Position size
            entry_price: Entry price per unit

        Returns:
            Tuple of (can_open, reason_if_not)
        """
        # Check if position already exists
        if portfolio.has_position(symbol):
            return False, f"Position already exists for {symbol}"

        # Check max positions limit
        open_positions = portfolio.get_open_positions()
        if len(open_positions) >= portfolio.max_positions:
            return False, f"Maximum positions limit reached ({portfolio.max_positions})"

        # Calculate required cash
        quantity_value = ValueObjectConverter.extract_value(quantity)
        price_value = ValueObjectConverter.extract_value(entry_price)
        position_cost = Money(str(abs(quantity_value) * price_value))

        # Check position size limit
        if position_cost > portfolio.max_position_size:
            return (
                False,
                f"Position size {position_cost} exceeds limit {portfolio.max_position_size}",
            )

        # Check cash availability
        if position_cost > portfolio.cash_balance:
            return (
                False,
                f"Insufficient cash: {portfolio.cash_balance} available, {position_cost} required",
            )

        # Check leverage limit
        from .portfolio_metrics_calculator import PortfolioMetricsCalculator

        total_value = PortfolioMetricsCalculator.get_total_value(portfolio)
        positions_value = PortfolioMetricsCalculator.get_positions_value(portfolio)
        new_positions_value = positions_value + position_cost
        cash_balance = ValueObjectConverter.extract_amount(portfolio.cash_balance)

        if cash_balance > 0:
            new_leverage = new_positions_value.amount / cash_balance
            if new_leverage > portfolio.max_leverage:
                return False, f"Would exceed max leverage of {portfolio.max_leverage}"

        # Check portfolio risk limit
        portfolio_risk = new_positions_value.amount / total_value.amount
        if portfolio_risk > portfolio.max_portfolio_risk:
            return (
                False,
                f"Position risk {portfolio_risk:.1%} exceeds portfolio limit {portfolio.max_portfolio_risk:.1%}",
            )

        return True, None

    @staticmethod
    def open_position(portfolio: "Portfolio", request: "PositionRequest") -> Position:
        """Open a new position in the portfolio.

        Args:
            portfolio: The portfolio to modify
            request: Position request with parameters

        Returns:
            Newly opened position

        Raises:
            ValueError: If position cannot be opened
        """
        # Validate ability to open
        can_open, reason = PortfolioTransactionService.can_open_position(
            portfolio, request.symbol, request.quantity, request.entry_price
        )
        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

        # Calculate costs
        quantity_value = ValueObjectConverter.extract_value(request.quantity)
        entry_price_value = ValueObjectConverter.extract_value(request.entry_price)
        position_cost = Money(str(abs(quantity_value) * entry_price_value))
        commission_value = ValueObjectConverter.to_decimal(request.commission)
        required_cash = position_cost + Money(str(commission_value))

        # Final cash check
        if required_cash > portfolio.cash_balance:
            raise ValueError(
                f"Insufficient cash: {portfolio.cash_balance} available, {required_cash} required"
            )

        # Create position object
        position = Position.open_position(
            symbol=request.symbol,
            quantity=request.quantity,
            entry_price=request.entry_price,
            commission=request.commission,
            strategy=request.strategy or portfolio.strategy,
        )

        # Update portfolio state (the Portfolio entity will handle the actual state updates)
        # This service just performs the business logic and validations
        return position

    @staticmethod
    def close_position(
        portfolio: "Portfolio",
        symbol: str,
        exit_price: Price,
        commission: Money = Money(Decimal("0")),
    ) -> tuple[Money, Money]:
        """Close a position and calculate P&L.

        Args:
            portfolio: The portfolio containing the position
            symbol: Symbol to close
            exit_price: Exit price per unit
            commission: Commission for the trade

        Returns:
            Tuple of (Realized P&L, Net proceeds after commission)

        Raises:
            ValueError: If position doesn't exist or is already closed
        """
        if symbol not in portfolio.positions:
            raise ValueError(f"No position found for {symbol}")

        position = portfolio.positions[symbol]
        if position.is_closed():
            raise ValueError(f"Position for {symbol} is already closed")

        # Calculate proceeds BEFORE closing (while quantity is still available)
        quantity_value = ValueObjectConverter.extract_value(position.quantity)
        exit_price_value = ValueObjectConverter.extract_value(exit_price)
        position_proceeds = Money(str(abs(quantity_value) * exit_price_value))

        # Close position and get P&L
        pnl = position.close_position(exit_price, commission)

        # Calculate net proceeds for portfolio cash update
        net_proceeds = position_proceeds - commission

        return pnl, net_proceeds

    @staticmethod
    def update_position_price(portfolio: "Portfolio", symbol: str, price: Price) -> None:
        """Update market price for a position.

        Args:
            portfolio: The portfolio containing the position
            symbol: Symbol to update
            price: New market price

        Raises:
            ValueError: If position doesn't exist
        """
        if symbol not in portfolio.positions:
            raise ValueError(f"No position found for {symbol}")

        portfolio.positions[symbol].update_market_price(price)

    @staticmethod
    def record_trade_statistics(portfolio: "Portfolio", pnl: Money) -> None:
        """Record trade statistics for win/loss tracking.

        Args:
            portfolio: The portfolio to update
            pnl: P&L from the trade
        """
        pnl_amount = ValueObjectConverter.extract_amount(pnl)
        if pnl_amount > 0:
            portfolio.winning_trades += 1
        elif pnl_amount < 0:
            portfolio.losing_trades += 1
        # If pnl_amount == 0, it's a breakeven trade (no update)
