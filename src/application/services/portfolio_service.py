"""Portfolio Service - Application layer orchestration for portfolio operations."""

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ...domain.entities.position import Position
from ...domain.services.portfolio_calculator import PortfolioCalculator
from ...domain.services.portfolio_validator_consolidated import PortfolioValidator
from ...domain.value_objects import Money, Price, Quantity

if TYPE_CHECKING:
    from ...domain.entities.portfolio import Portfolio, PositionRequest


class PortfolioService:
    """Application layer service for orchestrating portfolio operations.

    This service coordinates between:
    - Domain entities (Portfolio, Position)
    - Domain services (PortfolioCalculator, PortfolioValidator)
    - Infrastructure concerns (repositories, events, etc.)
    """

    def __init__(self) -> None:
        """Initialize the portfolio service."""
        self.calculator = PortfolioCalculator()
        self.validator = PortfolioValidator()

    # --- Position Management ---

    def can_open_position(
        self, portfolio: "Portfolio", symbol: str, quantity: Quantity, price: Price
    ) -> tuple[bool, str | None]:
        """Check if a new position can be opened based on portfolio constraints."""
        return self.validator.can_open_position(portfolio, symbol, quantity, price)

    def open_position(self, portfolio: "Portfolio", request: "PositionRequest") -> Position:
        """Open a new position with full validation and state management."""
        # Validate the request
        self.validator.validate_position_request(portfolio, request)

        # Create the position
        position = Position(
            symbol=request.symbol,
            quantity=request.quantity,
            average_entry_price=request.entry_price,
            current_price=request.entry_price,
            strategy=request.strategy,
        )

        # Calculate total cost
        position_cost = Money(request.quantity.value * request.entry_price.value)
        total_cost = position_cost + request.commission

        # Update portfolio state
        # Deduct cash
        portfolio.cash_balance = portfolio.cash_balance - total_cost

        # Add position
        portfolio.positions[request.symbol] = position

        # Update commission tracking
        portfolio.total_commission_paid = portfolio.total_commission_paid + request.commission

        # Update trade count
        portfolio.trades_count += 1

        # Increment version for optimistic locking
        portfolio.increment_version()

        return position

    def close_position(
        self,
        portfolio: "Portfolio",
        symbol: str,
        exit_price: Price,
        commission: Money = Money(Decimal("0")),
        quantity: Quantity | None = None,
    ) -> Money:
        """Close a position (fully or partially) and return realized P&L."""
        # Validate
        can_close, reason = self.validator.can_close_position(portfolio, symbol, quantity)
        if not can_close:
            raise ValueError(f"Cannot close position: {reason}")

        position = portfolio.get_position(symbol)
        if not position:
            raise ValueError(f"No position found for {symbol}")

        if quantity and quantity < position.quantity:
            # Partial close
            pnl, remaining_position = self._close_partial_position(
                position, quantity, exit_price, commission
            )
            portfolio.positions[symbol] = remaining_position
        else:
            # Full close
            pnl = self._close_full_position(position, exit_price, commission)
            # Mark position as closed but keep for history
            position.mark_as_closed(exit_price)

        # Update portfolio state
        gross_proceeds = Money(
            (quantity.value if quantity else position.quantity.value) * exit_price.value
        )
        net_proceeds = gross_proceeds - commission

        portfolio.cash_balance = portfolio.cash_balance + net_proceeds
        portfolio.total_realized_pnl = portfolio.total_realized_pnl + pnl
        portfolio.total_commission_paid = portfolio.total_commission_paid + commission
        portfolio.trades_count += 1

        if pnl.amount > 0:
            portfolio.winning_trades += 1
        elif pnl.amount < 0:
            portfolio.losing_trades += 1

        portfolio.increment_version()

        return pnl

    def _close_full_position(
        self, position: Position, exit_price: Price, commission: Money
    ) -> Money:
        """Close a position completely."""
        position.close_position(exit_price, commission)
        return position.realized_pnl or Money(Decimal("0"))

    def _close_partial_position(
        self, position: Position, quantity: Quantity, exit_price: Price, commission: Money
    ) -> tuple[Money, Position]:
        """Close part of a position."""
        # Calculate P&L for the partial close
        entry_cost = Money(quantity.value * position.average_entry_price.value)
        exit_value = Money(quantity.value * exit_price.value)
        pnl = exit_value - entry_cost - commission

        # Create new position with remaining quantity
        remaining_quantity = Quantity(position.quantity.value - quantity.value)
        remaining_position = Position(
            symbol=position.symbol,
            quantity=remaining_quantity,
            average_entry_price=position.average_entry_price,
            current_price=position.current_price,
            entry_time=position.entry_time,
            strategy=position.strategy,
        )

        return pnl, remaining_position

    # --- Portfolio Metrics ---

    def get_portfolio_metrics(self, portfolio: "Portfolio") -> dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        return {
            "value": {
                "total": float(self.calculator.get_total_value(portfolio).amount),
                "cash": float(portfolio.cash_balance.amount),
                "positions": float(self.calculator.get_positions_value(portfolio).amount),
            },
            "performance": {
                "total_pnl": float(self.calculator.get_total_pnl(portfolio).amount),
                "unrealized_pnl": float(self.calculator.get_unrealized_pnl(portfolio).amount),
                "realized_pnl": float(portfolio.total_realized_pnl.amount),
                "return_pct": (
                    float(ret_pct)
                    if (ret_pct := self.calculator.get_return_percentage(portfolio)) is not None
                    else 0.0
                ),
                "win_rate": (
                    float(win_rt)
                    if (win_rt := self.calculator.get_win_rate(portfolio)) is not None
                    else 0.0
                ),
                "profit_factor": (
                    float(pf)
                    if (pf := self.calculator.get_profit_factor(portfolio)) is not None
                    else 0.0
                ),
            },
            "risk": {
                "sharpe_ratio": (
                    float(sr)
                    if (sr := self.calculator.get_sharpe_ratio(portfolio)) is not None
                    else 0.0
                ),
                "max_drawdown": float(self.calculator.get_max_drawdown(portfolio)),
                "var_95": float(self.calculator.calculate_value_at_risk(portfolio).amount),
                "warnings": self.validator.validate_portfolio_risk(portfolio),
            },
            "statistics": {
                "positions_open": portfolio.get_position_count(),
                "max_positions": portfolio.max_positions,
                "trades_count": portfolio.trades_count,
                "winning_trades": portfolio.winning_trades,
                "losing_trades": portfolio.losing_trades,
                "commission_paid": float(portfolio.total_commission_paid.amount),
            },
        }

    # --- Risk Management ---

    def validate_portfolio_health(self, portfolio: "Portfolio") -> dict[str, Any]:
        """Perform comprehensive portfolio health check."""
        basic_warnings = self.validator.validate_portfolio_risk(portfolio)
        advanced_warnings = self.validator.validate_advanced_risk_metrics(portfolio)

        try:
            self.validator.validate_regulatory_compliance(portfolio)
            regulatory_status = "compliant"
            regulatory_issues = []
        except ValueError as e:
            regulatory_status = "non-compliant"
            regulatory_issues = [str(e)]

        return {
            "status": (
                "healthy"
                if not (basic_warnings + advanced_warnings + regulatory_issues)
                else "warning"
            ),
            "risk_warnings": basic_warnings,
            "advanced_warnings": advanced_warnings,
            "regulatory_status": regulatory_status,
            "regulatory_issues": regulatory_issues,
        }

    # --- Price Updates ---

    def update_portfolio_prices(self, portfolio: "Portfolio", prices: dict[str, Price]) -> None:
        """Update prices for all positions in the portfolio."""
        portfolio.update_all_prices(prices)

    # --- Portfolio Analysis ---

    def get_top_performers(
        self, portfolio: "Portfolio", current_prices: dict[str, Price], limit: int = 5
    ) -> list[tuple[Position, Money]]:
        """Get top performing positions."""
        positions_by_profit = self.calculator.get_positions_by_profit(portfolio, current_prices)
        return positions_by_profit[:limit]

    def get_worst_performers(
        self, portfolio: "Portfolio", current_prices: dict[str, Price], limit: int = 5
    ) -> list[tuple[Position, Money]]:
        """Get worst performing positions."""
        positions_by_profit = self.calculator.get_positions_by_profit(portfolio, current_prices)
        return (
            positions_by_profit[-limit:]
            if len(positions_by_profit) >= limit
            else positions_by_profit
        )

    def get_portfolio_allocation(self, portfolio: "Portfolio") -> dict[str, Any]:
        """Get portfolio allocation breakdown."""
        total_value = self.calculator.get_total_value(portfolio)
        cash_pct = (
            (portfolio.cash_balance.amount / total_value.amount * 100)
            if total_value.amount > 0
            else Decimal("0")
        )

        allocations: dict[str, Any] = {
            "cash": {"value": float(portfolio.cash_balance.amount), "percentage": float(cash_pct)},
            "positions": {},
        }

        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value and total_value.amount > 0:
                pct = position_value.amount / total_value.amount * 100
                allocations["positions"][position.symbol] = {
                    "value": float(position_value.amount),
                    "percentage": float(pct),
                    "shares": float(position.quantity.value),
                }

        return allocations

    # --- Additional Methods from PortfolioOperationsService ---

    def total_portfolio_value(
        self, portfolio: "Portfolio", current_prices: dict[str, Price]
    ) -> Money:
        """Calculate total portfolio value using provided current prices."""
        # Update all position prices first
        for position in portfolio.get_open_positions():
            if position.symbol not in current_prices:
                raise ValueError(f"Price not found for symbol: {position.symbol}")
            position.update_market_price(current_prices[position.symbol])

        # Now calculate total value using updated prices
        return self.calculator.get_total_value(portfolio)

    def unrealized_pnl(self, portfolio: "Portfolio", current_prices: dict[str, Price]) -> Money:
        """Calculate unrealized P&L using provided current prices."""
        # Update all position prices first
        for position in portfolio.get_open_positions():
            if position.symbol not in current_prices:
                raise ValueError(f"Price not found for symbol: {position.symbol}")
            position.update_market_price(current_prices[position.symbol])

        # Now calculate unrealized PnL using updated prices
        return self.calculator.get_unrealized_pnl(portfolio)

    def to_dict(self, portfolio: "Portfolio") -> dict[str, Any]:
        """Convert portfolio to dictionary for serialization."""
        return self.calculator.portfolio_to_dict(portfolio)

    def to_string(self, portfolio: "Portfolio") -> str:
        """Get detailed string representation of the portfolio."""
        return self.calculator.portfolio_to_string(portfolio)

    # --- Metric Delegation Methods (for compatibility) ---

    def get_total_value(self, portfolio: "Portfolio") -> Money:
        """Calculate total portfolio value."""
        return self.calculator.get_total_value(portfolio)

    def get_positions_value(self, portfolio: "Portfolio") -> Money:
        """Calculate total value of all open positions."""
        return self.calculator.get_positions_value(portfolio)

    def get_unrealized_pnl(self, portfolio: "Portfolio") -> Money:
        """Calculate total unrealized P&L."""
        return self.calculator.get_unrealized_pnl(portfolio)

    def get_total_pnl(self, portfolio: "Portfolio") -> Money:
        """Calculate total P&L (realized + unrealized)."""
        return self.calculator.get_total_pnl(portfolio)

    def get_return_percentage(self, portfolio: "Portfolio") -> Decimal:
        """Calculate portfolio return percentage."""
        return self.calculator.get_return_percentage(portfolio)

    def get_total_return(self, portfolio: "Portfolio") -> Decimal:
        """Calculate portfolio total return as a ratio."""
        return self.calculator.get_total_return(portfolio)

    def get_win_rate(self, portfolio: "Portfolio") -> Decimal | None:
        """Calculate win rate percentage."""
        return self.calculator.get_win_rate(portfolio)

    def get_profit_factor(self, portfolio: "Portfolio") -> Decimal | None:
        """Calculate profit factor (gross profits / gross losses)."""
        return self.calculator.get_profit_factor(portfolio)

    def get_average_win(self, portfolio: "Portfolio") -> Money | None:
        """Calculate average winning trade amount."""
        return self.calculator.get_average_win(portfolio)

    def get_average_loss(self, portfolio: "Portfolio") -> Money | None:
        """Calculate average losing trade amount."""
        return self.calculator.get_average_loss(portfolio)

    def get_sharpe_ratio(
        self, portfolio: "Portfolio", risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal | None:
        """Calculate Sharpe ratio."""
        return self.calculator.get_sharpe_ratio(portfolio, risk_free_rate)

    def get_max_drawdown(
        self, portfolio: "Portfolio", historical_values: list[Money] | None = None
    ) -> Decimal:
        """Calculate maximum drawdown using historical portfolio values."""
        return self.calculator.get_max_drawdown(portfolio, historical_values)
