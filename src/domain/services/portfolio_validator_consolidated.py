"""Portfolio Validator Service - Handles all portfolio validation logic."""

from decimal import Decimal
from typing import TYPE_CHECKING

from ..value_objects import Money, Price, Quantity

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio, PositionRequest


class PortfolioValidator:
    """Consolidated service for all portfolio validation rules.

    This service combines functionality from:
    - PortfolioValidator (old)
    - PortfolioValidationService
    - PortfolioValidationServiceV2
    - Validation logic from PortfolioTransactionService
    """

    # --- Position Opening Validation ---

    @staticmethod
    def can_open_position(
        portfolio: "Portfolio", symbol: str, quantity: Quantity, price: Price
    ) -> tuple[bool, str | None]:
        """Check if a new position can be opened with comprehensive validation."""
        # Check if position already exists
        if portfolio.has_position(symbol):
            return False, f"Position already exists for {symbol}"

        # Check position limit
        if portfolio.is_position_limit_reached():
            return False, f"Maximum position limit ({portfolio.max_positions}) reached"

        # Calculate required capital
        required_capital = Money(quantity.value * price.value)
        commission_estimate = Money(Decimal("10"))  # Estimate commission
        total_required = required_capital + commission_estimate

        # Check position size limit first (more specific check)
        if required_capital > portfolio.max_position_size:
            return (
                False,
                f"Position size {required_capital} exceeds maximum {portfolio.max_position_size}",
            )

        # Check sufficient cash
        if total_required > portfolio.cash_balance:
            return (
                False,
                f"Insufficient cash: {portfolio.cash_balance} available, {total_required} required",
            )

        # Check portfolio risk limit
        portfolio_value = portfolio.cash_balance
        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value:
                portfolio_value = portfolio_value + position_value

        position_risk_ratio = required_capital.amount / portfolio_value.amount
        if position_risk_ratio > portfolio.max_portfolio_risk:
            return (
                False,
                f"Position risk {position_risk_ratio:.2%} exceeds maximum {portfolio.max_portfolio_risk:.2%}",
            )

        # Check leverage
        total_position_value = required_capital
        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value:
                total_position_value = total_position_value + position_value

        leverage = total_position_value.amount / portfolio_value.amount
        if leverage > portfolio.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds maximum {portfolio.max_leverage}"

        return True, None

    @staticmethod
    def validate_position_request(portfolio: "Portfolio", request: "PositionRequest") -> None:
        """Validate a position request."""
        # Validate request parameters
        if not request.symbol:
            raise ValueError("Symbol is required")

        if request.quantity.value <= 0:
            raise ValueError("Quantity must be positive")

        if request.entry_price.value <= 0:
            raise ValueError("Entry price must be positive")

        if request.commission.amount < 0:
            raise ValueError("Commission cannot be negative")

        # Check if position can be opened
        can_open, reason = PortfolioValidator.can_open_position(
            portfolio, request.symbol, request.quantity, request.entry_price
        )

        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

    # --- Position Closing Validation ---

    @staticmethod
    def can_close_position(
        portfolio: "Portfolio", symbol: str, quantity: Quantity | None = None
    ) -> tuple[bool, str | None]:
        """Check if a position can be closed."""
        position = portfolio.get_position(symbol)

        if not position:
            return False, f"No position found for {symbol}"

        if position.is_closed():
            return False, f"Position for {symbol} is already closed"

        if quantity:
            if quantity.value <= 0:
                return False, "Quantity must be positive"

            if quantity > position.quantity:
                return False, f"Cannot close {quantity} shares, only {position.quantity} available"

        return True, None

    # --- Cash Management Validation ---

    @staticmethod
    def validate_cash_addition(portfolio: "Portfolio", amount: Money) -> None:
        """Validate cash addition."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")

        # Check for overflow
        new_balance = portfolio.cash_balance + amount
        if new_balance.amount > Decimal("1000000000"):  # $1 billion limit
            raise ValueError("Cash balance would exceed maximum allowed")

    @staticmethod
    def validate_cash_deduction(portfolio: "Portfolio", amount: Money) -> None:
        """Validate cash deduction."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")

        if amount > portfolio.cash_balance:
            raise ValueError(
                f"Insufficient cash: {portfolio.cash_balance} available, {amount} required"
            )

    # --- Risk Validation ---

    @staticmethod
    def validate_portfolio_risk(portfolio: "Portfolio") -> list[str]:
        """Validate portfolio against risk limits and return warnings."""
        warnings = []

        # Check concentration risk
        portfolio_value = portfolio.cash_balance
        position_values = {}

        for position in portfolio.get_open_positions():
            position_value = position.get_position_value()
            if position_value:
                portfolio_value = portfolio_value + position_value
                position_values[position.symbol] = position_value

        for symbol, value in position_values.items():
            concentration = value.amount / portfolio_value.amount
            if concentration > Decimal("0.25"):  # 25% concentration warning
                warnings.append(
                    f"High concentration risk: {symbol} represents {concentration:.1%} of portfolio"
                )

        # Check cash ratio
        cash_ratio = portfolio.cash_balance.amount / portfolio_value.amount
        if cash_ratio < Decimal("0.05"):  # Less than 5% cash warning
            warnings.append(
                f"Low cash ratio: {cash_ratio:.1%} (consider maintaining 5-10% cash buffer)"
            )

        # Check position count
        position_count = portfolio.get_position_count()
        if position_count >= portfolio.max_positions * Decimal("0.9"):
            warnings.append(
                f"Near position limit: {position_count}/{portfolio.max_positions} positions open"
            )

        # Check drawdown (simplified)
        total_pnl = portfolio.total_realized_pnl
        for position in portfolio.get_open_positions():
            unrealized = position.get_unrealized_pnl()
            if unrealized:
                total_pnl = total_pnl + unrealized

        if total_pnl.amount < 0:
            drawdown_pct = abs(total_pnl.amount / portfolio.initial_capital.amount)
            if drawdown_pct > Decimal("0.10"):  # 10% drawdown warning
                warnings.append(f"Significant drawdown: {drawdown_pct:.1%}")

        return warnings

    # --- Advanced Validation ---

    @staticmethod
    def validate_advanced_risk_metrics(
        portfolio: "Portfolio",
        var_threshold: Money | None = None,
        max_correlation: Decimal = Decimal("0.7"),
    ) -> list[str]:
        """Validate advanced risk metrics."""
        warnings = []

        # Value at Risk check
        if var_threshold:
            from .portfolio_calculator import PortfolioCalculator

            var = PortfolioCalculator.calculate_value_at_risk(portfolio)
            if var > var_threshold:
                warnings.append(f"VaR exceeds threshold: {var} > {var_threshold}")

        # Check for correlated positions (simplified - would need price data)
        sectors: dict[str, list[str]] = {}
        for position in portfolio.get_open_positions():
            # Extract sector from symbol (simplified logic)
            sector = position.symbol[:2]  # First 2 chars as proxy for sector
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(position.symbol)

        for sector, symbols in sectors.items():
            if len(symbols) > 3:
                warnings.append(
                    f"High sector concentration: {len(symbols)} positions in similar assets ({', '.join(symbols[:3])}...)"
                )

        return warnings

    @staticmethod
    def validate_regulatory_compliance(portfolio: "Portfolio") -> None:
        """Validate portfolio meets regulatory requirements."""
        # Pattern Day Trading Rule (simplified)
        if portfolio.trades_count > 4:  # In reality, would check within 5 business days
            portfolio_value = portfolio.cash_balance
            for position in portfolio.get_open_positions():
                position_value = position.get_position_value()
                if position_value:
                    portfolio_value = portfolio_value + position_value

            if portfolio_value.amount < Decimal("25000"):
                raise ValueError(
                    f"Pattern Day Trading Rule: Portfolio value ${portfolio_value.amount:.2f} "
                    f"below $25,000 minimum with {portfolio.trades_count} day trades"
                )

        # Position limit check (simplified)
        if portfolio.get_position_count() > 200:
            raise ValueError("Exceeds regulatory position limit of 200 open positions")

        # Leverage check
        if portfolio.max_leverage > Decimal("4.0"):
            raise ValueError("Leverage exceeds regulatory limit of 4:1 for day trading")

    # --- Order Validation (moved from OrderValidator) ---

    @staticmethod
    def validate_order_for_portfolio(
        portfolio: "Portfolio", symbol: str, quantity: Quantity, price: Price, order_type: str
    ) -> tuple[bool, str | None]:
        """Validate if an order can be executed for the portfolio."""
        if order_type.upper() in ["BUY", "BUY_TO_OPEN"]:
            return PortfolioValidator.can_open_position(portfolio, symbol, quantity, price)
        elif order_type.upper() in ["SELL", "SELL_TO_CLOSE"]:
            return PortfolioValidator.can_close_position(portfolio, symbol, quantity)
        else:
            return False, f"Unknown order type: {order_type}"
