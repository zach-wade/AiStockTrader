"""Risk Limit Validator domain service for risk limit validation.

This module provides the RiskLimitValidator service which focuses specifically
on risk limit validation and business rule enforcement. It handles pre-trade
risk checks and portfolio risk limit validation.

The service follows Single Responsibility Principle by focusing solely on risk
limit validation, extracted from the original RiskCalculator to improve maintainability.
"""

# Standard library imports
from decimal import Decimal

from ...entities import Order, Portfolio
from ...value_objects import Money, Price


class RiskLimitValidator:
    """Domain service for validating risk limits and business rules.

    This service provides comprehensive risk limit validation functionality
    including pre-trade risk checks, portfolio limits validation, and
    business rule enforcement for risk management.

    The service is stateless and thread-safe, with all methods operating as pure
    functions on provided entities.
    """

    def check_risk_limits(self, portfolio: Portfolio, new_order: Order) -> tuple[bool, str]:
        """Check if a new order violates portfolio risk limits.

        Validates a proposed order against various risk management constraints
        including position limits, leverage limits, and concentration limits.
        This method serves as a pre-trade risk check to prevent excessive risk-taking.

        Args:
            portfolio: Current portfolio state against which to check limits.
            new_order: Proposed new order to validate.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if order is within all risk limits, False otherwise
                - str: Empty string if within limits, otherwise a description of
                    the violated limit

        Risk Checks:
            1. Position limits: Validates portfolio can accept new position
            2. Leverage limits: Ensures total exposure doesn't exceed leverage cap
            3. Concentration limits: Prevents excessive allocation to single position

        Default Limits:
            - Maximum leverage: Defined by portfolio.max_leverage
            - Maximum concentration: 20% of portfolio value per position

        Note:
            Uses limit_price for limit orders or assumes $100 for market orders
            when calculating position values. This is a conservative approach
            for market orders.
        """
        # Check position limits
        estimated_price = new_order.limit_price or Price(
            Decimal("100")
        )  # Use estimate if market order
        can_open, reason = portfolio.can_open_position(
            new_order.symbol, new_order.quantity, estimated_price
        )

        if not can_open:
            return False, reason or "Position cannot be opened"

        # Check leverage
        if portfolio.max_leverage > 1:
            positions_value = portfolio.get_positions_value()
            order_value = Money(new_order.quantity.value * estimated_price.value)
            total_exposure = positions_value + order_value

            leverage = (
                total_exposure.amount / portfolio.cash_balance.amount
                if portfolio.cash_balance.amount > 0
                else Decimal("999")
            )

            if leverage > portfolio.max_leverage:
                return (
                    False,
                    f"Order would exceed leverage limit: {leverage:.2f} > {portfolio.max_leverage}",
                )

        # Check concentration
        max_concentration = Decimal("0.20")  # Max 20% in single position
        portfolio_value = portfolio.get_total_value()

        if portfolio_value.amount > 0:
            order_value = Money(new_order.quantity.value * estimated_price.value)
            concentration = order_value.amount / portfolio_value.amount

            if concentration > max_concentration:
                return (
                    False,
                    f"Order would exceed concentration limit: {concentration:.1%} > {max_concentration:.1%}",
                )

        return True, ""
