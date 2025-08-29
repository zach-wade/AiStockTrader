"""
Portfolio Validator Service

Handles all portfolio validation and risk checking logic.
Extracted from Portfolio entity to follow Single Responsibility Principle.
"""

from typing import TYPE_CHECKING, Any

from ..value_objects import Price, Quantity

if TYPE_CHECKING:
    from ..entities.portfolio import Portfolio


class PortfolioValidator:
    """
    Service for advanced portfolio validation and risk management.

    Handles complex validation logic that requires external data or
    sophisticated risk calculations beyond basic portfolio constraints.
    Note: Basic validations are now handled directly by Portfolio entity.
    """

    @staticmethod
    def validate_advanced_risk_metrics(
        portfolio: "Portfolio",
        symbol: str,
        quantity: Quantity,
        price: Price,
        market_data: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """
        Perform advanced risk validation using market data and correlations.

        This method handles complex risk checks that require external market data,
        correlation analysis, or sophisticated risk models. Basic validation
        is handled by the Portfolio entity itself.

        Args:
            portfolio: The portfolio to check
            symbol: Symbol for the new position
            quantity: Quantity for the new position
            price: Entry price for the new position
            market_data: Optional market data for advanced risk calculations

        Returns:
            Tuple of (passes_advanced_checks, reason_if_not)
        """
        # Basic validation is now handled by Portfolio entity
        # This method focuses on advanced risk metrics

        # For now, delegate to the basic validation in portfolio
        # In a full implementation, this would include:
        # - Sector concentration limits
        # - Correlation-based risk limits
        # - VaR (Value at Risk) calculations
        # - Beta-weighted exposure limits
        # - Liquidity constraints

        return portfolio.can_open_position(symbol, quantity, price)

    @staticmethod
    def validate_regulatory_compliance(portfolio: "Portfolio") -> None:
        """Validate portfolio compliance with regulatory requirements.

        This method handles complex regulatory validations that go beyond
        basic business rules. Basic state validation is handled by Portfolio entity.

        Args:
            portfolio: The portfolio to validate

        Raises:
            ValueError: If regulatory compliance fails
        """
        # Basic validation is now handled by Portfolio entity
        # This method focuses on regulatory compliance:

        # Example regulatory checks (would be expanded in production):
        # - Pattern day trader rules
        # - Margin requirements
        # - Position limits per regulation
        # - Concentration limits
        # - Liquidity requirements

        # For now, ensure the portfolio passes basic validation
        try:
            portfolio._validate()
        except Exception as e:
            raise ValueError(f"Portfolio fails basic validation: {e}")

        # Add any specific regulatory checks here
        pass
