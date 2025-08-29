"""Position Risk Calculator domain service for individual position risk analysis.

This module provides the PositionRiskCalculator service which focuses specifically
on position-level risk analysis and metrics. It handles comprehensive risk assessment
for individual trading positions including P&L calculations, risk exposure, and
risk/reward analysis.

The service follows Single Responsibility Principle by focusing solely on position
risk calculations, extracted from the original RiskCalculator to improve maintainability.
"""

# Standard library imports
from decimal import Decimal

from ...entities import Position
from ...value_objects import Money, Price


class PositionRiskCalculator:
    """Domain service for calculating individual position risk metrics.

    This service provides comprehensive risk analysis functionality for individual
    positions, including profit/loss calculations, risk exposure analysis, and
    risk/reward ratio calculations.

    The service is stateless and thread-safe, with all methods operating as pure
    functions on provided entities.
    """

    def calculate_position_risk(
        self, position: Position, current_price: Price
    ) -> dict[str, Money | Decimal | None]:
        """Calculate comprehensive risk metrics for a single position.

        Computes various risk and performance metrics for an individual position,
        providing a complete risk profile for position analysis and decision-making.

        Args:
            position: Position to analyze. Can be open or closed.
            current_price: Current market price for the position's symbol.

        Returns:
            dict[str, Money | Decimal | None]: Dictionary containing risk metrics:
                - position_value: Current market value of the position (Money)
                - unrealized_pnl: Unrealized profit/loss (Money, open positions only)
                - realized_pnl: Realized profit/loss from closed portions (Money)
                - total_pnl: Sum of realized and unrealized P&L (Money)
                - return_pct: Percentage return on the position (Decimal)
                - risk_amount: Dollar amount at risk if stop loss is hit (Money)
                Note: Some values may be None if not applicable or computable.

        Behavior:
            - For closed positions: Only realized_pnl and total_pnl are non-zero
            - For open positions: All metrics are calculated based on current price
            - Metrics default to Money(0) or Decimal("0") when not applicable

        Note:
            This method updates the position's market price as a side effect
            to ensure consistent calculations.
        """
        position.update_market_price(current_price)

        metrics: dict[str, Money | Decimal | None] = {
            "position_value": Money(Decimal("0")),
            "unrealized_pnl": Money(Decimal("0")),
            "realized_pnl": position.realized_pnl,
            "total_pnl": Money(Decimal("0")),
            "return_pct": Decimal("0"),
            "risk_amount": Money(Decimal("0")),
        }

        if position.is_closed():
            metrics["total_pnl"] = position.realized_pnl
        else:
            # Position value
            position_value = position.get_position_value()
            if position_value is not None:
                metrics["position_value"] = position_value

            # P&L
            unrealized = position.get_unrealized_pnl()
            if unrealized is not None:
                metrics["unrealized_pnl"] = unrealized
                # Use the position's get_total_pnl method which properly accounts for commission
                total_pnl = position.get_total_pnl()
                if total_pnl is not None:
                    metrics["total_pnl"] = total_pnl
                else:
                    metrics["total_pnl"] = position.realized_pnl + unrealized

            # Return percentage
            return_pct = position.get_return_percentage()
            if return_pct is not None:
                metrics["return_pct"] = return_pct

            # Risk amount (distance to stop loss)
            if position.stop_loss_price:
                risk_per_share = abs(current_price.value - position.stop_loss_price.value)
                metrics["risk_amount"] = Money(risk_per_share * abs(position.quantity.value))

        return metrics

    def calculate_position_risk_reward(
        self, entry_price: Price, stop_loss: Price, take_profit: Price
    ) -> Decimal:
        """Calculate risk/reward ratio for a position.

        Computes the ratio of potential reward to potential risk for a trade setup.
        This is a fundamental metric for evaluating whether a trade offers favorable
        risk-adjusted returns.

        Args:
            entry_price: Planned entry price for the position.
            stop_loss: Stop loss price (risk level).
            take_profit: Take profit target price (reward level).

        Returns:
            Decimal: Risk/reward ratio. Values interpretation:
                - < 1: Risk exceeds reward (generally unfavorable)
                - 1: Risk equals reward (breakeven risk profile)
                - > 2: Reward is at least twice the risk (favorable)
                - > 3: Excellent risk/reward profile

        Raises:
            ValueError: If risk (distance to stop loss) is zero.

        Formula:
            Risk/Reward = (Take Profit - Entry) / (Entry - Stop Loss)

        Note:
            A favorable risk/reward ratio doesn't guarantee profitability.
            The probability of reaching the target vs stop loss must also
            be considered (see expectancy calculations).
        """
        risk = abs(entry_price.value - stop_loss.value)
        reward = abs(take_profit.value - entry_price.value)

        if risk == 0:
            raise ValueError("Risk cannot be zero")

        return reward / risk
