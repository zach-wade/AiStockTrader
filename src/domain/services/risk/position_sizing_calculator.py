"""Position Sizing Calculator domain service for optimal position sizing.

This module provides the PositionSizingCalculator service which focuses specifically
on position sizing calculations including Kelly Criterion and position sizing
optimization logic. It handles mathematical position sizing decisions.

The service follows Single Responsibility Principle by focusing solely on position
sizing calculations, extracted from the original RiskCalculator to improve maintainability.
"""

# Standard library imports
from decimal import Decimal

from ...value_objects import Money


class PositionSizingCalculator:
    """Domain service for calculating optimal position sizing.

    This service provides comprehensive position sizing functionality including
    Kelly Criterion calculations and position sizing optimization logic for
    risk-adjusted position sizing decisions.

    The service is stateless and thread-safe, with all methods operating as pure
    functions on provided data.
    """

    def calculate_kelly_criterion(
        self, win_probability: Decimal, win_amount: Money, loss_amount: Money
    ) -> Decimal:
        """Calculate optimal position size using Kelly Criterion.

        Determines the mathematically optimal fraction of capital to risk based on
        the probability and magnitude of wins and losses. The Kelly Criterion maximizes
        long-term growth rate while avoiding ruin.

        Args:
            win_probability: Probability of winning as decimal (0-1).
                Should be based on historical performance or backtesting.
            win_amount: Average win amount (Money object with positive value).
            loss_amount: Average loss amount (Money object with positive value).

        Returns:
            Decimal: Optimal fraction of capital to risk (0-0.25).
                Capped at 25% for safety as full Kelly can be too aggressive.
                - Negative values indicate unfavorable odds (don't trade)
                - 0-0.10: Conservative position sizing
                - 0.10-0.25: Aggressive position sizing

        Raises:
            ValueError: If win_probability is not between 0 and 1.
            ValueError: If win_amount or loss_amount are not positive.

        Formula:
            f* = (p × b - q) / b
            Where:
            - f* = Optimal fraction of capital to bet
            - p = Probability of winning
            - q = Probability of losing (1 - p)
            - b = Win/loss ratio

        Note:
            The Kelly Criterion assumes:
            - Accurate probability estimates
            - Consistent win/loss amounts
            - Independent trades
            Many traders use "fractional Kelly" (e.g., 25% of full Kelly) for safety.
        """
        if win_probability <= 0 or win_probability >= 1:
            raise ValueError("Win probability must be between 0 and 1")

        if win_amount.amount <= 0 or loss_amount.amount <= 0:
            raise ValueError("Win and loss amounts must be positive")

        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        loss_probability = 1 - win_probability
        win_loss_ratio = win_amount.amount / loss_amount.amount

        kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

        # Cap at 25% for safety (full Kelly can be too aggressive)
        return min(kelly_fraction, Decimal("0.25"))
