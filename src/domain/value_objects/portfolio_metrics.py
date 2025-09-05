"""
PortfolioMetrics Value Object

This value object encapsulates all portfolio performance metrics and calculations,
helping to reduce the responsibilities of the Portfolio entity (SRP compliance).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.domain.value_objects.money import Money

if TYPE_CHECKING:
    from src.domain.entities.portfolio import Portfolio


@dataclass(frozen=True)
class PortfolioMetrics:
    """
    Value object representing portfolio performance metrics.

    This immutable object contains all calculated metrics for a portfolio at a point in time.
    All calculations use FinancialSafety for safe division and overflow protection.
    """

    # Core metrics
    total_equity: Money
    cash_balance: Money
    positions_value: Money
    unrealized_pnl: Money
    realized_pnl: Money

    # Performance metrics
    return_percentage: Decimal | None = None
    win_rate: Decimal | None = None
    profit_factor: Decimal | None = None
    average_win: Money | None = None
    average_loss: Money | None = None
    sharpe_ratio: Decimal | None = None

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk metrics
    max_drawdown: Decimal | None = None
    current_drawdown: Decimal | None = None

    @classmethod
    def calculate_from_portfolio(cls, portfolio: Portfolio) -> PortfolioMetrics:
        """
        Calculate all metrics from a Portfolio entity.

        This factory method extracts calculation logic from the Portfolio entity,
        helping to maintain Single Responsibility Principle.
        """

        # Get basic values
        total_equity = portfolio.get_total_equity()
        cash_balance = portfolio.cash_balance
        positions_value = portfolio.get_positions_value()
        unrealized_pnl = portfolio.get_unrealized_pnl()
        realized_pnl = portfolio.total_realized_pnl

        # Calculate performance metrics using FinancialSafety
        return_percentage = cls._calculate_return_percentage(
            total_equity, portfolio.initial_capital
        )

        # Calculate profit/loss from closed positions
        total_profit = Money(Decimal("0"))
        total_loss = Money(Decimal("0"))
        closed_positions = portfolio.get_closed_positions()

        for pos in closed_positions:
            pnl = pos.get_total_pnl()
            if pnl is not None:
                if pnl.amount > 0:
                    total_profit = total_profit + pnl
                elif pnl.amount < 0:
                    total_loss = total_loss + pnl

        win_rate = cls._calculate_win_rate(portfolio.winning_trades, portfolio.losing_trades)

        profit_factor = cls._calculate_profit_factor(total_profit, total_loss)

        # Calculate average win/loss
        average_win = cls._calculate_average_win(total_profit, portfolio.winning_trades)

        average_loss = cls._calculate_average_loss(total_loss, portfolio.losing_trades)

        # Sharpe ratio would require returns history
        sharpe_ratio = cls._calculate_sharpe_ratio(portfolio)

        return cls(
            total_equity=total_equity,
            cash_balance=cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            return_percentage=return_percentage,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            sharpe_ratio=sharpe_ratio,
            total_trades=portfolio.trades_count,
            winning_trades=portfolio.winning_trades,
            losing_trades=portfolio.losing_trades,
        )

    @staticmethod
    def _calculate_return_percentage(current_value: Money, initial_value: Money) -> Decimal | None:
        """Calculate return percentage using safe division."""
        from src.domain.services.financial_safety import FinancialSafety

        if initial_value.amount <= 0:
            return None

        profit = current_value.amount - initial_value.amount
        return FinancialSafety.safe_divide(
            profit * Decimal("100"), initial_value.amount, default=Decimal("0")
        )

    @staticmethod
    def _calculate_win_rate(winning_trades: int, losing_trades: int) -> Decimal | None:
        """Calculate win rate using safe division."""
        from src.domain.services.financial_safety import FinancialSafety

        total_trades = winning_trades + losing_trades
        if total_trades == 0:
            return None

        return FinancialSafety.safe_divide(
            Decimal(winning_trades) * Decimal("100"), Decimal(total_trades), default=Decimal("0")
        )

    @staticmethod
    def _calculate_profit_factor(total_profit: Money, total_loss: Money) -> Decimal | None:
        """Calculate profit factor using safe division."""
        from src.domain.services.financial_safety import FinancialSafety

        if total_loss.amount == 0:
            return None if total_profit.amount == 0 else Decimal("999999")  # Max value for infinite

        return FinancialSafety.safe_divide(
            abs(total_profit.amount), abs(total_loss.amount), default=Decimal("0")
        )

    @staticmethod
    def _calculate_average_win(total_profit: Money, winning_trades: int) -> Money | None:
        """Calculate average win amount using safe division."""
        from src.domain.services.financial_safety import FinancialSafety

        if winning_trades == 0:
            return None

        avg_amount = FinancialSafety.safe_divide(
            total_profit.amount, Decimal(winning_trades), default=Decimal("0")
        )

        return Money(avg_amount) if avg_amount is not None else None

    @staticmethod
    def _calculate_average_loss(total_loss: Money, losing_trades: int) -> Money | None:
        """Calculate average loss amount using safe division."""
        from src.domain.services.financial_safety import FinancialSafety

        if losing_trades == 0:
            return None

        avg_amount = FinancialSafety.safe_divide(
            abs(total_loss.amount), Decimal(losing_trades), default=Decimal("0")
        )

        return Money(avg_amount) if avg_amount is not None else None

    @staticmethod
    def _calculate_sharpe_ratio(portfolio: Portfolio) -> Decimal | None:
        """
        Calculate Sharpe ratio (placeholder - requires returns history).

        Full implementation would need:
        - Historical returns data
        - Risk-free rate
        - Standard deviation of returns
        """
        # This would require historical returns data which Portfolio doesn't currently track
        # Returning None for now - would be implemented with returns history
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_equity": str(self.total_equity),
            "cash_balance": str(self.cash_balance),
            "positions_value": str(self.positions_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "return_percentage": str(self.return_percentage) if self.return_percentage else None,
            "win_rate": str(self.win_rate) if self.win_rate else None,
            "profit_factor": str(self.profit_factor) if self.profit_factor else None,
            "average_win": str(self.average_win) if self.average_win else None,
            "average_loss": str(self.average_loss) if self.average_loss else None,
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "max_drawdown": str(self.max_drawdown) if self.max_drawdown else None,
            "current_drawdown": str(self.current_drawdown) if self.current_drawdown else None,
        }
