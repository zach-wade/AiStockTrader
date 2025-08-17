"""
Portfolio Entity - Manages multiple positions and overall portfolio metrics
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from .position import Position


@dataclass
class PositionRequest:
    """Request parameters for opening a position."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    commission: Decimal = Decimal("0")
    strategy: str | None = None


@dataclass
class Portfolio:
    """
    Portfolio entity managing multiple positions and risk.

    Tracks positions, cash balance, and portfolio-level metrics.
    All financial values use Decimal for precision.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = "Default Portfolio"

    # Capital
    initial_capital: Decimal = Decimal("100000")
    cash_balance: Decimal = Decimal("100000")

    # Positions
    positions: dict[str, Position] = field(default_factory=dict)

    # Risk limits
    max_position_size: Decimal = Decimal("10000")  # Max $ per position
    max_portfolio_risk: Decimal = Decimal("0.02")  # Max 2% portfolio risk
    max_positions: int = 10
    max_leverage: Decimal = Decimal("1.0")  # No leverage by default

    # Performance tracking
    total_realized_pnl: Decimal = Decimal("0")
    total_commission_paid: Decimal = Decimal("0")
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime | None = None

    # Metadata
    strategy: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate portfolio after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate portfolio attributes"""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if self.cash_balance < 0:
            raise ValueError("Cash balance cannot be negative")

        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")

        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            raise ValueError("Max portfolio risk must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")

        if self.max_leverage < 1:
            raise ValueError("Max leverage must be at least 1.0")

    def can_open_position(
        self, symbol: str, quantity: Decimal, price: Decimal
    ) -> tuple[bool, str | None]:
        """
        Check if a new position can be opened.

        Returns:
            Tuple of (can_open, reason_if_not)
        """
        # Check if symbol already has position
        if symbol in self.positions and not self.positions[symbol].is_closed():
            return False, f"Position already exists for {symbol}"

        # Check max positions limit
        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_positions:
            return False, f"Maximum positions limit reached ({self.max_positions})"

        # Check position size limit
        position_value = abs(quantity) * price
        if position_value > self.max_position_size:
            return (
                False,
                f"Position size ${position_value:.2f} exceeds limit ${self.max_position_size:.2f}",
            )

        # Check available cash
        required_capital = position_value  # Assuming no leverage for now
        if required_capital > self.cash_balance:
            return (
                False,
                f"Insufficient cash: ${self.cash_balance:.2f} available, ${required_capital:.2f} required",
            )

        # Check portfolio risk limit
        portfolio_value = self.get_total_value()
        if portfolio_value > 0:
            risk_amount = position_value
            risk_ratio = risk_amount / portfolio_value
            if risk_ratio > self.max_portfolio_risk:
                return (
                    False,
                    f"Position risk {risk_ratio:.1%} exceeds portfolio limit {self.max_portfolio_risk:.1%}",
                )

        return True, None

    def open_position(
        self,
        request: PositionRequest,
    ) -> Position:
        """Open a new position in the portfolio.

        Args:
            request: Position request with parameters

        Returns:
            Newly opened position

        Raises:
            ValueError: If position cannot be opened
        """
        # Validate ability to open
        can_open, reason = self.can_open_position(request.symbol, request.quantity, request.entry_price)
        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

        # Create position
        position = Position.open_position(
            symbol=request.symbol,
            quantity=request.quantity,
            entry_price=request.entry_price,
            commission=request.commission,
            strategy=request.strategy or self.strategy,
        )

        # Update portfolio
        self.positions[request.symbol] = position
        self.cash_balance -= abs(request.quantity) * request.entry_price + request.commission
        self.total_commission_paid += request.commission
        self.trades_count += 1
        self.last_updated = datetime.now(UTC)

        return position

    def close_position(
        self, symbol: str, exit_price: Decimal, commission: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Close a position and update portfolio.

        Returns:
            Realized P&L from closing the position
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        position = self.positions[symbol]
        if position.is_closed():
            raise ValueError(f"Position for {symbol} is already closed")

        # Close position and get P&L
        position_value = abs(position.quantity) * exit_price
        pnl = position.close_position(exit_price, commission)

        # Update portfolio
        self.cash_balance += position_value - commission
        self.total_realized_pnl += pnl
        self.total_commission_paid += commission

        # Track win/loss
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

        self.last_updated = datetime.now(UTC)

        return pnl

    def update_position_price(self, symbol: str, price: Decimal) -> None:
        """Update market price for a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        self.positions[symbol].update_market_price(price)
        self.last_updated = datetime.now(UTC)

    def update_all_prices(self, prices: dict[str, Decimal]) -> None:
        """Update market prices for multiple positions"""
        for symbol, price in prices.items():
            if symbol in self.positions and not self.positions[symbol].is_closed():
                self.positions[symbol].update_market_price(price)

        self.last_updated = datetime.now(UTC)

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def get_open_positions(self) -> list[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if not p.is_closed()]

    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions"""
        return [p for p in self.positions.values() if p.is_closed()]

    def get_total_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)"""
        total = self.cash_balance

        for position in self.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total += position_value

        return total

    def get_positions_value(self) -> Decimal:
        """Calculate total value of all open positions"""
        total = Decimal("0")

        for position in self.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total += position_value

        return total

    def get_unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L"""
        total = Decimal("0")

        for position in self.get_open_positions():
            unrealized = position.get_unrealized_pnl()
            if unrealized is not None:
                total += unrealized

        return total

    def get_total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)"""
        return self.total_realized_pnl + self.get_unrealized_pnl()

    def get_return_percentage(self) -> Decimal:
        """Calculate portfolio return percentage"""
        if self.initial_capital == 0:
            return Decimal("0")

        current_value = self.get_total_value()
        return ((current_value - self.initial_capital) / self.initial_capital) * Decimal("100")

    def get_win_rate(self) -> Decimal | None:
        """Calculate win rate percentage"""
        total_closed = self.winning_trades + self.losing_trades
        if total_closed == 0:
            return None

        return (Decimal(self.winning_trades) / Decimal(total_closed)) * Decimal("100")

    def get_average_win(self) -> Decimal | None:
        """Calculate average winning trade"""
        if self.winning_trades == 0:
            return None

        total_wins = Decimal("0")
        for position in self.get_closed_positions():
            if position.realized_pnl > 0:
                total_wins += position.realized_pnl

        return total_wins / Decimal(self.winning_trades)

    def get_average_loss(self) -> Decimal | None:
        """Calculate average losing trade"""
        if self.losing_trades == 0:
            return None

        total_losses = Decimal("0")
        for position in self.get_closed_positions():
            if position.realized_pnl < 0:
                total_losses += abs(position.realized_pnl)

        return total_losses / Decimal(self.losing_trades)

    def get_profit_factor(self) -> Decimal | None:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = Decimal("0")
        gross_loss = Decimal("0")

        for position in self.get_closed_positions():
            if position.realized_pnl > 0:
                gross_profit += position.realized_pnl
            elif position.realized_pnl < 0:
                gross_loss += abs(position.realized_pnl)

        if gross_loss == 0:
            return None if gross_profit == 0 else Decimal("999.99")  # Cap at 999.99

        return gross_profit / gross_loss

    def get_sharpe_ratio(self, _risk_free_rate: Decimal = Decimal("0.02")) -> Decimal | None:
        """
        Calculate Sharpe ratio (simplified daily).

        Note: This is a simplified version. Production implementation
        would need historical returns data.
        """
        # This would need historical data for proper calculation
        # Placeholder implementation
        return None

    def get_max_drawdown(self) -> Decimal:
        """
        Calculate maximum drawdown.

        Note: This would need historical value tracking for proper calculation.
        """
        # Placeholder - would need historical tracking
        return Decimal("0")

    def to_dict(self) -> dict[str, Any]:
        """Convert portfolio to dictionary for serialization"""
        return {
            "id": str(self.id),
            "name": self.name,
            "cash_balance": float(self.cash_balance),
            "total_value": float(self.get_total_value()),
            "positions_value": float(self.get_positions_value()),
            "unrealized_pnl": float(self.get_unrealized_pnl()),
            "realized_pnl": float(self.total_realized_pnl),
            "total_pnl": float(self.get_total_pnl()),
            "return_pct": float(self.get_return_percentage()),
            "open_positions": len(self.get_open_positions()),
            "total_trades": self.trades_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(self.get_win_rate()) if self.get_win_rate() else None,
            "commission_paid": float(self.total_commission_paid),
        }

    def __str__(self) -> str:
        """String representation"""
        return (
            f"Portfolio({self.name}: Value=${self.get_total_value():.2f}, "
            f"Cash=${self.cash_balance:.2f}, Positions={len(self.get_open_positions())}, "
            f"P&L=${self.get_total_pnl():.2f}, Return={self.get_return_percentage():.2f}%)"
        )
