"""
Portfolio Entity - Manages multiple positions and overall portfolio metrics
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..exceptions import StaleDataException
from ..value_objects import Money, Price, Quantity
from .position import Position


@dataclass
class PositionRequest:
    """Request parameters for opening a position."""

    symbol: str
    quantity: Quantity
    entry_price: Price
    commission: Money = Money(Decimal("0"))
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
    initial_capital: Money = Money(Decimal("100000"))
    cash_balance: Money = Money(Decimal("100000"))

    # Positions
    positions: dict[str, Position] = field(default_factory=dict)

    # Risk limits
    max_position_size: Money = Money(Decimal("10000"))  # Max $ per position
    max_portfolio_risk: Decimal = Decimal("0.02")  # Max 2% portfolio risk
    max_positions: int = 10
    max_leverage: Decimal = Decimal("1.0")  # No leverage by default

    # Performance tracking
    total_realized_pnl: Money = Money(Decimal("0"))
    total_commission_paid: Money = Money(Decimal("0"))
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime | None = None

    # Metadata
    strategy: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    version: int = 1  # For optimistic locking

    def __post_init__(self) -> None:
        """Validate portfolio after initialization"""
        self._validate()
        # Ensure version is initialized
        if not hasattr(self, "version") or self.version is None:
            self.version = 1

    def _increment_version(self) -> None:
        """Increment the version number for optimistic locking."""
        self.version = getattr(self, "version", 1) + 1
        self.last_updated = datetime.now(UTC)

    def _check_version(self, expected_version: int | None = None) -> None:
        """Check version for optimistic locking.

        Args:
            expected_version: The expected version number. If provided, will raise
                            StaleDataException if current version doesn't match.
        """
        if expected_version is not None:
            current_version = getattr(self, "version", 1)
            if current_version != expected_version:
                raise StaleDataException(
                    entity_type="Portfolio",
                    entity_id=self.id,
                    expected_version=expected_version,
                    actual_version=current_version,
                )

    def _validate(self) -> None:
        """Validate portfolio attributes"""
        # Validate initial capital (Money)
        if self.initial_capital.amount <= 0:
            raise ValueError("Initial capital must be positive")

        # Validate cash balance (Money)
        if self.cash_balance.amount < 0:
            raise ValueError("Cash balance cannot be negative")

        # Validate max position size (Money)
        if self.max_position_size and self.max_position_size.amount <= 0:
            raise ValueError("Max position size must be positive")

        if self.max_portfolio_risk is not None and (
            self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1
        ):
            raise ValueError("Max portfolio risk must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")

        if self.max_leverage < 1:
            raise ValueError("Max leverage must be at least 1.0")

    def can_open_position(
        self, symbol: str, quantity: Quantity, price: Price
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
        position_value = Money(abs(quantity.value) * price.value)
        if position_value > self.max_position_size:
            return (
                False,
                f"Position size {position_value} exceeds limit {self.max_position_size}",
            )

        # Check available cash
        required_capital = position_value  # Assuming no leverage for now
        if required_capital > self.cash_balance:
            return (
                False,
                f"Insufficient cash: {self.cash_balance} available, {required_capital} required",
            )

        # Check portfolio risk limit
        portfolio_value = self.get_total_value()
        if portfolio_value.amount > 0:
            risk_amount = position_value
            risk_ratio = risk_amount.amount / portfolio_value.amount
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
        can_open, reason = self.can_open_position(
            request.symbol, request.quantity, request.entry_price
        )
        if not can_open:
            raise ValueError(f"Cannot open position: {reason}")

        # Double-check position doesn't already exist
        if request.symbol in self.positions and not self.positions[request.symbol].is_closed():
            raise ValueError(f"Position already exists for {request.symbol}")

        # Check cash availability
        position_cost = Money(abs(request.quantity.value) * request.entry_price.value)
        required_cash = position_cost + request.commission
        if required_cash > self.cash_balance:
            raise ValueError(
                f"Insufficient cash: {self.cash_balance} available, {required_cash} required"
            )

        # Create position object
        position = Position.open_position(
            symbol=request.symbol,
            quantity=request.quantity,
            entry_price=request.entry_price,
            commission=request.commission,
            strategy=request.strategy or self.strategy,
        )

        # Update portfolio state
        self.cash_balance = self.cash_balance - required_cash
        self.total_commission_paid = self.total_commission_paid + request.commission
        self.trades_count += 1
        self.positions[request.symbol] = position
        self._increment_version()

        return position

    def close_position(
        self, symbol: str, exit_price: Price, commission: Money = Money(Decimal("0"))
    ) -> Money:
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
        position_proceeds = Money(abs(position.quantity.value) * exit_price.value)
        pnl = position.close_position(exit_price, commission)

        # Update portfolio state
        self.cash_balance = self.cash_balance + position_proceeds - commission
        self.total_realized_pnl = self.total_realized_pnl + pnl
        self.total_commission_paid = self.total_commission_paid + commission

        # Track win/loss
        if pnl.amount > 0:
            self.winning_trades += 1
        elif pnl.amount < 0:
            self.losing_trades += 1

        self._increment_version()

        return pnl

    def update_position_price(self, symbol: str, price: Price) -> None:
        """Update market price for a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        self.positions[symbol].update_market_price(price)
        self._increment_version()

    def update_all_prices(self, prices: dict[str, Price]) -> None:
        """Update market prices for multiple positions"""
        updated = False
        for symbol, price in prices.items():
            if symbol in self.positions and not self.positions[symbol].is_closed():
                self.positions[symbol].update_market_price(price)
                updated = True

        if updated:
            self._increment_version()

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def get_open_positions(self) -> list[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if not p.is_closed()]

    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions"""
        return [p for p in self.positions.values() if p.is_closed()]

    def get_total_value(self) -> Money:
        """Calculate total portfolio value (cash + positions)"""
        total = self.cash_balance

        for position in self.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total = total + position_value

        return total

    def get_positions_value(self) -> Money:
        """Calculate total value of all open positions"""
        total = Money(Decimal("0"))

        for position in self.get_open_positions():
            position_value = position.get_position_value()
            if position_value is not None:
                total = total + position_value

        return total

    def get_unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L"""
        total = Money(Decimal("0"))

        for position in self.get_open_positions():
            unrealized = position.get_unrealized_pnl()
            if unrealized is not None:
                total = total + unrealized

        return total

    def get_total_pnl(self) -> Money:
        """Calculate total P&L (realized + unrealized)"""
        return self.total_realized_pnl + self.get_unrealized_pnl()

    def get_return_percentage(self) -> Decimal:
        """Calculate portfolio return percentage"""
        if self.initial_capital.amount == 0:
            return Decimal("0")

        current_value = self.get_total_value()
        return (
            (current_value.amount - self.initial_capital.amount) / self.initial_capital.amount
        ) * Decimal("100")

    def get_win_rate(self) -> Decimal | None:
        """Calculate win rate percentage"""
        total_closed = self.winning_trades + self.losing_trades
        if total_closed == 0:
            return None

        return (Decimal(self.winning_trades) / Decimal(total_closed)) * Decimal("100")

    def get_average_win(self) -> Money | None:
        """Calculate average winning trade"""
        if self.winning_trades == 0:
            return None

        total_wins = Money(Decimal("0"))
        for position in self.get_closed_positions():
            if position.realized_pnl.amount > 0:
                total_wins = total_wins + position.realized_pnl

        return total_wins / Decimal(self.winning_trades)

    def get_average_loss(self) -> Money | None:
        """Calculate average losing trade"""
        if self.losing_trades == 0:
            return None

        total_losses = Money(Decimal("0"))
        for position in self.get_closed_positions():
            if position.realized_pnl.amount < 0:
                total_losses = total_losses + abs(position.realized_pnl)

        return total_losses / Decimal(self.losing_trades)

    def get_profit_factor(self) -> Decimal | None:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = Money(Decimal("0"))
        gross_loss = Money(Decimal("0"))

        for position in self.get_closed_positions():
            if position.realized_pnl.amount > 0:
                gross_profit = gross_profit + position.realized_pnl
            elif position.realized_pnl.amount < 0:
                gross_loss = gross_loss + abs(position.realized_pnl)

        if gross_loss.amount == 0:
            return None if gross_profit.amount == 0 else Decimal("999.99")  # Cap at 999.99

        return gross_profit.amount / gross_loss.amount

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
            "cash_balance": float(self.cash_balance.amount),
            "total_value": float(self.get_total_value().amount),
            "positions_value": float(self.get_positions_value().amount),
            "unrealized_pnl": float(self.get_unrealized_pnl().amount),
            "realized_pnl": float(self.total_realized_pnl.amount),
            "total_pnl": float(self.get_total_pnl().amount),
            "return_pct": float(self.get_return_percentage()),
            "open_positions": len(self.get_open_positions()),
            "total_trades": self.trades_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(win_rate) if (win_rate := self.get_win_rate()) is not None else None,
            "commission_paid": float(self.total_commission_paid.amount),
        }

    def __str__(self) -> str:
        """String representation"""
        return (
            f"Portfolio({self.name}: Value={self.get_total_value()}, "
            f"Cash={self.cash_balance}, Positions={len(self.get_open_positions())}, "
            f"P&L={self.get_total_pnl()}, Return={self.get_return_percentage():.2f}%)"
        )
