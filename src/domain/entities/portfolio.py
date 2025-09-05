"""Portfolio Entity - Pure domain entity for portfolio data and essential domain behavior"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..exceptions import StaleDataException
from ..value_objects import Money, Price, Quantity
from ..value_objects.portfolio_metrics import PortfolioMetrics
from .position import Position


@dataclass
class PositionRequest:
    """Request parameters for opening a position."""

    symbol: str
    quantity: Quantity
    entry_price: Price
    commission: Money = field(default_factory=lambda: Money(Decimal("0")))
    strategy: str | None = None


@dataclass
class Portfolio:
    """Pure domain entity for portfolio data and essential domain behavior.

    This entity maintains portfolio state and provides essential domain methods.
    Complex operations like opening/closing positions are delegated to services
    to follow the Single Responsibility Principle.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = "Default Portfolio"

    # Capital
    initial_capital: Money = field(default_factory=lambda: Money(Decimal("100000")))
    cash_balance: Money = field(default_factory=lambda: Money(Decimal("100000")))

    # Positions
    positions: dict[str, Position] = field(default_factory=dict)

    # Risk limits
    max_position_size: Money = field(default_factory=lambda: Money(Decimal("10000")))
    max_portfolio_risk: Decimal = Decimal("0.02")
    max_positions: int = 10
    max_leverage: Decimal = Decimal("1.0")

    # Performance tracking
    total_realized_pnl: Money = field(default_factory=lambda: Money(Decimal("0")))
    total_commission_paid: Money = field(default_factory=lambda: Money(Decimal("0")))
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_updated: datetime | None = None

    # Metadata
    strategy: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        """Validate portfolio after initialization"""
        # Convert raw Decimal values to Money objects for backward compatibility
        if hasattr(self.initial_capital, "real") and not hasattr(self.initial_capital, "amount"):
            self.initial_capital = Money(self.initial_capital)  # type: ignore
        if hasattr(self.cash_balance, "real") and not hasattr(self.cash_balance, "amount"):
            self.cash_balance = Money(self.cash_balance)  # type: ignore
        if hasattr(self.total_realized_pnl, "real") and not hasattr(
            self.total_realized_pnl, "amount"
        ):
            self.total_realized_pnl = Money(self.total_realized_pnl)  # type: ignore

        self._validate()
        if not hasattr(self, "version") or self.version is None:
            self.version = 1

    def increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self.version = getattr(self, "version", 1) + 1
        self.last_updated = datetime.now(UTC)

    def _check_version(self, expected_version: int | None = None) -> None:
        """Check version for optimistic locking."""
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
        # Basic validation only - complex validation should be done by services
        if self.initial_capital.amount <= 0:
            raise ValueError("Initial capital must be positive")
        if self.cash_balance.amount < 0:
            raise ValueError("Cash balance cannot be negative")
        if self.max_position_size.amount <= 0:
            raise ValueError("Max position size must be positive")
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            raise ValueError("Max portfolio risk must be between 0 and 1")
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        if self.max_leverage < 1:
            raise ValueError("Max leverage must be at least 1.0")

    # Price Updates

    def update_position_price(self, symbol: str, price: Price) -> None:
        """Update market price for a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        if not self.positions[symbol].is_closed():
            self.positions[symbol].update_market_price(price)
            self.increment_version()

    def update_all_prices(self, prices: dict[str, Price]) -> None:
        """Update market prices for multiple positions"""
        updated = False
        for symbol, price in prices.items():
            if symbol in self.positions and not self.positions[symbol].is_closed():
                self.positions[symbol].update_market_price(price)
                updated = True
        if updated:
            self.increment_version()

    # State Management - Simple state changes only

    def add_cash(self, amount: Money) -> None:
        """Add cash to the portfolio."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")
        self.cash_balance = self.cash_balance + amount
        self.increment_version()

    def deduct_cash(self, amount: Money) -> None:
        """Deduct cash from the portfolio."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")
        if amount > self.cash_balance:
            raise ValueError(f"Insufficient cash: {self.cash_balance} available, {amount} required")
        self.cash_balance = self.cash_balance - amount
        self.increment_version()

    # Queries - Essential data access methods

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol"""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has an open position for the symbol."""
        return symbol in self.positions and not self.positions[symbol].is_closed()

    def get_open_positions(self) -> list[Position]:
        """Get all open positions"""
        return [p for p in self.positions.values() if not p.is_closed()]

    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions"""
        return [p for p in self.positions.values() if p.is_closed()]

    def get_position_count(self) -> int:
        """Get the number of open positions."""
        return len(self.get_open_positions())

    def is_position_limit_reached(self) -> bool:
        """Check if position limit is reached"""
        return self.get_position_count() >= self.max_positions

    # Simple domain calculation that doesn't violate SOLID
    # Calculation methods removed - use PortfolioCalculator service instead
    # Following Single Responsibility Principle:
    # - Portfolio entity manages state
    # - PortfolioCalculator service performs calculations

    # Validation method removed - use PortfolioValidator service instead

    # Temporary bridge methods for backward compatibility
    # TODO: Remove these once all callers are updated to use PortfolioCalculator

    def get_total_value(self) -> Money:
        """DEPRECATED: Use PortfolioCalculator.get_total_value() instead.
        Temporary bridge method to prevent runtime errors during refactoring."""
        from ..services.portfolio_calculator import PortfolioCalculator

        return PortfolioCalculator.get_total_value(self)

    def get_total_equity(self) -> Money:
        """DEPRECATED: Use PortfolioCalculator.get_total_value() instead.
        Temporary bridge method to prevent runtime errors during refactoring."""
        return self.get_total_value()

    @property
    def total_equity(self) -> Money:
        """Get total equity as a property for backward compatibility."""
        return self.get_total_equity()

    def get_positions_value(self) -> Money:
        """DEPRECATED: Use PortfolioCalculator.get_positions_value() instead.
        Temporary bridge method to prevent runtime errors during refactoring."""
        from ..services.portfolio_calculator import PortfolioCalculator

        return PortfolioCalculator.get_positions_value(self)

    def get_unrealized_pnl(self) -> Money:
        """DEPRECATED: Use PortfolioCalculator.get_unrealized_pnl() instead.
        Temporary bridge method to prevent runtime errors during refactoring."""
        from ..services.portfolio_calculator import PortfolioCalculator

        return PortfolioCalculator.get_unrealized_pnl(self)

    def can_open_position(
        self, symbol: str, quantity: Quantity, entry_price: Price
    ) -> tuple[bool, str | None]:
        """DEPRECATED: Use PortfolioValidator.can_open_position() instead.
        Temporary bridge method to prevent runtime errors during refactoring."""
        from ..services.portfolio_validator_consolidated import PortfolioValidator

        return PortfolioValidator.can_open_position(self, symbol, quantity, entry_price)

    def get_metrics(self) -> PortfolioMetrics:
        """Get comprehensive portfolio metrics.

        This method delegates all metric calculations to the PortfolioMetrics value object,
        maintaining Single Responsibility Principle for the Portfolio entity.
        """
        return PortfolioMetrics.calculate_from_portfolio(self)

    # Convenience methods for backward compatibility - delegating to PortfolioMetrics
    def get_return_percentage(self) -> Decimal | None:
        """Get portfolio return percentage."""
        metrics = self.get_metrics()
        return metrics.return_percentage

    def get_win_rate(self) -> Decimal | None:
        """Get win rate of closed positions."""
        metrics = self.get_metrics()
        return metrics.win_rate

    def get_profit_factor(self) -> Decimal | None:
        """Get profit factor."""
        metrics = self.get_metrics()
        return metrics.profit_factor

    def get_average_win(self) -> Decimal | None:
        """Get average winning trade amount."""
        metrics = self.get_metrics()
        return metrics.average_win.amount if metrics.average_win else None

    def get_average_loss(self) -> Decimal | None:
        """Get average losing trade amount."""
        metrics = self.get_metrics()
        return metrics.average_loss.amount if metrics.average_loss else None

    def get_sharpe_ratio(self) -> Decimal | None:
        """Get Sharpe ratio."""
        metrics = self.get_metrics()
        return metrics.sharpe_ratio

    def __str__(self) -> str:
        """String representation of the portfolio."""
        return f"Portfolio(id={self.id}, cash={self.cash_balance}, positions={len(self.positions)})"
