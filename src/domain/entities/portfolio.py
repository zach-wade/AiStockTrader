"""Portfolio Entity - Manages multiple positions and overall portfolio metrics"""

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
    commission: Money = field(default_factory=lambda: Money(Decimal("0")))
    strategy: str | None = None


@dataclass
class Portfolio:
    """Portfolio entity managing positions and risk. Delegates complex operations to services."""

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
        self._validate()
        if not hasattr(self, "version") or self.version is None:
            self.version = 1

    def _increment_version(self) -> None:
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
        from ..services.portfolio_validation_service import PortfolioValidationService

        PortfolioValidationService.validate_portfolio(self)

    # Position Management - Delegated to Service

    def can_open_position(
        self, symbol: str, quantity: Quantity, price: Price
    ) -> tuple[bool, str | None]:
        """Check if a new position can be opened."""
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        return PortfolioTransactionService.can_open_position(self, symbol, quantity, price)

    def open_position(self, request: PositionRequest) -> Position:
        """Open a new position in the portfolio."""
        from ..services.portfolio_state_service import PortfolioStateService
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        position = PortfolioTransactionService.open_position(self, request)
        PortfolioStateService.update_state_after_open_position(self, request, position)

        return position

    def close_position(
        self, symbol: str, exit_price: Price, commission: Money = Money(Decimal("0"))
    ) -> Money:
        """Close a position and update portfolio."""
        from ..services.portfolio_state_service import PortfolioStateService
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        pnl, net_proceeds = PortfolioTransactionService.close_position(
            self, symbol, exit_price, commission
        )
        PortfolioStateService.update_cash_after_close_position(self, net_proceeds, pnl, commission)

        return pnl

    # Price Updates

    def update_position_price(self, symbol: str, price: Price) -> None:
        """Update market price for a position"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        if not self.positions[symbol].is_closed():
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

    # State Management

    def add_cash(self, amount: Money) -> None:
        """Add cash to the portfolio."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")
        self.cash_balance = self.cash_balance + amount
        self._increment_version()

    def deduct_cash(self, amount: Money) -> None:
        """Deduct cash from the portfolio."""
        if amount.amount <= 0:
            raise ValueError("Cash amount must be positive")
        if amount > self.cash_balance:
            raise ValueError(f"Insufficient cash: {self.cash_balance} available, {amount} required")
        self.cash_balance = self.cash_balance - amount
        self._increment_version()

    # Queries

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

    # Core Metrics - Delegated to Service

    def _get_metrics_calculator(self):
        """Get the metrics calculator service."""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator

    # Core Value Calculations
    def get_total_value(self) -> Money:
        """Calculate total portfolio value"""
        return self._get_metrics_calculator().get_total_value(self)

    def get_positions_value(self) -> Money:
        """Calculate total value of all open positions"""
        return self._get_metrics_calculator().get_positions_value(self)

    def get_unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L"""
        return self._get_metrics_calculator().get_unrealized_pnl(self)

    def get_total_pnl(self) -> Money:
        """Calculate total P&L (realized + unrealized)"""
        return self._get_metrics_calculator().get_total_pnl(self)

    # Return Calculations
    def get_return_percentage(self) -> Decimal:
        """Calculate portfolio return percentage"""
        return self._get_metrics_calculator().get_return_percentage(self)

    def get_total_return(self) -> Decimal:
        """Calculate portfolio total return as a ratio (not percentage)"""
        return self._get_metrics_calculator().get_total_return(self)

    # Trade Statistics
    def get_win_rate(self) -> Decimal | None:
        """Calculate win rate percentage"""
        return self._get_metrics_calculator().get_win_rate(self)

    def get_profit_factor(self) -> Decimal | None:
        """Calculate profit factor (gross profits / gross losses)"""
        return self._get_metrics_calculator().get_profit_factor(self)

    def get_average_win(self) -> Money | None:
        """Calculate average winning trade amount"""
        return self._get_metrics_calculator().get_average_win(self)

    def get_average_loss(self) -> Money | None:
        """Calculate average losing trade amount"""
        return self._get_metrics_calculator().get_average_loss(self)

    # Risk Metrics
    def get_sharpe_ratio(self, risk_free_rate: Decimal = Decimal("0.02")) -> Decimal | None:
        """Calculate Sharpe ratio"""
        return self._get_metrics_calculator().get_sharpe_ratio(self, risk_free_rate)

    def get_max_drawdown(self, historical_values: list[Money] | None = None) -> Decimal:
        """Calculate maximum drawdown using historical portfolio values"""
        return self._get_metrics_calculator().get_max_drawdown(self, historical_values)

    # Serialization
    def to_dict(self) -> dict[str, Any]:
        """Convert portfolio to dictionary for serialization"""
        return self._get_metrics_calculator().portfolio_to_dict(self)

    def __str__(self) -> str:
        """String representation of the portfolio"""
        return self._get_metrics_calculator().portfolio_to_string(self)
