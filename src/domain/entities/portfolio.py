"""Portfolio Entity - Manages multiple positions and overall portfolio metrics"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from ..exceptions import StaleDataException
from ..value_objects import Money, Price, Quantity
from ..value_objects.converter import ValueObjectConverter
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
    """Portfolio entity managing positions and risk. Delegates complex operations to services."""

    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = "Default Portfolio"

    # Capital
    initial_capital: Money = Money(Decimal("100000"))
    cash_balance: Money = Money(Decimal("100000"))

    # Positions
    positions: dict[str, Position] = field(default_factory=dict)

    # Risk limits
    max_position_size: Money = Money(Decimal("10000"))
    max_portfolio_risk: Decimal = Decimal("0.02")
    max_positions: int = 10
    max_leverage: Decimal = Decimal("1.0")

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
        if self.max_position_size and ValueObjectConverter.to_decimal(self.max_position_size) <= 0:
            raise ValueError("Max position size must be positive")
        if self.max_portfolio_risk is not None and (
            self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1
        ):
            raise ValueError("Max portfolio risk must be between 0 and 1")
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        if self.max_leverage < 1:
            raise ValueError("Max leverage must be at least 1.0")
        if ValueObjectConverter.to_decimal(self.initial_capital) <= 0:
            raise ValueError("Initial capital must be positive")
        if ValueObjectConverter.to_decimal(self.cash_balance) < 0:
            raise ValueError("Cash balance cannot be negative")

    # Position Management - Delegated to Service

    def can_open_position(
        self, symbol: str, quantity: Quantity, price: Price
    ) -> tuple[bool, str | None]:
        """Check if a new position can be opened."""
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        return PortfolioTransactionService.can_open_position(self, symbol, quantity, price)

    def open_position(self, request: PositionRequest) -> Position:
        """Open a new position in the portfolio."""
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        position = PortfolioTransactionService.open_position(self, request)

        # Update state
        qty_val = ValueObjectConverter.extract_value(request.quantity)
        price_val = ValueObjectConverter.extract_value(request.entry_price)
        cost = Money(str(abs(qty_val) * price_val))
        comm = ValueObjectConverter.to_decimal(request.commission)
        required = cost + Money(str(comm))

        self.cash_balance = self.cash_balance - required
        self.total_commission_paid = self.total_commission_paid + Money(str(comm))
        self.trades_count += 1
        self.positions[position.symbol] = position
        self._increment_version()

        return position

    def close_position(
        self, symbol: str, exit_price: Price, commission: Money = Money(Decimal("0"))
    ) -> Money:
        """Close a position and update portfolio."""
        from ..services.portfolio_transaction_service import PortfolioTransactionService

        # Service returns both pnl and net proceeds
        pnl, net_proceeds = PortfolioTransactionService.close_position(
            self, symbol, exit_price, commission
        )

        # Update cash with net proceeds
        if net_proceeds.amount > 0:
            self.cash_balance = self.cash_balance + net_proceeds
        elif net_proceeds.amount < 0:
            deduct = Money(abs(net_proceeds.amount))
            if deduct > self.cash_balance:
                raise ValueError(
                    f"Insufficient cash: {self.cash_balance} available, {deduct} required"
                )
            self.cash_balance = self.cash_balance - deduct

        self.total_realized_pnl = self.total_realized_pnl + pnl
        self.total_commission_paid = self.total_commission_paid + commission
        PortfolioTransactionService.record_trade_statistics(self, pnl)
        self._increment_version()

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

    def get_total_value(self) -> Money:
        """Calculate total portfolio value"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_total_value(self)

    def get_positions_value(self) -> Money:
        """Calculate total value of all open positions"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_positions_value(self)

    def get_unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_unrealized_pnl(self)

    def get_total_pnl(self) -> Money:
        """Calculate total P&L (realized + unrealized)"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_total_pnl(self)

    def get_return_percentage(self) -> Decimal:
        """Calculate portfolio return percentage"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_return_percentage(self)

    def get_total_return(self) -> Decimal:
        """Calculate portfolio total return as a ratio (not percentage)"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        # Get return percentage and convert to ratio
        return_pct = PortfolioMetricsCalculator.get_return_percentage(self)
        return return_pct / Decimal("100")

    def get_win_rate(self) -> Decimal | None:
        """Calculate win rate percentage"""
        total_trades = self.winning_trades + self.losing_trades
        return (
            Decimal(self.winning_trades) / Decimal(total_trades) * 100 if total_trades > 0 else None
        )

    def get_profit_factor(self) -> Decimal | None:
        """Calculate profit factor (gross profits / gross losses)"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_profit_factor(self)

    def get_average_win(self) -> Money | None:
        """Calculate average winning trade amount"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_average_win(self)

    def get_average_loss(self) -> Money | None:
        """Calculate average losing trade amount"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_average_loss(self)

    def get_sharpe_ratio(self, risk_free_rate: Decimal = Decimal("0.02")) -> Decimal | None:
        """Calculate Sharpe ratio"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_sharpe_ratio(self, risk_free_rate)

    def get_max_drawdown(self, historical_values: list[Money] | None = None) -> Decimal:
        """Calculate maximum drawdown using historical portfolio values"""
        from ..services.portfolio_metrics_calculator import PortfolioMetricsCalculator

        return PortfolioMetricsCalculator.get_max_drawdown(self, historical_values)

    def to_dict(self) -> dict[str, Any]:
        """Convert portfolio to dictionary for serialization"""
        total_trades = self.winning_trades + self.losing_trades
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
            "win_rate": (self.winning_trades / total_trades * 100) if total_trades > 0 else None,
            "commission_paid": float(self.total_commission_paid.amount),
        }

    def __str__(self) -> str:
        """String representation of the portfolio"""
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()
        return_pct = self.get_return_percentage()
        open_positions = len(self.get_open_positions())

        return (
            f"{self.name} - "
            f"Value=${total_value.amount:,.2f} | "
            f"Cash=${self.cash_balance.amount:,.2f} | "
            f"Positions={open_positions} | "
            f"P&L=${total_pnl.amount:,.2f} | "
            f"Return={return_pct:.2f}%"
        )
