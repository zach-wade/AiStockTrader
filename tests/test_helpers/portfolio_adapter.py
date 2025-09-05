"""Test adapter to help migrate tests to new service architecture."""

from decimal import Decimal

from src.application.services.portfolio_service import PortfolioService
from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.services.portfolio_calculator import PortfolioCalculator
from src.domain.services.portfolio_validator_consolidated import PortfolioValidator
from src.domain.value_objects import Money, Price, Quantity


class PortfolioTestAdapter:
    """Adapter to provide backward compatibility for tests during migration."""

    def __init__(self):
        self.service = PortfolioService()
        self.calculator = PortfolioCalculator()
        self.validator = PortfolioValidator()

    @staticmethod
    def open_position(portfolio: Portfolio, request: PositionRequest):
        """Open position using the new service architecture."""
        service = PortfolioService()
        return service.open_position(portfolio, request)

    @staticmethod
    def close_position(
        portfolio: Portfolio,
        symbol: str,
        exit_price: Price,
        commission: Money = Money(Decimal("0")),
        quantity: Quantity | None = None,
    ):
        """Close position using the new service architecture."""
        service = PortfolioService()
        return service.close_position(portfolio, symbol, exit_price, commission, quantity)

    @staticmethod
    def can_open_position(portfolio: Portfolio, symbol: str, quantity: Quantity, price: Price):
        """Check if position can be opened using the new validator."""
        return PortfolioValidator.can_open_position(portfolio, symbol, quantity, price)

    @staticmethod
    def get_total_value(portfolio: Portfolio):
        """Get total value using the new calculator."""
        return PortfolioCalculator.get_total_value(portfolio)

    @staticmethod
    def get_positions_value(portfolio: Portfolio):
        """Get positions value using the new calculator."""
        return PortfolioCalculator.get_positions_value(portfolio)

    @staticmethod
    def get_unrealized_pnl(portfolio: Portfolio):
        """Get unrealized P&L using the new calculator."""
        return PortfolioCalculator.get_unrealized_pnl(portfolio)

    @staticmethod
    def get_total_pnl(portfolio: Portfolio):
        """Get total P&L using the new calculator."""
        return PortfolioCalculator.get_total_pnl(portfolio)

    @staticmethod
    def get_return_percentage(portfolio: Portfolio):
        """Get return percentage using the new calculator."""
        return PortfolioCalculator.get_return_percentage(portfolio)

    @staticmethod
    def get_win_rate(portfolio: Portfolio):
        """Get win rate using the new calculator."""
        return PortfolioCalculator.get_win_rate(portfolio)

    @staticmethod
    def get_profit_factor(portfolio: Portfolio):
        """Get profit factor using the new calculator."""
        return PortfolioCalculator.get_profit_factor(portfolio)

    @staticmethod
    def get_sharpe_ratio(portfolio: Portfolio, risk_free_rate: Decimal = Decimal("0.02")):
        """Get Sharpe ratio using the new calculator."""
        return PortfolioCalculator.get_sharpe_ratio(portfolio, risk_free_rate)

    @staticmethod
    def get_max_drawdown(portfolio: Portfolio, historical_values=None):
        """Get max drawdown using the new calculator."""
        return PortfolioCalculator.get_max_drawdown(portfolio, historical_values)
