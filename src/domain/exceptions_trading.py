"""
Trading-specific exception hierarchy for the AI Trading System.

This module provides a comprehensive exception hierarchy for trading operations,
ensuring proper error handling and reporting in a financial system.
"""

from decimal import Decimal
from typing import Any
from uuid import UUID

from .exceptions import DomainException

# ============================================================================
# Trading Exceptions
# ============================================================================


class TradingException(DomainException):
    """Base exception for all trading-related errors."""

    pass


class OrderException(TradingException):
    """Base exception for order-related errors."""

    def __init__(
        self,
        message: str,
        order_id: UUID | str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"order_id": str(order_id)} if order_id else {}
        if symbol:
            details["symbol"] = symbol
        details.update(kwargs)
        super().__init__(message, details)
        self.order_id = order_id
        self.symbol = symbol


class OrderValidationException(OrderException):
    """Raised when order validation fails."""

    pass


class OrderExecutionException(OrderException):
    """Raised when order execution fails."""

    pass


class OrderNotFoundException(OrderException):
    """Raised when an order cannot be found."""

    pass


class OrderAlreadyCancelledException(OrderException):
    """Raised when attempting to cancel an already cancelled order."""

    pass


class OrderAlreadyFilledException(OrderException):
    """Raised when attempting to modify a filled order."""

    pass


class InsufficientFundsException(OrderException):
    """Raised when there are insufficient funds for an order."""

    def __init__(
        self,
        required_amount: Decimal,
        available_amount: Decimal,
        order_id: UUID | str | None = None,
        symbol: str | None = None,
    ) -> None:
        message = (
            f"Insufficient funds: required {required_amount}, " f"available {available_amount}"
        )
        super().__init__(
            message,
            order_id=order_id,
            symbol=symbol,
            required_amount=str(required_amount),
            available_amount=str(available_amount),
        )
        self.required_amount = required_amount
        self.available_amount = available_amount


# ============================================================================
# Position Exceptions
# ============================================================================


class PositionException(TradingException):
    """Base exception for position-related errors."""

    def __init__(
        self,
        message: str,
        position_id: UUID | str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = {"position_id": str(position_id)} if position_id else {}
        if symbol:
            details["symbol"] = symbol
        details.update(kwargs)
        super().__init__(message, details)
        self.position_id = position_id
        self.symbol = symbol


class PositionNotFoundException(PositionException):
    """Raised when a position cannot be found."""

    pass


class PositionAlreadyClosedException(PositionException):
    """Raised when attempting to modify a closed position."""

    pass


class InvalidPositionSizeException(PositionException):
    """Raised when position size is invalid."""

    def __init__(
        self,
        size: Decimal,
        max_size: Decimal | None = None,
        min_size: Decimal | None = None,
        position_id: UUID | str | None = None,
        symbol: str | None = None,
    ) -> None:
        if max_size and size > max_size:
            message = f"Position size {size} exceeds maximum {max_size}"
        elif min_size and size < min_size:
            message = f"Position size {size} below minimum {min_size}"
        else:
            message = f"Invalid position size: {size}"

        super().__init__(
            message,
            position_id=position_id,
            symbol=symbol,
            size=str(size),
            max_size=str(max_size) if max_size else None,
            min_size=str(min_size) if min_size else None,
        )
        self.size = size
        self.max_size = max_size
        self.min_size = min_size


# ============================================================================
# Portfolio Exceptions
# ============================================================================


class PortfolioException(TradingException):
    """Base exception for portfolio-related errors."""

    def __init__(self, message: str, portfolio_id: UUID | str | None = None, **kwargs: Any) -> None:
        details = {"portfolio_id": str(portfolio_id)} if portfolio_id else {}
        details.update(kwargs)
        super().__init__(message, details)
        self.portfolio_id = portfolio_id


class PortfolioNotFoundException(PortfolioException):
    """Raised when a portfolio cannot be found."""

    pass


class PortfolioLimitExceededException(PortfolioException):
    """Raised when portfolio limits are exceeded."""

    pass


class MaxPositionsExceededException(PortfolioException):
    """Raised when maximum number of positions is exceeded."""

    def __init__(
        self, current_positions: int, max_positions: int, portfolio_id: UUID | str | None = None
    ) -> None:
        message = (
            f"Maximum positions exceeded: current {current_positions}, " f"maximum {max_positions}"
        )
        super().__init__(
            message,
            portfolio_id=portfolio_id,
            current_positions=current_positions,
            max_positions=max_positions,
        )
        self.current_positions = current_positions
        self.max_positions = max_positions


# ============================================================================
# Risk Management Exceptions
# ============================================================================


class RiskException(TradingException):
    """Base exception for risk management errors."""

    pass


class RiskLimitExceededException(RiskException):
    """Raised when risk limits are exceeded."""

    def __init__(
        self, risk_type: str, current_value: Decimal, limit_value: Decimal, **kwargs: Any
    ) -> None:
        message = (
            f"{risk_type} risk limit exceeded: " f"current {current_value}, limit {limit_value}"
        )
        details = {
            "risk_type": risk_type,
            "current_value": str(current_value),
            "limit_value": str(limit_value),
        }
        details.update(kwargs)
        super().__init__(message, details)
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value


class MarginCallException(RiskException):
    """Raised when margin requirements are not met."""

    def __init__(self, required_margin: Decimal, available_margin: Decimal, **kwargs: Any) -> None:
        message = f"Margin call: required {required_margin}, " f"available {available_margin}"
        details = {
            "required_margin": str(required_margin),
            "available_margin": str(available_margin),
        }
        details.update(kwargs)
        super().__init__(message, details)
        self.required_margin = required_margin
        self.available_margin = available_margin


class StopLossTriggedException(RiskException):
    """Raised when a stop loss is triggered."""

    def __init__(
        self,
        symbol: str,
        stop_price: Decimal,
        current_price: Decimal,
        position_id: UUID | str | None = None,
    ) -> None:
        message = (
            f"Stop loss triggered for {symbol}: "
            f"stop price {stop_price}, current price {current_price}"
        )
        details = {
            "symbol": symbol,
            "stop_price": str(stop_price),
            "current_price": str(current_price),
            "position_id": str(position_id) if position_id else None,
        }
        super().__init__(message, details)
        self.symbol = symbol
        self.stop_price = stop_price
        self.current_price = current_price
        self.position_id = position_id


# ============================================================================
# Market Data Exceptions
# ============================================================================


class MarketDataException(TradingException):
    """Base exception for market data errors."""

    pass


class DataUnavailableException(MarketDataException):
    """Raised when market data is unavailable."""

    def __init__(
        self, symbol: str | None = None, data_type: str | None = None, **kwargs: Any
    ) -> None:
        message_parts = ["Market data unavailable"]
        if data_type:
            message_parts.append(f"for {data_type}")
        if symbol:
            message_parts.append(f"for symbol {symbol}")

        details = {"symbol": symbol, "data_type": data_type}
        details.update(kwargs)
        super().__init__(" ".join(message_parts), details)
        self.symbol = symbol
        self.data_type = data_type


class InvalidPriceException(MarketDataException):
    """Raised when price data is invalid."""

    def __init__(self, price: Any, reason: str | None = None, symbol: str | None = None) -> None:
        message = f"Invalid price: {price}"
        if reason:
            message += f" - {reason}"

        details = {"price": str(price), "reason": reason, "symbol": symbol}
        super().__init__(message, details)
        self.price = price
        self.reason = reason
        self.symbol = symbol


class StaleDataException(MarketDataException):
    """Raised when market data is stale."""

    def __init__(self, symbol: str, last_update: str, max_age_seconds: int, **kwargs: Any) -> None:
        message = (
            f"Stale market data for {symbol}: "
            f"last update {last_update}, max age {max_age_seconds}s"
        )
        details = {"symbol": symbol, "last_update": last_update, "max_age_seconds": max_age_seconds}
        details.update(kwargs)
        super().__init__(message, details)
        self.symbol = symbol
        self.last_update = last_update
        self.max_age_seconds = max_age_seconds


# ============================================================================
# Broker/Infrastructure Exceptions
# ============================================================================


class BrokerException(TradingException):
    """Base exception for broker-related errors."""

    pass


class BrokerConnectionException(BrokerException):
    """Raised when broker connection fails."""

    def __init__(self, broker_name: str, reason: str | None = None, **kwargs: Any) -> None:
        message = f"Failed to connect to broker {broker_name}"
        if reason:
            message += f": {reason}"

        details = {"broker_name": broker_name, "reason": reason}
        details.update(kwargs)
        super().__init__(message, details)
        self.broker_name = broker_name
        self.reason = reason


class BrokerAPIException(BrokerException):
    """Raised when broker API calls fail."""

    def __init__(
        self, broker_name: str, api_error: str, api_code: str | None = None, **kwargs: Any
    ) -> None:
        message = f"Broker API error ({broker_name}): {api_error}"
        if api_code:
            message += f" [Code: {api_code}]"

        details = {"broker_name": broker_name, "api_error": api_error, "api_code": api_code}
        details.update(kwargs)
        super().__init__(message, details)
        self.broker_name = broker_name
        self.api_error = api_error
        self.api_code = api_code


class BrokerRateLimitException(BrokerException):
    """Raised when broker rate limits are exceeded."""

    def __init__(self, broker_name: str, retry_after: int | None = None, **kwargs: Any) -> None:
        message = f"Rate limit exceeded for broker {broker_name}"
        if retry_after:
            message += f", retry after {retry_after}s"

        details = {"broker_name": broker_name, "retry_after": retry_after}
        details.update(kwargs)
        super().__init__(message, details)
        self.broker_name = broker_name
        self.retry_after = retry_after


# ============================================================================
# Strategy Exceptions
# ============================================================================


class StrategyException(TradingException):
    """Base exception for strategy-related errors."""

    def __init__(self, message: str, strategy_name: str | None = None, **kwargs: Any) -> None:
        details = {"strategy_name": strategy_name} if strategy_name else {}
        details.update(kwargs)
        super().__init__(message, details)
        self.strategy_name = strategy_name


class StrategyInitializationException(StrategyException):
    """Raised when strategy initialization fails."""

    pass


class StrategyExecutionException(StrategyException):
    """Raised when strategy execution fails."""

    pass


class InvalidSignalException(StrategyException):
    """Raised when strategy generates invalid signals."""

    pass


# ============================================================================
# Backtesting Exceptions
# ============================================================================


class BacktestException(TradingException):
    """Base exception for backtesting errors."""

    pass


class InsufficientHistoricalDataException(BacktestException):
    """Raised when there's insufficient historical data for backtesting."""

    def __init__(self, required_bars: int, available_bars: int, symbol: str | None = None) -> None:
        message = (
            f"Insufficient historical data: "
            f"required {required_bars}, available {available_bars}"
        )
        if symbol:
            message += f" for {symbol}"

        details = {
            "required_bars": required_bars,
            "available_bars": available_bars,
            "symbol": symbol,
        }
        super().__init__(message, details)
        self.required_bars = required_bars
        self.available_bars = available_bars
        self.symbol = symbol


class BacktestConfigurationException(BacktestException):
    """Raised when backtest configuration is invalid."""

    pass
