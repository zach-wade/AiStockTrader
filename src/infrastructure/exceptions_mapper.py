"""
Exception mapper for converting between different exception types.

This module provides utilities for mapping exceptions between layers
and external systems (e.g., broker APIs to domain exceptions).
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

from ..application.exceptions_application import UseCaseException
from ..domain.exceptions import DomainException
from ..domain.exceptions_trading import (
    BrokerAPIException,
    BrokerConnectionException,
    BrokerRateLimitException,
    DataUnavailableException,
    InsufficientFundsException,
    InvalidPriceException,
    MarketDataException,
    OrderExecutionException,
    OrderNotFoundException,
    TradingException,
)
from .exceptions_infrastructure import (
    DatabaseException,
    ExternalServiceException,
    ServiceTimeoutException,
    ServiceUnavailableException,
)


class ExceptionMapper:
    """Maps exceptions between different layers and external systems."""

    def __init__(self) -> None:
        self._broker_mappings: dict[str, dict[str | int, type[Exception]]] = {
            "alpaca": self._get_alpaca_mappings(),
            "interactive_brokers": self._get_ib_mappings(),
            "paper": self._get_paper_mappings(),
        }

        self._http_status_mappings = self._get_http_status_mappings()

    def map_broker_exception(
        self, broker_name: str, error: Exception, context: dict[str, Any] | None = None
    ) -> TradingException:
        """
        Map broker-specific exceptions to domain exceptions.

        Args:
            broker_name: Name of the broker
            error: The original exception
            context: Additional context for error mapping

        Returns:
            A domain-level trading exception
        """
        context = context or {}
        broker_mappings = self._broker_mappings.get(broker_name, {})

        # Try to map by error code or type
        error_code = getattr(error, "code", None) or getattr(error, "error_code", None)
        exception_class: type[Exception] | None = (
            broker_mappings.get(error_code) if error_code else None
        )

        if not exception_class:
            # Try to map by exception type
            exception_class = broker_mappings.get(type(error).__name__)

        if exception_class:
            return self._create_exception(exception_class, error, broker_name, context)

        # Default to generic broker API exception
        return BrokerAPIException(
            broker_name=broker_name,
            api_error=str(error),
            api_code=str(error_code) if error_code else None,
            **context,
        )

    def map_http_status_to_exception(
        self,
        status_code: int,
        service_name: str,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Exception:
        """
        Map HTTP status codes to appropriate exceptions.

        Args:
            status_code: HTTP status code
            service_name: Name of the service
            message: Error message
            context: Additional context

        Returns:
            An appropriate exception based on status code
        """
        context = context or {}
        exception_class = self._http_status_mappings.get(status_code)

        if exception_class:
            if issubclass(exception_class, ServiceUnavailableException):
                return ServiceUnavailableException(
                    service_name=service_name,
                    reason=message,
                    retry_after=context.get("retry_after"),
                )
            elif issubclass(exception_class, ServiceTimeoutException):
                return ServiceTimeoutException(
                    service_name=service_name,
                    timeout_seconds=context.get("timeout", 30),
                    operation=context.get("operation"),
                )
            else:
                return exception_class(message or f"HTTP {status_code}")

        # Default to external service exception
        return ExternalServiceException(
            service_name=service_name,
            message=message or f"HTTP {status_code}",
            status_code=status_code,
            **context,
        )

    def map_database_exception(
        self, error: Exception, operation: str | None = None
    ) -> DatabaseException:
        """
        Map database-specific exceptions to infrastructure exceptions.

        Args:
            error: The original database exception
            operation: The database operation being performed

        Returns:
            An infrastructure database exception
        """
        error_str = str(error).lower()

        # Check for common database error patterns
        if "deadlock" in error_str:
            from .exceptions_infrastructure import DatabaseDeadlockException

            return DatabaseDeadlockException(transaction_id=operation)

        if "connection" in error_str or "connect" in error_str:
            from .exceptions_infrastructure import DatabaseConnectionException

            return DatabaseConnectionException(database="trading_db", reason=str(error))

        if "not found" in error_str or "does not exist" in error_str:
            from .exceptions_infrastructure import EntityNotFoundException

            # Try to extract entity info from error
            return EntityNotFoundException(
                entity_type="entity", entity_id="unknown", error=str(error)
            )

        # Default to generic database exception
        from .exceptions_infrastructure import DatabaseQueryException

        return DatabaseQueryException(query=operation or "unknown", error=str(error))

    def wrap_with_context(self, exception: Exception, context: dict[str, Any]) -> Exception:
        """
        Wrap an exception with additional context.

        Args:
            exception: The original exception
            context: Additional context to add

        Returns:
            The exception with added context
        """
        if hasattr(exception, "details"):
            exception.details.update(context)
        elif isinstance(exception, DomainException):
            exception.details = {**exception.details, **context}

        return exception

    # Private methods for broker-specific mappings

    def _get_alpaca_mappings(self) -> dict[str | int, type[Exception]]:
        """Get Alpaca-specific error mappings."""
        return {
            "insufficient_balance": InsufficientFundsException,
            "order_not_found": OrderNotFoundException,
            "invalid_order": OrderExecutionException,
            "market_closed": MarketDataException,
            "rate_limit": BrokerRateLimitException,
            "connection_error": BrokerConnectionException,
            "api_key_invalid": BrokerConnectionException,
            403: BrokerConnectionException,  # Forbidden - likely auth issue
            404: OrderNotFoundException,
            422: OrderExecutionException,  # Unprocessable entity
            429: BrokerRateLimitException,  # Too many requests
            500: BrokerAPIException,  # Server error
            503: ServiceUnavailableException,  # Service unavailable
        }

    def _get_ib_mappings(self) -> dict[str | int, type[Exception]]:
        """Get Interactive Brokers-specific error mappings."""
        return {
            201: InsufficientFundsException,  # Order rejected - margin
            202: OrderExecutionException,  # Order canceled
            203: OrderExecutionException,  # Security not available
            399: OrderNotFoundException,  # Order not found
            504: BrokerConnectionException,  # Not connected
            1100: BrokerConnectionException,  # Connectivity lost
            1101: BrokerConnectionException,  # Connectivity restored
            1102: BrokerConnectionException,  # Connectivity lost - data
            2103: DataUnavailableException,  # Market data farm connection broken
            2104: DataUnavailableException,  # Market data farm OK
            2105: DataUnavailableException,  # HMDS data farm connection broken
            2106: DataUnavailableException,  # HMDS data farm OK
        }

    def _get_paper_mappings(self) -> dict[str | int, type[Exception]]:
        """Get paper trading-specific error mappings."""
        return {
            "insufficient_balance": InsufficientFundsException,
            "order_not_found": OrderNotFoundException,
            "invalid_order": OrderExecutionException,
            "position_not_found": OrderNotFoundException,
            "invalid_price": InvalidPriceException,
            "market_closed": MarketDataException,
        }

    def _get_http_status_mappings(self) -> dict[int, type[Exception]]:
        """Get HTTP status code to exception mappings."""
        return {
            400: UseCaseException,  # Bad request
            401: BrokerConnectionException,  # Unauthorized
            403: BrokerConnectionException,  # Forbidden
            404: OrderNotFoundException,  # Not found
            408: ServiceTimeoutException,  # Request timeout
            422: OrderExecutionException,  # Unprocessable entity
            429: BrokerRateLimitException,  # Too many requests
            500: BrokerAPIException,  # Internal server error
            502: ServiceUnavailableException,  # Bad gateway
            503: ServiceUnavailableException,  # Service unavailable
            504: ServiceTimeoutException,  # Gateway timeout
        }

    def _create_exception(
        self,
        exception_class: type[Exception],
        original_error: Exception,
        broker_name: str,
        context: dict[str, Any],
    ) -> TradingException:
        """
        Create an exception instance with appropriate parameters.

        Args:
            exception_class: The exception class to instantiate
            original_error: The original error
            broker_name: Name of the broker
            context: Additional context

        Returns:
            An instance of the exception class
        """
        error_message = str(original_error)

        # Handle specific exception types
        if exception_class == InsufficientFundsException:
            return InsufficientFundsException(
                required_amount=Decimal(context.get("required", 0)),
                available_amount=Decimal(context.get("available", 0)),
                order_id=context.get("order_id"),
                symbol=context.get("symbol"),
            )

        if exception_class == OrderNotFoundException:
            return OrderNotFoundException(
                message=error_message,
                order_id=context.get("order_id"),
                symbol=context.get("symbol"),
            )

        if exception_class == BrokerRateLimitException:
            return BrokerRateLimitException(
                broker_name=broker_name, retry_after=context.get("retry_after")
            )

        if exception_class == BrokerConnectionException:
            return BrokerConnectionException(broker_name=broker_name, reason=error_message)

        # Default to broker API exception
        return BrokerAPIException(
            broker_name=broker_name,
            api_error=error_message,
            api_code=getattr(original_error, "code", None),
        )


# Global exception mapper instance
exception_mapper = ExceptionMapper()


def with_exception_mapping(
    broker_name: str | None = None, map_to: type[Exception] | None = None
) -> Callable[..., Any]:
    """
    Decorator for automatic exception mapping.

    Args:
        broker_name: Name of the broker for broker-specific mapping
        map_to: Target exception class to map to

    Returns:
        Decorated function with exception mapping
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if broker_name:
                    # Map broker-specific exceptions
                    raise exception_mapper.map_broker_exception(
                        broker_name=broker_name, error=e, context=kwargs
                    ) from e
                elif map_to:
                    # Map to specific exception type
                    if isinstance(e, map_to):
                        raise
                    raise map_to(str(e)) from e
                else:
                    # No mapping, re-raise original
                    raise

        return wrapper

    return decorator
