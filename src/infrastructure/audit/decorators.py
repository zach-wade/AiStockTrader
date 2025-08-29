"""
Audit decorators for automatic operation logging.

This module provides decorators that automatically capture and log
audit events for financial operations, providing seamless integration
with minimal code changes.
"""

import functools
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypeVar, cast

from .events import AuditEvent, EventSeverity, OrderEvent, PortfolioEvent, PositionEvent, RiskEvent
from .logger import AuditContext, AuditLogger

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class AuditDecorator:
    """
    Base class for audit decorators.

    Provides common functionality for all audit decorators including
    parameter extraction, result processing, and error handling.
    """

    def __init__(
        self,
        logger: AuditLogger,
        event_type: str,
        resource_type: str,
        action: str,
        include_args: bool = True,
        include_result: bool = True,
        include_errors: bool = True,
        severity: EventSeverity = EventSeverity.MEDIUM,
        is_critical: bool = False,
    ):
        """
        Initialize audit decorator.

        Args:
            logger: Audit logger instance
            event_type: Type of audit event
            resource_type: Type of resource being operated on
            action: Action being performed
            include_args: Whether to include function arguments in audit
            include_result: Whether to include function result in audit
            include_errors: Whether to include error details in audit
            severity: Event severity level
            is_critical: Whether this is a critical event
        """
        self.logger = logger
        self.event_type = event_type
        self.resource_type = resource_type
        self.action = action
        self.include_args = include_args
        self.include_result = include_result
        self.include_errors = include_errors
        self.severity = severity
        self.is_critical = is_critical

    def __call__(self, func: F) -> F:
        """Apply audit decorator to function."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            function_name = func.__name__
            module_name = func.__module__

            # Extract audit context
            audit_context = self._extract_audit_context(args, kwargs)

            # Create base audit event
            event_data = {
                "event_type": self.event_type,
                "resource_type": self.resource_type,
                "action": self.action,
                "function_name": function_name,
                "module_name": module_name,
                "severity": self.severity,
                "is_critical": self.is_critical,
                "execution_start": datetime.now(UTC).isoformat(),
            }

            # Add function arguments if requested
            if self.include_args:
                event_data["function_args"] = self._serialize_args(args, kwargs)

            try:
                # Execute the original function
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time
                event_data["execution_time_ms"] = round(execution_time * 1000, 3)
                event_data["execution_status"] = "success"

                # Add result if requested
                if self.include_result:
                    event_data["function_result"] = self._serialize_result(result)

                # Create and log audit event
                audit_event = self._create_audit_event(event_data, args, kwargs, result)

                try:
                    self.logger.log_event(audit_event, audit_context)
                except Exception as audit_error:
                    # Don't let audit failures break the original function
                    print(f"Audit logging failed: {audit_error}")

                return result

            except Exception as e:
                # Calculate execution time for error case
                execution_time = time.time() - start_time
                event_data["execution_time_ms"] = round(execution_time * 1000, 3)
                event_data["execution_status"] = "error"

                # Add error details if requested
                if self.include_errors:
                    event_data["error"] = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }

                # Create and log audit event for error
                audit_event = self._create_audit_event(event_data, args, kwargs, None, e)

                try:
                    self.logger.log_event(audit_event, audit_context)
                except Exception as audit_error:
                    # Don't let audit failures break the original function
                    print(f"Audit logging failed: {audit_error}")

                # Re-raise the original exception
                raise

        return cast(F, wrapper)

    def _extract_audit_context(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> AuditContext | None:
        """Extract audit context from function arguments."""
        # Look for common context parameter names
        context_keys = ["audit_context", "context", "session_context"]

        for key in context_keys:
            if key in kwargs and isinstance(kwargs[key], AuditContext):
                return cast(AuditContext, kwargs[key])

        # Look for individual context fields
        context_data = {}
        context_fields = ["user_id", "session_id", "request_id", "ip_address"]

        for field in context_fields:
            if field in kwargs:
                context_data[field] = kwargs[field]

        if context_data:
            return AuditContext(**context_data)

        return None

    def _serialize_args(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Serialize function arguments for audit logging."""
        try:
            serialized_args = []
            for i, arg in enumerate(args):
                serialized_args.append(self._serialize_value(arg, f"arg_{i}"))

            serialized_kwargs = {}
            for key, value in kwargs.items():
                serialized_kwargs[key] = self._serialize_value(value, key)

            return {"positional": serialized_args, "keyword": serialized_kwargs}
        except Exception:
            return {"error": "Failed to serialize arguments"}

    def _serialize_result(self, result: Any) -> Any:
        """Serialize function result for audit logging."""
        try:
            return self._serialize_value(result, "result")
        except Exception:
            return {"error": "Failed to serialize result"}

    def _serialize_value(self, value: Any, context: str) -> Any:
        """Serialize a value for audit logging."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v, f"{context}.{k}") for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item, f"{context}[{i}]") for i, item in enumerate(value)]
        elif hasattr(value, "__dict__"):
            # For objects, serialize their __dict__
            return self._serialize_value(value.__dict__, context)
        else:
            return str(value)

    def _create_audit_event(
        self,
        event_data: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
    ) -> AuditEvent:
        """Create specific audit event based on decorator type."""
        # This base implementation creates a generic audit event
        # Subclasses should override this method for specific event types
        from .events import AuditEvent

        class GenericAuditEvent(AuditEvent):
            def get_resource_details(self) -> dict[str, Any]:
                return event_data

            def _validate_resource_data(self) -> None:
                pass

        return GenericAuditEvent(
            event_type=self.event_type,
            resource_type=self.resource_type,
            resource_id=event_data.get("resource_id", "unknown"),
            action=self.action,
            severity=self.severity,
            is_critical=self.is_critical,
            metadata=event_data,
        )


def audit_financial_operation(
    logger: AuditLogger,
    event_type: str,
    action: str,
    resource_type: str = "financial_operation",
    severity: EventSeverity = EventSeverity.HIGH,
    is_critical: bool = False,
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for auditing general financial operations.

    Args:
        logger: Audit logger instance
        event_type: Type of audit event
        action: Action being performed
        resource_type: Type of resource
        severity: Event severity
        is_critical: Whether this is a critical event
        include_args: Include function arguments
        include_result: Include function result

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        audit_decorator = AuditDecorator(
            logger=logger,
            event_type=event_type,
            resource_type=resource_type,
            action=action,
            severity=severity,
            is_critical=is_critical,
            include_args=include_args,
            include_result=include_result,
        )
        return audit_decorator(func)

    return decorator


class OrderAuditDecorator(AuditDecorator):
    """Specialized audit decorator for order operations."""

    def _create_audit_event(
        self,
        event_data: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
    ) -> OrderEvent:
        """Create order-specific audit event."""
        # Extract order information from arguments or result
        order_id = self._extract_order_id(args, kwargs, result)
        symbol = self._extract_symbol(args, kwargs, result)
        quantity = self._extract_quantity(args, kwargs, result)
        price = self._extract_price(args, kwargs, result)
        side = self._extract_side(args, kwargs, result)
        order_type = self._extract_order_type(args, kwargs, result)

        return OrderEvent(
            event_type=self.event_type,
            resource_type=self.resource_type,
            resource_id=order_id or "unknown",
            action=self.action,
            order_id=order_id or "",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            severity=self.severity,
            is_critical=self.is_critical,
            metadata=event_data,
        )

    def _extract_order_id(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> str | None:
        """Extract order ID from function parameters or result."""
        # Look in kwargs first
        if "order_id" in kwargs:
            return str(kwargs["order_id"])
        if "id" in kwargs:
            return str(kwargs["id"])

        # Look in result
        if result and hasattr(result, "order_id"):
            return str(result.order_id)
        if result and hasattr(result, "id"):
            return str(result.id)
        if isinstance(result, dict) and "order_id" in result:
            return str(result["order_id"])

        return None

    def _extract_symbol(self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> str:
        """Extract symbol from function parameters or result."""
        if "symbol" in kwargs:
            return str(kwargs["symbol"])
        if "ticker" in kwargs:
            return str(kwargs["ticker"])

        if result and hasattr(result, "symbol"):
            return str(result.symbol)
        if isinstance(result, dict) and "symbol" in result:
            return str(result["symbol"])

        return "UNKNOWN"

    def _extract_quantity(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> Decimal | None:
        """Extract quantity from function parameters or result."""
        for key in ["quantity", "qty", "shares", "size"]:
            if key in kwargs:
                try:
                    return Decimal(str(kwargs[key]))
                except (ValueError, TypeError):
                    continue

        if result and hasattr(result, "quantity"):
            try:
                return Decimal(str(result.quantity))
            except (ValueError, TypeError):
                pass

        return None

    def _extract_price(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> Decimal | None:
        """Extract price from function parameters or result."""
        for key in ["price", "limit_price", "stop_price"]:
            if key in kwargs:
                try:
                    return Decimal(str(kwargs[key]))
                except (ValueError, TypeError):
                    continue

        if result and hasattr(result, "price"):
            try:
                return Decimal(str(result.price))
            except (ValueError, TypeError):
                pass

        return None

    def _extract_side(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> str | None:
        """Extract order side from function parameters or result."""
        if "side" in kwargs:
            return str(kwargs["side"]).lower()
        if "direction" in kwargs:
            direction = str(kwargs["direction"]).lower()
            return "buy" if direction in ["buy", "long"] else "sell"

        return None

    def _extract_order_type(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> str | None:
        """Extract order type from function parameters or result."""
        if "order_type" in kwargs:
            return str(kwargs["order_type"])
        if "type" in kwargs:
            return str(kwargs["type"])

        return None


def audit_order_operation(
    logger: AuditLogger,
    action: str,
    event_type: str = "order_operation",
    severity: EventSeverity = EventSeverity.HIGH,
    is_critical: bool = False,
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for auditing order operations.

    Args:
        logger: Audit logger instance
        action: Order action (create, modify, cancel, fill)
        event_type: Type of audit event
        severity: Event severity
        is_critical: Whether this is a critical event
        include_args: Include function arguments
        include_result: Include function result

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        audit_decorator = OrderAuditDecorator(
            logger=logger,
            event_type=event_type,
            resource_type="order",
            action=action,
            severity=severity,
            is_critical=is_critical,
            include_args=include_args,
            include_result=include_result,
        )
        return audit_decorator(func)

    return decorator


class PositionAuditDecorator(AuditDecorator):
    """Specialized audit decorator for position operations."""

    def _create_audit_event(
        self,
        event_data: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
    ) -> PositionEvent:
        """Create position-specific audit event."""
        position_id = self._extract_position_id(args, kwargs, result)
        symbol = self._extract_symbol(args, kwargs, result)
        quantity = self._extract_quantity(args, kwargs, result)

        return PositionEvent(
            event_type=self.event_type,
            resource_type=self.resource_type,
            resource_id=position_id or "unknown",
            action=self.action,
            position_id=position_id or "",
            symbol=symbol,
            quantity=quantity,
            severity=self.severity,
            is_critical=self.is_critical,
            metadata=event_data,
        )

    def _extract_position_id(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> str | None:
        """Extract position ID from function parameters or result."""
        if "position_id" in kwargs:
            return str(kwargs["position_id"])
        if "id" in kwargs:
            return str(kwargs["id"])

        if result and hasattr(result, "position_id"):
            return str(result.position_id)

        return None

    def _extract_symbol(self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> str:
        """Extract symbol from function parameters or result."""
        if "symbol" in kwargs:
            return str(kwargs["symbol"])

        if result and hasattr(result, "symbol"):
            return str(result.symbol)

        return "UNKNOWN"

    def _extract_quantity(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> Decimal | None:
        """Extract quantity from function parameters or result."""
        if "quantity" in kwargs:
            try:
                return Decimal(str(kwargs["quantity"]))
            except (ValueError, TypeError):
                pass

        if result and hasattr(result, "quantity"):
            try:
                return Decimal(str(result.quantity))
            except (ValueError, TypeError):
                pass

        return None


def audit_position_operation(
    logger: AuditLogger,
    action: str,
    event_type: str = "position_operation",
    severity: EventSeverity = EventSeverity.HIGH,
    is_critical: bool = False,
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for auditing position operations.

    Args:
        logger: Audit logger instance
        action: Position action (open, close, modify, update)
        event_type: Type of audit event
        severity: Event severity
        is_critical: Whether this is a critical event
        include_args: Include function arguments
        include_result: Include function result

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        audit_decorator = PositionAuditDecorator(
            logger=logger,
            event_type=event_type,
            resource_type="position",
            action=action,
            severity=severity,
            is_critical=is_critical,
            include_args=include_args,
            include_result=include_result,
        )
        return audit_decorator(func)

    return decorator


class PortfolioAuditDecorator(AuditDecorator):
    """Specialized audit decorator for portfolio operations."""

    def _create_audit_event(
        self,
        event_data: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
    ) -> PortfolioEvent:
        """Create portfolio-specific audit event."""
        portfolio_id = self._extract_portfolio_id(args, kwargs, result)

        return PortfolioEvent(
            event_type=self.event_type,
            resource_type=self.resource_type,
            resource_id=portfolio_id or "unknown",
            action=self.action,
            portfolio_id=portfolio_id or "",
            severity=self.severity,
            is_critical=self.is_critical,
            metadata=event_data,
        )

    def _extract_portfolio_id(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> str | None:
        """Extract portfolio ID from function parameters or result."""
        if "portfolio_id" in kwargs:
            return str(kwargs["portfolio_id"])
        if "id" in kwargs:
            return str(kwargs["id"])

        if result and hasattr(result, "portfolio_id"):
            return str(result.portfolio_id)

        return None


def audit_portfolio_operation(
    logger: AuditLogger,
    action: str,
    event_type: str = "portfolio_operation",
    severity: EventSeverity = EventSeverity.HIGH,
    is_critical: bool = False,
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for auditing portfolio operations.

    Args:
        logger: Audit logger instance
        action: Portfolio action (create, update, rebalance, etc.)
        event_type: Type of audit event
        severity: Event severity
        is_critical: Whether this is a critical event
        include_args: Include function arguments
        include_result: Include function result

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        audit_decorator = PortfolioAuditDecorator(
            logger=logger,
            event_type=event_type,
            resource_type="portfolio",
            action=action,
            severity=severity,
            is_critical=is_critical,
            include_args=include_args,
            include_result=include_result,
        )
        return audit_decorator(func)

    return decorator


class RiskAuditDecorator(AuditDecorator):
    """Specialized audit decorator for risk operations."""

    def _create_audit_event(
        self,
        event_data: dict[str, Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any = None,
        error: Exception | None = None,
    ) -> RiskEvent:
        """Create risk-specific audit event."""
        risk_type = self._extract_risk_type(args, kwargs, result)
        threshold_value = self._extract_threshold_value(args, kwargs, result)
        current_value = self._extract_current_value(args, kwargs, result)

        return RiskEvent(
            event_type=self.event_type,
            resource_type=self.resource_type,
            resource_id=f"risk_{risk_type}",
            action=self.action,
            risk_type=risk_type,
            threshold_value=threshold_value,
            current_value=current_value,
            severity=self.severity,
            is_critical=self.is_critical,
            metadata=event_data,
        )

    def _extract_risk_type(self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> str:
        """Extract risk type from function parameters or result."""
        if "risk_type" in kwargs:
            return str(kwargs["risk_type"])
        if "type" in kwargs:
            return str(kwargs["type"])

        return "unknown"

    def _extract_threshold_value(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> Decimal | None:
        """Extract threshold value from function parameters or result."""
        for key in ["threshold", "limit", "threshold_value"]:
            if key in kwargs:
                try:
                    return Decimal(str(kwargs[key]))
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_current_value(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any
    ) -> Decimal | None:
        """Extract current value from function parameters or result."""
        for key in ["current_value", "value", "amount"]:
            if key in kwargs:
                try:
                    return Decimal(str(kwargs[key]))
                except (ValueError, TypeError):
                    continue

        return None


def audit_risk_operation(
    logger: AuditLogger,
    action: str,
    event_type: str = "risk_operation",
    severity: EventSeverity = EventSeverity.CRITICAL,
    is_critical: bool = True,
    include_args: bool = True,
    include_result: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for auditing risk operations.

    Args:
        logger: Audit logger instance
        action: Risk action (check, breach, alert, mitigate)
        event_type: Type of audit event
        severity: Event severity
        is_critical: Whether this is a critical event
        include_args: Include function arguments
        include_result: Include function result

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        audit_decorator = RiskAuditDecorator(
            logger=logger,
            event_type=event_type,
            resource_type="risk",
            action=action,
            severity=severity,
            is_critical=is_critical,
            include_args=include_args,
            include_result=include_result,
        )
        return audit_decorator(func)

    return decorator


# Convenience decorators for common operations
def audit_create_order(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for order creation."""
    return audit_order_operation(logger, action="create", is_critical=True)


def audit_cancel_order(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for order cancellation."""
    return audit_order_operation(logger, action="cancel", is_critical=False)


def audit_fill_order(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for order fills."""
    return audit_order_operation(logger, action="fill", is_critical=True)


def audit_open_position(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for opening positions."""
    return audit_position_operation(logger, action="open", is_critical=True)


def audit_close_position(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for closing positions."""
    return audit_position_operation(logger, action="close", is_critical=True)


def audit_risk_breach(logger: AuditLogger) -> Callable[[F], F]:
    """Convenience decorator for risk breaches."""
    return audit_risk_operation(logger, action="breach", severity=EventSeverity.CRITICAL)
