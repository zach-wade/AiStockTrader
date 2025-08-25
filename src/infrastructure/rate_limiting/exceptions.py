"""
Rate limiting exceptions for the AI Trading System.

Provides comprehensive error handling for rate limiting scenarios.
"""

from datetime import datetime
from typing import Any


class RateLimitError(Exception):
    """Base exception for all rate limiting errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class RateLimitExceeded(RateLimitError):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: int,
        window_size: str,
        current_count: int,
        retry_after: int | None = None,
        limit_type: str = "requests",
        identifier: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, kwargs)
        self.limit = limit
        self.window_size = window_size
        self.current_count = current_count
        self.retry_after = retry_after
        self.limit_type = limit_type
        self.identifier = identifier

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": "rate_limit_exceeded",
            "message": self.message,
            "limit": self.limit,
            "window_size": self.window_size,
            "current_count": self.current_count,
            "retry_after": self.retry_after,
            "limit_type": self.limit_type,
            "identifier": self.identifier,
            "timestamp": self.timestamp.isoformat(),
            **self.details,
        }


class RateLimitConfigError(RateLimitError):
    """Raised when rate limit configuration is invalid."""

    def __init__(self, message: str, config_field: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, kwargs)
        self.config_field = config_field


class RateLimitStorageError(RateLimitError):
    """Raised when rate limit storage operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        storage_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, kwargs)
        self.operation = operation
        self.storage_backend = storage_backend


class RateLimitServiceUnavailable(RateLimitError):
    """Raised when rate limiting service is unavailable."""

    def __init__(self, message: str, service: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, kwargs)
        self.service = service


class TradingRateLimitExceeded(RateLimitExceeded):
    """Specialized exception for trading-specific rate limits."""

    def __init__(
        self,
        message: str,
        trading_action: str,
        user_id: str | None = None,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.trading_action = trading_action
        self.user_id = user_id
        self.symbol = symbol

    def to_dict(self) -> dict[str, Any]:
        """Convert trading rate limit exception to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "trading_action": self.trading_action,
                "user_id": self.user_id,
                "symbol": self.symbol,
            }
        )
        return base_dict


class APIRateLimitExceeded(RateLimitExceeded):
    """Specialized exception for API rate limits."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        method: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.method = method
        self.api_key = api_key

    def to_dict(self) -> dict[str, Any]:
        """Convert API rate limit exception to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "endpoint": self.endpoint,
                "method": self.method,
                "api_key": self.api_key,
            }
        )
        return base_dict


class IPRateLimitExceeded(RateLimitExceeded):
    """Specialized exception for IP-based rate limits."""

    def __init__(self, message: str, ip_address: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.ip_address = ip_address

    def to_dict(self) -> dict[str, Any]:
        """Convert IP rate limit exception to dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "ip_address": self.ip_address,
            }
        )
        return base_dict
