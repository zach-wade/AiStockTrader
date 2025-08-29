"""
Rate Limiting Service - Domain service for rate limiting business rules.

This service handles business logic for determining rate limits and cooldowns
for different endpoints and actions, implementing the Single Responsibility Principle.
"""

from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_requests: int
    window_seconds: int
    burst_allowance: int = 0

    @property
    def requests_per_minute(self) -> int:
        """Get requests per minute based on configuration."""
        if self.window_seconds == 60:
            return self.max_requests
        return int(self.max_requests * 60 / self.window_seconds)

    @property
    def burst_size(self) -> int:
        """Get burst size allowance."""
        return self.burst_allowance


class RateLimitingService:
    """
    Domain service for rate limiting business logic.

    This service contains business rules for determining rate limits,
    separated from the infrastructure rate limiting implementation.
    """

    # Default rate limiting constants
    DEFAULT_COOLDOWN_SECONDS = 60

    # Trading-specific rate limits (business rules)
    TRADING_RATE_LIMITS = {
        "place_order": {"max_requests": 10, "window_seconds": 60, "burst_allowance": 2},
        "cancel_order": {"max_requests": 20, "window_seconds": 60, "burst_allowance": 5},
        "get_positions": {"max_requests": 60, "window_seconds": 60, "burst_allowance": 10},
        "get_market_data": {"max_requests": 100, "window_seconds": 60, "burst_allowance": 20},
    }

    # Endpoint-specific rate limits
    ENDPOINT_RATE_LIMITS = {
        "/api/orders": {"max_requests": 60, "window_seconds": 60, "burst_allowance": 10},
        "/api/positions": {"max_requests": 60, "window_seconds": 60, "burst_allowance": 10},
        "/api/market/quotes": {"max_requests": 300, "window_seconds": 60, "burst_allowance": 50},
        "/api/market/bars": {"max_requests": 300, "window_seconds": 60, "burst_allowance": 50},
        "/api/admin/users": {"max_requests": 30, "window_seconds": 60, "burst_allowance": 5},
        "/api/admin/settings": {"max_requests": 30, "window_seconds": 60, "burst_allowance": 5},
    }

    # Default rate limit
    DEFAULT_RATE_LIMIT = {"max_requests": 120, "window_seconds": 60, "burst_allowance": 20}

    @classmethod
    def get_rate_limit_for_endpoint(cls, endpoint: str) -> dict[str, int]:
        """
        Get rate limit configuration for a specific endpoint.

        This encapsulates business rules about rate limiting.

        Args:
            endpoint: The API endpoint name

        Returns:
            Dictionary with rate limit configuration
        """
        # Check endpoint-specific limits first
        if endpoint in cls.ENDPOINT_RATE_LIMITS:
            return cls.ENDPOINT_RATE_LIMITS[endpoint]

        # Check trading action limits
        if endpoint in cls.TRADING_RATE_LIMITS:
            return cls.TRADING_RATE_LIMITS[endpoint]

        # Default rate limit
        return cls.DEFAULT_RATE_LIMIT

    @classmethod
    def get_rate_limit_config(cls, endpoint: str) -> RateLimitConfig:
        """
        Get rate limit configuration as RateLimitConfig object.

        Args:
            endpoint: The API endpoint name

        Returns:
            RateLimitConfig object with rate limit settings
        """
        config = cls.get_rate_limit_for_endpoint(endpoint)
        return RateLimitConfig(
            max_requests=config["max_requests"],
            window_seconds=config["window_seconds"],
            burst_allowance=config.get("burst_allowance", 0),
        )

    @classmethod
    def get_request_priority(cls, endpoint: str, method: str) -> str:
        """
        Determine request priority based on endpoint and method.

        Args:
            endpoint: The API endpoint path
            method: HTTP method (GET, POST, etc.)

        Returns:
            Priority level: 'high', 'medium', or 'low'
        """
        # High priority - trading operations
        if endpoint.startswith("/api/orders") and method in ["POST", "DELETE"]:
            return "high"

        if endpoint.startswith("/api/positions") and method in ["POST", "DELETE"]:
            return "high"

        # Low priority - admin, health, and reports
        if endpoint.startswith("/api/admin"):
            return "low"

        if endpoint.startswith("/api/health"):
            return "low"

        # Medium priority - data queries
        if endpoint.startswith("/api/market"):
            return "medium"

        if method == "GET":
            return "medium"

        # Default to medium
        return "medium"

    @classmethod
    def get_cooldown_period(cls, endpoint: str) -> int:
        """
        Get cooldown period after rate limit is exceeded.

        Args:
            endpoint: API endpoint

        Returns:
            Cooldown period in seconds
        """
        # Business rule: Trading endpoints have longer cooldowns
        if endpoint.startswith("/api/trading/"):
            return 120  # 2 minutes for trading endpoints

        return cls.DEFAULT_COOLDOWN_SECONDS

    @classmethod
    def get_request_identifier(cls, headers: dict[str, str], fallback: str = "unknown") -> str:
        """
        Extract request identifier from headers according to business rules.

        Args:
            headers: Request headers
            fallback: Default value if no identifier found

        Returns:
            Request identifier (IP address or API key)
        """
        # Priority order (business rule):
        # 1. API Key
        # 2. X-Real-IP
        # 3. X-Forwarded-For (first IP)
        # 4. Remote address
        # 5. Fallback

        # Check for API key
        if "X-API-Key" in headers:
            return f"api:{headers['X-API-Key']}"

        # Check for real IP
        if "X-Real-IP" in headers:
            ip = headers["X-Real-IP"]
            if cls._validate_ip_format(ip):
                return f"ip:{ip}"

        # Check for forwarded IP
        if "X-Forwarded-For" in headers:
            ips = headers["X-Forwarded-For"].split(",")
            if ips:
                ip = ips[0].strip()
                if cls._validate_ip_format(ip):
                    return f"ip:{ip}"

        # Check for remote address
        if "Remote-Addr" in headers:
            ip = headers["Remote-Addr"]
            if cls._validate_ip_format(ip):
                return f"ip:{ip}"

        return fallback

    @classmethod
    def _validate_ip_format(cls, ip: str) -> bool:
        """
        Validate IP address format (helper method).

        Args:
            ip: IP address string

        Returns:
            True if IP format is valid
        """
        import re

        # Simple IPv4 validation
        ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
        if re.match(ipv4_pattern, ip):
            # Check each octet is within valid range
            octets = ip.split(".")
            return all(0 <= int(octet) <= 255 for octet in octets)

        # Simple IPv6 validation (basic check)
        ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
        if re.match(ipv6_pattern, ip):
            return True

        return False
