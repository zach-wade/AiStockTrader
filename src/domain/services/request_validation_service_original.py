"""
Request Validation Service - Domain service for request validation business rules.

This service handles business logic for validating requests, including
business rules about what makes valid requests, headers, and IP addresses.
The actual security enforcement remains in the infrastructure layer.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


class RequestValidationError(Exception):
    """Exception raised when request validation fails."""

    pass


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


class RequestValidationService:
    """
    Domain service for request validation business logic.

    This service contains business rules for validating requests,
    separated from the infrastructure security implementation.
    """

    # Business rules for request validation
    MAX_REQUESTS_PER_MINUTE = 100
    MAX_BURST_REQUESTS = 10
    DEFAULT_COOLDOWN_SECONDS = 60
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB default

    def __init__(self) -> None:
        """Initialize the service with default settings."""
        # IP whitelist (instance attribute for testing)
        self.whitelist_ips: list[str] = []
        # IP blacklist (instance attribute for testing)
        self.blacklist_ips: list[str] = []
        # Allowed CORS origins (instance attribute for testing)
        self.allowed_origins: list[str] = []

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

    # Endpoints requiring signature (business rule)
    SIGNATURE_REQUIRED_ENDPOINTS = [
        "/api/orders",
        "/api/positions",
        "/api/admin",
    ]

    # Supported API versions (business rule)
    SUPPORTED_API_VERSIONS = ["v1", "v2"]

    # Pagination limits (business rule)
    MAX_PAGE_SIZE = 100
    DEFAULT_PAGE_SIZE = 50

    # Date range limits (business rule)
    MAX_DATE_RANGE_DAYS = 365  # Maximum 1 year

    # Endpoint timeouts in milliseconds (business rule)
    ENDPOINT_TIMEOUTS = {
        "/api/orders": 5000,  # 5 seconds for trading
        "/api/positions": 5000,  # 5 seconds for trading
        "/api/market": 10000,  # 10 seconds for market data
        "/api/admin": 30000,  # 30 seconds for admin operations
    }
    DEFAULT_TIMEOUT = 15000  # 15 seconds default

    # Allowed content types (business rule)
    ALLOWED_CONTENT_TYPES = [
        "application/json",
        "application/xml",
        "multipart/form-data",
        "application/x-www-form-urlencoded",
    ]

    # Business rules for header validation
    SUSPICIOUS_USER_AGENTS = [
        "bot",
        "crawler",
        "scanner",
        "sqlmap",
        "nikto",
        "curl",
        "wget",
        "python-requests",
    ]

    # Allowed forwarding headers (business decision)
    ALLOWED_FORWARDING_HEADERS = ["X-Forwarded-For", "X-Real-IP", "X-Originating-IP"]

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
    def is_suspicious_user_agent(cls, user_agent: str) -> bool:
        """
        Check if User-Agent looks suspicious according to business rules.

        Args:
            user_agent: The User-Agent string to validate

        Returns:
            True if the User-Agent is considered suspicious
        """
        if not user_agent:
            # Business rule: missing User-Agent is suspicious
            return True

        ua_lower = user_agent.lower()
        for pattern in cls.SUSPICIOUS_USER_AGENTS:
            if pattern in ua_lower:
                return True

        return False

    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """
        Validate IP address format according to business rules.

        Args:
            ip: IP address string

        Returns:
            True if IP format is valid

        Raises:
            RequestValidationError: If IP address is invalid
        """
        if not ip:
            raise RequestValidationError("IP address cannot be empty")

        # Basic IPv4 validation (business rule: only support IPv4 for now)
        ipv4_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        if not re.match(ipv4_pattern, ip):
            raise RequestValidationError(f"Invalid IP address format: {ip}")

        # Validate each octet
        parts = ip.split(".")
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    raise RequestValidationError(f"IP address octet out of range: {num}")
            except ValueError:
                raise RequestValidationError(f"Invalid IP address: {ip}")

        return True

    @classmethod
    def validate_ip_list(cls, ip_string: str) -> bool:
        """
        Validate comma-separated list of IP addresses.

        Args:
            ip_string: Comma-separated IP addresses

        Returns:
            True if all IPs in the list are valid
        """
        if not ip_string:
            return False

        ips = [ip.strip() for ip in ip_string.split(",")]
        for ip in ips:
            try:
                cls.validate_ip_address(ip)
            except RequestValidationError:
                return False

        return True

    def is_whitelisted_ip(self, ip: str) -> bool:
        """
        Check if an IP address is whitelisted.

        Args:
            ip: IP address to check

        Returns:
            True if IP is in whitelist
        """
        return ip in self.whitelist_ips

    def is_blacklisted_ip(self, ip: str) -> bool:
        """
        Check if an IP address is blacklisted.

        Args:
            ip: IP address to check

        Returns:
            True if IP is in blacklist
        """
        return ip in self.blacklist_ips

    def validate_request_id(self, request_id: str) -> bool:
        """
        Validate request ID format according to business rules.

        Args:
            request_id: Request ID string

        Returns:
            True if request ID is valid

        Raises:
            RequestValidationError: If request ID format is invalid
        """
        if not request_id:
            raise RequestValidationError("Request ID is required")

        # Minimum length check
        if len(request_id) < 5:
            raise RequestValidationError(f"Request ID too short: {request_id}")

        # Maximum length check
        if len(request_id) > 100:
            raise RequestValidationError(f"Request ID too long: {request_id}")

        # Check for invalid characters (basic alphanumeric + hyphen + underscore)
        if not re.match(r"^[a-zA-Z0-9_-]+$", request_id):
            raise RequestValidationError(f"Request ID contains invalid characters: {request_id}")

        return True

    def validate_authorization_header(self, auth_header: str) -> bool:
        """
        Validate authorization header according to business rules.

        Args:
            auth_header: Authorization header value

        Returns:
            True if authorization is valid

        Raises:
            RequestValidationError: If authorization format is invalid
        """
        if not auth_header:
            raise RequestValidationError("Authorization header is required")

        # Check for Bearer token
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if len(token) < 5:  # Minimum token length
                raise RequestValidationError("Invalid Bearer token")
            return True

        # Check for API key
        if auth_header.startswith("ApiKey "):
            key = auth_header[7:].strip()
            if len(key) < 5:  # Minimum key length
                raise RequestValidationError("Invalid API key")
            return True

        raise RequestValidationError("Invalid authorization format")

    def validate_content_type(self, content_type: str) -> bool:
        """
        Validate content type according to business rules.

        Args:
            content_type: Content-Type header value

        Returns:
            True if content type is valid

        Raises:
            RequestValidationError: If content type is not allowed
        """
        # Extract main content type (ignore charset and other parameters)
        main_type = content_type.split(";")[0].strip().lower()

        for allowed in self.ALLOWED_CONTENT_TYPES:
            if main_type == allowed.lower():
                return True

        raise RequestValidationError(f"Unsupported content type: {content_type}")

    def validate_api_version(self, version: str) -> bool:
        """
        Validate API version according to business rules.

        Args:
            version: API version string

        Returns:
            True if version is valid

        Raises:
            RequestValidationError: If version is not supported
        """
        if version not in self.SUPPORTED_API_VERSIONS:
            raise RequestValidationError(f"Unsupported API version: {version}")
        return True

    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate date range according to business rules.

        Args:
            start_date: Start date of the range
            end_date: End date of the range

        Returns:
            True if date range is valid

        Raises:
            RequestValidationError: If date range is invalid
        """
        # Check if end date is after start date
        if end_date < start_date:
            raise RequestValidationError("End date must be after start date")

        # Check if range is not too large
        date_diff = (end_date - start_date).days
        if date_diff > self.MAX_DATE_RANGE_DAYS:
            raise RequestValidationError(
                f"Date range exceeds maximum of {self.MAX_DATE_RANGE_DAYS} days"
            )

        # Check if dates are not in the future
        now = datetime.now()
        if start_date > now:
            raise RequestValidationError("Start date cannot be in the future")

        return True

    def validate_pagination_params(self, params: dict[str, Any]) -> bool:
        """
        Validate pagination parameters according to business rules.

        Args:
            params: Dictionary containing pagination parameters

        Returns:
            True if pagination is valid

        Raises:
            RequestValidationError: If pagination parameters are invalid
        """
        # Check page number
        page = params.get("page", 1)
        if not isinstance(page, int) or page < 1:
            raise RequestValidationError(f"Invalid page number: {page}")

        # Check limit
        limit = params.get("limit", self.DEFAULT_PAGE_SIZE)
        if not isinstance(limit, int) or limit < 1:
            raise RequestValidationError(f"Invalid page limit: {limit}")

        if limit > self.MAX_PAGE_SIZE:
            raise RequestValidationError(f"Limit exceeds maximum of {self.MAX_PAGE_SIZE}")

        return True

    def get_request_priority(self, endpoint: str, method: str) -> str:
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

    def get_endpoint_timeout(self, endpoint: str) -> int:
        """
        Get timeout for a specific endpoint.

        Args:
            endpoint: The API endpoint path

        Returns:
            Timeout in milliseconds
        """
        # Check exact match first
        if endpoint in self.ENDPOINT_TIMEOUTS:
            return self.ENDPOINT_TIMEOUTS[endpoint]

        # Check prefix match
        for path_prefix, timeout in self.ENDPOINT_TIMEOUTS.items():
            if endpoint.startswith(path_prefix):
                return timeout

        return self.DEFAULT_TIMEOUT

    def requires_signature(self, endpoint: str) -> bool:
        """
        Check if an endpoint requires request signature.

        Args:
            endpoint: The API endpoint path

        Returns:
            True if signature is required
        """
        for pattern in self.SIGNATURE_REQUIRED_ENDPOINTS:
            if endpoint.startswith(pattern):
                return True
        return False

    def validate_request_size(self, size_bytes: int, max_size: int | None = None) -> bool:
        """
        Validate request size against business rules.

        Args:
            size_bytes: Size of request in bytes
            max_size: Optional custom maximum size (defaults to MAX_REQUEST_SIZE)

        Returns:
            True if size is valid

        Raises:
            RequestValidationError: If size exceeds limits
        """
        limit = max_size if max_size is not None else self.MAX_REQUEST_SIZE
        if size_bytes > limit:
            raise RequestValidationError(
                f"Request size exceeds maximum allowed size of {limit} bytes"
            )
        return True

    @classmethod
    def is_allowed_forwarding_header(cls, header_name: str) -> bool:
        """
        Check if a forwarding header is allowed according to business rules.

        Args:
            header_name: The header name to check

        Returns:
            True if the header is allowed
        """
        return header_name in cls.ALLOWED_FORWARDING_HEADERS

    @classmethod
    def validate_request_headers(cls, headers: dict[str, str]) -> list[str]:
        """
        Validate request headers according to business rules.

        Args:
            headers: Dictionary of request headers

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for User-Agent (required)
        user_agent = headers.get("User-Agent", "")
        if not user_agent:
            errors.append("User-Agent header is required")
        elif len(user_agent) < 5:
            # Check if User-Agent is too short (business rule)
            errors.append(f"Invalid User-Agent: '{user_agent}' is too short")
        elif cls.is_suspicious_user_agent(user_agent):
            # Check for suspicious User-Agent
            errors.append(f"Invalid User-Agent: '{user_agent}' appears suspicious")

        # Check for suspicious forwarding headers
        for header in cls.ALLOWED_FORWARDING_HEADERS:
            if header in headers:
                value = headers[header]
                if not cls.validate_ip_list(value):
                    errors.append(f"Invalid {header} header format: {value}")

        # Business rule: Check for suspicious header combinations
        if "X-Forwarded-For" in headers and "X-Real-IP" in headers:
            # Multiple forwarding headers might indicate spoofing attempt
            errors.append("Multiple forwarding headers detected")

        return errors

    def validate_webhook_payload(self, payload: dict[str, Any]) -> bool:
        """
        Validate webhook payload according to business rules.

        Args:
            payload: Webhook payload dictionary

        Returns:
            True if payload is valid

        Raises:
            RequestValidationError: If payload is invalid
        """
        # Check required fields
        required_fields = ["event", "timestamp", "data"]
        for field in required_fields:
            if field not in payload:
                raise RequestValidationError(f"Missing required field: {field}")

        # Validate event type
        valid_events = [
            "order.filled",
            "order.cancelled",
            "order.rejected",
            "position.opened",
            "position.closed",
            "alert.triggered",
        ]
        event = payload.get("event", "")
        if event not in valid_events:
            raise RequestValidationError(f"Invalid webhook event: {event}")

        # Validate timestamp format (ISO format)
        timestamp = payload.get("timestamp", "")
        try:
            # Try to parse the timestamp
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise RequestValidationError(f"Invalid timestamp format: {timestamp}")

        # Validate data is a dictionary
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RequestValidationError("Webhook data must be a dictionary")

        return True

    def validate_cors_origin(self, origin: str) -> bool:
        """
        Validate CORS origin according to business rules.

        Args:
            origin: Origin header value

        Returns:
            True if origin is allowed

        Raises:
            RequestValidationError: If origin is not allowed
        """
        if not self.allowed_origins:
            # If no origins configured, allow all (for development)
            return True

        if origin not in self.allowed_origins:
            raise RequestValidationError(f"Origin not allowed: {origin}")

        return True

    def validate_market_data_request(self, request_data: dict[str, Any]) -> bool:
        """
        Validate market data request according to business rules.

        Args:
            request_data: Dictionary containing market data request

        Returns:
            True if request is valid

        Raises:
            RequestValidationError: If validation fails
        """
        # Check for required fields
        required_fields = ["symbols", "interval"]
        for field in required_fields:
            if field not in request_data:
                raise RequestValidationError(f"{field} is required for market data request")

        # Validate symbols
        symbols = request_data.get("symbols", [])
        if not symbols or not isinstance(symbols, list):
            raise RequestValidationError("symbols must be a non-empty list")

        # Check for too many symbols
        if len(symbols) > 100:
            raise RequestValidationError(f"Too many symbols: {len(symbols)} (max 100)")

        for symbol in symbols:
            if not isinstance(symbol, str) or not re.match(r"^[A-Z0-9]{1,5}$", symbol):
                raise RequestValidationError(f"Invalid symbol: {symbol}")

        # Validate interval
        valid_intervals = ["1min", "5min", "15min", "30min", "1hour", "1day"]
        interval = request_data.get("interval", "")
        if interval not in valid_intervals:
            raise RequestValidationError(f"Invalid interval: {interval}")

        return True

    @classmethod
    def validate_trading_request(cls, request_data: dict[str, Any]) -> bool:
        """
        Validate trading request data according to business rules.

        Args:
            request_data: Dictionary containing trading request data

        Returns:
            True if request is valid

        Raises:
            RequestValidationError: If validation fails
        """
        # Required fields for trading requests
        required_fields = ["symbol", "quantity", "order_type", "side"]

        for field in required_fields:
            if field not in request_data:
                raise RequestValidationError(f"{field} is required for trading request")

        # Validate symbol format (uppercase letters, 1-5 chars)
        symbol = request_data.get("symbol", "")
        if not re.match(r"^[A-Z]{1,5}$", symbol):
            raise RequestValidationError(f"Invalid symbol format: {symbol}")

        # Validate quantity (positive integer)
        quantity = request_data.get("quantity")
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            raise RequestValidationError(f"Invalid quantity: {quantity}")

        # Validate order type
        valid_order_types = ["market", "limit", "stop", "stop_limit"]
        order_type = request_data.get("order_type", "").lower()
        if order_type not in valid_order_types:
            raise RequestValidationError(f"Invalid order type: {order_type}")

        # Validate side
        valid_sides = ["buy", "sell"]
        side = request_data.get("side", "").lower()
        if side not in valid_sides:
            raise RequestValidationError(f"Invalid side: {side}")

        return True

    @classmethod
    def validate_ip_format(cls, ip: str) -> bool:
        """
        Validate IP address format.

        Args:
            ip: IP address string

        Returns:
            True if IP format is valid
        """
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
            if cls.validate_ip_format(ip):
                return f"ip:{ip}"

        # Check for forwarded IP
        if "X-Forwarded-For" in headers:
            ips = headers["X-Forwarded-For"].split(",")
            if ips:
                ip = ips[0].strip()
                if cls.validate_ip_format(ip):
                    return f"ip:{ip}"

        # Check for remote address
        if "Remote-Addr" in headers:
            ip = headers["Remote-Addr"]
            if cls.validate_ip_format(ip):
                return f"ip:{ip}"

        return fallback

    @classmethod
    def should_enforce_signature(cls, endpoint: str, method: str) -> bool:
        """
        Determine if request signature should be enforced for an endpoint.

        This is a business rule decision.

        Args:
            endpoint: API endpoint
            method: HTTP method

        Returns:
            True if signature should be enforced
        """
        # Business rule: Enforce signatures for trading operations
        if endpoint.startswith("/api/trading/"):
            # All trading endpoints require signatures for POST/PUT/DELETE
            return method in ["POST", "PUT", "DELETE", "PATCH"]

        # Business rule: Admin endpoints always require signatures
        if endpoint.startswith("/api/admin/"):
            return True

        # Business rule: Public endpoints don't require signatures
        if endpoint.startswith("/api/public/"):
            return False

        # Default: No signature required for GET requests
        return method != "GET"

    @classmethod
    def get_max_request_size(cls, endpoint: str) -> int:
        """
        Get maximum allowed request size for an endpoint.

        This is a business rule about data limits.

        Args:
            endpoint: API endpoint

        Returns:
            Maximum request size in bytes
        """
        # Business rules for request sizes
        if endpoint.startswith("/api/upload/"):
            return 10 * 1024 * 1024  # 10MB for uploads
        elif endpoint.startswith("/api/trading/"):
            return 512 * 1024  # 512KB for trading requests
        elif endpoint.startswith("/api/market_data/"):
            return 256 * 1024  # 256KB for market data requests
        else:
            return 1024 * 1024  # 1MB default

    @classmethod
    def is_private_endpoint(cls, endpoint: str) -> bool:
        """
        Check if an endpoint is private (requires authentication).

        This is a business rule about endpoint access.

        Args:
            endpoint: API endpoint

        Returns:
            True if endpoint is private
        """
        # Business rule: Public endpoints don't require authentication
        public_prefixes = ["/api/public/", "/health/", "/status/"]
        for prefix in public_prefixes:
            if endpoint.startswith(prefix):
                return False

        # All other endpoints are private
        return True

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
