"""
API Versioning Service - Domain service for API versioning and endpoint policy business rules.

This service handles business logic for API version validation, endpoint policies,
and access control rules, implementing the Single Responsibility Principle.
"""


class ApiVersioningError(Exception):
    """Exception raised when API versioning validation fails."""

    pass


class ApiVersioningService:
    """
    Domain service for API versioning and endpoint policy business logic.

    This service contains business rules for API versions, endpoint access,
    timeouts, and security policies.
    """

    # Supported API versions (business rule)
    SUPPORTED_API_VERSIONS = ["v1", "v2"]

    # Endpoints requiring signature (business rule)
    SIGNATURE_REQUIRED_ENDPOINTS = [
        "/api/orders",
        "/api/positions",
        "/api/admin",
    ]

    # Endpoint timeouts in milliseconds (business rule)
    ENDPOINT_TIMEOUTS = {
        "/api/orders": 5000,  # 5 seconds for trading
        "/api/positions": 5000,  # 5 seconds for trading
        "/api/market": 10000,  # 10 seconds for market data
        "/api/admin": 30000,  # 30 seconds for admin operations
    }
    DEFAULT_TIMEOUT = 15000  # 15 seconds default

    def validate_api_version(self, version: str) -> bool:
        """
        Validate API version according to business rules.

        Args:
            version: API version string

        Returns:
            True if version is valid

        Raises:
            ApiVersioningError: If version is not supported
        """
        if version not in self.SUPPORTED_API_VERSIONS:
            raise ApiVersioningError(f"Unsupported API version: {version}")
        return True

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
