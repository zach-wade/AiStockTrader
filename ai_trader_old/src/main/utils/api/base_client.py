# File: utils/api/base_client.py

"""
Base API client with resilience patterns for all external API interactions.
"""

# Standard library imports
from abc import ABC
import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any

# Third-party imports
import aiohttp
import backoff

from .rate_monitor import record_api_request

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods for API clients."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    OAUTH2 = "oauth2"
    BASIC = "basic"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10
    burst_multiplier: float = 1.5

    @property
    def burst_limit(self) -> int:
        """Calculate burst limit."""
        return int(self.requests_per_second * self.burst_multiplier)


class BaseAPIClient(ABC):
    """
    Base class for all API clients with built-in resilience patterns.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting
    - Circuit breaker pattern
    - Request/response logging
    - Error handling
    """

    def __init__(
        self,
        base_url: str,
        auth_method: AuthMethod = AuthMethod.NONE,
        auth_token: str | None = None,
        headers: dict[str, str] | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize base API client.

        Args:
            base_url: Base URL for the API
            auth_method: Authentication method to use
            auth_token: Authentication token/key
            headers: Additional headers to include
            rate_limit_config: Rate limiting configuration
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.auth_method = auth_method
        self.auth_token = auth_token
        self.headers = headers or {}
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Setup authentication headers
        self._setup_auth_headers()

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_last_failure = None
        self._circuit_threshold = 5
        self._circuit_timeout = 60  # seconds

        # Rate limiting
        self._rate_limiter = None
        self._last_request_time = None

        # Session management
        self._session = None
        self._source_name = None  # Will be set by subclasses
        self._closed = False

    def _setup_auth_headers(self):
        """Setup authentication headers based on auth method."""
        if self.auth_method == AuthMethod.API_KEY and self.auth_token:
            self.headers["X-API-Key"] = self.auth_token
        elif self.auth_method == AuthMethod.BEARER and self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_method == AuthMethod.BASIC and self.auth_token:
            self.headers["Authorization"] = f"Basic {self.auth_token}"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self._session

    def _check_circuit_breaker(self):
        """Check if circuit breaker is open."""
        if not self._circuit_open:
            return

        # Check if timeout has passed
        if self._circuit_last_failure:
            time_since_failure = (datetime.utcnow() - self._circuit_last_failure).total_seconds()
            if time_since_failure > self._circuit_timeout:
                self._circuit_open = False
                self._circuit_failures = 0
                logger.info(f"Circuit breaker closed for {self.base_url}")
            else:
                raise Exception(f"Circuit breaker open for {self.base_url}")

    def _record_failure(self):
        """Record a failure for circuit breaker."""
        self._circuit_failures += 1
        self._circuit_last_failure = datetime.utcnow()

        if self._circuit_failures >= self._circuit_threshold:
            self._circuit_open = True
            logger.warning(f"Circuit breaker opened for {self.base_url}")

    def _record_success(self):
        """Record a success, reset failure count."""
        self._circuit_failures = 0

    async def _apply_rate_limit(self):
        """Apply rate limiting."""
        if self._last_request_time is None:
            self._last_request_time = datetime.utcnow()
            return

        time_since_last = (datetime.utcnow() - self._last_request_time).total_seconds()
        min_interval = 1.0 / self.rate_limit_config.requests_per_second

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self._last_request_time = datetime.utcnow()

    @backoff.on_exception(
        backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3, max_time=60
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: Any | None = None,
        custom_timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request with retry and error handling.

        Args:
            method: HTTP method
            url: Full URL or endpoint
            params: Query parameters
            json: JSON body
            data: Form data

        Returns:
            Response data as dictionary
        """
        # Check circuit breaker
        self._check_circuit_breaker()

        # Apply rate limiting
        await self._apply_rate_limit()

        # Build full URL if needed
        if not url.startswith("http"):
            url = f"{self.base_url}/{url.lstrip('/')}"

        try:
            session = await self._get_session()

            logger.debug(f"{method} {url}")

            # Record request for rate monitoring
            if self._source_name:
                await record_api_request(self._source_name)

            async with session.request(
                method=method, url=url, params=params, json=json, data=data
            ) as response:
                response.raise_for_status()

                # Record success
                self._record_success()

                # Return JSON response
                return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"API error: {e.status} - {e.message}")
            self._record_failure()
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self._record_failure()
            raise

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make GET request."""
        return await self._make_request("GET", endpoint, params=params)

    async def post(self, endpoint: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make POST request."""
        return await self._make_request("POST", endpoint, json=json)

    async def put(self, endpoint: str, json: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make PUT request."""
        return await self._make_request("PUT", endpoint, json=json)

    async def delete(self, endpoint: str) -> dict[str, Any]:
        """Make DELETE request."""
        return await self._make_request("DELETE", endpoint)

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._closed = True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
