"""
Base Ingestion Client

Abstract base class for all data ingestion clients providing standardized
framework with rate limiting, retries, circuit breaker, and monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypeVar, Generic
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass

from main.utils.core import (
    get_logger,
    RateLimiter,
    AsyncCircuitBreaker,
    async_retry,
    timer
)
from main.utils.api import managed_aiohttp_session

T = TypeVar('T')


@dataclass
class ClientConfig:
    """Configuration for ingestion client."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_second: int = 5
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 30
    circuit_breaker_failures: int = 5
    circuit_breaker_timeout: int = 60
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class FetchResult(Generic[T]):
    """Result of a fetch operation."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseIngestionClient(ABC, Generic[T]):
    """
    Abstract base class for data ingestion clients.
    
    Provides:
    - Rate limiting
    - Automatic retries with exponential backoff
    - Circuit breaker for fault tolerance
    - Session management
    - Monitoring and metrics
    """
    
    def __init__(self, config: ClientConfig):
        """
        Initialize the base client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize rate limiter
        # RateLimiter expects rate per time period, not per second
        self.rate_limiter = RateLimiter(
            rate=int(config.rate_limit_per_second * 60),  # Convert to requests per minute
            per=60.0  # per minute
        )
        
        # Initialize circuit breaker
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=config.circuit_breaker_failures,
            recovery_timeout=config.circuit_breaker_timeout,
            expected_exception=aiohttp.ClientError
        )
        
        # Session will be created on demand
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Cache for frequently accessed data
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info(
            f"{self.__class__.__name__} initialized with "
            f"rate_limit={config.rate_limit_per_second}/s"
        )
    
    @abstractmethod
    def get_base_url(self) -> str:
        """Get the base URL for the API."""
        pass
    
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        pass
    
    @abstractmethod
    async def validate_response(self, response: aiohttp.ClientResponse) -> bool:
        """
        Validate API response.
        
        Args:
            response: The API response
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    async def parse_response(self, response: aiohttp.ClientResponse) -> T:
        """
        Parse API response into the expected type.
        
        Args:
            response: The API response
            
        Returns:
            Parsed data of type T
        """
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = await managed_aiohttp_session(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ).__aenter__()
        return self._session
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{param_str}"
    
    def _check_cache(self, cache_key: str) -> Optional[T]:
        """Check if data exists in cache and is still valid."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config.cache_ttl_seconds):
                self._cache_hits += 1
                self.logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _update_cache(self, cache_key: str, data: T):
        """Update cache with new data."""
        if self.config.enable_caching:
            self._cache[cache_key] = (data, datetime.now())
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1]
                )
                for key, _ in sorted_items[:100]:
                    del self._cache[key]
    
    @timer
    @async_retry(max_attempts=3, delay=1.0)
    async def fetch(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> FetchResult[T]:
        """
        Fetch data from API endpoint.
        
        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            use_cache: Whether to use caching
            
        Returns:
            FetchResult containing the data or error
        """
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params)
        
        # Check cache first
        if use_cache:
            cached_data = self._check_cache(cache_key)
            if cached_data is not None:
                return FetchResult(success=True, data=cached_data)
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Use circuit breaker
            async with self.circuit_breaker:
                session = await self._get_session()
                url = f"{self.get_base_url()}/{endpoint}"
                
                self._request_count += 1
                
                async with session.get(
                    url,
                    params=params,
                    headers=self.get_headers()
                ) as response:
                    # Validate response
                    if not await self.validate_response(response):
                        error_msg = f"Invalid response: {response.status}"
                        self._error_count += 1
                        return FetchResult(
                            success=False,
                            error=error_msg,
                            metadata={'status': response.status}
                        )
                    
                    # Parse response
                    data = await self.parse_response(response)
                    
                    # Update cache
                    if use_cache:
                        self._update_cache(cache_key, data)
                    
                    return FetchResult(
                        success=True,
                        data=data,
                        metadata={
                            'status': response.status,
                            'cached': False
                        }
                    )
        
        except asyncio.TimeoutError:
            self._error_count += 1
            return FetchResult(
                success=False,
                error="Request timeout"
            )
        
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Fetch error: {e}")
            return FetchResult(
                success=False,
                error=str(e)
            )
    
    async def batch_fetch(
        self,
        requests: List[tuple[str, Dict[str, Any]]],
        max_concurrent: int = 5
    ) -> List[FetchResult[T]]:
        """
        Fetch multiple requests in parallel with concurrency control.
        
        Args:
            requests: List of (endpoint, params) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of FetchResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(endpoint: str, params: Dict[str, Any]) -> FetchResult[T]:
            async with semaphore:
                return await self.fetch(endpoint, params)
        
        tasks = [
            fetch_with_semaphore(endpoint, params)
            for endpoint, params in requests
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )
        
        return {
            'request_count': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(1, self._request_count),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._cache),
            'circuit_breaker_state': self.circuit_breaker.state
        }
    
    async def health_check(self) -> bool:
        """Check if the client is healthy."""
        try:
            # Try a simple request
            result = await self.fetch('', {})
            return result.success
        except Exception:
            return False
    
    async def close(self):
        """Close the client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self.logger.info(
            f"{self.__class__.__name__} closed. "
            f"Requests: {self._request_count}, Errors: {self._error_count}"
        )