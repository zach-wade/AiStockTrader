"""
Base Service Class

Provides common functionality for service-oriented components that provide
specific functionality without complex state management.
"""

# Standard library imports
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.utils.core import (
    AsyncCircuitBreaker,
    RateLimiter,
    async_lru_cache,
    ensure_utc,
    get_logger,
)
from main.utils.data import StreamingConfig
from main.utils.monitoring import MetricsCollector, log_performance

from ..enums import DataLayer
from ..exceptions import DataPipelineError, convert_exception


class BaseService(ABC):
    """
    Abstract base class for service components.

    Services are stateless components that provide specific functionality
    and can be easily composed into larger systems.

    Provides common functionality including:
    - Standardized logging and monitoring
    - Layer-aware operations
    - Configuration management
    - Service health tracking
    - Performance monitoring
    """

    def __init__(
        self,
        service_name: str,
        config: dict[str, Any] | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        self.service_name = service_name
        self.config = config or {}
        self.metrics_collector = metrics_collector
        self.logger = get_logger(f"data_pipeline.{service_name}")

        # Service statistics
        self._service_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "creation_time": ensure_utc(datetime.now(UTC)),
            "last_request": None,
        }

        # Health status
        self._health_status = "healthy"
        self._last_health_check = ensure_utc(datetime.now(UTC))

        # Circuit breaker for resilient external calls
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_timeout", 60),
            expected_exception=Exception,
        )

        # Rate limiter for service-level rate limiting
        self.rate_limiter = RateLimiter(
            rate=self.config.get("rate_limit", 100),
            per=self.config.get("rate_limit_period", 60),  # per minute
        )

        # Streaming configuration
        self.streaming_config = StreamingConfig(
            chunk_size=self.config.get("streaming_chunk_size", 10000),
            max_memory_mb=self.config.get("streaming_max_memory_mb", 500),
            parallel_workers=self.config.get("streaming_workers", 1),
        )

        # Cache configuration
        self._cache_enabled = self.config.get("cache_enabled", True)
        self._cache_ttl = self.config.get("cache_ttl_seconds", 300)

        self.logger.debug(f"Initialized {service_name} service")

    @log_performance
    async def execute(
        self,
        request_data: Any,
        layer: DataLayer | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute a service request with monitoring and error handling.

        Args:
            request_data: Data for the service request
            layer: Data layer for layer-aware services
            context: Additional context for the request

        Returns:
            Service response

        Raises:
            DataPipelineError: If service execution fails
        """
        context = context or {}
        request_id = context.get(
            "request_id", f"{self.service_name}_{ensure_utc(datetime.now(UTC)).isoformat()}"
        )

        self.logger.debug(f"Executing service request: {request_id} (layer: {layer})")

        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Pre-execution validation
            await self._validate_request(request_data, layer, context)

            # Execute service logic with caching if enabled
            if self._cache_enabled and context.get("use_cache", True):
                result = await self._execute_with_cache(request_data, layer, context)
            else:
                result = await self._execute_service(request_data, layer, context)

            # Post-execution validation
            await self._validate_response(result, layer, context)

            # Update statistics
            self._update_stats(success=True)

            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_service_success(
                    service=self.service_name, layer=layer.value if layer else None, context=context
                )

            self.logger.debug(f"Service request completed successfully: {request_id}")
            return result

        except Exception as e:
            self._update_stats(success=False)

            # Record metrics for failure
            if self.metrics_collector:
                self.metrics_collector.record_service_failure(
                    service=self.service_name,
                    error=str(e),
                    layer=layer.value if layer else None,
                    context=context,
                )

            # Convert to DataPipelineError with context
            error = DataPipelineError(
                message=f"Service execution failed in {self.service_name}",
                component="service",
                original_error=e,
                context={
                    "request_id": request_id,
                    "service": self.service_name,
                    "layer": layer.value if layer else None,
                    **context,
                },
            )

            self.logger.error(f"Service execution failed: {error}")
            raise error

    @abstractmethod
    async def _execute_service(
        self, request_data: Any, layer: DataLayer | None, context: dict[str, Any]
    ) -> Any:
        """Core service execution logic. Override in subclasses."""
        pass

    async def _validate_request(
        self, request_data: Any, layer: DataLayer | None, context: dict[str, Any]
    ) -> None:
        """Validate service request. Override in subclasses for specific validation."""
        if request_data is None:
            raise DataPipelineError(
                "Service request data cannot be None",
                component="service",
                context={"service": self.service_name},
            )

    async def _validate_response(
        self, result: Any, layer: DataLayer | None, context: dict[str, Any]
    ) -> None:
        """Validate service response. Override in subclasses for specific validation."""
        pass

    def _update_stats(self, success: bool) -> None:
        """Update service statistics."""
        self._service_stats["total_requests"] += 1
        if success:
            self._service_stats["successful_requests"] += 1
        else:
            self._service_stats["failed_requests"] += 1
        self._service_stats["last_request"] = ensure_utc(datetime.now(UTC))
        self._last_health_check = ensure_utc(datetime.now(UTC))

    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the service.

        Returns:
            Health status information
        """
        try:
            # Perform service-specific health check
            health_details = await self._perform_health_check()

            self._health_status = "healthy"
            self._last_health_check = ensure_utc(datetime.now(UTC))

            return {
                "service_name": self.service_name,
                "status": self._health_status,
                "last_check": self._last_health_check.isoformat(),
                "statistics": self.get_stats(),
                "details": health_details,
            }

        except Exception as e:
            self._health_status = "unhealthy"
            self.logger.error(f"Health check failed for service {self.service_name}: {e}")

            return {
                "service_name": self.service_name,
                "status": self._health_status,
                "last_check": self._last_health_check.isoformat(),
                "error": str(e),
                "statistics": self.get_stats(),
            }

    async def _perform_health_check(self) -> dict[str, Any]:
        """Service-specific health check logic. Override in subclasses."""
        return {"status": "ok"}

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        stats = self._service_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 1.0
        return stats

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._health_status == "healthy"

    def get_layer_config(self, layer: DataLayer) -> dict[str, Any]:
        """Get configuration for a specific layer."""
        layer_configs = self.config.get("layers", {})
        default_config = self.config.get("default_layer_config", {})
        return layer_configs.get(str(layer.value), default_config)

    def is_layer_supported(self, layer: DataLayer) -> bool:
        """Check if a layer is supported by this service."""
        supported_layers = self.config.get(
            "supported_layers", list(range(4))
        )  # Default: all layers
        return layer.value in supported_layers

    def get_layer_limits(self, layer: DataLayer) -> dict[str, Any]:
        """Get processing limits for a specific layer."""
        layer_config = self.get_layer_config(layer)
        return {
            "max_concurrent_requests": layer_config.get("max_concurrent_requests", 10),
            "timeout_seconds": layer_config.get("timeout_seconds", 60),
            "retry_attempts": layer_config.get("retry_attempts", 3),
            "rate_limit_per_minute": layer_config.get("rate_limit_per_minute", 100),
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.service_name})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.service_name}, health={self._health_status})"

    async def _execute_with_cache(
        self, request_data: Any, layer: DataLayer | None, context: dict[str, Any]
    ) -> Any:
        """Execute service with caching support."""
        # Create cache key from request data
        cache_key = self._create_cache_key(request_data, layer, context)

        # Wrap the service execution with caching
        @async_lru_cache(maxsize=128, ttl=self._cache_ttl)
        async def cached_execute(key: str) -> Any:
            return await self._execute_service(request_data, layer, context)

        return await cached_execute(cache_key)

    def _create_cache_key(
        self, request_data: Any, layer: DataLayer | None, context: dict[str, Any]
    ) -> str:
        """Create a cache key from request parameters."""
        # Simple key creation - override in subclasses for better keys
        # Standard library imports
        import hashlib
        import json

        key_data = {
            "service": self.service_name,
            "layer": layer.value if layer else None,
            "request": str(request_data),
            "context_keys": sorted(context.keys()),
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def execute_with_circuit_breaker(self, external_call: Any, *args, **kwargs) -> Any:
        """Execute an external call with circuit breaker protection."""
        return await self.circuit_breaker.call(external_call, *args, **kwargs)

    def get_streaming_config(self) -> StreamingConfig:
        """Get streaming configuration for the service."""
        return self.streaming_config

    async def warmup(self) -> None:
        """Warm up the service (e.g., pre-load caches, establish connections)."""
        try:
            self.logger.info(f"Warming up {self.service_name} service")
            await self._warmup_service()
            self._health_status = "healthy"
            self.logger.info(f"Service {self.service_name} warmed up successfully")
        except Exception as e:
            self._health_status = "unhealthy"
            error = convert_exception(e, f"Failed to warm up service {self.service_name}")
            self.logger.error(f"Service warmup failed: {error}")
            raise error

    async def _warmup_service(self) -> None:
        """Service-specific warmup logic. Override in subclasses."""
        pass
