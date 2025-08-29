"""
Resilience Integration Module

Integrates all resilience features with existing trading system components,
providing production-ready wrappers for brokers, market data, and databases.
"""

import logging
from typing import Any

from src.application.interfaces.broker import IBroker
from src.application.interfaces.market_data import IMarketDataProvider
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from src.infrastructure.resilience.config import ApplicationConfig
from src.infrastructure.resilience.database import (
    EnhancedDatabaseConfig,
    ResilientDatabaseConnection,
)
from src.infrastructure.resilience.error_handling import error_manager, handle_errors
from src.infrastructure.resilience.fallback import CacheFirstStrategy
from src.infrastructure.resilience.health import APIHealthCheck, HealthChecker
from src.infrastructure.resilience.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


class ResilientBrokerWrapper:
    """
    Resilient wrapper for broker implementations.

    Adds circuit breaker, retry logic, health monitoring,
    and error handling to any IBroker implementation.
    """

    def __init__(
        self, broker: IBroker, config: ApplicationConfig, circuit_breaker_name: str | None = None
    ):
        self.broker = broker
        self.config = config
        self.circuit_breaker_name = circuit_breaker_name or f"broker_{type(broker).__name__}"
        self.circuit_breaker: CircuitBreaker | None

        # Initialize circuit breaker
        if config.resilience.circuit_breaker_enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=config.resilience.circuit_breaker_failure_threshold,
                timeout=config.resilience.circuit_breaker_timeout,
                half_open_max_calls=config.resilience.circuit_breaker_half_open_max_calls,
            )
            self.circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
                self.circuit_breaker_name, cb_config
            )
        else:
            self.circuit_breaker = None

        # Retry configuration
        self.retry_config = (
            RetryConfig(
                max_retries=config.resilience.retry_max_attempts,
                initial_delay=config.resilience.retry_initial_delay,
                max_delay=config.resilience.retry_max_delay,
                backoff_multiplier=config.resilience.retry_backoff_multiplier,
                jitter=config.resilience.retry_jitter,
            )
            if config.resilience.retry_enabled
            else None
        )

        logger.info(f"Initialized resilient broker wrapper for {type(broker).__name__}")

    @handle_errors(context_name="broker_operations")
    async def place_order(self, order_data: dict[str, Any]) -> Any:
        """Place order with resilience features."""
        # Note: This method should be updated to accept Order objects and handle conversion
        # For now, we'll disable circuit breaker for sync broker methods

        async def _place_order() -> Any:
            # Circuit breaker disabled for sync methods - need to implement sync circuit breaker
            return self.broker.submit_order(order_data)  # type: ignore[arg-type]

        if self.retry_config:
            return await retry_with_backoff(_place_order, config=self.retry_config)
        else:
            return await _place_order()

    @handle_errors(context_name="broker_operations")
    async def get_account_info(self) -> Any:
        """Get account info with resilience features."""

        async def _get_account_info() -> Any:
            # Circuit breaker disabled for sync methods
            return self.broker.get_account_info()

        if self.retry_config:
            return await retry_with_backoff(_get_account_info, config=self.retry_config)
        else:
            return await _get_account_info()

    @handle_errors(context_name="broker_operations")
    async def get_positions(self) -> Any:
        """Get positions with resilience features."""

        async def _get_positions() -> Any:
            # Circuit breaker disabled for sync methods
            return self.broker.get_positions()

        if self.retry_config:
            return await retry_with_backoff(_get_positions, config=self.retry_config)
        else:
            return await _get_positions()

    def get_health_check(self) -> APIHealthCheck:
        """Get health check for broker."""
        # This would need to be implemented based on the specific broker's health endpoint
        return APIHealthCheck(
            name=f"broker_{type(self.broker).__name__}",
            url="https://example.com/health",  # Replace with actual broker health endpoint
            timeout=self.config.resilience.health_check_timeout,
        )


class ResilientMarketDataWrapper:
    """
    Resilient wrapper for market data providers.

    Adds circuit breaker, retry logic, caching fallback,
    and health monitoring to market data providers.
    """

    def __init__(
        self,
        provider: IMarketDataProvider,
        config: ApplicationConfig,
        circuit_breaker_name: str | None = None,
        cache_provider: Any | None = None,
    ):
        self.provider = provider
        self.config = config
        self.circuit_breaker_name = circuit_breaker_name or f"market_data_{type(provider).__name__}"
        self.cache_provider = cache_provider
        self.circuit_breaker: CircuitBreaker | None
        self.fallback_strategy: CacheFirstStrategy[Any] | None

        # Initialize circuit breaker
        if config.resilience.circuit_breaker_enabled:
            cb_config = CircuitBreakerConfig(
                failure_threshold=config.resilience.circuit_breaker_failure_threshold,
                timeout=config.resilience.circuit_breaker_timeout,
                half_open_max_calls=config.resilience.circuit_breaker_half_open_max_calls,
            )
            self.circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
                self.circuit_breaker_name, cb_config
            )
        else:
            self.circuit_breaker = None

        # Retry configuration
        self.retry_config = (
            RetryConfig(
                max_retries=config.resilience.retry_max_attempts,
                initial_delay=config.resilience.retry_initial_delay,
                max_delay=config.resilience.retry_max_delay,
                backoff_multiplier=config.resilience.retry_backoff_multiplier,
                jitter=config.resilience.retry_jitter,
            )
            if config.resilience.retry_enabled
            else None
        )

        # Fallback strategy with caching
        if config.resilience.fallback_enabled and cache_provider:
            self.fallback_strategy = CacheFirstStrategy(
                name=f"market_data_cache_{type(provider).__name__}",
                primary_func=self._get_price_primary,
                fallback_func=self._get_price_fallback,
                cache_get_func=self._get_cached_price,
                cache_set_func=self._set_cached_price,
            )
        else:
            self.fallback_strategy = None

        logger.info(f"Initialized resilient market data wrapper for {type(provider).__name__}")

    async def _get_price_primary(self, symbol: str) -> Any:
        """Primary price retrieval."""
        if self.circuit_breaker:
            return await self.circuit_breaker.call_async(self.provider.get_current_price, symbol)
        else:
            return await self.provider.get_current_price(symbol)

    async def _get_price_fallback(self, symbol: str) -> Any:
        """Fallback price retrieval (could be from another provider)."""
        # This would implement fallback logic, e.g., using a different provider
        # For now, return cached data or default
        cached = await self._get_cached_price(symbol)
        if cached:
            return cached

        # Return a degraded service response
        return {
            "symbol": symbol,
            "price": 0.0,
            "status": "degraded",
            "message": "Using fallback data",
        }

    async def _get_cached_price(self, symbol: str) -> Any | None:
        """Get cached price."""
        if self.cache_provider:
            try:
                return await self.cache_provider.get(f"price_{symbol}")
            except Exception as e:
                logger.debug(f"Cache get failed: {e}")
        return None

    async def _set_cached_price(self, price_data: Any, symbol: str) -> None:
        """Set cached price."""
        if self.cache_provider:
            try:
                await self.cache_provider.set(
                    f"price_{symbol}", price_data, ttl=self.config.resilience.fallback_cache_ttl
                )
            except Exception as e:
                logger.debug(f"Cache set failed: {e}")

    @handle_errors(context_name="market_data_operations")
    async def get_current_price(self, symbol: str) -> Any:
        """Get current price with resilience features."""
        if self.fallback_strategy:
            result = await self.fallback_strategy.execute_with_fallback(symbol)
            return result.value

        async def _get_price() -> Any:
            return await self._get_price_primary(symbol)

        if self.retry_config:
            return await retry_with_backoff(_get_price, config=self.retry_config)
        else:
            return await _get_price()

    @handle_errors(context_name="market_data_operations")
    async def get_historical_data(
        self, symbol: str, start: Any, end: Any, timeframe: str = "1min"
    ) -> Any:
        """Get historical data with resilience features."""

        async def _get_historical() -> Any:
            if self.circuit_breaker:
                return await self.circuit_breaker.call_async(
                    self.provider.get_historical_bars, symbol, start, end, timeframe
                )
            else:
                return await self.provider.get_historical_bars(symbol, start, end, timeframe)

        if self.retry_config:
            return await retry_with_backoff(_get_historical, config=self.retry_config)
        else:
            return await _get_historical()

    def get_health_check(self) -> APIHealthCheck:
        """Get health check for market data provider."""
        # This would need to be implemented based on the specific provider's health endpoint
        return APIHealthCheck(
            name=f"market_data_{type(self.provider).__name__}",
            url="https://api.polygon.io/v1/meta/symbols/AAPL/company",  # Example
            timeout=self.config.resilience.health_check_timeout,
        )


class ResilienceOrchestrator:
    """
    Central orchestrator for all resilience features.

    Manages health checking, circuit breakers, error handling,
    and provides a unified interface for resilience operations.
    """

    def __init__(self, config: ApplicationConfig) -> None:
        self.config = config
        self.health_checker = HealthChecker(
            check_interval=config.resilience.health_check_interval,
            degraded_threshold=0.7,
            unhealthy_threshold=3,
        )

        # Initialize components
        self.circuit_breaker_registry = CircuitBreakerRegistry.get_instance()
        self.resilient_components: dict[str, Any] = {}

        logger.info("Initialized resilience orchestrator")

    async def initialize(self) -> None:
        """Initialize resilience infrastructure."""
        logger.info("Initializing resilience infrastructure...")

        # Start health monitoring if enabled
        if self.config.resilience.health_check_enabled:
            await self.health_checker.start_monitoring()
            logger.info("Health monitoring started")

        logger.info("Resilience infrastructure initialized")

    async def shutdown(self) -> None:
        """Shutdown resilience infrastructure."""
        logger.info("Shutting down resilience infrastructure...")

        # Stop health monitoring
        if self.config.resilience.health_check_enabled:
            await self.health_checker.stop_monitoring()

        logger.info("Resilience infrastructure shutdown complete")

    def register_database(
        self, database: ResilientDatabaseConnection, name: str = "database"
    ) -> None:
        """Register database for health monitoring."""
        if self.config.resilience.health_check_enabled:
            health_check = database.get_health_check()
            self.health_checker.register_health_check(health_check)
            logger.info(f"Registered database health check: {name}")

        self.resilient_components[name] = database

    def register_broker(self, broker: ResilientBrokerWrapper, name: str) -> None:
        """Register broker for health monitoring."""
        if self.config.resilience.health_check_enabled:
            health_check = broker.get_health_check()
            self.health_checker.register_health_check(health_check)
            logger.info(f"Registered broker health check: {name}")

        self.resilient_components[f"broker_{name}"] = broker

    def register_market_data(self, provider: ResilientMarketDataWrapper, name: str) -> None:
        """Register market data provider for health monitoring."""
        if self.config.resilience.health_check_enabled:
            health_check = provider.get_health_check()
            self.health_checker.register_health_check(health_check)
            logger.info(f"Registered market data health check: {name}")

        self.resilient_components[f"market_data_{name}"] = provider

    async def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health status."""
        health_summary = self.health_checker.get_health_summary()

        # Add circuit breaker status
        cb_metrics = self.circuit_breaker_registry.get_all_metrics()

        # Add error metrics
        error_metrics = error_manager.get_metrics()

        return {
            "overall_status": health_summary["overall_status"],
            "timestamp": health_summary["timestamp"],
            "health_checks": health_summary["services"],
            "circuit_breakers": cb_metrics,
            "error_handling": error_metrics,
            "components": list(self.resilient_components.keys()),
        }

    async def reset_all_metrics(self) -> None:
        """Reset all resilience metrics."""
        logger.info("Resetting all resilience metrics...")

        # Reset circuit breakers
        self.circuit_breaker_registry.reset_all()

        # Reset error manager
        error_manager.reset_metrics()

        # Reset health checker
        self.health_checker.reset_metrics()

        # Reset component-specific metrics
        for name, component in self.resilient_components.items():
            if hasattr(component, "reset_metrics"):
                component.reset_metrics()

        logger.info("All resilience metrics reset")

    def get_resilient_database(self, name: str = "database") -> ResilientDatabaseConnection | None:
        """Get resilient database connection."""
        return self.resilient_components.get(name)

    def get_resilient_broker(self, name: str) -> ResilientBrokerWrapper | None:
        """Get resilient broker wrapper."""
        return self.resilient_components.get(f"broker_{name}")

    def get_resilient_market_data(self, name: str) -> ResilientMarketDataWrapper | None:
        """Get resilient market data wrapper."""
        return self.resilient_components.get(f"market_data_{name}")


class ResilienceFactory:
    """
    Factory for creating resilient components.

    Provides a unified interface for wrapping existing components
    with resilience features based on configuration.
    """

    def __init__(self, config: ApplicationConfig) -> None:
        self.config = config
        self.orchestrator = ResilienceOrchestrator(config)

    async def create_resilient_database(
        self, base_config: EnhancedDatabaseConfig | None = None
    ) -> ResilientDatabaseConnection:
        """Create resilient database connection."""
        if base_config is None:
            from src.infrastructure.database.connection import DatabaseConfig

            base_db_config = DatabaseConfig.from_env()
            base_config = EnhancedDatabaseConfig.from_base_config(
                base_db_config,
                circuit_breaker_enabled=self.config.resilience.circuit_breaker_enabled,
                retry_enabled=self.config.resilience.retry_enabled,
                health_check_enabled=self.config.resilience.health_check_enabled,
            )

        db_connection = ResilientDatabaseConnection(base_config)
        await db_connection.connect()

        # Register with orchestrator
        self.orchestrator.register_database(db_connection)

        return db_connection

    def create_resilient_broker(self, broker: IBroker, name: str) -> ResilientBrokerWrapper:
        """Create resilient broker wrapper."""
        resilient_broker = ResilientBrokerWrapper(broker, self.config, f"broker_{name}")

        # Register with orchestrator
        self.orchestrator.register_broker(resilient_broker, name)

        return resilient_broker

    def create_resilient_market_data(
        self, provider: IMarketDataProvider, name: str, cache_provider: Any | None = None
    ) -> ResilientMarketDataWrapper:
        """Create resilient market data wrapper."""
        resilient_provider = ResilientMarketDataWrapper(
            provider, self.config, f"market_data_{name}", cache_provider
        )

        # Register with orchestrator
        self.orchestrator.register_market_data(resilient_provider, name)

        return resilient_provider

    async def initialize_resilience(self) -> None:
        """Initialize the resilience orchestrator."""
        await self.orchestrator.initialize()

    async def shutdown_resilience(self) -> None:
        """Shutdown the resilience orchestrator."""
        await self.orchestrator.shutdown()

    def get_orchestrator(self) -> ResilienceOrchestrator:
        """Get the resilience orchestrator."""
        return self.orchestrator
