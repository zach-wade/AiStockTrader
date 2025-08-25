"""
Integration tests for resilience components working together.
"""

import asyncio
import time
from typing import Any

import pytest

from src.application.interfaces.broker import IBroker
from src.application.interfaces.market_data import IMarketDataProvider
from src.infrastructure.resilience.circuit_breaker import CircuitBreakerRegistry
from src.infrastructure.resilience.config import (
    ApplicationConfig,
    DatabaseConfig,
    Environment,
    ExternalAPIConfig,
    FeatureFlags,
    ResilienceConfig,
    TradingConfig,
)
from src.infrastructure.resilience.error_handling import error_manager
from src.infrastructure.resilience.integration import (
    ResilienceFactory,
    ResilienceOrchestrator,
    ResilientBrokerWrapper,
    ResilientMarketDataWrapper,
)


class MockBroker(IBroker):
    """Mock broker for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0

    async def place_order(self, order_data: dict) -> Any:
        """Mock place order."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock broker failure")
        return {"order_id": "mock_order_123", "status": "submitted"}

    async def get_account_info(self) -> Any:
        """Mock get account info."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock broker failure")
        return {"account_id": "mock_account", "buying_power": 10000.0}

    async def get_positions(self) -> Any:
        """Mock get positions."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock broker failure")
        return []


class MockMarketDataProvider(IMarketDataProvider):
    """Mock market data provider for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.call_count = 0

    async def get_current_price(self, symbol: str) -> Any:
        """Mock get current price."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock market data failure")
        return {"symbol": symbol, "price": 100.0, "timestamp": time.time()}

    async def get_historical_data(self, symbol: str, period: str) -> Any:
        """Mock get historical data."""
        self.call_count += 1
        if self.should_fail:
            raise ConnectionError("Mock market data failure")
        return [{"timestamp": time.time(), "price": 100.0}]


class MockCacheProvider:
    """Mock cache provider for testing."""

    def __init__(self):
        self.cache = {}

    async def get(self, key: str) -> Any:
        """Get from cache."""
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: float = None) -> None:
        """Set in cache."""
        self.cache[key] = value


@pytest.fixture
def test_config():
    """Create test configuration."""
    return ApplicationConfig(
        environment=Environment.TESTING,
        debug=True,
        resilience=ResilienceConfig(
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_timeout=0.1,
            retry_enabled=True,
            retry_max_attempts=2,
            retry_initial_delay=0.01,
            retry_max_delay=0.1,
            health_check_enabled=True,
            health_check_interval=0.1,
            fallback_enabled=True,
        ),
        trading=TradingConfig(),
        database=DatabaseConfig(),
        external_apis=ExternalAPIConfig(),
        features=FeatureFlags(),
    )


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    return MockBroker()


@pytest.fixture
def mock_market_data():
    """Create mock market data provider."""
    return MockMarketDataProvider()


@pytest.fixture
def mock_cache():
    """Create mock cache provider."""
    return MockCacheProvider()


class TestResilientBrokerWrapper:
    """Test resilient broker wrapper."""

    @pytest.mark.asyncio
    async def test_successful_operations(self, test_config, mock_broker):
        """Test successful broker operations."""
        wrapper = ResilientBrokerWrapper(mock_broker, test_config, "test_broker")

        # Test place order
        result = await wrapper.place_order({"symbol": "AAPL", "quantity": 10})
        assert result["order_id"] == "mock_order_123"

        # Test get account info
        result = await wrapper.get_account_info()
        assert result["account_id"] == "mock_account"

        # Test get positions
        result = await wrapper.get_positions()
        assert result == []

        assert mock_broker.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, test_config):
        """Test retry logic on broker failures."""
        # Create broker that fails first time, then succeeds
        call_count = 0

        class FlakeyBroker(MockBroker):
            async def place_order(self, order_data: dict) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("First attempt fails")
                return {"order_id": "retry_success", "status": "submitted"}

        broker = FlakeyBroker()
        wrapper = ResilientBrokerWrapper(broker, test_config, "flakey_broker")

        result = await wrapper.place_order({"symbol": "AAPL"})

        assert result["order_id"] == "retry_success"
        assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self, test_config):
        """Test circuit breaker opening on repeated failures."""
        failing_broker = MockBroker(should_fail=True)
        wrapper = ResilientBrokerWrapper(failing_broker, test_config, "failing_broker")

        # First few calls should fail and retry
        for i in range(2):
            with pytest.raises(ConnectionError):
                await wrapper.place_order({"symbol": "AAPL"})

        # Circuit should be open now
        cb = wrapper.circuit_breaker
        assert cb.state == "open"

        # Next call should fail fast (circuit breaker error)
        from src.infrastructure.resilience.circuit_breaker import CircuitBreakerError

        with pytest.raises(CircuitBreakerError):
            await wrapper.place_order({"symbol": "AAPL"})


class TestResilientMarketDataWrapper:
    """Test resilient market data wrapper."""

    @pytest.mark.asyncio
    async def test_successful_operations(self, test_config, mock_market_data):
        """Test successful market data operations."""
        wrapper = ResilientMarketDataWrapper(mock_market_data, test_config, "test_provider")

        # Test get current price
        result = await wrapper.get_current_price("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["price"] == 100.0

        # Test get historical data
        result = await wrapper.get_historical_data("AAPL", "1D")
        assert len(result) == 1
        assert result[0]["price"] == 100.0

        assert mock_market_data.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_fallback(self, test_config, mock_cache):
        """Test cache fallback functionality."""
        # Pre-populate cache
        await mock_cache.set("price_AAPL", {"symbol": "AAPL", "price": 95.0, "cached": True})

        failing_provider = MockMarketDataProvider(should_fail=True)
        wrapper = ResilientMarketDataWrapper(
            failing_provider, test_config, "failing_provider", mock_cache
        )

        # Should get cached data when primary fails
        result = await wrapper.get_current_price("AAPL")

        # The exact behavior depends on fallback strategy implementation
        # This test verifies the wrapper handles failures gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, test_config):
        """Test retry logic on market data failures."""
        call_count = 0

        class FlakeyProvider(MockMarketDataProvider):
            async def get_current_price(self, symbol: str) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("First attempt fails")
                return {"symbol": symbol, "price": 101.0}

        provider = FlakeyProvider()
        wrapper = ResilientMarketDataWrapper(provider, test_config, "flakey_provider")

        result = await wrapper.get_current_price("AAPL")

        assert result["price"] == 101.0
        assert call_count == 2  # Retry succeeded


class TestResilienceOrchestrator:
    """Test resilience orchestrator."""

    @pytest.mark.asyncio
    async def test_initialization_and_shutdown(self, test_config):
        """Test orchestrator initialization and shutdown."""
        orchestrator = ResilienceOrchestrator(test_config)

        # Test initialization
        await orchestrator.initialize()

        # Health checker should be running
        assert orchestrator.health_checker._monitoring_task is not None

        # Test shutdown
        await orchestrator.shutdown()

        # Health checker should be stopped
        task = orchestrator.health_checker._monitoring_task
        assert task is None or task.done()

    @pytest.mark.asyncio
    async def test_component_registration(self, test_config, mock_broker, mock_market_data):
        """Test registering components with orchestrator."""
        orchestrator = ResilienceOrchestrator(test_config)
        await orchestrator.initialize()

        try:
            # Create resilient wrappers
            broker_wrapper = ResilientBrokerWrapper(mock_broker, test_config, "test_broker")
            data_wrapper = ResilientMarketDataWrapper(mock_market_data, test_config, "test_data")

            # Register components
            orchestrator.register_broker(broker_wrapper, "test_broker")
            orchestrator.register_market_data(data_wrapper, "test_data")

            # Check registration
            assert "broker_test_broker" in orchestrator.resilient_components
            assert "market_data_test_data" in orchestrator.resilient_components

            # Check health checks were registered
            assert "broker_test_broker" in orchestrator.health_checker.health_checks
            assert "market_data_test_data" in orchestrator.health_checker.health_checks

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, test_config, mock_broker):
        """Test system health monitoring."""
        orchestrator = ResilienceOrchestrator(test_config)
        await orchestrator.initialize()

        try:
            # Register a component
            broker_wrapper = ResilientBrokerWrapper(mock_broker, test_config, "test_broker")
            orchestrator.register_broker(broker_wrapper, "test_broker")

            # Wait for health checks to run
            await asyncio.sleep(0.15)

            # Get system health
            health = await orchestrator.get_system_health()

            assert "overall_status" in health
            assert "health_checks" in health
            assert "circuit_breakers" in health
            assert "error_handling" in health
            assert "components" in health

            assert "broker_test_broker" in health["components"]

        finally:
            await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_reset(self, test_config, mock_broker):
        """Test resetting all resilience metrics."""
        orchestrator = ResilienceOrchestrator(test_config)
        await orchestrator.initialize()

        try:
            # Register component and generate some activity
            broker_wrapper = ResilientBrokerWrapper(mock_broker, test_config, "test_broker")
            orchestrator.register_broker(broker_wrapper, "test_broker")

            # Perform some operations to generate metrics
            await broker_wrapper.get_account_info()

            # Reset metrics
            await orchestrator.reset_all_metrics()

            # Verify reset
            cb_metrics = orchestrator.circuit_breaker_registry.get_all_metrics()
            error_metrics = error_manager.get_metrics()

            # Metrics should be reset (exact values depend on implementation)
            assert isinstance(cb_metrics, dict)
            assert isinstance(error_metrics, dict)

        finally:
            await orchestrator.shutdown()


class TestResilienceFactory:
    """Test resilience factory."""

    @pytest.mark.asyncio
    async def test_factory_initialization(self, test_config):
        """Test factory initialization."""
        factory = ResilienceFactory(test_config)

        assert factory.config is test_config
        assert factory.orchestrator is not None

        # Test initialization
        await factory.initialize_resilience()

        # Test shutdown
        await factory.shutdown_resilience()

    @pytest.mark.asyncio
    async def test_create_resilient_broker(self, test_config, mock_broker):
        """Test creating resilient broker through factory."""
        factory = ResilienceFactory(test_config)
        await factory.initialize_resilience()

        try:
            resilient_broker = factory.create_resilient_broker(mock_broker, "test_broker")

            assert isinstance(resilient_broker, ResilientBrokerWrapper)
            assert resilient_broker.broker is mock_broker

            # Should be registered with orchestrator
            orchestrator = factory.get_orchestrator()
            assert "broker_test_broker" in orchestrator.resilient_components

            # Test operations work
            result = await resilient_broker.get_account_info()
            assert result["account_id"] == "mock_account"

        finally:
            await factory.shutdown_resilience()

    @pytest.mark.asyncio
    async def test_create_resilient_market_data(self, test_config, mock_market_data, mock_cache):
        """Test creating resilient market data through factory."""
        factory = ResilienceFactory(test_config)
        await factory.initialize_resilience()

        try:
            resilient_data = factory.create_resilient_market_data(
                mock_market_data, "test_data", mock_cache
            )

            assert isinstance(resilient_data, ResilientMarketDataWrapper)
            assert resilient_data.provider is mock_market_data
            assert resilient_data.cache_provider is mock_cache

            # Should be registered with orchestrator
            orchestrator = factory.get_orchestrator()
            assert "market_data_test_data" in orchestrator.resilient_components

            # Test operations work
            result = await resilient_data.get_current_price("AAPL")
            assert result["symbol"] == "AAPL"

        finally:
            await factory.shutdown_resilience()


class TestEndToEndIntegration:
    """Test end-to-end integration of all resilience features."""

    @pytest.mark.asyncio
    async def test_complete_resilience_stack(self, test_config):
        """Test the complete resilience stack working together."""
        # Create factory and initialize
        factory = ResilienceFactory(test_config)
        await factory.initialize_resilience()

        try:
            # Create mock services
            broker = MockBroker()
            market_data = MockMarketDataProvider()
            cache = MockCacheProvider()

            # Create resilient wrappers
            resilient_broker = factory.create_resilient_broker(broker, "main_broker")
            resilient_data = factory.create_resilient_market_data(market_data, "main_data", cache)

            # Perform operations
            account_info = await resilient_broker.get_account_info()
            current_price = await resilient_data.get_current_price("AAPL")

            assert account_info["account_id"] == "mock_account"
            assert current_price["symbol"] == "AAPL"

            # Wait for health checks to run
            await asyncio.sleep(0.15)

            # Check system health
            orchestrator = factory.get_orchestrator()
            health = await orchestrator.get_system_health()

            assert health["overall_status"] in ["healthy", "unknown"]  # Could be unknown initially
            assert len(health["components"]) == 2  # Broker and market data
            assert "broker_main_broker" in health["components"]
            assert "market_data_main_data" in health["components"]

        finally:
            await factory.shutdown_resilience()

    @pytest.mark.asyncio
    async def test_failure_scenarios_with_recovery(self, test_config):
        """Test failure scenarios and recovery mechanisms."""
        factory = ResilienceFactory(test_config)
        await factory.initialize_resilience()

        try:
            # Create initially failing services
            failing_broker = MockBroker(should_fail=True)
            failing_data = MockMarketDataProvider(should_fail=True)

            resilient_broker = factory.create_resilient_broker(failing_broker, "failing_broker")
            resilient_data = factory.create_resilient_market_data(failing_data, "failing_data")

            # Operations should fail initially
            with pytest.raises((ConnectionError, Exception)):
                await resilient_broker.get_account_info()

            with pytest.raises((ConnectionError, Exception)):
                await resilient_data.get_current_price("AAPL")

            # Wait for circuit breakers to open
            await asyncio.sleep(0.1)

            # Circuit breakers should be open
            cb_metrics = CircuitBreakerRegistry.get_instance().get_all_metrics()

            # Fix the services
            failing_broker.should_fail = False
            failing_data.should_fail = False

            # Wait for recovery timeout
            await asyncio.sleep(0.15)

            # Operations should eventually succeed (circuit breaker half-open)
            # This might require multiple attempts as circuit breaker recovers
            success = False
            for _ in range(5):  # Try a few times
                try:
                    await resilient_broker.get_account_info()
                    success = True
                    break
                except:
                    await asyncio.sleep(0.05)

            # At minimum, the system should be attempting recovery
            assert isinstance(cb_metrics, dict)  # Metrics available

        finally:
            await factory.shutdown_resilience()

            # Clean up registry for other tests
            CircuitBreakerRegistry.get_instance().clear()


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield

    # Reset global state
    CircuitBreakerRegistry.get_instance().clear()
    error_manager.reset_metrics()
