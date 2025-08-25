"""
Comprehensive tests for the Dependency Injection Container.

This test suite covers:
- Container initialization and configuration
- Component registration and retrieval
- Singleton vs factory patterns
- Scoped container creation
- Error handling and edge cases
- Async initialization and cleanup
- Thread safety for concurrent access
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from src.application.coordinators.broker_coordinator import BrokerCoordinator
from src.application.interfaces.broker import IBroker
from src.application.interfaces.repositories import IOrderRepository, IPortfolioRepository
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases import GetMarketDataUseCase, GetPortfolioUseCase, PlaceOrderUseCase
from src.domain.services import OrderValidator, RiskCalculator, TradingCalendar
from src.infrastructure.container import (
    ContainerConfig,
    DIContainer,
    get_container,
    reset_container,
)
from src.infrastructure.database.adapter import PostgreSQLAdapter


class TestContainerConfig:
    """Test ContainerConfig dataclass."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = ContainerConfig()

        assert config.db_host == "localhost"
        assert config.db_port == 5432
        assert config.db_name == "ai_trader"
        assert config.db_user == "zachwade"
        assert config.db_password == ""
        assert config.broker_type == "paper"
        assert config.broker_config is None
        assert config.service_config is None
        assert config.enable_caching is True
        assert config.enable_metrics is True
        assert config.enable_tracing is False

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ContainerConfig(
            db_host="remote-host",
            db_port=5433,
            db_name="test_db",
            db_user="test_user",
            db_password="secret",
            broker_type="alpaca",
            broker_config={"api_key": "test_key"},
            service_config={"commission_rate": 0.001},
            enable_caching=False,
            enable_metrics=False,
            enable_tracing=True,
        )

        assert config.db_host == "remote-host"
        assert config.db_port == 5433
        assert config.db_name == "test_db"
        assert config.db_user == "test_user"
        assert config.db_password == "secret"
        assert config.broker_type == "alpaca"
        assert config.broker_config == {"api_key": "test_key"}
        assert config.service_config == {"commission_rate": 0.001}
        assert config.enable_caching is False
        assert config.enable_metrics is False
        assert config.enable_tracing is True


class TestDIContainer:
    """Test DIContainer functionality."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        with patch("src.infrastructure.container.AsyncConnectionPool") as mock:
            yield mock.return_value

    @pytest.fixture
    def container(self, mock_pool):
        """Create a container instance for testing."""
        with patch("src.infrastructure.container.BrokerFactory") as mock_factory:
            mock_broker = MagicMock(spec=IBroker)
            mock_factory.return_value.create_broker.return_value = mock_broker

            with patch("src.infrastructure.container.ServiceFactory") as mock_service_factory:
                mock_services = {
                    "commission_calculator": MagicMock(),
                    "market_microstructure": MagicMock(),
                    "order_validator": MagicMock(spec=OrderValidator),
                    "trading_calendar": MagicMock(spec=TradingCalendar),
                    "domain_validator": MagicMock(),
                }
                mock_service_factory.create_all_services.return_value = mock_services

                container = DIContainer()
                yield container

    def test_container_initialization(self, mock_pool):
        """Test container initializes correctly."""
        with (
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = DIContainer()

            assert container.config is not None
            assert len(container._singletons) == 0
            assert len(container._factories) > 0

    def test_container_with_custom_config(self, mock_pool):
        """Test container with custom configuration."""
        config = ContainerConfig(db_host="custom-host", broker_type="alpaca")

        with (
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = DIContainer(config)

            assert container.config.db_host == "custom-host"
            assert container.config.broker_type == "alpaca"

    def test_get_singleton_component(self, container):
        """Test retrieving singleton components."""
        # First retrieval creates the instance
        adapter1 = container.get(PostgreSQLAdapter)
        assert adapter1 is not None

        # Second retrieval returns the same instance
        adapter2 = container.get(PostgreSQLAdapter)
        assert adapter1 is adapter2

    def test_get_factory_component(self, container):
        """Test retrieving factory-created components."""
        # Each retrieval creates a new instance for use cases
        use_case1 = container.get(GetMarketDataUseCase)
        use_case2 = container.get(GetMarketDataUseCase)

        assert use_case1 is not None
        assert use_case2 is not None
        # Use cases are created via factory, not singletons
        assert use_case1 is not use_case2

    def test_has_registered_component(self, container):
        """Test checking if a component is registered."""
        assert container.has(PostgreSQLAdapter) is True
        assert container.has(RiskCalculator) is True
        assert container.has(GetPortfolioUseCase) is True

        # Check for non-registered component
        class UnregisteredClass:
            pass

        assert container.has(UnregisteredClass) is False

    def test_get_unregistered_component_raises_error(self, container):
        """Test that getting unregistered component raises KeyError."""

        class UnregisteredClass:
            pass

        with pytest.raises(KeyError, match="No registration found for UnregisteredClass"):
            container.get(UnregisteredClass)

    def test_register_custom_instance(self, container):
        """Test registering a custom instance."""

        class CustomService:
            def __init__(self, value):
                self = value

        custom_instance = CustomService(42)
        container.register(CustomService, custom_instance)

        # Should retrieve the registered instance
        retrieved = container.get(CustomService)
        assert retrieved is custom_instance
        assert retrieved == 42

    def test_infrastructure_components_are_singletons(self, container):
        """Test that infrastructure components are singletons."""
        # Get repositories multiple times
        repo1 = container.get(IOrderRepository)
        repo2 = container.get(IOrderRepository)

        assert repo1 is repo2

    def test_domain_services_are_singletons(self, container):
        """Test that domain services are singletons."""
        # Get domain services multiple times
        calc1 = container.get(RiskCalculator)
        calc2 = container.get(RiskCalculator)

        assert calc1 is calc2

    @pytest.mark.asyncio
    async def test_container_initialize(self, container):
        """Test async initialization of container."""
        mock_broker = MagicMock(spec=IBroker)
        mock_broker.is_connected.return_value = False
        mock_broker.connect = MagicMock()

        container.register(IBroker, mock_broker)

        await container.initialize()

        mock_broker.is_connected.assert_called_once()
        mock_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_container_initialize_already_connected(self, container):
        """Test initialization when broker is already connected."""
        mock_broker = MagicMock(spec=IBroker)
        mock_broker.is_connected.return_value = True
        mock_broker.connect = MagicMock()

        container.register(IBroker, mock_broker)

        await container.initialize()

        mock_broker.is_connected.assert_called_once()
        mock_broker.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_container_cleanup(self, container):
        """Test cleanup of container resources."""
        mock_broker = MagicMock(spec=IBroker)
        mock_broker.is_connected.return_value = True
        mock_broker.disconnect = MagicMock()

        container.register(IBroker, mock_broker)

        # Add some singletons
        container._singletons[PostgreSQLAdapter] = MagicMock()

        await container.cleanup()

        mock_broker.is_connected.assert_called_once()
        mock_broker.disconnect.assert_called_once()
        assert len(container._singletons) == 0

    @pytest.mark.asyncio
    async def test_container_cleanup_disconnected_broker(self, container):
        """Test cleanup when broker is already disconnected."""
        mock_broker = MagicMock(spec=IBroker)
        mock_broker.is_connected.return_value = False
        mock_broker.disconnect = MagicMock()

        container.register(IBroker, mock_broker)

        await container.cleanup()

        mock_broker.is_connected.assert_called_once()
        mock_broker.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_container_cleanup_no_broker(self, container):
        """Test cleanup when no broker is registered."""
        # Remove broker registration
        if IBroker in container._factories:
            del container._factories[IBroker]

        # Should not raise error
        await container.cleanup()
        assert len(container._singletons) == 0

    def test_create_scope(self, container):
        """Test creating a scoped container."""
        # Add some infrastructure singletons
        mock_adapter = MagicMock()
        container._singletons[PostgreSQLAdapter] = mock_adapter

        # Create scope
        scoped = container.create_scope()

        # Should share configuration
        assert scoped.config == container.config

        # Should share infrastructure singletons
        assert PostgreSQLAdapter in scoped._singletons
        assert scoped._singletons[PostgreSQLAdapter] is mock_adapter

    def test_scoped_container_independent_app_instances(self, container):
        """Test that scoped containers have independent app instances."""
        # Add infrastructure singleton
        container._singletons[PostgreSQLAdapter] = MagicMock()

        # Create two scopes
        scope1 = container.create_scope()
        scope2 = container.create_scope()

        # Get use cases from each scope
        use_case1 = scope1.get(GetMarketDataUseCase)
        use_case2 = scope2.get(GetMarketDataUseCase)

        # Should be different instances
        assert use_case1 is not use_case2

    def test_thread_safety_singleton_creation(self, container):
        """Test thread-safe singleton creation."""
        results = []

        def get_singleton():
            adapter = container.get(PostgreSQLAdapter)
            results.append(adapter)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_singleton)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should be the same instance
        first = results[0]
        for adapter in results[1:]:
            assert adapter is first

    def test_concurrent_factory_creation(self, container):
        """Test concurrent factory component creation."""
        results = []

        def get_use_case():
            use_case = container.get(GetMarketDataUseCase)
            results.append(use_case)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_use_case) for _ in range(10)]
            for future in futures:
                future.result()

        # All should be different instances (factory pattern)
        for i, use_case1 in enumerate(results):
            for j, use_case2 in enumerate(results):
                if i != j:
                    assert use_case1 is not use_case2

    def test_registration_order_dependencies(self, mock_pool):
        """Test that components are registered in correct dependency order."""
        with (
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = DIContainer()

            # Infrastructure should be registered first
            assert PostgreSQLAdapter in container._factories

            # Then repositories (depend on adapter)
            assert IOrderRepository in container._factories
            assert IPortfolioRepository in container._factories

            # Then domain services
            assert RiskCalculator in container._factories

            # Finally use cases (depend on everything else)
            assert PlaceOrderUseCase in container._factories

    def test_get_with_missing_optional_dependency(self, container):
        """Test getting component with missing optional dependency."""
        # Remove IMarketDataProvider if registered
        from src.application.interfaces.market_data import IMarketDataProvider

        if IMarketDataProvider in container._factories:
            del container._factories[IMarketDataProvider]

        # Should still create use case with None for optional dependency
        use_case = container.get(GetMarketDataUseCase)
        assert use_case is not None


class TestGlobalContainer:
    """Test global container functions."""

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_get_container_creates_singleton(self):
        """Test that get_container creates and returns singleton."""
        with (
            patch("src.infrastructure.container.AsyncConnectionPool"),
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container1 = get_container()
            container2 = get_container()

            assert container1 is container2

    def test_get_container_with_config(self):
        """Test get_container with custom config."""
        config = ContainerConfig(db_host="custom-host")

        with (
            patch("src.infrastructure.container.AsyncConnectionPool"),
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = get_container(config)

            assert container.config.db_host == "custom-host"

    def test_reset_container(self):
        """Test resetting global container."""
        with (
            patch("src.infrastructure.container.AsyncConnectionPool"),
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container1 = get_container()
            reset_container()
            container2 = get_container()

            assert container1 is not container2


class TestContainerIntegration:
    """Integration tests for container with real component creation."""

    @pytest.fixture
    def integrated_container(self):
        """Create container with minimal mocking for integration tests."""
        with patch("src.infrastructure.container.AsyncConnectionPool") as mock_pool:
            mock_pool.return_value = MagicMock()

            with patch("src.infrastructure.brokers.broker_factory.BrokerFactory") as mock_factory:
                from src.infrastructure.brokers.paper_broker import PaperBroker

                mock_factory.return_value.create_broker.return_value = PaperBroker()

                with patch(
                    "src.infrastructure.container.ServiceFactory.create_all_services"
                ) as mock_services:
                    mock_services.return_value = {
                        "commission_calculator": MagicMock(),
                        "market_microstructure": MagicMock(),
                        "order_validator": OrderValidator(),
                        "trading_calendar": TradingCalendar(),
                        "domain_validator": MagicMock(),
                    }

                    container = DIContainer()
                    yield container

    def test_full_use_case_creation(self, integrated_container):
        """Test creating a complete use case with all dependencies."""
        use_case = integrated_container.get(PlaceOrderUseCase)

        assert use_case is not None
        assert hasattr(use_case, "unit_of_work")
        assert hasattr(use_case, "broker")
        assert hasattr(use_case, "order_validator")
        assert hasattr(use_case, "risk_calculator")

    def test_broker_coordinator_creation(self, integrated_container):
        """Test creating broker coordinator with services."""
        coordinator = integrated_container.get(BrokerCoordinator)

        assert coordinator is not None
        assert hasattr(coordinator, "broker")
        assert hasattr(coordinator, "services")

    def test_risk_calculator_creation(self, integrated_container):
        """Test creating risk calculator."""
        calculator = integrated_container.get(RiskCalculator)

        assert calculator is not None
        assert isinstance(calculator, RiskCalculator)

    def test_repository_creation_chain(self, integrated_container):
        """Test creating repositories with adapter dependency."""
        # Get adapter first
        adapter = integrated_container.get(PostgreSQLAdapter)
        assert adapter is not None

        # Get repository that depends on adapter
        repo = integrated_container.get(IOrderRepository)
        assert repo is not None

    def test_unit_of_work_creation(self, integrated_container):
        """Test creating unit of work with all repositories."""
        uow = integrated_container.get(IUnitOfWork)

        assert uow is not None

    @pytest.mark.asyncio
    async def test_async_initialization_flow(self, integrated_container):
        """Test full async initialization flow."""
        await integrated_container.initialize()

        # Verify broker was initialized
        broker = integrated_container.get(IBroker)
        assert broker is not None

    @pytest.mark.asyncio
    async def test_cleanup_flow(self, integrated_container):
        """Test full cleanup flow."""
        await integrated_container.initialize()
        await integrated_container.cleanup()

        # Verify singletons were cleared
        assert len(integrated_container._singletons) == 0


class TestContainerErrorHandling:
    """Test error handling in container."""

    @pytest.fixture
    def container(self):
        """Create container for error testing."""
        with (
            patch("src.infrastructure.container.AsyncConnectionPool"),
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = DIContainer()
            yield container

    def test_factory_raises_error(self, container):
        """Test handling when factory raises error."""

        class FailingService:
            pass

        def failing_factory():
            raise ValueError("Factory failed")

        container._register_factory(FailingService, failing_factory)

        with pytest.raises(ValueError, match="Factory failed"):
            container.get(FailingService)

    def test_circular_dependency_detection(self, container):
        """Test detection of circular dependencies."""

        class ServiceA:
            pass

        class ServiceB:
            pass

        # Create circular dependency
        def create_a():
            container.get(ServiceB)  # A depends on B
            return ServiceA()

        def create_b():
            container.get(ServiceA)  # B depends on A
            return ServiceB()

        container._register_factory(ServiceA, create_a)
        container._register_factory(ServiceB, create_b)

        # This will cause infinite recursion, should be caught
        with pytest.raises(RecursionError):
            container.get(ServiceA)

    def test_invalid_configuration(self):
        """Test container with invalid configuration."""
        config = ContainerConfig(db_port=-1, broker_type="invalid_broker")  # Invalid port

        with patch("src.infrastructure.container.AsyncConnectionPool") as mock_pool:
            mock_pool.side_effect = ValueError("Invalid connection string")

            with pytest.raises(ValueError):
                container = DIContainer(config)

    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self, container):
        """Test cleanup handles errors gracefully."""
        mock_broker = MagicMock(spec=IBroker)
        mock_broker.is_connected.return_value = True
        mock_broker.disconnect.side_effect = Exception("Disconnect failed")

        container.register(IBroker, mock_broker)

        # Should not raise, errors are handled
        await container.cleanup()

        # Singletons should still be cleared
        assert len(container._singletons) == 0


class TestContainerPerformance:
    """Performance tests for container."""

    @pytest.fixture
    def container(self):
        """Create container for performance testing."""
        with (
            patch("src.infrastructure.container.AsyncConnectionPool"),
            patch("src.infrastructure.container.BrokerFactory"),
            patch("src.infrastructure.container.ServiceFactory"),
        ):
            container = DIContainer()
            yield container

    def test_singleton_retrieval_performance(self, container):
        """Test performance of singleton retrieval."""
        import time

        # First retrieval (creation)
        start = time.time()
        container.get(RiskCalculator)
        creation_time = time.time() - start

        # Subsequent retrievals (cached)
        start = time.time()
        for _ in range(1000):
            container.get(RiskCalculator)
        retrieval_time = time.time() - start

        # Cached retrieval should be much faster
        avg_retrieval = retrieval_time / 1000
        assert avg_retrieval < creation_time

    def test_concurrent_access_performance(self, container):
        """Test performance under concurrent access."""
        import time

        def get_components():
            for _ in range(100):
                container.get(RiskCalculator)
                container.get(OrderValidator)
                container.get(GetMarketDataUseCase)

        start = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_components) for _ in range(20)]
            for future in futures:
                future.result()
        elapsed = time.time() - start

        # Should complete in reasonable time even with high concurrency
        assert elapsed < 5.0  # 5 seconds for 40,000 component retrievals
