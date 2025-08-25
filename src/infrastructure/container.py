"""
Dependency Injection Container - Central container for application dependencies.

This module provides a comprehensive dependency injection container that manages
the lifecycle and wiring of all application components.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from src.application.coordinators.broker_coordinator import BrokerCoordinator
from src.application.coordinators.service_factory import CoordinatorFactory, ServiceFactory
from src.application.interfaces.broker import IBroker
from src.application.interfaces.market_data import IMarketDataProvider
from src.application.interfaces.repositories import (
    IMarketDataRepository,
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
)
from src.application.interfaces.unit_of_work import IUnitOfWork
from src.application.use_cases import (
    CalculateRiskUseCase,
    CancelOrderUseCase,
    ClosePositionUseCase,
    GetHistoricalDataUseCase,
    GetLatestPriceUseCase,
    GetMarketDataUseCase,
    GetOrderStatusUseCase,
    GetPortfolioUseCase,
    GetPositionsUseCase,
    GetRiskMetricsUseCase,
    ModifyOrderUseCase,
    PlaceOrderUseCase,
    UpdatePortfolioUseCase,
    ValidateOrderRiskUseCase,
)
from src.domain.services import (
    ICommissionCalculator,
    IMarketMicrostructure,
    OrderValidator,
    PositionManager,
    RiskCalculator,
    TradingCalendar,
)
from src.domain.services.validation_service import DomainValidator
from src.infrastructure.brokers.broker_factory import BrokerFactory
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.repositories import (
    MarketDataRepository,
    PostgreSQLOrderRepository,
    PostgreSQLPortfolioRepository,
    PostgreSQLPositionRepository,
    PostgreSQLUnitOfWork,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ContainerConfig:
    """Configuration for the DI container."""

    # Database configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ai_trader"
    db_user: str = "zachwade"
    db_password: str = ""

    # Broker configuration
    broker_type: str = "paper"
    broker_config: dict[str, Any] | None = None

    # Service configuration
    service_config: dict[str, Any] | None = None

    # Feature flags
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_tracing: bool = False


class DIContainer:
    """
    Dependency Injection Container for the AI Trading System.

    This container manages the creation and wiring of all application components,
    ensuring proper dependency injection and lifecycle management.
    """

    def __init__(self, config: ContainerConfig | None = None) -> None:
        """Initialize the container with configuration."""
        self.config = config or ContainerConfig()
        self._singletons: dict[type[Any], Any] = {}
        self._factories: dict[type[Any], Callable[[], Any]] = {}

        # Register all components
        self._register_infrastructure()
        self._register_domain_services()
        self._register_application_services()
        self._register_use_cases()

        logger.info("Dependency injection container initialized")

    def _register_infrastructure(self) -> None:
        """Register infrastructure components."""
        # Create database connection pool
        from psycopg_pool import AsyncConnectionPool

        connection_string = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )

        pool = AsyncConnectionPool(connection_string, min_size=5, max_size=20)

        # Database adapter
        self._register_singleton(PostgreSQLAdapter, lambda: PostgreSQLAdapter(pool))

        # Repositories
        self._register_singleton(
            IMarketDataRepository,  # type: ignore[type-abstract]
            lambda: MarketDataRepository(self.get(PostgreSQLAdapter)),
        )
        self._register_singleton(
            IOrderRepository,  # type: ignore[type-abstract]
            lambda: PostgreSQLOrderRepository(self.get(PostgreSQLAdapter)),
        )
        self._register_singleton(
            IPortfolioRepository,  # type: ignore[type-abstract]
            lambda: PostgreSQLPortfolioRepository(self.get(PostgreSQLAdapter)),
        )
        self._register_singleton(
            IPositionRepository,  # type: ignore[type-abstract]
            lambda: PostgreSQLPositionRepository(self.get(PostgreSQLAdapter)),
        )

        # Unit of Work
        self._register_singleton(
            IUnitOfWork,  # type: ignore[type-abstract]
            lambda: PostgreSQLUnitOfWork(adapter=self.get(PostgreSQLAdapter)),
        )

        # Broker
        self._register_singleton(
            IBroker,  # type: ignore[type-abstract]
            lambda: BrokerFactory().create_broker(
                self.config.broker_type, **(self.config.broker_config or {})
            ),
        )

    def _register_domain_services(self) -> None:
        """Register domain services."""
        # Create all domain services
        services = ServiceFactory.create_all_services(
            self.config.broker_type, **(self.config.service_config or {})
        )

        # Register each service
        self._register_singleton(
            ICommissionCalculator,  # type: ignore[type-abstract]
            lambda: services["commission_calculator"],
        )
        self._register_singleton(
            IMarketMicrostructure,  # type: ignore[type-abstract]
            lambda: services["market_microstructure"],
        )
        self._register_singleton(OrderValidator, lambda: services["order_validator"])
        self._register_singleton(TradingCalendar, lambda: services["trading_calendar"])
        self._register_singleton(DomainValidator, lambda: services["domain_validator"])

        # Risk calculator
        self._register_singleton(RiskCalculator, lambda: RiskCalculator())

    def _register_application_services(self) -> None:
        """Register application-level services."""
        # Broker coordinator
        self._register_singleton(
            BrokerCoordinator,
            lambda: CoordinatorFactory.create_broker_coordinator(
                self.get(IBroker),  # type: ignore[type-abstract]
                {
                    "order_validator": self.get(OrderValidator),
                    "commission_calculator": self.get(ICommissionCalculator),  # type: ignore[type-abstract]
                    "market_microstructure": self.get(IMarketMicrostructure),  # type: ignore[type-abstract]
                    "trading_calendar": self.get(TradingCalendar),
                },
            ),
        )

    def _register_use_cases(self) -> None:
        """Register all use cases."""
        # Market Data Use Cases
        self._register_factory(
            GetMarketDataUseCase,
            lambda: GetMarketDataUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                market_data_provider=(
                    self.get(IMarketDataProvider) if self.has(IMarketDataProvider) else None
                ),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            GetLatestPriceUseCase,
            lambda: GetLatestPriceUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                market_data_provider=(
                    self.get(IMarketDataProvider) if self.has(IMarketDataProvider) else None
                ),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            GetHistoricalDataUseCase,
            lambda: GetHistoricalDataUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )

        # Trading Use Cases
        self._register_factory(
            PlaceOrderUseCase,
            lambda: PlaceOrderUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                broker=self.get(IBroker),  # type: ignore[type-abstract]
                order_validator=self.get(OrderValidator),
                risk_calculator=self.get(RiskCalculator),
            ),
        )
        self._register_factory(
            CancelOrderUseCase,
            lambda: CancelOrderUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                broker=self.get(IBroker),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            ModifyOrderUseCase,
            lambda: ModifyOrderUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                broker=self.get(IBroker),  # type: ignore[type-abstract]
                order_validator=self.get(OrderValidator),
            ),
        )
        self._register_factory(
            GetOrderStatusUseCase,
            lambda: GetOrderStatusUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                broker=self.get(IBroker),  # type: ignore[type-abstract]
            ),
        )

        # Portfolio Use Cases
        self._register_factory(
            GetPortfolioUseCase,
            lambda: GetPortfolioUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                risk_calculator=self.get(RiskCalculator),
            ),
        )
        self._register_factory(
            UpdatePortfolioUseCase,
            lambda: UpdatePortfolioUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            GetPositionsUseCase,
            lambda: GetPositionsUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            ClosePositionUseCase,
            lambda: ClosePositionUseCase(
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
                position_manager=self.get(PositionManager),
            ),
        )

        # Risk Use Cases
        self._register_factory(
            CalculateRiskUseCase,
            lambda: CalculateRiskUseCase(
                risk_calculator=self.get(RiskCalculator),
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            ValidateOrderRiskUseCase,
            lambda: ValidateOrderRiskUseCase(
                risk_calculator=self.get(RiskCalculator),
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )
        self._register_factory(
            GetRiskMetricsUseCase,
            lambda: GetRiskMetricsUseCase(
                risk_calculator=self.get(RiskCalculator),
                unit_of_work=self.get(IUnitOfWork),  # type: ignore[type-abstract]
            ),
        )

    def _register_singleton(self, cls: type[T], factory: Callable[[], Any]) -> None:
        """Register a singleton component."""
        self._factories[cls] = factory

    def _register_factory(self, cls: type[T], factory: Callable[[], Any]) -> None:
        """Register a factory for creating instances."""
        self._factories[cls] = factory

    def get(self, cls: type[T]) -> T:
        """
        Get an instance of a registered component.

        Args:
            cls: The class type to retrieve

        Returns:
            Instance of the requested class

        Raises:
            KeyError: If the class is not registered
        """
        # Check if it's a singleton and already created
        if cls in self._singletons:
            return cast(T, self._singletons[cls])

        # Check if we have a factory for it
        if cls not in self._factories:
            raise KeyError(f"No registration found for {cls.__name__}")

        # Create the instance
        instance = self._factories[cls]()

        # Store as singleton if it's an infrastructure component
        if cls.__module__.startswith("src.infrastructure") or cls.__module__.startswith(
            "src.domain"
        ):
            self._singletons[cls] = instance

        return cast(T, instance)

    def has(self, cls: type[T]) -> bool:
        """
        Check if a component is registered.

        Args:
            cls: The class type to check

        Returns:
            True if registered, False otherwise
        """
        return cls in self._factories

    def register(self, cls: type[T], instance: T) -> None:
        """
        Register a pre-created instance.

        Args:
            cls: The class type
            instance: The instance to register
        """
        self._singletons[cls] = instance
        self._factories[cls] = lambda: instance

    async def initialize(self) -> None:
        """Initialize all async components."""
        # Database adapter uses connection pool - no explicit connect needed
        adapter = self.get(PostgreSQLAdapter)

        # Initialize broker
        broker = self.get(IBroker)  # type: ignore[type-abstract]
        if not broker.is_connected():
            broker.connect()

        logger.info("Container initialized successfully")

    async def cleanup(self) -> None:
        """Clean up all resources."""
        # Disconnect broker
        try:
            broker = self.get(IBroker)  # type: ignore[type-abstract]
            if broker.is_connected():
                broker.disconnect()
        except KeyError:
            pass

        # Database adapter uses connection pool - handled by pool lifecycle
        try:
            adapter = self.get(PostgreSQLAdapter)
            # Pool cleanup is handled separately
        except KeyError:
            pass

        # Clear singletons
        self._singletons.clear()

        logger.info("Container cleaned up successfully")

    def create_scope(self) -> "DIContainer":
        """
        Create a scoped container for request-specific instances.

        Returns:
            New container instance with shared configuration
        """
        scoped = DIContainer(self.config)
        # Share infrastructure singletons
        for cls, instance in self._singletons.items():
            if cls.__module__.startswith("src.infrastructure"):
                scoped._singletons[cls] = instance
        return scoped


# Global container instance
_container: DIContainer | None = None


def get_container(config: ContainerConfig | None = None) -> DIContainer:
    """
    Get the global container instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        The global container instance
    """
    global _container
    if _container is None:
        _container = DIContainer(config)
    return _container


def reset_container() -> None:
    """Reset the global container instance."""
    global _container
    _container = None
