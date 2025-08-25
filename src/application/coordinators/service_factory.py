"""
Service Factory - Application layer factory for creating and wiring services.

This factory handles the complex initialization and dependency injection
of domain services and application coordinators.
"""

import logging
from decimal import Decimal
from typing import Any

from src.domain.services import (
    CommissionCalculatorFactory,
    CommissionSchedule,
    CommissionType,
    Exchange,
    ICommissionCalculator,
    IMarketMicrostructure,
    MarketImpactModel,
    MarketMicrostructureFactory,
    OrderConstraints,
    OrderValidator,
    SlippageConfig,
    TradingCalendar,
)
from src.domain.services.validation_service import DomainValidator

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating and configuring domain services.

    This factory encapsulates the complex logic for creating domain services
    with proper configuration and dependencies.
    """

    @staticmethod
    def create_commission_calculator(
        broker_type: str, commission_config: dict[str, Any] | None = None, **kwargs: Any
    ) -> ICommissionCalculator:
        """
        Create a commission calculator based on broker type.

        Args:
            broker_type: Type of broker (alpaca, paper, backtest)
            commission_config: Optional commission configuration
            **kwargs: Additional configuration options

        Returns:
            Configured commission calculator
        """
        if commission_config:
            # Convert config dict to CommissionSchedule
            comm_type = commission_config.get("type", "PER_SHARE")
            # Handle both enum and string values
            if isinstance(comm_type, CommissionType):
                commission_type = comm_type
            else:
                commission_type = CommissionType[str(comm_type).upper()]

            schedule = CommissionSchedule(
                commission_type=commission_type,
                rate=Decimal(str(commission_config.get("rate", "0"))),
                minimum=Decimal(str(commission_config.get("minimum", "0"))),
                maximum=Decimal(str(commission_config.get("maximum", "10000"))),
            )
            return CommissionCalculatorFactory.create(schedule)

        if broker_type == "alpaca":
            # Alpaca has zero commission for stocks
            schedule = CommissionSchedule(
                commission_type=CommissionType.PER_SHARE,
                rate=Decimal("0"),
                minimum=Decimal("0"),
            )
        else:
            # Default commission for paper/backtest
            commission_per_share = kwargs.get("commission_per_share", Decimal("0.01"))
            min_commission = kwargs.get("min_commission", Decimal("1.0"))
            schedule = CommissionSchedule(
                commission_type=CommissionType.PER_SHARE,
                rate=commission_per_share,
                minimum=min_commission,
            )

        return CommissionCalculatorFactory.create(schedule)

    @staticmethod
    def create_market_microstructure(
        broker_type: str, market_config: dict[str, Any] | None = None, **kwargs: Any
    ) -> IMarketMicrostructure:
        """
        Create a market microstructure model based on broker type.

        Args:
            broker_type: Type of broker (alpaca, paper, backtest)
            market_config: Optional market configuration
            **kwargs: Additional configuration options

        Returns:
            Configured market microstructure model
        """
        if market_config:
            return MarketMicrostructureFactory.create(
                market_config["model"],
                market_config["slippage"],
            )

        if broker_type == "alpaca":
            # Realistic market model for live trading
            slippage_config = SlippageConfig(
                base_bid_ask_bps=Decimal("1"),
                impact_coefficient=Decimal("0.05"),
                add_randomness=False,
            )
            model = MarketImpactModel.SQUARE_ROOT
        elif broker_type == "backtest":
            # Minimal slippage for backtesting
            slippage_pct = kwargs.get("slippage_pct", Decimal("0.0005"))
            slippage_config = SlippageConfig(
                base_bid_ask_bps=slippage_pct * Decimal("10000"),
                impact_coefficient=Decimal("0.01"),
                add_randomness=False,
            )
            model = MarketImpactModel.LINEAR
        else:
            # Default for paper trading
            slippage_pct = kwargs.get("slippage_pct", Decimal("0.001"))
            slippage_config = SlippageConfig(
                base_bid_ask_bps=slippage_pct * Decimal("10000"),
                impact_coefficient=Decimal("0.1"),
                add_randomness=True,
            )
            model = MarketImpactModel.LINEAR

        return MarketMicrostructureFactory.create(model, slippage_config)

    @staticmethod
    def create_order_validator(
        commission_calculator: ICommissionCalculator, constraints: OrderConstraints | None = None
    ) -> OrderValidator:
        """
        Create an order validator with dependencies.

        Args:
            commission_calculator: Commission calculator to use
            constraints: Optional order constraints

        Returns:
            Configured order validator
        """
        if constraints is None:
            constraints = OrderConstraints()

        return OrderValidator(commission_calculator, constraints)

    @staticmethod
    def create_trading_calendar(exchange: Exchange | None = None) -> TradingCalendar:
        """
        Create a trading calendar for an exchange.

        Args:
            exchange: Exchange to create calendar for

        Returns:
            Configured trading calendar
        """
        from src.infrastructure.time.timezone_service import PythonTimeService

        if exchange is None:
            exchange = Exchange.NYSE

        # Create time service dependency
        time_service = PythonTimeService()

        return TradingCalendar(time_service, exchange)

    @staticmethod
    def create_domain_validator() -> DomainValidator:
        """
        Create a domain validator.

        Returns:
            Configured domain validator
        """
        return DomainValidator()

    @staticmethod
    def create_risk_calculator(risk_limits: dict[str, Any] | None = None) -> Any:
        """
        Create a risk calculator.

        Args:
            risk_limits: Optional risk limits configuration

        Returns:
            Configured risk calculator
        """
        from src.domain.services.risk_calculator import RiskCalculator

        calculator = RiskCalculator()
        # Note: Risk limits would be passed to check_risk_limits method as needed
        # rather than being stored as an attribute
        return calculator

    @staticmethod
    def create_position_manager() -> Any:
        """
        Create a position manager.

        Returns:
            Configured position manager
        """
        from src.domain.services.position_manager import PositionManager

        return PositionManager()

    @staticmethod
    def create_order_processor() -> Any:
        """
        Create an order processor.

        Returns:
            Configured order processor
        """
        from src.domain.services.order_processor import OrderProcessor

        return OrderProcessor()

    @staticmethod
    def create_all_services(broker_type: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """
        Create all domain services for a broker type.

        Args:
            broker_type: Type of broker
            **kwargs: Additional configuration

        Returns:
            Dictionary of configured services
        """
        # Get broker_type from kwargs if not provided as positional arg
        if broker_type is None:
            broker_type = kwargs.pop("broker_type", "paper")

        # Create commission calculator
        commission_calculator = ServiceFactory.create_commission_calculator(
            broker_type,
            kwargs.get("commission_config"),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "commission_config",
                    "market_config",
                    "order_constraints",
                    "exchange",
                    "risk_limits",
                ]
            },
        )

        # Create market microstructure
        market_microstructure = ServiceFactory.create_market_microstructure(
            broker_type,
            kwargs.get("market_config"),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "commission_config",
                    "market_config",
                    "order_constraints",
                    "exchange",
                    "risk_limits",
                ]
            },
        )

        # Create order validator
        order_validator = ServiceFactory.create_order_validator(
            commission_calculator, kwargs.get("order_constraints")
        )

        # Create trading calendar
        trading_calendar = ServiceFactory.create_trading_calendar(kwargs.get("exchange"))

        # Create domain validator
        domain_validator = ServiceFactory.create_domain_validator()

        # Create risk calculator
        risk_calculator = ServiceFactory.create_risk_calculator(kwargs.get("risk_limits"))

        # Create position manager
        position_manager = ServiceFactory.create_position_manager()

        # Create order processor
        order_processor = ServiceFactory.create_order_processor()

        return {
            "commission_calculator": commission_calculator,
            "market_microstructure": market_microstructure,
            "order_validator": order_validator,
            "trading_calendar": trading_calendar,
            "domain_validator": domain_validator,
            "risk_calculator": risk_calculator,
            "position_manager": position_manager,
            "order_processor": order_processor,
        }


class CoordinatorFactory:
    """
    Factory for creating application coordinators.

    This factory creates and wires application-level coordinators
    with their dependencies.
    """

    @staticmethod
    def create_broker_coordinator(broker: Any, services: dict[str, Any]) -> Any:
        """
        Create a broker coordinator with dependencies.

        Args:
            broker: Broker instance
            services: Dictionary of domain services

        Returns:
            Configured broker coordinator
        """
        from src.application.coordinators.broker_coordinator import (
            BrokerCoordinator,
            UseCaseFactory,
        )

        # Create use case factory with all required dependencies
        # Note: unit_of_work must be provided by the caller
        unit_of_work = services.get("unit_of_work")
        if not unit_of_work:
            raise ValueError("unit_of_work must be provided in services")

        use_case_factory = UseCaseFactory(
            unit_of_work=unit_of_work,
            order_processor=services.get("order_processor"),
            commission_calculator=services["commission_calculator"],
            market_microstructure=services["market_microstructure"],
            risk_calculator=services.get("risk_calculator"),
            order_validator=services["order_validator"],
            position_manager=services.get("position_manager"),
        )

        return BrokerCoordinator(
            broker=broker,
            use_case_factory=use_case_factory,
        )
