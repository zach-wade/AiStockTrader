"""
Comprehensive unit tests for Service Factory.

Tests all service factory functionality including:
- Commission calculator creation
- Market microstructure creation
- Order validator creation
- Trading calendar creation
- Domain validator creation
- All services creation
- Coordinator factory
"""

from decimal import Decimal
from unittest.mock import Mock, patch

from src.application.coordinators.service_factory import ServiceFactory
from src.domain.services import (
    CommissionSchedule,
    CommissionType,
    Exchange,
    ICommissionCalculator,
    MarketImpactModel,
    OrderConstraints,
    OrderValidator,
    SlippageConfig,
)


class TestServiceFactoryCommissionCalculator:
    """Test commission calculator creation."""

    @patch("src.application.coordinators.service_factory.CommissionCalculatorFactory")
    def test_create_commission_calculator_alpaca(self, mock_factory):
        """Test creating commission calculator for Alpaca broker."""
        mock_calculator = Mock()
        mock_factory.create.return_value = mock_calculator

        calculator = ServiceFactory.create_commission_calculator("alpaca")

        assert calculator == mock_calculator

        # Verify Alpaca has zero commission
        call_args = mock_factory.create.call_args[0][0]
        assert isinstance(call_args, CommissionSchedule)
        assert call_args.rate == Decimal("0")
        assert call_args.minimum == Decimal("0")
        assert call_args.commission_type == CommissionType.PER_SHARE

    @patch("src.application.coordinators.service_factory.CommissionCalculatorFactory")
    def test_create_commission_calculator_paper(self, mock_factory):
        """Test creating commission calculator for paper broker."""
        mock_calculator = Mock()
        mock_factory.create.return_value = mock_calculator

        calculator = ServiceFactory.create_commission_calculator("paper")

        assert calculator == mock_calculator

        # Verify default paper commission
        call_args = mock_factory.create.call_args[0][0]
        assert isinstance(call_args, CommissionSchedule)
        assert call_args.rate == Decimal("0.01")
        assert call_args.minimum == Decimal("1.0")
        assert call_args.commission_type == CommissionType.PER_SHARE

    @patch("src.application.coordinators.service_factory.CommissionCalculatorFactory")
    def test_create_commission_calculator_custom(self, mock_factory):
        """Test creating commission calculator with custom config."""
        mock_calculator = Mock()
        mock_factory.create.return_value = mock_calculator

        calculator = ServiceFactory.create_commission_calculator(
            "paper", commission_per_share=Decimal("0.005"), min_commission=Decimal("0.50")
        )

        assert calculator == mock_calculator

        # Verify custom commission
        call_args = mock_factory.create.call_args[0][0]
        assert call_args.rate == Decimal("0.005")
        assert call_args.minimum == Decimal("0.50")

    @patch("src.application.coordinators.service_factory.CommissionCalculatorFactory")
    def test_create_commission_calculator_with_config(self, mock_factory):
        """Test creating commission calculator with full config."""
        mock_calculator = Mock()
        mock_factory.create.return_value = mock_calculator

        commission_config = {
            "type": CommissionType.PERCENTAGE,
            "rate": Decimal("0.001"),
            "minimum": Decimal("5.0"),
            "maximum": Decimal("20.0"),
        }

        calculator = ServiceFactory.create_commission_calculator(
            "alpaca", commission_config=commission_config
        )

        assert calculator == mock_calculator
        # Verify the factory was called with a CommissionSchedule
        mock_factory.create.assert_called_once()
        schedule = mock_factory.create.call_args[0][0]
        assert isinstance(schedule, CommissionSchedule)
        assert schedule.commission_type == CommissionType.PERCENTAGE
        assert schedule.rate == Decimal("0.001")
        assert schedule.minimum == Decimal("5.0")
        assert schedule.maximum == Decimal("20.0")

    @patch("src.application.coordinators.service_factory.MarketMicrostructureFactory")
    def test_create_market_microstructure_default(self, mock_factory):
        """Test creating market microstructure with defaults."""
        mock_microstructure = Mock()
        mock_factory.create.return_value = mock_microstructure

        microstructure = ServiceFactory.create_market_microstructure("paper")

        assert microstructure == mock_microstructure

        # Verify default configuration
        mock_factory.create.assert_called_once()
        call_args = mock_factory.create.call_args[0]
        assert call_args[0] == MarketImpactModel.LINEAR
        assert isinstance(call_args[1], SlippageConfig)

    @patch("src.application.coordinators.service_factory.MarketMicrostructureFactory")
    def test_create_market_microstructure_alpaca(self, mock_factory):
        """Test creating market microstructure for Alpaca."""
        mock_microstructure = Mock()
        mock_factory.create.return_value = mock_microstructure

        microstructure = ServiceFactory.create_market_microstructure("alpaca")

        assert microstructure == mock_microstructure

        # Alpaca uses square root model
        call_args = mock_factory.create.call_args[0]
        assert call_args[0] == MarketImpactModel.SQUARE_ROOT

    @patch("src.application.coordinators.service_factory.MarketMicrostructureFactory")
    def test_create_market_microstructure_with_config(self, mock_factory):
        """Test creating market microstructure with custom config."""
        mock_microstructure = Mock()
        mock_factory.create.return_value = mock_microstructure

        market_config = {
            "model": MarketImpactModel.SQUARE_ROOT,
            "slippage": SlippageConfig(
                base_bid_ask_bps=Decimal("20"),  # 0.002 * 10000 = 20 bps
                impact_coefficient=Decimal("0.1"),
            ),
        }

        microstructure = ServiceFactory.create_market_microstructure(
            "backtest", market_config=market_config
        )

        assert microstructure == mock_microstructure
        mock_factory.create.assert_called_once_with(
            market_config["model"], market_config["slippage"]
        )

    @patch("src.application.coordinators.service_factory.MarketMicrostructureFactory")
    def test_create_market_microstructure_custom_params(self, mock_factory):
        """Test creating market microstructure with custom parameters."""
        mock_microstructure = Mock()
        mock_factory.create.return_value = mock_microstructure

        microstructure = ServiceFactory.create_market_microstructure(
            "backtest",
            slippage_pct=Decimal("0.003"),  # This is what ServiceFactory expects for backtest
        )

        assert microstructure == mock_microstructure

        # Verify custom slippage config
        call_args = mock_factory.create.call_args[0]
        slippage_config = call_args[1]
        # ServiceFactory converts slippage_pct to base_bid_ask_bps
        assert slippage_config.base_bid_ask_bps == Decimal("30")  # 0.003 * 10000
        assert slippage_config.impact_coefficient == Decimal("0.01")  # Default for backtest

    def test_create_order_validator(self):
        """Test creating order validator."""
        mock_calculator = Mock(spec=ICommissionCalculator)
        validator = ServiceFactory.create_order_validator(mock_calculator)

        assert validator is not None
        assert isinstance(validator, OrderValidator)
        assert hasattr(validator, "validate_order")

    def test_create_order_validator_with_constraints(self):
        """Test creating order validator with custom constraints."""
        from src.domain.value_objects import Money

        mock_calculator = Mock(spec=ICommissionCalculator)
        constraints = OrderConstraints(
            max_position_size=10000,
            max_order_value=Money(Decimal("100000")),
            min_order_value=Money(Decimal("10")),
            max_portfolio_concentration=Decimal("0.15"),
        )

        validator = ServiceFactory.create_order_validator(mock_calculator, constraints=constraints)

        assert validator is not None
        assert isinstance(validator, OrderValidator)
        assert validator.constraints == constraints

    @patch("src.infrastructure.time.timezone_service.PythonTimeService")
    @patch("src.application.coordinators.service_factory.TradingCalendar")
    def test_create_trading_calendar(self, mock_calendar_class, mock_time_service_class):
        """Test creating trading calendar."""
        mock_calendar = Mock()
        mock_calendar_class.return_value = mock_calendar
        mock_time_service = Mock()
        mock_time_service_class.return_value = mock_time_service

        calendar = ServiceFactory.create_trading_calendar()

        assert calendar == mock_calendar
        mock_time_service_class.assert_called_once()
        mock_calendar_class.assert_called_once_with(mock_time_service, Exchange.NYSE)

    @patch("src.infrastructure.time.timezone_service.PythonTimeService")
    @patch("src.application.coordinators.service_factory.TradingCalendar")
    def test_create_trading_calendar_custom_exchange(
        self, mock_calendar_class, mock_time_service_class
    ):
        """Test creating trading calendar for custom exchange."""
        mock_calendar = Mock()
        mock_calendar_class.return_value = mock_calendar
        mock_time_service = Mock()
        mock_time_service_class.return_value = mock_time_service

        calendar = ServiceFactory.create_trading_calendar(exchange=Exchange.NASDAQ)

        assert calendar == mock_calendar
        mock_time_service_class.assert_called_once()
        mock_calendar_class.assert_called_once_with(mock_time_service, Exchange.NASDAQ)

    def test_create_domain_validator(self):
        """Test creating domain validator."""
        validator = ServiceFactory.create_domain_validator()

        assert validator is not None
        assert hasattr(validator, "validate_order")
        assert hasattr(validator, "validate_portfolio")
        assert hasattr(validator, "validate_position")
        assert hasattr(validator, "validate_trading_request")

    def test_create_risk_calculator(self):
        """Test creating risk calculator."""
        calculator = ServiceFactory.create_risk_calculator()

        assert calculator is not None
        assert hasattr(calculator, "calculate_position_risk")
        assert hasattr(calculator, "calculate_portfolio_var")
        assert hasattr(calculator, "check_risk_limits")

    def test_create_risk_calculator_with_limits(self):
        """Test creating risk calculator with custom limits."""
        risk_limits = {
            "max_position_size": Decimal("50000"),
            "max_portfolio_risk": Decimal("0.02"),
            "max_daily_loss": Decimal("1000"),
        }

        calculator = ServiceFactory.create_risk_calculator(risk_limits=risk_limits)

        assert calculator is not None
        # Risk limits are passed to methods when needed, not stored as attributes
        # This test just verifies the calculator is created successfully
        assert hasattr(calculator, "check_risk_limits")

    def test_create_position_manager(self):
        """Test creating position manager."""
        manager = ServiceFactory.create_position_manager()

        assert manager is not None
        assert hasattr(manager, "open_position")
        assert hasattr(manager, "close_position")
        assert hasattr(manager, "update_position")

    def test_create_order_processor(self):
        """Test creating order processor."""
        processor = ServiceFactory.create_order_processor()

        assert processor is not None
        assert hasattr(processor, "process_fill")
        assert hasattr(processor, "calculate_fill_price")
        assert hasattr(processor, "should_fill_order")

    def test_create_all_services(self):
        """Test creating all services at once."""
        services = ServiceFactory.create_all_services("paper")

        assert "commission_calculator" in services
        assert "market_microstructure" in services
        assert "order_validator" in services
        assert "trading_calendar" in services
        assert "domain_validator" in services
        assert "risk_calculator" in services
        assert "position_manager" in services
        assert "order_processor" in services

        # Verify all services are not None
        for service_name, service in services.items():
            assert service is not None, f"{service_name} should not be None"

    def test_create_all_services_with_config(self):
        """Test creating all services with configuration."""
        config = {
            "broker_type": "alpaca",
            "commission_config": {"type": CommissionType.PERCENTAGE, "rate": Decimal("0.001")},
            "risk_limits": {"max_position_size": Decimal("100000")},
        }

        services = ServiceFactory.create_all_services(**config)

        assert len(services) > 0
        assert all(service is not None for service in services.values())

    def test_create_services_for_backtesting(self):
        """Test creating services optimized for backtesting."""
        services = ServiceFactory.create_all_services(
            broker_type="backtest",
            commission_per_share=Decimal("0.005"),
            base_slippage=Decimal("0.001"),
            impact_coefficient=Decimal("0.1"),
        )

        assert services["commission_calculator"] is not None
        assert services["market_microstructure"] is not None

        # Backtesting should have configurable commission and slippage

    def test_service_factory_thread_safety(self):
        """Test that service factory is thread-safe."""
        import threading

        results = []

        def create_service():
            service = ServiceFactory.create_commission_calculator("paper")
            results.append(service)

        threads = [threading.Thread(target=create_service) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(r is not None for r in results)
