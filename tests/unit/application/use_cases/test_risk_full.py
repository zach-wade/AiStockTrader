"""
Comprehensive Unit Tests for Risk Management Use Cases

Tests all risk use cases with complete coverage including:
- CalculateRiskUseCase
- ValidateOrderRiskUseCase
- GetRiskMetricsUseCase

Achieves 80%+ coverage with focus on business logic validation.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.risk import (
    CalculateRiskRequest,
    CalculateRiskResponse,
    CalculateRiskUseCase,
    GetRiskMetricsRequest,
    GetRiskMetricsResponse,
    GetRiskMetricsUseCase,
    ValidateOrderRiskRequest,
    ValidateOrderRiskResponse,
    ValidateOrderRiskUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.risk_calculator import RiskCalculator
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity


# Fixtures
@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work with all repositories."""
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Mock repositories
    uow.orders = AsyncMock()
    uow.portfolios = AsyncMock()

    return uow


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    calculator.calculate_portfolio_var = Mock(return_value=Money(Decimal("500.00")))
    calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("1.5"))
    calculator.calculate_max_drawdown = Mock(return_value=Decimal("0.15"))
    calculator.check_risk_limits = Mock(return_value=(True, None))
    return calculator


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        name="Test Portfolio",
        initial_capital=Money(Decimal("10000.00")),
    )
    portfolio.id = uuid4()
    portfolio.cash_balance = Money(Decimal("5000.00"))
    portfolio.total_realized_pnl = Money(Decimal("100.00"))

    # Add some positions
    position1 = Position(
        symbol="AAPL",
        quantity=Quantity(Decimal("100")),
        average_entry_price=Price(Decimal("150.00")),
    )
    position1.current_price = Price(Decimal("155.00"))

    position2 = Position(
        symbol="GOOGL",
        quantity=Quantity(Decimal("50")),
        average_entry_price=Price(Decimal("2800.00")),
    )
    position2.current_price = Price(Decimal("2850.00"))

    portfolio.positions = {
        position1.id: position1,
        position2.id: position2,
    }

    return portfolio


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Quantity(Decimal("100")),
        limit_price=Price(Decimal("150.00")),
    )
    order.id = uuid4()
    return order


# Test CalculateRiskUseCase
class TestCalculateRiskUseCase:
    """Test CalculateRiskUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_risk_calculator):
        """Test use case initialization."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "CalculateRiskUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_risk_calculator):
        """Test successful validation."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = CalculateRiskRequest(
            portfolio_id=uuid4(),
            confidence_level=0.95,
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_zero(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test validation with confidence level of zero."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = CalculateRiskRequest(
            portfolio_id=uuid4(),
            confidence_level=0.0,
        )

        result = await use_case.validate(request)
        assert result == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_one(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test validation with confidence level of one."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = CalculateRiskRequest(
            portfolio_id=uuid4(),
            confidence_level=1.0,
        )

        result = await use_case.validate(request)
        assert result == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_negative(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test validation with negative confidence level."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = CalculateRiskRequest(
            portfolio_id=uuid4(),
            confidence_level=-0.5,
        )

        result = await use_case.validate(request)
        assert result == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_process_all_metrics(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with all metrics requested."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_drawdown=True,
            confidence_level=0.95,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.value_at_risk == Decimal("500.00")
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown == Decimal("0.15")
        assert response.risk_score == Decimal("0.5")

        # Verify calculations were called
        mock_risk_calculator.calculate_portfolio_var.assert_called_once()
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once()
        mock_risk_calculator.calculate_max_drawdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_var_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with only VaR requested."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=False,
            include_drawdown=False,
            confidence_level=0.99,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.value_at_risk == Decimal("500.00")
        assert response.sharpe_ratio is None
        assert response.max_drawdown is None

        # Verify only VaR was calculated
        mock_risk_calculator.calculate_portfolio_var.assert_called_once()
        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_sharpe_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with only Sharpe ratio requested."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=True,
            include_drawdown=False,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.value_at_risk is None
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown is None

    @pytest.mark.asyncio
    async def test_process_drawdown_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with only max drawdown requested."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=False,
            include_drawdown=True,
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.value_at_risk is None
        assert response.sharpe_ratio is None
        assert response.max_drawdown == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(self, mock_unit_of_work, mock_risk_calculator):
        """Test processing when portfolio not found."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = CalculateRiskRequest(portfolio_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with metadata and correlation ID."""
        use_case = CalculateRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            correlation_id=uuid4(),
            metadata={"source": "risk_monitor"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test ValidateOrderRiskUseCase
class TestValidateOrderRiskUseCase:
    """Test ValidateOrderRiskUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_risk_calculator):
        """Test use case initialization."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "ValidateOrderRiskUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_risk_calculator):
        """Test successful validation."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = ValidateOrderRiskRequest(
            order_id=uuid4(),
            portfolio_id=uuid4(),
            current_price=Decimal("150.00"),
        )

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_negative_price(self, mock_unit_of_work, mock_risk_calculator):
        """Test validation with negative current price."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = ValidateOrderRiskRequest(
            order_id=uuid4(),
            portfolio_id=uuid4(),
            current_price=Decimal("-150.00"),
        )

        result = await use_case.validate(request)
        assert result == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_price(self, mock_unit_of_work, mock_risk_calculator):
        """Test validation with zero current price."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        request = ValidateOrderRiskRequest(
            order_id=uuid4(),
            portfolio_id=uuid4(),
            current_price=Decimal("0"),
        )

        result = await use_case.validate(request)
        assert result == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_process_valid_order(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test processing valid order with no risk violations."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.is_valid is True
        assert response.risk_violations == []
        assert response.risk_metrics is not None
        assert "position_size_pct" in response.risk_metrics
        assert "leverage" in response.risk_metrics
        assert "concentration" in response.risk_metrics
        assert "max_loss" in response.risk_metrics

    @pytest.mark.asyncio
    async def test_process_order_with_risk_violations(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test processing order with risk violations."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (False, "Position size exceeds limit")

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.is_valid is False
        assert response.risk_violations == ["Position size exceeds limit"]
        assert response.risk_metrics is not None

    @pytest.mark.asyncio
    async def test_process_order_not_found(self, mock_unit_of_work, mock_risk_calculator):
        """Test processing when order not found."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        request = ValidateOrderRiskRequest(
            order_id=uuid4(),
            portfolio_id=uuid4(),
            current_price=Decimal("150.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Order not found"
        assert response.is_valid is False

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(
        self, mock_unit_of_work, mock_risk_calculator, sample_order
    ):
        """Test processing when portfolio not found."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=uuid4(),
            current_price=Decimal("150.00"),
        )

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.is_valid is False

    @pytest.mark.asyncio
    async def test_process_risk_metrics_calculation(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test risk metrics calculation details."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        response = await use_case.process(request)

        assert response.success is True

        # Check metrics calculations
        from src.domain.value_objects.converter import ValueObjectConverter

        quantity_val = ValueObjectConverter.extract_value(sample_order.quantity)
        position_value = quantity_val * request.current_price
        portfolio_value = ValueObjectConverter.extract_amount(sample_portfolio.get_total_value())
        cash_balance = ValueObjectConverter.extract_amount(sample_portfolio.cash_balance)

        expected_position_size_pct = float(position_value / portfolio_value * 100)
        expected_leverage = float(portfolio_value / cash_balance)
        expected_concentration = float(position_value / portfolio_value)
        expected_max_loss = float(position_value * Decimal("0.1"))

        assert response.risk_metrics["position_size_pct"] == pytest.approx(
            expected_position_size_pct, rel=0.01
        )
        assert response.risk_metrics["leverage"] == pytest.approx(expected_leverage, rel=0.01)
        assert response.risk_metrics["concentration"] == pytest.approx(
            expected_concentration, rel=0.01
        )
        assert response.risk_metrics["max_loss"] == pytest.approx(expected_max_loss, rel=0.01)

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test processing with metadata and correlation ID."""
        use_case = ValidateOrderRiskUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
            correlation_id=uuid4(),
            metadata={"check_type": "pre_trade"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test GetRiskMetricsUseCase
class TestGetRiskMetricsUseCase:
    """Test GetRiskMetricsUseCase class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_unit_of_work, mock_risk_calculator):
        """Test use case initialization."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.risk_calculator == mock_risk_calculator
        assert use_case.name == "GetRiskMetricsUseCase"

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_unit_of_work, mock_risk_calculator):
        """Test successful validation."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        result = await use_case.validate(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_successful_metrics_retrieval(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful risk metrics retrieval."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetRiskMetricsRequest(portfolio_id=sample_portfolio.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.metrics is not None

        # Check all expected metrics are present
        expected_metrics = [
            "portfolio_value",
            "positions_value",
            "cash_balance",
            "leverage",
            "position_count",
            "concentration",
            "unrealized_pnl",
            "realized_pnl",
            "total_return_pct",
        ]

        for metric in expected_metrics:
            assert metric in response.metrics

    @pytest.mark.asyncio
    async def test_process_portfolio_not_found(self, mock_unit_of_work, mock_risk_calculator):
        """Test processing when portfolio not found."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        response = await use_case.process(request)

        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.metrics == {}

    @pytest.mark.asyncio
    async def test_process_metrics_calculation(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test detailed metrics calculation."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetRiskMetricsRequest(portfolio_id=sample_portfolio.id)

        response = await use_case.process(request)

        assert response.success is True

        # Verify metric calculations
        from src.domain.value_objects.converter import ValueObjectConverter

        portfolio_value = ValueObjectConverter.extract_amount(sample_portfolio.get_total_value())
        positions_value = ValueObjectConverter.extract_amount(
            sample_portfolio.get_positions_value()
        )
        cash_balance = ValueObjectConverter.extract_amount(sample_portfolio.cash_balance)

        assert response.metrics["portfolio_value"] == float(portfolio_value)
        assert response.metrics["positions_value"] == float(positions_value)
        assert response.metrics["cash_balance"] == float(cash_balance)
        assert response.metrics["position_count"] == len(sample_portfolio.get_open_positions())
        assert response.metrics["unrealized_pnl"] == float(
            ValueObjectConverter.extract_amount(sample_portfolio.get_unrealized_pnl())
        )
        assert response.metrics["realized_pnl"] == float(
            ValueObjectConverter.extract_amount(sample_portfolio.total_realized_pnl)
        )
        assert response.metrics["total_return_pct"] == float(
            sample_portfolio.get_total_return() * 100
        )

    @pytest.mark.asyncio
    async def test_process_with_zero_cash_balance(self, mock_unit_of_work, mock_risk_calculator):
        """Test metrics calculation with zero cash balance."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Create portfolio with zero cash
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("10000.00")),
        )
        portfolio.id = uuid4()
        portfolio.cash_balance = Money(Decimal("0"))

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.metrics["leverage"] == 0  # Should handle division by zero

    @pytest.mark.asyncio
    async def test_process_with_zero_portfolio_value(self, mock_unit_of_work, mock_risk_calculator):
        """Test metrics calculation with zero portfolio value."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Create portfolio with minimal value (zero would violate validation)
        portfolio = Portfolio(
            name="Test Portfolio",
            initial_capital=Money(Decimal("0.01")),
        )
        portfolio.id = uuid4()
        portfolio.cash_balance = Money(Decimal("0.01"))

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)

        response = await use_case.process(request)

        assert response.success is True
        assert response.metrics["concentration"] == 0  # Should handle division by zero

    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test processing with metadata and correlation ID."""
        use_case = GetRiskMetricsUseCase(mock_unit_of_work, mock_risk_calculator)

        # Setup mocks
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        request = GetRiskMetricsRequest(
            portfolio_id=sample_portfolio.id,
            correlation_id=uuid4(),
            metadata={"monitor": "dashboard"},
        )

        response = await use_case.process(request)

        assert response.success is True
        assert response.request_id == request.request_id


# Test Request/Response DTOs
class TestRequestResponseDTOs:
    """Test request and response data classes."""

    def test_calculate_risk_request_init(self):
        """Test CalculateRiskRequest initialization."""
        request = CalculateRiskRequest(portfolio_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}
        assert request.include_var is True
        assert request.include_sharpe is True
        assert request.include_drawdown is True
        assert request.confidence_level == 0.95

    def test_calculate_risk_request_with_values(self):
        """Test CalculateRiskRequest with custom values."""
        portfolio_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()

        request = CalculateRiskRequest(
            portfolio_id=portfolio_id,
            include_var=False,
            include_sharpe=True,
            include_drawdown=False,
            confidence_level=0.99,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata={"type": "daily"},
        )

        assert request.portfolio_id == portfolio_id
        assert request.include_var is False
        assert request.include_sharpe is True
        assert request.include_drawdown is False
        assert request.confidence_level == 0.99
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == {"type": "daily"}

    def test_validate_order_risk_request_init(self):
        """Test ValidateOrderRiskRequest initialization."""
        request = ValidateOrderRiskRequest(
            order_id=uuid4(),
            portfolio_id=uuid4(),
            current_price=Decimal("150.00"),
        )

        assert request.request_id is not None
        assert request.metadata == {}

    def test_get_risk_metrics_request_init(self):
        """Test GetRiskMetricsRequest initialization."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        assert request.request_id is not None
        assert request.metadata == {}

    def test_calculate_risk_response(self):
        """Test CalculateRiskResponse initialization."""
        response = CalculateRiskResponse(
            success=True,
            value_at_risk=Decimal("500.00"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.15"),
            portfolio_beta=Decimal("1.2"),
            risk_score=Decimal("0.6"),
        )

        assert response.success is True
        assert response.value_at_risk == Decimal("500.00")
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown == Decimal("0.15")
        assert response.portfolio_beta == Decimal("1.2")
        assert response.risk_score == Decimal("0.6")

    def test_validate_order_risk_response_init(self):
        """Test ValidateOrderRiskResponse initialization with defaults."""
        response = ValidateOrderRiskResponse(success=True)

        assert response.success is True
        assert response.is_valid is False
        assert response.risk_violations == []
        assert response.risk_metrics is None

    def test_validate_order_risk_response_with_values(self):
        """Test ValidateOrderRiskResponse with values."""
        response = ValidateOrderRiskResponse(
            success=True,
            is_valid=True,
            risk_violations=["Position too large"],
            risk_metrics={"leverage": 2.0},
        )

        assert response.success is True
        assert response.is_valid is True
        assert response.risk_violations == ["Position too large"]
        assert response.risk_metrics == {"leverage": 2.0}

    def test_get_risk_metrics_response_init(self):
        """Test GetRiskMetricsResponse initialization with defaults."""
        response = GetRiskMetricsResponse(success=True)

        assert response.success is True
        assert response.metrics == {}

    def test_get_risk_metrics_response_with_values(self):
        """Test GetRiskMetricsResponse with values."""
        metrics = {
            "portfolio_value": 10000.0,
            "leverage": 1.5,
            "position_count": 5,
        }

        response = GetRiskMetricsResponse(
            success=True,
            metrics=metrics,
        )

        assert response.success is True
        assert response.metrics == metrics
