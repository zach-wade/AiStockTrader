"""
Comprehensive tests for risk management use cases with full coverage.

Tests all risk-related use cases including:
- CalculateRiskUseCase
- ValidateOrderRiskUseCase
- GetRiskMetricsUseCase

Covers all scenarios including success, failure, validation, and edge cases.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.interfaces.repositories import IOrderRepository, IPortfolioRepository
from src.application.interfaces.unit_of_work import IUnitOfWork
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


@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = Mock(spec=IUnitOfWork)

    # Setup repository mocks
    uow.orders = AsyncMock(spec=IOrderRepository)
    uow.portfolios = AsyncMock(spec=IPortfolioRepository)
    uow.positions = AsyncMock()

    # Setup transaction methods
    uow.begin = AsyncMock()
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()

    # Make it work as async context manager
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)

    return uow


@pytest.fixture
def mock_risk_calculator():
    """Create a mock risk calculator."""
    calculator = Mock(spec=RiskCalculator)
    calculator.calculate_portfolio_var = Mock(return_value=Money(Decimal("5000")))
    calculator.calculate_sharpe_ratio = Mock(return_value=Decimal("1.5"))
    calculator.calculate_max_drawdown = Mock(return_value=Decimal("0.15"))
    calculator.check_risk_limits = Mock(return_value=(True, None))
    return calculator


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    portfolio = Portfolio(
        id=uuid4(),
        name="Test Portfolio",
        initial_capital=Decimal("100000"),
        cash_balance=Decimal("50000"),
    )
    portfolio.total_realized_pnl = Decimal("5000")

    # Add some positions
    position1 = Position(
        symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
    )
    position1.current_price = Decimal("155.00")

    position2 = Position(
        symbol="GOOGL", quantity=Decimal("50"), average_entry_price=Decimal("2800.00")
    )
    position2.current_price = Decimal("2850.00")

    portfolio.positions = {position1.id: position1, position2.id: position2}

    return portfolio


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    order = Order(
        symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
    )
    order.portfolio_id = uuid4()
    return order


class TestCalculateRiskUseCase:
    """Test CalculateRiskUseCase."""

    @pytest.mark.asyncio
    async def test_calculate_risk_all_metrics(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating all risk metrics successfully."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_drawdown=True,
            confidence_level=0.95,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk == Decimal("5000")
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown == Decimal("0.15")
        assert response.risk_score == Decimal("0.5")

        # Verify calculations were called
        mock_risk_calculator.calculate_portfolio_var.assert_called_once()
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once()
        mock_risk_calculator.calculate_max_drawdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_risk_var_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating only VaR metric."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=False,
            include_drawdown=False,
            confidence_level=0.99,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk == Decimal("5000")
        assert response.sharpe_ratio is None
        assert response.max_drawdown is None

        # Verify only VaR was calculated
        mock_risk_calculator.calculate_portfolio_var.assert_called_once()
        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_risk_sharpe_only(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating only Sharpe ratio."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=True,
            include_drawdown=False,
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk is None
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown is None

    @pytest.mark.asyncio
    async def test_calculate_risk_portfolio_not_found(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test risk calculation when portfolio doesn't exist."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        mock_risk_calculator.calculate_portfolio_var.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_low(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test validation with confidence level too low."""
        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.0)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Confidence level must be between 0 and 1" in response.error

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_high(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test validation with confidence level too high."""
        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=1.0)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Confidence level must be between 0 and 1" in response.error

    @pytest.mark.asyncio
    async def test_calculate_risk_with_different_confidence_levels(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test risk calculation with different confidence levels."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        # Test with 90% confidence
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=False,
            include_drawdown=False,
            confidence_level=0.90,
        )

        response = await use_case.execute(request)
        assert response.success is True

        # Verify confidence level was passed correctly
        call_args = mock_risk_calculator.calculate_portfolio_var.call_args
        assert call_args[1]["confidence_level"] == Decimal("0.90")

    @pytest.mark.asyncio
    async def test_calculate_risk_exception_handling(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test risk calculation with exception in calculator."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.side_effect = Exception("Calculation error")

        use_case = CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = CalculateRiskRequest(portfolio_id=sample_portfolio.id, include_var=True)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Calculation error" in response.error
        mock_unit_of_work.rollback.assert_called_once()


class TestValidateOrderRiskUseCase:
    """Test ValidateOrderRiskUseCase."""

    @pytest.mark.asyncio
    async def test_validate_order_risk_success(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test successful order risk validation."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("155.00"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is True
        assert len(response.risk_violations) == 0
        assert response.risk_metrics is not None
        assert "position_size_pct" in response.risk_metrics
        assert "leverage" in response.risk_metrics
        assert "concentration" in response.risk_metrics
        assert "max_loss" in response.risk_metrics

        # Verify risk check was called
        mock_risk_calculator.check_risk_limits.assert_called_once_with(
            portfolio=sample_portfolio, new_order=sample_order
        )

    @pytest.mark.asyncio
    async def test_validate_order_risk_violation(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test order risk validation with violations."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds 10% of portfolio",
        )

        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("155.00"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is False
        assert len(response.risk_violations) == 1
        assert "Position size exceeds 10% of portfolio" in response.risk_violations
        assert response.risk_metrics is not None

    @pytest.mark.asyncio
    async def test_validate_order_risk_order_not_found(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test risk validation when order doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("155.00")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order not found" in response.error
        mock_risk_calculator.check_risk_limits.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_order_risk_portfolio_not_found(
        self, mock_unit_of_work, mock_risk_calculator, sample_order
    ):
        """Test risk validation when portfolio doesn't exist."""
        # Setup
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id, portfolio_id=uuid4(), current_price=Decimal("155.00")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        mock_risk_calculator.check_risk_limits.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_invalid_price(self, mock_unit_of_work, mock_risk_calculator):
        """Test validation with invalid price."""
        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("0")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Current price must be positive" in response.error

    @pytest.mark.asyncio
    async def test_validate_order_risk_metrics_calculation(
        self, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test detailed risk metrics calculation."""
        # Setup
        sample_order.quantity = Decimal("200")
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        # Execute
        response = await use_case.execute(request)

        # Assert metrics calculation
        assert response.success is True
        metrics = response.risk_metrics

        # Position value = 200 * 150 = 30,000
        position_value = Decimal("30000")
        portfolio_value = sample_portfolio.get_total_value_sync()

        expected_position_pct = float(position_value / portfolio_value * 100)
        assert abs(metrics["position_size_pct"] - expected_position_pct) < 0.01

        expected_concentration = float(position_value / portfolio_value)
        assert abs(metrics["concentration"] - expected_concentration) < 0.01

        expected_max_loss = float(position_value * Decimal("0.1"))
        assert metrics["max_loss"] == expected_max_loss


class TestGetRiskMetricsUseCase:
    """Test GetRiskMetricsUseCase."""

    @pytest.mark.asyncio
    async def test_get_risk_metrics_success(
        self, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful risk metrics retrieval."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        use_case = GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetRiskMetricsRequest(portfolio_id=sample_portfolio.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics is not None

        metrics = response.metrics
        assert "portfolio_value" in metrics
        assert "positions_value" in metrics
        assert "cash_balance" in metrics
        assert "leverage" in metrics
        assert "position_count" in metrics
        assert "concentration" in metrics
        assert "unrealized_pnl" in metrics
        assert "realized_pnl" in metrics
        assert "total_return_pct" in metrics

        # Verify values
        assert metrics["position_count"] == 2  # Two positions in sample portfolio
        assert metrics["realized_pnl"] == 5000.0

    @pytest.mark.asyncio
    async def test_get_risk_metrics_empty_portfolio(self, mock_unit_of_work, mock_risk_calculator):
        """Test risk metrics for empty portfolio."""
        # Setup empty portfolio
        empty_portfolio = Portfolio(
            id=uuid4(),
            name="Empty Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100000")),
        )
        empty_portfolio.positions = {}
        empty_portfolio.total_realized_pnl = Money(Decimal("0"))

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = empty_portfolio

        use_case = GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetRiskMetricsRequest(portfolio_id=empty_portfolio.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics is not None

        metrics = response.metrics
        assert metrics["position_count"] == 0
        assert metrics["positions_value"] == 0.0
        assert metrics["leverage"] == 1.0  # No leverage with no positions
        assert metrics["concentration"] == 0.0
        assert metrics["unrealized_pnl"] == 0.0

    @pytest.mark.asyncio
    async def test_get_risk_metrics_portfolio_not_found(
        self, mock_unit_of_work, mock_risk_calculator
    ):
        """Test risk metrics when portfolio doesn't exist."""
        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        use_case = GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Portfolio not found" in response.error
        assert response.metrics == {}

    @pytest.mark.asyncio
    async def test_get_risk_metrics_with_zero_cash(self, mock_unit_of_work, mock_risk_calculator):
        """Test risk metrics calculation with zero cash balance."""
        # Setup portfolio with zero cash
        portfolio = Portfolio(
            id=uuid4(),
            name="Zero Cash Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("0")),
        )
        portfolio.positions = {}

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        use_case = GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        metrics = response.metrics
        assert metrics["leverage"] == 0  # Division by zero handled
        assert metrics["cash_balance"] == 0.0


class TestRiskRequestResponseDTOs:
    """Test risk management request and response DTOs."""

    def test_calculate_risk_request_defaults(self):
        """Test CalculateRiskRequest default values."""
        request = CalculateRiskRequest(portfolio_id=uuid4())

        assert request.include_var is True
        assert request.include_sharpe is True
        assert request.include_drawdown is True
        assert request.confidence_level == 0.95
        assert request.request_id is not None
        assert request.metadata == {}

    def test_calculate_risk_request_custom_values(self):
        """Test CalculateRiskRequest with custom values."""
        portfolio_id = uuid4()
        request_id = uuid4()

        request = CalculateRiskRequest(
            portfolio_id=portfolio_id,
            include_var=False,
            include_sharpe=True,
            include_drawdown=False,
            confidence_level=0.99,
            request_id=request_id,
            metadata={"source": "risk_monitor"},
        )

        assert request.portfolio_id == portfolio_id
        assert request.include_var is False
        assert request.include_sharpe is True
        assert request.include_drawdown is False
        assert request.confidence_level == 0.99
        assert request.request_id == request_id
        assert request.metadata == {"source": "risk_monitor"}

    def test_validate_order_risk_response_defaults(self):
        """Test ValidateOrderRiskResponse default values."""
        response = ValidateOrderRiskResponse(success=True)

        assert response.is_valid is False
        assert response.risk_violations == []
        assert response.risk_metrics is None

    def test_get_risk_metrics_response_defaults(self):
        """Test GetRiskMetricsResponse default values."""
        response = GetRiskMetricsResponse(success=True)

        assert response.metrics == {}

    def test_risk_response_with_metrics(self):
        """Test risk response with full metrics."""
        response = CalculateRiskResponse(
            success=True,
            value_at_risk=Decimal("5000"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.15"),
            portfolio_beta=Decimal("1.2"),
            risk_score=Decimal("0.6"),
            request_id=uuid4(),
        )

        assert response.success is True
        assert response.value_at_risk == Decimal("5000")
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown == Decimal("0.15")
        assert response.portfolio_beta == Decimal("1.2")
        assert response.risk_score == Decimal("0.6")
