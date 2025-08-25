"""
Comprehensive tests for risk management use cases.

Tests all risk-related use cases including risk calculation, order validation,
and risk metrics retrieval with full coverage.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
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
from src.domain.entities.portfolio import Portfolio
from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price


class TestCalculateRiskRequest:
    """Test CalculateRiskRequest data class."""

    def test_request_with_defaults(self):
        """Test request initialization with defaults."""
        portfolio_id = uuid4()
        request = CalculateRiskRequest(portfolio_id=portfolio_id)

        assert request.portfolio_id == portfolio_id
        assert request.include_var is True
        assert request.include_sharpe is True
        assert request.include_drawdown is True
        assert request.confidence_level == 0.95
        assert request.request_id is not None
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_request_with_custom_values(self):
        """Test request with custom values."""
        portfolio_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"key": "value"}

        request = CalculateRiskRequest(
            portfolio_id=portfolio_id,
            include_var=False,
            include_sharpe=False,
            include_drawdown=False,
            confidence_level=0.99,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.portfolio_id == portfolio_id
        assert request.include_var is False
        assert request.include_sharpe is False
        assert request.include_drawdown is False
        assert request.confidence_level == 0.99
        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_request_post_init_with_none_metadata(self):
        """Test that None metadata gets initialized as empty dict."""
        request = CalculateRiskRequest(portfolio_id=uuid4(), metadata=None)
        assert request.metadata == {}


class TestValidateOrderRiskRequest:
    """Test ValidateOrderRiskRequest data class."""

    def test_request_with_defaults(self):
        """Test request initialization with defaults."""
        order_id = uuid4()
        portfolio_id = uuid4()
        current_price = Decimal("150.00")

        request = ValidateOrderRiskRequest(
            order_id=order_id, portfolio_id=portfolio_id, current_price=current_price
        )

        assert request.order_id == order_id
        assert request.portfolio_id == portfolio_id
        assert request.current_price == current_price
        assert request.request_id is not None
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_request_with_custom_values(self):
        """Test request with custom values."""
        order_id = uuid4()
        portfolio_id = uuid4()
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"source": "api"}

        request = ValidateOrderRiskRequest(
            order_id=order_id,
            portfolio_id=portfolio_id,
            current_price=Decimal("200.00"),
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata,
        )

        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata


class TestGetRiskMetricsRequest:
    """Test GetRiskMetricsRequest data class."""

    def test_request_with_defaults(self):
        """Test request initialization with defaults."""
        portfolio_id = uuid4()
        request = GetRiskMetricsRequest(portfolio_id=portfolio_id)

        assert request.portfolio_id == portfolio_id
        assert request.request_id is not None
        assert request.correlation_id is None
        assert request.metadata == {}


class TestCalculateRiskResponse:
    """Test CalculateRiskResponse data class."""

    def test_response_defaults(self):
        """Test response with default values."""
        response = CalculateRiskResponse(success=True)

        assert response.success is True
        assert response.value_at_risk is None
        assert response.sharpe_ratio is None
        assert response.max_drawdown is None
        assert response.portfolio_beta is None
        assert response.risk_score is None


class TestValidateOrderRiskResponse:
    """Test ValidateOrderRiskResponse data class."""

    def test_response_defaults(self):
        """Test response with default values."""
        response = ValidateOrderRiskResponse(success=True)

        assert response.success is True
        assert response.is_valid is False
        assert response.risk_violations == []
        assert response.risk_metrics is None

    def test_response_with_none_violations(self):
        """Test response initializes None violations as empty list."""
        response = ValidateOrderRiskResponse(success=True, risk_violations=None)
        assert response.risk_violations == []


class TestGetRiskMetricsResponse:
    """Test GetRiskMetricsResponse data class."""

    def test_response_defaults(self):
        """Test response with default values."""
        response = GetRiskMetricsResponse(success=True)

        assert response.success is True
        assert response.metrics == {}

    def test_response_with_none_metrics(self):
        """Test response initializes None metrics as empty dict."""
        response = GetRiskMetricsResponse(success=True, metrics=None)
        assert response.metrics == {}


class TestCalculateRiskUseCase:
    """Test CalculateRiskUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock()
        calculator.calculate_portfolio_var = Mock()
        calculator.calculate_sharpe_ratio = Mock()
        calculator.calculate_max_drawdown = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio with positions."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        portfolio.cash_balance = Money(Decimal("50000"))
        portfolio.total_realized_pnl = Money(Decimal("5000"))

        # Add positions for value calculation
        position = Mock()
        position.id = uuid4()
        position.portfolio_id = portfolio.id
        position.quantity = 100
        position.entry_price = Price(Decimal("150.00"))
        position.current_price = Price(Decimal("155.00"))
        position.get_current_value = Mock(return_value=Money(Decimal("15500")))
        position.is_open = True

        portfolio.positions = {position.id: position}
        portfolio.get_total_return = Mock(return_value=Decimal("0.05"))
        portfolio.get_total_value = Mock(return_value=Money(Decimal("65500")))
        portfolio.get_total_value_sync = Mock(return_value=Money(Decimal("65500")))

        return portfolio

    @pytest.mark.asyncio
    async def test_calculate_all_risk_metrics(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating all risk metrics."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_drawdown=True,
            confidence_level=0.95,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("5000.00"))
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("1.25")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.15")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk == Decimal("5000.00")
        assert response.sharpe_ratio == Decimal("1.25")
        assert response.max_drawdown == Decimal("0.15")
        assert response.risk_score == Decimal("0.5")  # Default score

        mock_risk_calculator.calculate_portfolio_var.assert_called_once_with(
            portfolio=sample_portfolio, confidence_level=Decimal("0.95"), time_horizon=1
        )
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once_with(
            returns=[Decimal("0.05")], risk_free_rate=Decimal("0.02")
        )
        mock_risk_calculator.calculate_max_drawdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_var_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating VaR only."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=False,
            include_drawdown=False,
            confidence_level=0.99,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("7500.00"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk == Decimal("7500.00")
        assert response.sharpe_ratio is None
        assert response.max_drawdown is None

        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating Sharpe ratio only."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=True,
            include_drawdown=False,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("2.0")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.sharpe_ratio == Decimal("2.0")
        assert response.value_at_risk is None
        assert response.max_drawdown is None

    @pytest.mark.asyncio
    async def test_calculate_drawdown_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating max drawdown only."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=False,
            include_drawdown=True,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.25")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.max_drawdown == Decimal("0.25")
        assert response.value_at_risk is None
        assert response.sharpe_ratio is None

    @pytest.mark.asyncio
    async def test_calculate_no_metrics_requested(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test when no metrics are requested."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=False,
            include_drawdown=False,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk is None
        assert response.sharpe_ratio is None
        assert response.max_drawdown is None
        assert response.risk_score == Decimal("0.5")  # Still calculated

        mock_risk_calculator.calculate_portfolio_var.assert_not_called()
        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_risk_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test risk calculation when portfolio doesn't exist."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_zero(self, use_case):
        """Test validation with zero confidence level."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.0)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_one(self, use_case):
        """Test validation with confidence level of 1."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=1.0)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_negative(self, use_case):
        """Test validation with negative confidence level."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=-0.5)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level_above_one(self, use_case):
        """Test validation with confidence level above 1."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=1.5)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_valid_confidence_level(self, use_case):
        """Test validation with valid confidence level."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.95)

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_edge_confidence_levels(self, use_case):
        """Test validation with edge case confidence levels."""
        # Just above 0
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.001)
        assert await use_case.validate(request) is None

        # Just below 1
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.999)
        assert await use_case.validate(request) is None

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, use_case, mock_unit_of_work):
        """Test exception handling during processing."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.side_effect = Exception("Database error")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Database error" in response.error
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_commit_on_success(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test that transaction is committed on success."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=False,
            include_sharpe=False,
            include_drawdown=False,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, use_case, mock_unit_of_work):
        """Test that transaction is rolled back on failure."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()


class TestValidateOrderRiskUseCase:
    """Test ValidateOrderRiskUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock()
        calculator.check_risk_limits = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        order = Mock()
        order.id = uuid4()
        order.portfolio_id = uuid4()
        order.quantity = 100
        return order

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Mock()
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("50000")
        portfolio.get_total_value = Mock(return_value=Decimal("65500"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("65500"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("65500"))
        return portfolio

    @pytest.mark.asyncio
    async def test_validate_order_risk_passes(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test order risk validation that passes."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (True, None)  # Passes

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

    @pytest.mark.asyncio
    async def test_validate_order_risk_fails_with_single_violation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test order risk validation that fails with single violation."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds maximum",
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is False
        assert len(response.risk_violations) == 1
        assert "Position size exceeds maximum" in response.risk_violations

    @pytest.mark.asyncio
    async def test_validate_order_risk_metrics_calculation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test risk metrics calculation."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True

        # Check metric calculations
        position_value = 100 * 150  # quantity * price
        portfolio_value = 65500

        assert response.risk_metrics["position_size_pct"] == pytest.approx(
            float(position_value / portfolio_value * 100), rel=0.01
        )
        assert response.risk_metrics["leverage"] == pytest.approx(
            float(portfolio_value / 50000), rel=0.01
        )
        assert response.risk_metrics["concentration"] == pytest.approx(
            float(position_value / portfolio_value), rel=0.01
        )
        assert response.risk_metrics["max_loss"] == float(position_value * Decimal("0.1"))

    @pytest.mark.asyncio
    async def test_validate_order_not_found(self, use_case, mock_unit_of_work):
        """Test validation when order doesn't exist."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_validate_portfolio_not_found(self, use_case, mock_unit_of_work, sample_order):
        """Test validation when portfolio doesn't exist."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id, portfolio_id=uuid4(), current_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_validate_negative_current_price(self, use_case):
        """Test validation with negative current price."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("-150.00")
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_current_price(self, use_case):
        """Test validation with zero current price."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("0")
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_validate_positive_current_price(self, use_case):
        """Test validation with positive current price."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("150.00")
        )

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None

    @pytest.mark.asyncio
    async def test_process_with_price_object_creation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test that Price object is created correctly."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("999.99"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.check_risk_limits.return_value = (True, None)

        # Execute
        with patch("src.application.use_cases.risk.Price") as mock_price_class:
            mock_price = Mock()
            mock_price = Decimal("999.99")
            mock_price_class.return_value = mock_price

            response = await use_case.execute(request)

            # Assert Price was created with correct value
            mock_price_class.assert_called_once_with(Decimal("999.99"))

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, use_case, mock_unit_of_work, sample_order):
        """Test exception handling during processing."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=sample_order.id, portfolio_id=uuid4(), current_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.side_effect = Exception(
            "Database connection lost"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Database connection lost" in response.error
        mock_unit_of_work.rollback.assert_called_once()


class TestGetRiskMetricsUseCase:
    """Test GetRiskMetricsUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio_with_positions(self):
        """Create sample portfolio with multiple positions."""
        portfolio = Mock()
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("30000")
        portfolio.total_realized_pnl = Decimal("5000")
        portfolio.initial_capital = Money(Decimal("100000"))

        # Mock portfolio methods
        portfolio.get_total_value = Mock(return_value=Decimal("175500"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("175500"))
        portfolio.get_positions_value = Mock(return_value=Decimal("145500"))
        portfolio.get_open_positions = Mock(return_value=[Mock(), Mock()])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("10500"))
        portfolio.get_total_return = Mock(return_value=Decimal("0.755"))

        return portfolio

    @pytest.mark.asyncio
    async def test_get_risk_metrics_comprehensive(
        self, use_case, mock_unit_of_work, sample_portfolio_with_positions
    ):
        """Test getting comprehensive risk metrics."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=sample_portfolio_with_positions.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = (
            sample_portfolio_with_positions
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics is not None

        # Check all expected metrics
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

        # Verify some calculations
        assert response.metrics["portfolio_value"] == 175500.0
        assert response.metrics["positions_value"] == 145500.0
        assert response.metrics["cash_balance"] == 30000.0
        assert response.metrics["leverage"] == pytest.approx(175500.0 / 30000.0, rel=0.01)
        assert response.metrics["position_count"] == 2
        assert response.metrics["concentration"] == pytest.approx(145500.0 / 175500.0, rel=0.01)
        assert response.metrics["unrealized_pnl"] == 10500.0
        assert response.metrics["realized_pnl"] == 5000.0
        assert response.metrics["total_return_pct"] == 75.5

    @pytest.mark.asyncio
    async def test_get_risk_metrics_empty_portfolio(self, use_case, mock_unit_of_work):
        """Test getting risk metrics for empty portfolio."""
        # Setup
        empty_portfolio = Mock()
        empty_portfolio.id = uuid4()
        empty_portfolio.cash_balance = Decimal("100000")
        empty_portfolio.total_realized_pnl = Decimal("0")
        empty_portfolio.get_total_value = Mock(return_value=Decimal("100000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("100000"))
        empty_portfolio.get_positions_value = Mock(return_value=Decimal("0"))
        empty_portfolio.get_open_positions = Mock(return_value=[])
        empty_portfolio.get_unrealized_pnl = Mock(return_value=Decimal("0"))
        empty_portfolio.get_total_return = Mock(return_value=Decimal("0"))

        request = GetRiskMetricsRequest(portfolio_id=empty_portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = empty_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics["portfolio_value"] == 100000.0
        assert response.metrics["positions_value"] == 0.0
        assert response.metrics["cash_balance"] == 100000.0
        assert response.metrics["leverage"] == 1.0  # No leverage
        assert response.metrics["position_count"] == 0
        assert response.metrics["concentration"] == 0.0

    @pytest.mark.asyncio
    async def test_get_risk_metrics_high_leverage(self, use_case, mock_unit_of_work):
        """Test risk metrics with high leverage."""
        # Setup
        leveraged_portfolio = Mock()
        leveraged_portfolio.id = uuid4()
        leveraged_portfolio.cash_balance = Decimal("10000")  # Low cash
        leveraged_portfolio.total_realized_pnl = Decimal("0")
        leveraged_portfolio.get_total_value = Mock(return_value=Decimal("860000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("860000"))
        leveraged_portfolio.get_positions_value = Mock(return_value=Decimal("850000"))
        leveraged_portfolio.get_open_positions = Mock(return_value=[Mock()])
        leveraged_portfolio.get_unrealized_pnl = Mock(return_value=Decimal("50000"))
        leveraged_portfolio.get_total_return = Mock(return_value=Decimal("7.6"))

        request = GetRiskMetricsRequest(portfolio_id=leveraged_portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = leveraged_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        leverage = 860000.0 / 10000.0
        assert response.metrics["leverage"] == leverage
        assert response.metrics["leverage"] == 86.0  # High leverage

    @pytest.mark.asyncio
    async def test_get_risk_metrics_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test getting risk metrics when portfolio doesn't exist."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_get_risk_metrics_zero_cash_balance(self, use_case, mock_unit_of_work):
        """Test risk metrics with zero cash balance."""
        # Setup
        portfolio = Mock()
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("0")
        portfolio.total_realized_pnl = Decimal("1000")
        portfolio.get_total_value = Mock(return_value=Decimal("41000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("41000"))
        portfolio.get_positions_value = Mock(return_value=Decimal("41000"))
        portfolio.get_open_positions = Mock(return_value=[Mock()])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("1000"))
        portfolio.get_total_return = Mock(return_value=Decimal("0.025"))

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics["cash_balance"] == 0.0
        assert response.metrics["leverage"] == 0  # Division by zero handled
        assert response.metrics["concentration"] == 1.0  # 100% in positions

    @pytest.mark.asyncio
    async def test_get_risk_metrics_zero_portfolio_value(self, use_case, mock_unit_of_work):
        """Test risk metrics with zero portfolio value."""
        # Setup
        portfolio = Mock()
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("0")
        portfolio.total_realized_pnl = Decimal("0")
        portfolio.get_total_value = Mock(return_value=Decimal("0"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("0"))
        portfolio.get_positions_value = Mock(return_value=Decimal("0"))
        portfolio.get_open_positions = Mock(return_value=[])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("0"))
        portfolio.get_total_return = Mock(return_value=Decimal("0"))

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics["portfolio_value"] == 0.0
        assert response.metrics["leverage"] == 0
        assert response.metrics["concentration"] == 0  # Division by zero handled

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test validation always passes for get risk metrics."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        # Execute
        error = await use_case.validate(request)

        # Assert
        assert error is None

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, use_case, mock_unit_of_work):
        """Test exception handling during processing."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.side_effect = Exception("Network timeout")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Network timeout" in response.error
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_commit_on_success(self, use_case, mock_unit_of_work):
        """Test that transaction is committed on success."""
        # Setup
        portfolio = Mock()
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("50000")
        portfolio.total_realized_pnl = Decimal("0")
        portfolio.get_total_value = Mock(return_value=Decimal("50000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("50000"))
        portfolio.get_positions_value = Mock(return_value=Decimal("0"))
        portfolio.get_open_positions = Mock(return_value=[])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("0"))
        portfolio.get_total_return = Mock(return_value=Decimal("0"))

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()


# Additional edge case tests for complete coverage
class TestEdgeCasesAndCornerScenarios:
    """Test edge cases and corner scenarios for comprehensive coverage."""

    @pytest.mark.asyncio
    async def test_calculate_risk_with_none_request_id(self):
        """Test that None request_id gets generated."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.portfolios.get_portfolio_by_id.return_value = None
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)

        calculator = Mock()
        use_case = CalculateRiskUseCase(uow, calculator)

        request = CalculateRiskRequest(portfolio_id=uuid4(), request_id=None)

        response = await use_case.execute(request)
        assert response.request_id is not None

    @pytest.mark.asyncio
    async def test_validate_order_risk_with_metadata(self):
        """Test order risk validation with custom metadata."""
        uow = AsyncMock()
        uow.orders = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)

        calculator = Mock()
        use_case = ValidateOrderRiskUseCase(uow, calculator)

        metadata = {"source": "api", "user_id": "123"}
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("100"), metadata=metadata
        )

        assert request.metadata == metadata

    @pytest.mark.asyncio
    async def test_get_risk_metrics_with_correlation_id(self):
        """Test get risk metrics with correlation ID."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.portfolios.get_portfolio_by_id.return_value = None
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)

        calculator = Mock()
        use_case = GetRiskMetricsUseCase(uow, calculator)

        correlation_id = uuid4()
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), correlation_id=correlation_id)

        assert request.correlation_id == correlation_id

        response = await use_case.execute(request)
        assert response.success is False
