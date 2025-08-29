"""
Comprehensive unit tests for risk management use cases.

Tests all risk-related use cases including risk calculation, order validation,
and risk metrics retrieval with complete coverage of success scenarios, failure
scenarios, edge cases, and transaction management.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from src.application.use_cases.risk import (
    CalculateRiskRequest,
    CalculateRiskUseCase,
    GetRiskMetricsRequest,
    GetRiskMetricsUseCase,
    ValidateOrderRiskRequest,
    ValidateOrderRiskUseCase,
)
from src.domain.entities.order import Order, OrderSide, OrderType
from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.value_objects import Quantity
from src.domain.value_objects.money import Money


class TestCalculateRiskUseCase:
    """
    Comprehensive tests for CalculateRiskUseCase.

    Tests risk calculation accuracy, VaR, Sharpe ratio, drawdown calculations,
    portfolio retrieval, error handling, and edge cases.
    """

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create a mock unit of work with transaction support."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create a mock risk calculator with all necessary methods."""
        calculator = Mock()
        calculator.calculate_portfolio_var = Mock()
        calculator.calculate_sharpe_ratio = Mock()
        calculator.calculate_max_drawdown = Mock()
        calculator.calculate_portfolio_beta = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create the CalculateRiskUseCase instance."""
        return CalculateRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio with positions for testing."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("50000")
        portfolio.total_realized_pnl = Decimal("5000")

        # Add diverse positions
        position1 = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        position1.current_price = Decimal("155.00")

        position2 = Position(
            symbol="GOOGL", quantity=Decimal("50"), average_entry_price=Decimal("2500.00")
        )
        position2.current_price = Decimal("2550.00")

        portfolio.positions = {position1.id: position1, position2.id: position2}

        # Mock the methods that will be called
        portfolio.get_total_return = Mock(return_value=Decimal("0.05"))  # 5% return
        portfolio.get_total_value = Mock(return_value=Decimal("195000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("195000"))
        portfolio.get_positions_value = Mock(return_value=Decimal("145000"))
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("7500"))
        portfolio.get_open_positions = Mock(return_value=[position1, position2])

        return portfolio

    @pytest.fixture
    def empty_portfolio(self):
        """Create an empty portfolio with no positions."""
        portfolio = Portfolio(name="Empty Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("100000")
        portfolio.positions = {}

        # Mock the methods that will be called
        portfolio.get_total_return = Mock(return_value=Decimal("0"))
        portfolio.get_total_value = Mock(return_value=Decimal("100000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("100000"))
        portfolio.get_positions_value = Mock(return_value=Decimal("0"))
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("0"))
        portfolio.get_open_positions = Mock(return_value=[])

        return portfolio

    @pytest.mark.asyncio
    async def test_calculate_all_risk_metrics_success(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful calculation of all risk metrics."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_drawdown=True,
            confidence_level=0.95,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Mock calculator returns with Money objects
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("5000.00"))
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("1.25")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.15")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.error is None
        assert response.value_at_risk == Decimal("5000.00")
        assert response.sharpe_ratio == Decimal("1.25")
        assert response.max_drawdown == Decimal("0.15")
        assert response.risk_score == Decimal("0.5")
        assert response.request_id == request.request_id

        # Verify all calculations were made
        mock_risk_calculator.calculate_portfolio_var.assert_called_once_with(
            portfolio=sample_portfolio, confidence_level=Decimal("0.95"), time_horizon=1
        )
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once()
        mock_risk_calculator.calculate_max_drawdown.assert_called_once()

        # Verify transaction was committed
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_var_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating only Value at Risk (VaR)."""
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
        assert response.risk_score == Decimal("0.5")

        # Verify only VaR was calculated
        mock_risk_calculator.calculate_portfolio_var.assert_called_once_with(
            portfolio=sample_portfolio, confidence_level=Decimal("0.99"), time_horizon=1
        )
        mock_risk_calculator.calculate_sharpe_ratio.assert_not_called()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating only Sharpe ratio."""
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
        assert response.value_at_risk is None
        assert response.sharpe_ratio == Decimal("2.0")
        assert response.max_drawdown is None

        # Verify only Sharpe was calculated
        mock_risk_calculator.calculate_portfolio_var.assert_not_called()
        mock_risk_calculator.calculate_sharpe_ratio.assert_called_once()
        mock_risk_calculator.calculate_max_drawdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_calculate_max_drawdown_only(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculating only maximum drawdown."""
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
        assert response.value_at_risk is None
        assert response.sharpe_ratio is None
        assert response.max_drawdown == Decimal("0.25")

    @pytest.mark.asyncio
    async def test_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test error handling when portfolio doesn't exist."""
        # Setup
        non_existent_id = uuid4()
        request = CalculateRiskRequest(portfolio_id=non_existent_id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.request_id == request.request_id

        # Verify transaction was rolled back
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_portfolio_risk_calculation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, empty_portfolio
    ):
        """Test risk calculation for portfolio with no positions."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=empty_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_drawdown=True,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = empty_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("0"))
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("0")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.value_at_risk == Decimal("0")
        assert response.sharpe_ratio == Decimal("0")
        assert response.max_drawdown == Decimal("0")

    @pytest.mark.asyncio
    async def test_extreme_confidence_levels(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test risk calculation with extreme but valid confidence levels."""
        # Test near-zero confidence level
        request_low = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id, include_var=True, confidence_level=0.01
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("100"))

        response_low = await use_case.execute(request_low)
        assert response_low.success is True

        # Test near-one confidence level
        request_high = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id, include_var=True, confidence_level=0.999
        )

        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("10000"))

        response_high = await use_case.execute(request_high)
        assert response_high.success is True

    @pytest.mark.asyncio
    async def test_validation_invalid_confidence_level_zero(self, use_case):
        """Test validation with confidence level of exactly 0."""
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=0.0)

        error = await use_case.validate(request)
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validation_invalid_confidence_level_one(self, use_case):
        """Test validation with confidence level of exactly 1."""
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=1.0)

        error = await use_case.validate(request)
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validation_negative_confidence_level(self, use_case):
        """Test validation with negative confidence level."""
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=-0.5)

        error = await use_case.validate(request)
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validation_valid_confidence_levels(self, use_case):
        """Test validation with various valid confidence levels."""
        valid_levels = [0.01, 0.05, 0.1, 0.5, 0.95, 0.99, 0.999]

        for level in valid_levels:
            request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=level)
            error = await use_case.validate(request)
            assert error is None

    @pytest.mark.asyncio
    async def test_request_metadata_handling(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test that request metadata is properly handled."""
        # Setup
        correlation_id = uuid4()
        metadata = {"source": "API", "user_id": "test123"}

        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id, correlation_id=correlation_id, metadata=metadata
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Money(Decimal("5000"))

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_exception_handling_in_calculation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test exception handling during risk calculation."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=sample_portfolio.id, include_var=True)

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.side_effect = ValueError(
            "Invalid portfolio data"
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Invalid portfolio data" in response.error
        mock_unit_of_work.rollback.assert_called()


class TestValidateOrderRiskUseCase:
    """
    Comprehensive tests for ValidateOrderRiskUseCase.

    Tests validation logic for risk limits, portfolio and order retrieval,
    risk metrics generation, error handling, and edge cases.
    """

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create a mock unit of work with transaction support."""
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
        """Create a mock risk calculator."""
        calculator = Mock()
        calculator.check_risk_limits = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create the ValidateOrderRiskUseCase instance."""
        return ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("100")),
        )
        order.id = uuid4()
        order.portfolio_id = uuid4()
        return order

    @pytest.fixture
    def large_order(self):
        """Create a large order that might violate risk limits."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("10000")),
        )
        order.id = uuid4()
        order.portfolio_id = uuid4()
        return order

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("50000")

        # Add existing position
        position = Position(
            symbol="GOOGL", quantity=Decimal("20"), average_entry_price=Decimal("2500.00")
        )
        position.current_price = Decimal("2600.00")
        portfolio.positions = {position.id: position}

        return portfolio

    @pytest.mark.asyncio
    async def test_validate_order_risk_passes(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_order, sample_portfolio
    ):
        """Test successful order risk validation with no violations."""
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
        assert response.is_valid is True
        assert len(response.risk_violations) == 0
        assert response.risk_metrics is not None

        # Verify risk metrics
        assert "position_size_pct" in response.risk_metrics
        assert "leverage" in response.risk_metrics
        assert "concentration" in response.risk_metrics
        assert "max_loss" in response.risk_metrics

        # Verify calculations
        position_value = 100 * 150  # quantity * price
        portfolio_value = 50000 + 52000  # cash + existing position value

        assert response.risk_metrics["position_size_pct"] == pytest.approx(
            float(position_value / portfolio_value * 100), rel=0.01
        )

        # Verify transaction was committed
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_order_risk_fails_single_violation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, large_order, sample_portfolio
    ):
        """Test order risk validation with a single violation."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=large_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = large_order
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
    async def test_validate_order_risk_multiple_violations(
        self, use_case, mock_unit_of_work, mock_risk_calculator, large_order, sample_portfolio
    ):
        """Test order risk validation with multiple violations."""
        # Setup
        request = ValidateOrderRiskRequest(
            order_id=large_order.id,
            portfolio_id=sample_portfolio.id,
            current_price=Decimal("150.00"),
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = large_order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Mock multiple violations
        mock_risk_calculator.check_risk_limits.return_value = (
            False,
            "Position size exceeds maximum; Concentration limit exceeded; Insufficient margin",
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is False
        assert len(response.risk_violations) == 1  # Current implementation stores as single string
        assert "Position size exceeds maximum" in response.risk_violations[0]

    @pytest.mark.asyncio
    async def test_order_not_found(self, use_case, mock_unit_of_work):
        """Test error handling when order doesn't exist."""
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
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_not_found(self, use_case, mock_unit_of_work, sample_order):
        """Test error handling when portfolio doesn't exist."""
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
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_current_price_validation(self, use_case):
        """Test validation with zero current price."""
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("0")
        )

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_negative_current_price_validation(self, use_case):
        """Test validation with negative current price."""
        request = ValidateOrderRiskRequest(
            order_id=uuid4(), portfolio_id=uuid4(), current_price=Decimal("-150.00")
        )

        error = await use_case.validate(request)
        assert error == "Current price must be positive"

    @pytest.mark.asyncio
    async def test_valid_current_price_validation(self, use_case):
        """Test validation with valid current prices."""
        valid_prices = [Decimal("0.01"), Decimal("1"), Decimal("150.50"), Decimal("10000")]

        for price in valid_prices:
            request = ValidateOrderRiskRequest(
                order_id=uuid4(), portfolio_id=uuid4(), current_price=price
            )
            error = await use_case.validate(request)
            assert error is None

    @pytest.mark.asyncio
    async def test_extreme_leverage_calculation(
        self, use_case, mock_unit_of_work, mock_risk_calculator
    ):
        """Test risk metrics with extreme leverage scenarios."""
        # Setup - Portfolio with very low cash
        portfolio = Portfolio(name="High Leverage Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("1000")  # Very low cash

        # Mock methods - portfolio value is just cash since no positions
        portfolio.get_total_value = Mock(return_value=Decimal("1000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("1000"))

        order = Order(
            symbol="TSLA",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("1000")),
        )
        order.portfolio_id = portfolio.id

        request = ValidateOrderRiskRequest(
            order_id=order.id, portfolio_id=portfolio.id, current_price=Decimal("800.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio
        mock_risk_calculator.check_risk_limits.return_value = (False, "Leverage too high")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is False
        # Leverage is portfolio_value / cash_balance = 1000 / 1000 = 1.0
        assert response.risk_metrics["leverage"] == 1.0

    @pytest.mark.asyncio
    async def test_zero_portfolio_value_edge_case(
        self, use_case, mock_unit_of_work, mock_risk_calculator
    ):
        """Test edge case with zero portfolio value."""
        # Setup - Empty portfolio with no cash
        portfolio = Portfolio(
            name="Empty Portfolio",
            initial_capital=Decimal("1"),  # Must be positive, but we'll set cash to 0
        )
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("0.01")  # Small non-zero to avoid division by zero
        portfolio.positions = {}

        # Mock methods
        portfolio.get_total_value = Mock(return_value=Decimal("0.01"))
        portfolio.get_total_value_sync = Mock(
            return_value=Decimal("0.01")
        )  # Small non-zero to avoid division by zero

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Quantity(Decimal("10")),
        )
        order.portfolio_id = portfolio.id

        request = ValidateOrderRiskRequest(
            order_id=order.id, portfolio_id=portfolio.id, current_price=Decimal("150.00")
        )

        mock_unit_of_work.orders.get_order_by_id.return_value = order
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio
        mock_risk_calculator.check_risk_limits.return_value = (False, "Insufficient funds")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.is_valid is False
        # With very small portfolio value, concentration will be very high
        # position_value = 10 * 150 = 1500, portfolio_value = 0.01
        # concentration = 1500 / 0.01 = 150000
        assert response.risk_metrics["concentration"] > 1000  # Very high concentration
        # leverage = portfolio_value / cash_balance = 0.01 / 0.01 = 1.0
        assert response.risk_metrics["leverage"] == 1.0


class TestGetRiskMetricsUseCase:
    """
    Comprehensive tests for GetRiskMetricsUseCase.

    Tests risk metrics generation, portfolio management, error handling
    for missing data, and edge cases like zero positions and extreme values.
    """

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create a mock unit of work with transaction support."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create a mock risk calculator."""
        return Mock()

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create the GetRiskMetricsUseCase instance."""
        return GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def portfolio_with_positions(self):
        """Create a portfolio with multiple positions."""
        portfolio = Portfolio(name="Diversified Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("30000")
        portfolio.total_realized_pnl = Decimal("5000")

        # Add multiple positions with different characteristics
        positions = []

        # Profitable position
        position1 = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        position1.current_price = Decimal("160.00")
        position1.closed_at = None  # Open position
        positions.append(position1)

        # Losing position
        position2 = Position(
            symbol="GOOGL", quantity=Decimal("50"), average_entry_price=Decimal("2600.00")
        )
        position2.current_price = Decimal("2500.00")
        position2.closed_at = None  # Open position
        positions.append(position2)

        # Closed position (should not be counted)
        position3 = Position(
            symbol="MSFT", quantity=Decimal("75"), average_entry_price=Decimal("300.00")
        )
        position3.current_price = Decimal("310.00")
        position3.closed_at = datetime.now(UTC)  # Closed position
        positions.append(position3)

        portfolio.positions = {p.id: p for p in positions}

        # Mock methods
        portfolio.get_total_value = Mock(return_value=Decimal("171000"))
        portfolio.get_total_value_sync = Mock(
            return_value=Decimal("171000")
        )  # 30k cash + positions
        portfolio.get_positions_value = Mock(
            return_value=Decimal("141000")
        )  # Sum of open positions
        portfolio.get_open_positions = Mock(return_value=[position1, position2])  # Only open ones
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("6000"))  # Unrealized P&L
        portfolio.get_total_return = Mock(return_value=Decimal("0.05"))  # 5% return

        return portfolio

    @pytest.fixture
    def empty_portfolio(self):
        """Create an empty portfolio with no positions."""
        portfolio = Portfolio(name="Empty Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("100000")
        portfolio.total_realized_pnl = Decimal("0")
        portfolio.positions = {}

        # Mock methods
        portfolio.get_total_value = Mock(return_value=Decimal("100000"))
        portfolio.get_total_value_sync = Mock(return_value=Decimal("100000"))
        portfolio.get_positions_value = Mock(return_value=Decimal("0"))
        portfolio.get_open_positions = Mock(return_value=[])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("0"))
        portfolio.get_total_return = Mock(return_value=Decimal("0"))

        return portfolio

    @pytest.fixture
    def leveraged_portfolio(self):
        """Create a highly leveraged portfolio."""
        portfolio = Portfolio(name="Leveraged Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("5000")  # Very low cash

        # Large position relative to cash
        position = Position(
            symbol="TSLA", quantity=Decimal("1000"), average_entry_price=Decimal("800.00")
        )
        position.current_price = Decimal("850.00")
        position.closed_at = None  # Open position

        portfolio.positions = {position.id: position}

        # Mock methods
        portfolio.get_total_value = Mock(return_value=Decimal("855000"))
        portfolio.get_total_value_sync = Mock(
            return_value=Decimal("855000")
        )  # 5k cash + 850k position
        portfolio.get_positions_value = Mock(return_value=Decimal("850000"))
        portfolio.get_open_positions = Mock(return_value=[position])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("50000"))  # Profit
        portfolio.get_total_return = Mock(return_value=Decimal("0.75"))  # 75% return

        return portfolio

    @pytest.mark.asyncio
    async def test_get_risk_metrics_comprehensive(
        self, use_case, mock_unit_of_work, portfolio_with_positions
    ):
        """Test getting comprehensive risk metrics for a diversified portfolio."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=portfolio_with_positions.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio_with_positions

        # Execute
        response = await use_case.execute(request)

        # Assert basic response
        assert response.success is True
        assert response.error is None
        assert response.metrics is not None

        # Verify all expected metrics are present
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
            assert metric in response.metrics, f"Missing metric: {metric}"

        # Verify specific calculations
        assert response.metrics["cash_balance"] == 30000.0
        assert response.metrics["realized_pnl"] == 5000.0
        assert response.metrics["position_count"] == 2  # Only open positions

        # Calculate expected values
        position_values = (100 * 160) + (50 * 2500)  # AAPL + GOOGL
        portfolio_value = 30000 + position_values

        assert response.metrics["portfolio_value"] == float(portfolio_value)
        assert response.metrics["positions_value"] == float(position_values)

        # Verify transaction was committed
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_risk_metrics_empty_portfolio(
        self, use_case, mock_unit_of_work, empty_portfolio
    ):
        """Test getting risk metrics for an empty portfolio."""
        # Setup
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
        assert response.metrics["unrealized_pnl"] == 0.0
        assert response.metrics["realized_pnl"] == 0.0
        assert response.metrics["total_return_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_get_risk_metrics_high_leverage(
        self, use_case, mock_unit_of_work, leveraged_portfolio
    ):
        """Test risk metrics calculation for highly leveraged portfolio."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=leveraged_portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = leveraged_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True

        # Calculate expected leverage
        position_value = 1000 * 850
        portfolio_value = 5000 + position_value
        expected_leverage = portfolio_value / 5000

        assert response.metrics["leverage"] == float(expected_leverage)
        assert response.metrics["leverage"] > 100  # Very high leverage
        assert response.metrics["concentration"] == pytest.approx(
            float(position_value / portfolio_value), rel=0.01
        )

    @pytest.mark.asyncio
    async def test_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test error handling when portfolio doesn't exist."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"
        assert response.metrics == {}
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_cash_balance_edge_case(self, use_case, mock_unit_of_work):
        """Test edge case with zero cash balance."""
        # Setup portfolio with zero cash
        portfolio = Portfolio(name="No Cash Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("0")

        position = Position(
            symbol="SPY", quantity=Decimal("100"), average_entry_price=Decimal("400.00")
        )
        position.current_price = Decimal("410.00")
        position.get_position_value = Mock(return_value=Decimal("41000"))
        position.get_unrealized_pnl = Mock(return_value=Decimal("1000"))
        position.is_closed = Mock(return_value=False)
        portfolio.positions = {position.id: position}

        # Mock portfolio methods
        portfolio.get_total_value = Mock(return_value=Decimal("41000"))
        portfolio.get_total_value_sync = Mock(
            return_value=Decimal("41000")
        )  # Just the position value, no cash
        portfolio.get_positions_value = Mock(return_value=Decimal("41000"))
        portfolio.get_open_positions = Mock(return_value=[position])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("1000"))
        portfolio.get_total_return = Mock(return_value=Decimal("-0.59"))  # Lost money overall

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
    async def test_negative_pnl_handling(self, use_case, mock_unit_of_work):
        """Test handling of negative P&L values."""
        # Setup portfolio with losses
        portfolio = Portfolio(name="Losing Portfolio", initial_capital=Decimal("100000"))
        portfolio.id = uuid4()
        portfolio.cash_balance = Decimal("80000")
        portfolio.total_realized_pnl = Decimal("-10000")  # Loss

        position = Position(
            symbol="XYZ", quantity=Decimal("100"), average_entry_price=Decimal("200.00")
        )
        position.current_price = Decimal("150.00")  # 25% loss
        position.closed_at = None  # Open position
        position.get_position_value = Mock(return_value=Decimal("15000"))
        position.get_unrealized_pnl = Mock(return_value=Decimal("-5000"))
        position.is_closed = Mock(return_value=False)
        portfolio.positions = {position.id: position}

        # Mock portfolio methods
        portfolio.get_total_value = Mock(return_value=Decimal("95000"))
        portfolio.get_total_value_sync = Mock(
            return_value=Decimal("95000")
        )  # 80k cash + 15k position
        portfolio.get_positions_value = Mock(return_value=Decimal("15000"))
        portfolio.get_open_positions = Mock(return_value=[position])
        portfolio.get_unrealized_pnl = Mock(return_value=Decimal("-5000"))
        portfolio.get_total_return = Mock(return_value=Decimal("-0.05"))  # 5% loss

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.metrics["realized_pnl"] == -10000.0
        assert response.metrics["unrealized_pnl"] < 0  # Negative unrealized P&L
        assert response.metrics["total_return_pct"] < 0  # Negative return

    @pytest.mark.asyncio
    async def test_validation_always_passes(self, use_case):
        """Test that validation always passes for GetRiskMetricsUseCase."""
        requests = [
            GetRiskMetricsRequest(portfolio_id=uuid4()),
            GetRiskMetricsRequest(portfolio_id=uuid4(), metadata={"source": "monitoring"}),
            GetRiskMetricsRequest(portfolio_id=uuid4(), correlation_id=uuid4()),
        ]

        for request in requests:
            error = await use_case.validate(request)
            assert error is None

    @pytest.mark.asyncio
    async def test_request_id_propagation(self, use_case, mock_unit_of_work, empty_portfolio):
        """Test that request IDs are properly propagated through the response."""
        # Setup
        request_id = uuid4()
        request = GetRiskMetricsRequest(portfolio_id=empty_portfolio.id, request_id=request_id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = empty_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.request_id == request_id

    @pytest.mark.asyncio
    async def test_exception_handling_during_metric_calculation(self, use_case, mock_unit_of_work):
        """Test exception handling during metric calculation."""
        # Setup portfolio that will cause an error
        portfolio = Mock(spec=Portfolio)
        portfolio.id = uuid4()
        portfolio.get_total_value.side_effect = RuntimeError("Calculation error")

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Calculation error" in response.error
        mock_unit_of_work.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_metric_requests(
        self, use_case, mock_unit_of_work, portfolio_with_positions
    ):
        """Test handling of concurrent metric requests for the same portfolio."""
        import asyncio

        # Setup
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio_with_positions

        # Create multiple concurrent requests
        requests = [
            GetRiskMetricsRequest(portfolio_id=portfolio_with_positions.id) for _ in range(5)
        ]

        # Execute concurrently
        responses = await asyncio.gather(*[use_case.execute(req) for req in requests])

        # Assert all succeeded
        for response in responses:
            assert response.success is True
            assert response.metrics is not None
