"""
Comprehensive tests for risk management use cases.

Tests all risk-related use cases including risk calculation, limit checking,
and portfolio risk analysis with full coverage.
"""

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
from src.domain.value_objects.money import Money


class TestRequestPostInit:
    """Test __post_init__ methods of request classes."""

    def test_calculate_portfolio_risk_request_post_init(self):
        """Test CalculateRiskRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = CalculateRiskRequest(portfolio_id=uuid4())
        assert request.request_id is not None
        assert request.metadata == {}

        # Test with provided values
        req_id = uuid4()
        metadata = {"calculation": "var"}
        request = CalculateRiskRequest(portfolio_id=uuid4(), request_id=req_id, metadata=metadata)
        assert request.request_id == req_id
        assert request.metadata == metadata

    def test_check_risk_limits_request_post_init(self):
        """Test ValidateOrderRiskRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = ValidateOrderRiskRequest(portfolio_id=uuid4(), order_id=uuid4())
        assert request.request_id is not None
        assert request.metadata == {}

        # Test with provided values
        req_id = uuid4()
        metadata = {"check_type": "pre_trade"}
        request = ValidateOrderRiskRequest(
            portfolio_id=uuid4(), order_id=uuid4(), request_id=req_id, metadata=metadata
        )
        assert request.request_id == req_id
        assert request.metadata == metadata

    def test_update_risk_parameters_request_post_init(self):
        """Test GetRiskMetricsRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        assert request.request_id is not None
        assert request.metadata == {}

    def test_get_risk_metrics_request_post_init(self):
        """Test GetRiskMetricsRequest __post_init__ method."""
        # Test with no request_id and metadata
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        assert request.request_id is not None
        assert request.metadata == {}


class TestCalculateRiskUseCase:
    """Test CalculateRiskUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.market_data = AsyncMock()
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
        calculator.calculate_position_risk = Mock()
        calculator.calculate_sharpe_ratio = Mock()
        calculator.calculate_max_drawdown = Mock()
        calculator.calculate_beta = Mock()
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

        # Add positions
        position1 = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        position1.current_price = Decimal("155.00")

        position2 = Position(
            symbol="GOOGL", quantity=Decimal("50"), average_entry_price=Decimal("2500.00")
        )
        position2.current_price = Decimal("2550.00")

        portfolio.positions = {position1.id: position1, position2.id: position2}

        return portfolio

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_success(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful portfolio risk calculation."""
        # Setup
        request = CalculateRiskRequest(
            portfolio_id=sample_portfolio.id,
            include_var=True,
            include_sharpe=True,
            include_beta=True,
            confidence_level=Decimal("0.95"),
            lookback_days=30,
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("5000.00")
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("1.5")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.15")
        mock_risk_calculator.calculate_beta.return_value = Decimal("1.2")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.var_95 == Decimal("5000.00")
        assert response.sharpe_ratio == Decimal("1.5")
        assert response.max_drawdown == Decimal("0.15")
        assert response.beta == Decimal("1.2")
        assert response.position_risks is not None
        assert len(response.position_risks) == 2

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_no_positions(
        self, use_case, mock_unit_of_work, mock_risk_calculator
    ):
        """Test risk calculation for portfolio with no positions."""
        # Setup
        portfolio = Portfolio(name="Empty Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        portfolio.positions = {}

        request = CalculateRiskRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("0")
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("0")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.total_exposure == Decimal("0")
        assert len(response.position_risks) == 0

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_not_found(self, use_case, mock_unit_of_work):
        """Test risk calculation when portfolio doesn't exist."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_with_market_data(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test risk calculation with market data for price updates."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=sample_portfolio.id, use_current_prices=True)

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Mock market data
        mock_bar1 = Mock()
        mock_bar1.close = Decimal("160.00")
        mock_bar2 = Mock()
        mock_bar2.close = Decimal("2600.00")

        mock_unit_of_work.market_data.get_latest_bar.side_effect = [mock_bar1, mock_bar2]

        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("6000.00")
        mock_risk_calculator.calculate_position_risk.return_value = Decimal("1000.00")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        # Verify market data was fetched for positions
        assert mock_unit_of_work.market_data.get_latest_bar.call_count == 2

    @pytest.mark.asyncio
    async def test_calculate_position_specific_risks(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test calculation of position-specific risk metrics."""
        # Setup
        request = CalculateRiskRequest(portfolio_id=sample_portfolio.id)

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Mock position risk calculations
        mock_risk_calculator.calculate_position_risk.side_effect = [
            Decimal("500.00"),  # AAPL risk
            Decimal("2500.00"),  # GOOGL risk
        ]

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert len(response.position_risks) == 2

        # Check position risk structure
        aapl_risk = next(r for r in response.position_risks if r["symbol"] == "AAPL")
        assert aapl_risk["risk_amount"] == Decimal("500.00")
        assert "weight" in aapl_risk
        assert "value" in aapl_risk

    @pytest.mark.asyncio
    async def test_validate_invalid_confidence_level(self, use_case):
        """Test validation with invalid confidence level."""
        # Test confidence level > 1
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=Decimal("1.5"))
        error = await use_case.validate(request)
        assert error == "Confidence level must be between 0 and 1"

        # Test negative confidence level
        request = CalculateRiskRequest(portfolio_id=uuid4(), confidence_level=Decimal("-0.5"))
        error = await use_case.validate(request)
        assert error == "Confidence level must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_lookback_days(self, use_case):
        """Test validation with invalid lookback days."""
        request = CalculateRiskRequest(portfolio_id=uuid4(), lookback_days=0)
        error = await use_case.validate(request)
        assert error == "Lookback days must be positive"

        request = CalculateRiskRequest(portfolio_id=uuid4(), lookback_days=-10)
        error = await use_case.validate(request)
        assert error == "Lookback days must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = CalculateRiskRequest(
            portfolio_id=uuid4(), confidence_level=Decimal("0.95"), lookback_days=30
        )
        error = await use_case.validate(request)
        assert error is None


class TestValidateOrderRiskUseCase:
    """Test ValidateOrderRiskUseCase with comprehensive scenarios."""

    @pytest.fixture
    def mock_unit_of_work(self):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.orders = AsyncMock()
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_risk_calculator(self):
        """Create mock risk calculator."""
        calculator = Mock()
        calculator.check_position_limit = Mock()
        calculator.check_concentration_limit = Mock()
        calculator.check_leverage_limit = Mock()
        calculator.check_daily_loss_limit = Mock()
        calculator.calculate_order_impact = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return ValidateOrderRiskUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        portfolio.cash_balance = Money(Decimal("50000"))
        portfolio.max_position_size = Money(Decimal("20000"))
        portfolio.max_leverage = Decimal("2.0")
        portfolio.max_portfolio_risk = Decimal("0.1")
        return portfolio

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("100")
        )
        order.portfolio_id = uuid4()
        return order

    @pytest.mark.asyncio
    async def test_check_risk_limits_all_pass(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio, sample_order
    ):
        """Test risk limit check when all limits pass."""
        # Setup
        request = ValidateOrderRiskRequest(
            portfolio_id=sample_portfolio.id, order_id=sample_order.id
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        # All checks pass
        mock_risk_calculator.check_position_limit.return_value = (True, None)
        mock_risk_calculator.check_concentration_limit.return_value = (True, None)
        mock_risk_calculator.check_leverage_limit.return_value = (True, None)
        mock_risk_calculator.check_daily_loss_limit.return_value = (True, None)
        mock_risk_calculator.calculate_order_impact.return_value = {
            "estimated_cost": Decimal("15000.00"),
            "new_exposure": Decimal("65000.00"),
            "new_leverage": Decimal("0.65"),
        }

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.limits_passed is True
        assert len(response.violations) == 0
        assert response.order_impact is not None

    @pytest.mark.asyncio
    async def test_check_risk_limits_position_limit_violation(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio, sample_order
    ):
        """Test risk limit check with position limit violation."""
        # Setup
        request = ValidateOrderRiskRequest(
            portfolio_id=sample_portfolio.id, order_id=sample_order.id
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        # Position limit fails
        mock_risk_calculator.check_position_limit.return_value = (
            False,
            "Position size exceeds maximum",
        )
        mock_risk_calculator.check_concentration_limit.return_value = (True, None)
        mock_risk_calculator.check_leverage_limit.return_value = (True, None)
        mock_risk_calculator.check_daily_loss_limit.return_value = (True, None)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.limits_passed is False
        assert len(response.violations) == 1
        assert response.violations[0]["limit"] == "position_size"
        assert "Position size exceeds maximum" in response.violations[0]["message"]

    @pytest.mark.asyncio
    async def test_check_risk_limits_multiple_violations(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio, sample_order
    ):
        """Test risk limit check with multiple violations."""
        # Setup
        request = ValidateOrderRiskRequest(
            portfolio_id=sample_portfolio.id, order_id=sample_order.id
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.orders.get_order_by_id.return_value = sample_order

        # Multiple limits fail
        mock_risk_calculator.check_position_limit.return_value = (False, "Position too large")
        mock_risk_calculator.check_concentration_limit.return_value = (
            False,
            "Concentration too high",
        )
        mock_risk_calculator.check_leverage_limit.return_value = (True, None)
        mock_risk_calculator.check_daily_loss_limit.return_value = (
            False,
            "Daily loss limit exceeded",
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.limits_passed is False
        assert len(response.violations) == 3

    @pytest.mark.asyncio
    async def test_check_risk_limits_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test risk limit check when portfolio doesn't exist."""
        # Setup
        request = ValidateOrderRiskRequest(portfolio_id=uuid4(), order_id=uuid4())

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_check_risk_limits_order_not_found(
        self, use_case, mock_unit_of_work, sample_portfolio
    ):
        """Test risk limit check when order doesn't exist."""
        # Setup
        request = ValidateOrderRiskRequest(portfolio_id=sample_portfolio.id, order_id=uuid4())

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio
        mock_unit_of_work.orders.get_order_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Order not found"

    @pytest.mark.asyncio
    async def test_check_risk_limits_proposed_trade(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test risk limit check for proposed trade without order."""
        # Setup
        request = ValidateOrderRiskRequest(
            portfolio_id=sample_portfolio.id,
            proposed_trade={"symbol": "AAPL", "side": "buy", "quantity": 100, "price": 150.00},
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # All checks pass
        mock_risk_calculator.check_position_limit.return_value = (True, None)
        mock_risk_calculator.check_concentration_limit.return_value = (True, None)
        mock_risk_calculator.check_leverage_limit.return_value = (True, None)
        mock_risk_calculator.check_daily_loss_limit.return_value = (True, None)

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.limits_passed is True

    @pytest.mark.asyncio
    async def test_validate_always_passes(self, use_case):
        """Test validation always passes for check risk limits."""
        request = ValidateOrderRiskRequest(portfolio_id=uuid4(), order_id=uuid4())
        error = await use_case.validate(request)
        assert error is None


class TestGetRiskMetricsUseCaseAdditional:
    """Test GetRiskMetricsUseCase additional comprehensive scenarios."""

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
    def use_case(self, mock_unit_of_work):
        """Create use case instance."""
        return GetRiskMetricsUseCase(unit_of_work=mock_unit_of_work)

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        portfolio.max_position_size = Money(Decimal("10000"))
        portfolio.max_leverage = Decimal("1.5")
        portfolio.max_portfolio_risk = Decimal("0.05")
        return portfolio

    @pytest.mark.asyncio
    async def test_update_risk_parameters_success(
        self, use_case, mock_unit_of_work, sample_portfolio
    ):
        """Test successful risk parameters update."""
        # Setup
        request = GetRiskMetricsRequest(
            portfolio_id=sample_portfolio.id,
            max_position_size=Decimal("20000"),
            max_concentration=Decimal("0.25"),
            max_leverage=Decimal("2.0"),
            max_daily_loss=Decimal("5000"),
            var_limit=Decimal("10000"),
            stop_loss_percentage=Decimal("0.05"),
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.updated is True
        assert sample_portfolio.max_position_size == Money(Decimal("20000"))
        assert sample_portfolio.max_leverage == Decimal("2.0")
        mock_unit_of_work.portfolios.update_portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_risk_parameters_partial(
        self, use_case, mock_unit_of_work, sample_portfolio
    ):
        """Test partial risk parameters update."""
        # Setup
        original_leverage = sample_portfolio.max_leverage
        request = GetRiskMetricsRequest(
            portfolio_id=sample_portfolio.id, max_position_size=Decimal("15000")
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert sample_portfolio.max_position_size == Money(Decimal("15000"))
        assert sample_portfolio.max_leverage == original_leverage  # Unchanged

    @pytest.mark.asyncio
    async def test_update_risk_parameters_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test update when portfolio doesn't exist."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_position_size=Decimal("20000"))

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_validate_negative_max_position_size(self, use_case):
        """Test validation with negative max position size."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_position_size=Decimal("-10000"))
        error = await use_case.validate(request)
        assert error == "Max position size must be positive"

    @pytest.mark.asyncio
    async def test_validate_invalid_concentration(self, use_case):
        """Test validation with invalid concentration."""
        # Greater than 1
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_concentration=Decimal("1.5"))
        error = await use_case.validate(request)
        assert error == "Max concentration must be between 0 and 1"

        # Negative
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_concentration=Decimal("-0.1"))
        error = await use_case.validate(request)
        assert error == "Max concentration must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_invalid_leverage(self, use_case):
        """Test validation with invalid leverage."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_leverage=Decimal("0.5"))
        error = await use_case.validate(request)
        assert error == "Max leverage must be at least 1.0"

    @pytest.mark.asyncio
    async def test_validate_negative_daily_loss(self, use_case):
        """Test validation with negative daily loss."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), max_daily_loss=Decimal("-5000"))
        error = await use_case.validate(request)
        assert error == "Max daily loss must be positive"

    @pytest.mark.asyncio
    async def test_validate_negative_var_limit(self, use_case):
        """Test validation with negative VaR limit."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), var_limit=Decimal("-10000"))
        error = await use_case.validate(request)
        assert error == "VaR limit must be positive"

    @pytest.mark.asyncio
    async def test_validate_invalid_stop_loss(self, use_case):
        """Test validation with invalid stop loss percentage."""
        # Greater than 1
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), stop_loss_percentage=Decimal("1.5"))
        error = await use_case.validate(request)
        assert error == "Stop loss percentage must be between 0 and 1"

        # Negative
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), stop_loss_percentage=Decimal("-0.05"))
        error = await use_case.validate(request)
        assert error == "Stop loss percentage must be between 0 and 1"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = GetRiskMetricsRequest(
            portfolio_id=uuid4(),
            max_position_size=Decimal("20000"),
            max_concentration=Decimal("0.25"),
            max_leverage=Decimal("2.0"),
            max_daily_loss=Decimal("5000"),
            var_limit=Decimal("10000"),
            stop_loss_percentage=Decimal("0.05"),
        )
        error = await use_case.validate(request)
        assert error is None


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
        calculator.calculate_portfolio_var = Mock()
        calculator.calculate_sharpe_ratio = Mock()
        calculator.calculate_max_drawdown = Mock()
        calculator.calculate_beta = Mock()
        calculator.calculate_correlation_matrix = Mock()
        calculator.get_risk_contributions = Mock()
        return calculator

    @pytest.fixture
    def use_case(self, mock_unit_of_work, mock_risk_calculator):
        """Create use case instance."""
        return GetRiskMetricsUseCase(
            unit_of_work=mock_unit_of_work, risk_calculator=mock_risk_calculator
        )

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio with positions."""
        portfolio = Portfolio(name="Test Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        position1 = Position(
            symbol="AAPL", quantity=Decimal("100"), average_entry_price=Decimal("150.00")
        )
        position1.current_price = Decimal("155.00")

        position2 = Position(
            symbol="GOOGL", quantity=Decimal("50"), average_entry_price=Decimal("2500.00")
        )
        position2.current_price = Decimal("2550.00")

        portfolio.positions = {position1.id: position1, position2.id: position2}

        return portfolio

    @pytest.mark.asyncio
    async def test_get_risk_metrics_success(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test successful retrieval of risk metrics."""
        # Setup
        request = GetRiskMetricsRequest(
            portfolio_id=sample_portfolio.id, include_history=True, history_days=30
        )

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("5000.00")
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("1.5")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.15")
        mock_risk_calculator.calculate_beta.return_value = Decimal("1.2")
        mock_risk_calculator.calculate_correlation_matrix.return_value = {
            "AAPL-GOOGL": Decimal("0.65")
        }
        mock_risk_calculator.get_risk_contributions.return_value = {
            "AAPL": Decimal("0.4"),
            "GOOGL": Decimal("0.6"),
        }

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.current_metrics is not None
        assert response.current_metrics["var_95"] == Decimal("5000.00")
        assert response.current_metrics["sharpe_ratio"] == Decimal("1.5")
        assert response.current_metrics["max_drawdown"] == Decimal("0.15")
        assert response.current_metrics["beta"] == Decimal("1.2")
        assert response.risk_contributions is not None
        assert response.correlations is not None

    @pytest.mark.asyncio
    async def test_get_risk_metrics_without_history(
        self, use_case, mock_unit_of_work, mock_risk_calculator, sample_portfolio
    ):
        """Test getting risk metrics without historical data."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=sample_portfolio.id, include_history=False)

        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = sample_portfolio

        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("5000.00")
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("1.5")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0.15")
        mock_risk_calculator.calculate_beta.return_value = Decimal("1.2")

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.current_metrics is not None
        assert response.historical_metrics is None

    @pytest.mark.asyncio
    async def test_get_risk_metrics_portfolio_not_found(self, use_case, mock_unit_of_work):
        """Test getting metrics when portfolio doesn't exist."""
        # Setup
        request = GetRiskMetricsRequest(portfolio_id=uuid4())
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = None

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.error == "Portfolio not found"

    @pytest.mark.asyncio
    async def test_get_risk_metrics_empty_portfolio(
        self, use_case, mock_unit_of_work, mock_risk_calculator
    ):
        """Test getting metrics for empty portfolio."""
        # Setup
        portfolio = Portfolio(name="Empty Portfolio", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()
        portfolio.positions = {}

        request = GetRiskMetricsRequest(portfolio_id=portfolio.id)
        mock_unit_of_work.portfolios.get_portfolio_by_id.return_value = portfolio

        mock_risk_calculator.calculate_portfolio_var.return_value = Decimal("0")
        mock_risk_calculator.calculate_sharpe_ratio.return_value = Decimal("0")
        mock_risk_calculator.calculate_max_drawdown.return_value = Decimal("0")
        mock_risk_calculator.calculate_beta.return_value = Decimal("0")
        mock_risk_calculator.get_risk_contributions.return_value = {}

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.current_metrics["var_95"] == Decimal("0")
        assert len(response.risk_contributions) == 0

    @pytest.mark.asyncio
    async def test_validate_negative_history_days(self, use_case):
        """Test validation with negative history days."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), history_days=-30)
        error = await use_case.validate(request)
        assert error == "History days must be positive"

    @pytest.mark.asyncio
    async def test_validate_zero_history_days(self, use_case):
        """Test validation with zero history days."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), history_days=0)
        error = await use_case.validate(request)
        assert error == "History days must be positive"

    @pytest.mark.asyncio
    async def test_validate_valid_request(self, use_case):
        """Test validation with valid request."""
        request = GetRiskMetricsRequest(portfolio_id=uuid4(), history_days=30)
        error = await use_case.validate(request)
        assert error is None


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.asyncio
    async def test_calculate_risk_with_exception(self):
        """Test risk calculation when exception occurs."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.portfolios.get_portfolio_by_id.side_effect = Exception("Database error")

        calculator = Mock()
        use_case = CalculateRiskUseCase(unit_of_work=uow, risk_calculator=calculator)

        request = CalculateRiskRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Database error" in response.error

    @pytest.mark.asyncio
    async def test_check_limits_with_exception(self):
        """Test risk limit check when exception occurs."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.orders = AsyncMock()

        portfolio = Portfolio(name="Test", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        uow.portfolios.get_portfolio_by_id.return_value = portfolio
        uow.orders.get_order_by_id.side_effect = Exception("Order service error")

        calculator = Mock()
        use_case = ValidateOrderRiskUseCase(unit_of_work=uow, risk_calculator=calculator)

        request = ValidateOrderRiskRequest(portfolio_id=portfolio.id, order_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Order service error" in response.error

    @pytest.mark.asyncio
    async def test_update_parameters_with_exception(self):
        """Test parameter update when exception occurs."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.portfolios.update_portfolio.side_effect = Exception("Update failed")

        portfolio = Portfolio(name="Test", initial_capital=Money(Decimal("100000")))
        portfolio.id = uuid4()

        uow.portfolios.get_portfolio_by_id.return_value = portfolio

        use_case = GetRiskMetricsUseCase(unit_of_work=uow)

        request = GetRiskMetricsRequest(
            portfolio_id=portfolio.id, max_position_size=Decimal("20000")
        )

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Update failed" in response.error

    @pytest.mark.asyncio
    async def test_get_metrics_with_exception(self):
        """Test get metrics when exception occurs."""
        # Setup
        uow = AsyncMock()
        uow.portfolios = AsyncMock()
        uow.portfolios.get_portfolio_by_id.side_effect = Exception("Fetch error")

        calculator = Mock()
        use_case = GetRiskMetricsUseCase(unit_of_work=uow, risk_calculator=calculator)

        request = GetRiskMetricsRequest(portfolio_id=uuid4())

        # Execute
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert "Fetch error" in response.error
