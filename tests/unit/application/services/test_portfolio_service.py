"""Comprehensive tests for PortfolioService."""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from src.application.services.portfolio_service import PortfolioService
from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.entities.position import Position
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioService:
    """Test suite for PortfolioService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = PortfolioService()

        # Create mock portfolio
        self.portfolio = Mock(spec=Portfolio)
        self.portfolio.cash_balance = Money(Decimal("10000"))
        self.portfolio.initial_capital = Money(Decimal("100000"))
        self.portfolio.total_realized_pnl = Money(Decimal("0"))
        self.portfolio.total_commission_paid = Money(Decimal("0"))
        self.portfolio.trades_count = 0
        self.portfolio.winning_trades = 0
        self.portfolio.losing_trades = 0
        self.portfolio.max_positions = 10
        self.portfolio.positions = {}
        self.portfolio.increment_version = Mock()
        self.portfolio.get_position = Mock(return_value=None)
        self.portfolio.get_open_positions = Mock(return_value=[])
        self.portfolio.get_closed_positions = Mock(return_value=[])
        self.portfolio.get_position_count = Mock(return_value=0)
        self.portfolio.update_all_prices = Mock()

    # --- Initialization Tests ---

    def test_initialization(self):
        """Test service initialization."""
        service = PortfolioService()
        assert service.calculator is not None
        assert service.validator is not None

    # --- Position Management Tests ---

    @patch("src.application.services.portfolio_service.Position")
    def test_open_position_success(self, mock_position_class):
        """Test successfully opening a position."""
        # Setup
        mock_position = Mock(spec=Position)
        mock_position_class.return_value = mock_position

        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("10"))
        request.strategy = "momentum"

        # Execute
        with patch.object(self.service.validator, "validate_position_request"):
            position = self.service.open_position(self.portfolio, request)

        # Verify position creation
        mock_position_class.assert_called_once_with(
            symbol="AAPL",
            quantity=Quantity(Decimal("10")),
            average_entry_price=Price(Decimal("150")),
            current_price=Price(Decimal("150")),
            strategy="momentum",
        )

        # Verify portfolio state updates
        assert self.portfolio.positions["AAPL"] == mock_position
        assert self.portfolio.cash_balance.amount == Decimal("8490")  # 10000 - (10*150 + 10)
        assert self.portfolio.total_commission_paid.amount == Decimal("10")

        # Verify version increment
        self.portfolio.increment_version.assert_called_once()

        # Verify return value
        assert position == mock_position

    def test_open_position_validation_failure(self):
        """Test opening position with validation failure."""
        request = Mock(spec=PositionRequest)
        request.symbol = ""
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("10"))

        with patch.object(
            self.service.validator,
            "validate_position_request",
            side_effect=ValueError("Invalid request"),
        ):
            with pytest.raises(ValueError, match="Invalid request"):
                self.service.open_position(self.portfolio, request)

        # Verify no state changes
        assert len(self.portfolio.positions) == 0
        assert self.portfolio.cash_balance.amount == Decimal("10000")
        self.portfolio.increment_version.assert_not_called()

    def test_close_position_full_success(self):
        """Test successfully closing a full position."""
        # Setup position
        position = Mock(spec=Position)
        position.quantity = Quantity(Decimal("10"))
        position.average_entry_price = Price(Decimal("150"))
        position.realized_pnl = Money(Decimal("100"))
        position.close_position = Mock()
        position.mark_as_closed = Mock()

        self.portfolio.get_position.return_value = position

        # Execute
        with patch.object(self.service.validator, "can_close_position", return_value=(True, None)):
            pnl = self.service.close_position(
                self.portfolio, "AAPL", Price(Decimal("160")), Money(Decimal("10"))
            )

        # Verify close was called
        position.close_position.assert_called_once_with(Price(Decimal("160")), Money(Decimal("10")))
        position.mark_as_closed.assert_called_once_with(Price(Decimal("160")))

        # Verify P&L
        assert pnl.amount == Decimal("100")

        # Verify portfolio state updates
        expected_proceeds = Money(Decimal("1590"))  # 10*160 - 10
        assert self.portfolio.cash_balance.amount == Decimal("11590")
        assert self.portfolio.total_realized_pnl.amount == Decimal("100")
        assert self.portfolio.total_commission_paid.amount == Decimal("10")
        assert self.portfolio.trades_count == 1
        assert self.portfolio.winning_trades == 1

        # Verify version increment
        self.portfolio.increment_version.assert_called_once()

    def test_close_position_partial_success(self):
        """Test successfully closing a partial position."""
        # Setup position
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.quantity = Quantity(Decimal("10"))
        position.average_entry_price = Price(Decimal("150"))
        position.current_price = Price(Decimal("155"))
        position.entry_time = Mock()
        position.strategy = "momentum"

        self.portfolio.get_position.return_value = position

        # Execute partial close (5 shares)
        with patch.object(self.service.validator, "can_close_position", return_value=(True, None)):
            with patch.object(
                self.service, "_close_partial_position", return_value=(Money(Decimal("50")), Mock())
            ) as mock_partial:
                pnl = self.service.close_position(
                    self.portfolio,
                    "AAPL",
                    Price(Decimal("160")),
                    Money(Decimal("5")),
                    Quantity(Decimal("5")),
                )

        # Verify partial close was called
        mock_partial.assert_called_once()

        # Verify P&L
        assert pnl.amount == Decimal("50")

    def test_close_position_validation_failure(self):
        """Test closing position with validation failure."""
        with patch.object(
            self.service.validator, "can_close_position", return_value=(False, "Position not found")
        ):
            with pytest.raises(ValueError, match="Cannot close position: Position not found"):
                self.service.close_position(
                    self.portfolio, "AAPL", Price(Decimal("160")), Money(Decimal("10"))
                )

        # Verify no state changes
        self.portfolio.increment_version.assert_not_called()

    def test_close_position_not_found(self):
        """Test closing non-existent position."""
        self.portfolio.get_position.return_value = None

        with patch.object(self.service.validator, "can_close_position", return_value=(True, None)):
            with pytest.raises(ValueError, match="No position found for AAPL"):
                self.service.close_position(
                    self.portfolio, "AAPL", Price(Decimal("160")), Money(Decimal("10"))
                )

    def test_close_full_position_internal(self):
        """Test internal _close_full_position method."""
        position = Mock(spec=Position)
        position.realized_pnl = Money(Decimal("100"))
        position.close_position = Mock()

        pnl = self.service._close_full_position(
            position, Price(Decimal("160")), Money(Decimal("10"))
        )

        position.close_position.assert_called_once_with(Price(Decimal("160")), Money(Decimal("10")))
        assert pnl.amount == Decimal("100")

    def test_close_full_position_internal_no_pnl(self):
        """Test _close_full_position with no realized P&L."""
        position = Mock(spec=Position)
        position.realized_pnl = None
        position.close_position = Mock()

        pnl = self.service._close_full_position(
            position, Price(Decimal("160")), Money(Decimal("10"))
        )

        assert pnl.amount == Decimal("0")

    def test_close_partial_position_internal(self):
        """Test internal _close_partial_position method."""
        position = Mock(spec=Position)
        position.symbol = "AAPL"
        position.quantity = Quantity(Decimal("10"))
        position.average_entry_price = Price(Decimal("150"))
        position.current_price = Price(Decimal("155"))
        position.entry_time = Mock()
        position.strategy = "momentum"

        pnl, remaining = self.service._close_partial_position(
            position, Quantity(Decimal("5")), Price(Decimal("160")), Money(Decimal("5"))
        )

        # P&L: (5 * 160) - (5 * 150) - 5 = 800 - 750 - 5 = 45
        assert pnl.amount == Decimal("45")
        assert remaining.quantity.value == Decimal("5")
        assert remaining.symbol == "AAPL"
        assert remaining.average_entry_price.value == Decimal("150")

    # --- Portfolio Metrics Tests ---

    def test_get_portfolio_metrics_complete(self):
        """Test getting comprehensive portfolio metrics."""
        # Setup calculator mock returns
        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("15000"))
        ):
            with patch.object(
                self.service.calculator, "get_positions_value", return_value=Money(Decimal("5000"))
            ):
                with patch.object(
                    self.service.calculator, "get_total_pnl", return_value=Money(Decimal("2000"))
                ):
                    with patch.object(
                        self.service.calculator,
                        "get_unrealized_pnl",
                        return_value=Money(Decimal("1000")),
                    ):
                        with patch.object(
                            self.service.calculator,
                            "get_return_percentage",
                            return_value=Decimal("15.5"),
                        ):
                            with patch.object(
                                self.service.calculator,
                                "get_win_rate",
                                return_value=Decimal("0.65"),
                            ):
                                with patch.object(
                                    self.service.calculator,
                                    "get_profit_factor",
                                    return_value=Decimal("2.5"),
                                ):
                                    with patch.object(
                                        self.service.calculator,
                                        "get_sharpe_ratio",
                                        return_value=Decimal("1.25"),
                                    ):
                                        with patch.object(
                                            self.service.calculator,
                                            "get_max_drawdown",
                                            return_value=Decimal("0.15"),
                                        ):
                                            with patch.object(
                                                self.service.calculator,
                                                "calculate_value_at_risk",
                                                return_value=Money(Decimal("500")),
                                            ):
                                                with patch.object(
                                                    self.service.validator,
                                                    "validate_portfolio_risk",
                                                    return_value=["Warning 1"],
                                                ):
                                                    metrics = self.service.get_portfolio_metrics(
                                                        self.portfolio
                                                    )

        # Verify structure and values
        assert metrics["value"]["total"] == 15000.0
        assert metrics["value"]["cash"] == 10000.0
        assert metrics["value"]["positions"] == 5000.0

        assert metrics["performance"]["total_pnl"] == 2000.0
        assert metrics["performance"]["unrealized_pnl"] == 1000.0
        assert metrics["performance"]["realized_pnl"] == 0.0
        assert metrics["performance"]["return_pct"] == 15.5
        assert metrics["performance"]["win_rate"] == 0.65
        assert metrics["performance"]["profit_factor"] == 2.5

        assert metrics["risk"]["sharpe_ratio"] == 1.25
        assert metrics["risk"]["max_drawdown"] == 0.15
        assert metrics["risk"]["var_95"] == 500.0
        assert metrics["risk"]["warnings"] == ["Warning 1"]

        assert metrics["statistics"]["positions_open"] == 0
        assert metrics["statistics"]["max_positions"] == 10
        assert metrics["statistics"]["trades_count"] == 0
        assert metrics["statistics"]["winning_trades"] == 0
        assert metrics["statistics"]["losing_trades"] == 0
        assert metrics["statistics"]["commission_paid"] == 0.0

    def test_get_portfolio_metrics_with_none_values(self):
        """Test portfolio metrics when some calculations return None."""
        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("10000"))
        ):
            with patch.object(
                self.service.calculator, "get_positions_value", return_value=Money(Decimal("0"))
            ):
                with patch.object(
                    self.service.calculator, "get_total_pnl", return_value=Money(Decimal("0"))
                ):
                    with patch.object(
                        self.service.calculator,
                        "get_unrealized_pnl",
                        return_value=Money(Decimal("0")),
                    ):
                        with patch.object(
                            self.service.calculator, "get_return_percentage", return_value=None
                        ):
                            with patch.object(
                                self.service.calculator, "get_win_rate", return_value=None
                            ):
                                with patch.object(
                                    self.service.calculator, "get_profit_factor", return_value=None
                                ):
                                    with patch.object(
                                        self.service.calculator,
                                        "get_sharpe_ratio",
                                        return_value=None,
                                    ):
                                        with patch.object(
                                            self.service.calculator,
                                            "get_max_drawdown",
                                            return_value=Decimal("0"),
                                        ):
                                            with patch.object(
                                                self.service.calculator,
                                                "calculate_value_at_risk",
                                                return_value=Money(Decimal("0")),
                                            ):
                                                with patch.object(
                                                    self.service.validator,
                                                    "validate_portfolio_risk",
                                                    return_value=[],
                                                ):
                                                    metrics = self.service.get_portfolio_metrics(
                                                        self.portfolio
                                                    )

        # Verify None values are handled correctly
        assert metrics["performance"]["return_pct"] == 0.0
        assert metrics["performance"]["win_rate"] == 0.0
        assert metrics["performance"]["profit_factor"] == 0.0
        assert metrics["risk"]["sharpe_ratio"] == 0.0

    # --- Risk Management Tests ---

    def test_validate_portfolio_health_healthy(self):
        """Test portfolio health validation with healthy portfolio."""
        with patch.object(self.service.validator, "validate_portfolio_risk", return_value=[]):
            with patch.object(
                self.service.validator, "validate_advanced_risk_metrics", return_value=[]
            ):
                with patch.object(self.service.validator, "validate_regulatory_compliance"):
                    health = self.service.validate_portfolio_health(self.portfolio)

        assert health["status"] == "healthy"
        assert health["risk_warnings"] == []
        assert health["advanced_warnings"] == []
        assert health["regulatory_status"] == "compliant"
        assert health["regulatory_issues"] == []

    def test_validate_portfolio_health_with_warnings(self):
        """Test portfolio health validation with warnings."""
        with patch.object(
            self.service.validator, "validate_portfolio_risk", return_value=["High concentration"]
        ):
            with patch.object(
                self.service.validator,
                "validate_advanced_risk_metrics",
                return_value=["VaR exceeds threshold"],
            ):
                with patch.object(self.service.validator, "validate_regulatory_compliance"):
                    health = self.service.validate_portfolio_health(self.portfolio)

        assert health["status"] == "warning"
        assert health["risk_warnings"] == ["High concentration"]
        assert health["advanced_warnings"] == ["VaR exceeds threshold"]
        assert health["regulatory_status"] == "compliant"

    def test_validate_portfolio_health_regulatory_violation(self):
        """Test portfolio health with regulatory violations."""
        with patch.object(self.service.validator, "validate_portfolio_risk", return_value=[]):
            with patch.object(
                self.service.validator, "validate_advanced_risk_metrics", return_value=[]
            ):
                with patch.object(
                    self.service.validator,
                    "validate_regulatory_compliance",
                    side_effect=ValueError("PDT rule violation"),
                ):
                    health = self.service.validate_portfolio_health(self.portfolio)

        assert health["status"] == "warning"
        assert health["regulatory_status"] == "non-compliant"
        assert health["regulatory_issues"] == ["PDT rule violation"]

    # --- Price Update Tests ---

    def test_update_portfolio_prices(self):
        """Test updating prices for all positions."""
        prices = {"AAPL": Price(Decimal("160")), "MSFT": Price(Decimal("300"))}

        self.service.update_portfolio_prices(self.portfolio, prices)

        self.portfolio.update_all_prices.assert_called_once_with(prices)

    # --- Portfolio Analysis Tests ---

    def test_get_top_performers(self):
        """Test getting top performing positions."""
        # Setup positions with P&L
        position1 = Mock(spec=Position)
        position2 = Mock(spec=Position)
        position3 = Mock(spec=Position)

        positions_by_profit = [
            (position1, Money(Decimal("500"))),
            (position2, Money(Decimal("300"))),
            (position3, Money(Decimal("100"))),
        ]

        current_prices = {"AAPL": Price(Decimal("160"))}

        with patch.object(
            self.service.calculator, "get_positions_by_profit", return_value=positions_by_profit
        ):
            top = self.service.get_top_performers(self.portfolio, current_prices, limit=2)

        assert len(top) == 2
        assert top[0][1].amount == Decimal("500")
        assert top[1][1].amount == Decimal("300")

    def test_get_worst_performers(self):
        """Test getting worst performing positions."""
        # Setup positions with P&L
        position1 = Mock(spec=Position)
        position2 = Mock(spec=Position)
        position3 = Mock(spec=Position)
        position4 = Mock(spec=Position)

        positions_by_profit = [
            (position1, Money(Decimal("500"))),
            (position2, Money(Decimal("300"))),
            (position3, Money(Decimal("-100"))),
            (position4, Money(Decimal("-200"))),
        ]

        current_prices = {"AAPL": Price(Decimal("160"))}

        with patch.object(
            self.service.calculator, "get_positions_by_profit", return_value=positions_by_profit
        ):
            worst = self.service.get_worst_performers(self.portfolio, current_prices, limit=2)

        assert len(worst) == 2
        assert worst[0][1].amount == Decimal("-100")
        assert worst[1][1].amount == Decimal("-200")

    def test_get_worst_performers_insufficient_positions(self):
        """Test getting worst performers with fewer positions than limit."""
        position1 = Mock(spec=Position)
        position2 = Mock(spec=Position)

        positions_by_profit = [
            (position1, Money(Decimal("100"))),
            (position2, Money(Decimal("-50"))),
        ]

        current_prices = {"AAPL": Price(Decimal("160"))}

        with patch.object(
            self.service.calculator, "get_positions_by_profit", return_value=positions_by_profit
        ):
            worst = self.service.get_worst_performers(self.portfolio, current_prices, limit=5)

        assert len(worst) == 2

    def test_get_portfolio_allocation_with_positions(self):
        """Test getting portfolio allocation breakdown."""
        # Setup positions
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.quantity = Quantity(Decimal("10"))
        position1.get_position_value.return_value = Money(Decimal("3000"))

        position2 = Mock(spec=Position)
        position2.symbol = "MSFT"
        position2.quantity = Quantity(Decimal("5"))
        position2.get_position_value.return_value = Money(Decimal("2000"))

        self.portfolio.get_open_positions.return_value = [position1, position2]

        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("15000"))
        ):  # 10k cash + 5k positions
            allocation = self.service.get_portfolio_allocation(self.portfolio)

        # Verify cash allocation
        assert allocation["cash"]["value"] == 10000.0
        assert allocation["cash"]["percentage"] == pytest.approx(66.67, rel=0.01)

        # Verify position allocations
        assert "AAPL" in allocation["positions"]
        assert allocation["positions"]["AAPL"]["value"] == 3000.0
        assert allocation["positions"]["AAPL"]["percentage"] == 20.0
        assert allocation["positions"]["AAPL"]["shares"] == 10.0

        assert "MSFT" in allocation["positions"]
        assert allocation["positions"]["MSFT"]["value"] == 2000.0
        assert allocation["positions"]["MSFT"]["percentage"] == pytest.approx(13.33, rel=0.01)
        assert allocation["positions"]["MSFT"]["shares"] == 5.0

    def test_get_portfolio_allocation_cash_only(self):
        """Test portfolio allocation with no positions."""
        self.portfolio.get_open_positions.return_value = []

        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("10000"))
        ):
            allocation = self.service.get_portfolio_allocation(self.portfolio)

        assert allocation["cash"]["value"] == 10000.0
        assert allocation["cash"]["percentage"] == 100.0
        assert len(allocation["positions"]) == 0

    def test_get_portfolio_allocation_zero_total_value(self):
        """Test portfolio allocation with zero total value."""
        self.portfolio.cash_balance = Money(Decimal("0"))
        self.portfolio.get_open_positions.return_value = []

        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("0"))
        ):
            allocation = self.service.get_portfolio_allocation(self.portfolio)

        assert allocation["cash"]["value"] == 0.0
        assert allocation["cash"]["percentage"] == 0.0

    # --- Edge Cases and Financial Precision Tests ---

    @patch("src.application.services.portfolio_service.Position")
    def test_open_position_fractional_shares(self, mock_position_class):
        """Test opening position with fractional shares."""
        mock_position = Mock(spec=Position)
        mock_position_class.return_value = mock_position

        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("0.12345"))  # Fractional shares
        request.entry_price = Price(Decimal("150.999"))
        request.commission = Money(Decimal("0.01"))
        request.strategy = "dca"

        with patch.object(self.service.validator, "validate_position_request"):
            position = self.service.open_position(self.portfolio, request)

        # Verify precise calculation: 0.12345 * 150.999 + 0.01
        expected_cost = Decimal("0.12345") * Decimal("150.999") + Decimal("0.01")
        expected_balance = Decimal("10000") - expected_cost
        assert abs(self.portfolio.cash_balance.amount - expected_balance) < Decimal("0.01")

    def test_close_position_precise_pnl_calculation(self):
        """Test precise P&L calculation with many decimals."""
        position = Mock(spec=Position)
        position.quantity = Quantity(Decimal("3.14159"))
        position.average_entry_price = Price(Decimal("100.111"))
        position.realized_pnl = Money(Decimal("31.3845"))  # (110.111 - 100.111) * 3.14159
        position.close_position = Mock()
        position.mark_as_closed = Mock()

        self.portfolio.get_position.return_value = position

        with patch.object(self.service.validator, "can_close_position", return_value=(True, None)):
            pnl = self.service.close_position(
                self.portfolio, "AAPL", Price(Decimal("110.111")), Money(Decimal("0.99"))
            )

        # P&L should match position's realized P&L
        assert pnl.amount == Decimal("31.3845")

    def test_portfolio_metrics_extreme_values(self):
        """Test portfolio metrics with extreme values."""
        # Setup extreme values
        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("999999999"))
        ):
            with patch.object(
                self.service.calculator,
                "get_positions_value",
                return_value=Money(Decimal("999999998")),
            ):
                with patch.object(
                    self.service.calculator,
                    "get_return_percentage",
                    return_value=Decimal("9999.99"),
                ):
                    with patch.object(
                        self.service.calculator, "get_sharpe_ratio", return_value=Decimal("10.5")
                    ):
                        with patch.object(
                            self.service.calculator,
                            "get_max_drawdown",
                            return_value=Decimal("0.95"),
                        ):
                            with patch.object(
                                self.service.calculator,
                                "calculate_value_at_risk",
                                return_value=Money(Decimal("50000000")),
                            ):
                                # Add other required patches with default values
                                with patch.object(
                                    self.service.calculator,
                                    "get_total_pnl",
                                    return_value=Money(Decimal("0")),
                                ):
                                    with patch.object(
                                        self.service.calculator,
                                        "get_unrealized_pnl",
                                        return_value=Money(Decimal("0")),
                                    ):
                                        with patch.object(
                                            self.service.calculator,
                                            "get_win_rate",
                                            return_value=None,
                                        ):
                                            with patch.object(
                                                self.service.calculator,
                                                "get_profit_factor",
                                                return_value=None,
                                            ):
                                                with patch.object(
                                                    self.service.validator,
                                                    "validate_portfolio_risk",
                                                    return_value=[],
                                                ):
                                                    metrics = self.service.get_portfolio_metrics(
                                                        self.portfolio
                                                    )

        # Verify extreme values are handled
        assert metrics["value"]["total"] == 999999999.0
        assert metrics["performance"]["return_pct"] == 9999.99
        assert metrics["risk"]["sharpe_ratio"] == 10.5
        assert metrics["risk"]["max_drawdown"] == 0.95
        assert metrics["risk"]["var_95"] == 50000000.0

    def test_get_portfolio_allocation_rounding(self):
        """Test portfolio allocation percentage rounding."""
        # Setup positions with values that create rounding challenges
        position1 = Mock(spec=Position)
        position1.symbol = "AAPL"
        position1.quantity = Quantity(Decimal("1"))
        position1.get_position_value.return_value = Money(Decimal("3333.33"))

        position2 = Mock(spec=Position)
        position2.symbol = "MSFT"
        position2.quantity = Quantity(Decimal("1"))
        position2.get_position_value.return_value = Money(Decimal("3333.34"))

        self.portfolio.cash_balance = Money(Decimal("3333.33"))
        self.portfolio.get_open_positions.return_value = [position1, position2]

        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("10000"))
        ):
            allocation = self.service.get_portfolio_allocation(self.portfolio)

        # Verify percentages sum to approximately 100%
        total_pct = allocation["cash"]["percentage"] + sum(
            p["percentage"] for p in allocation["positions"].values()
        )
        assert abs(total_pct - 100.0) < 0.1

    # --- Thread Safety and Concurrency Tests ---

    @patch("src.application.services.portfolio_service.Position")
    def test_multiple_operations_increment_version(self, mock_position_class):
        """Test that multiple operations properly increment version."""
        mock_position = Mock(spec=Position)
        mock_position_class.return_value = mock_position

        # Open position
        request = Mock(spec=PositionRequest)
        request.symbol = "AAPL"
        request.quantity = Quantity(Decimal("10"))
        request.entry_price = Price(Decimal("150"))
        request.commission = Money(Decimal("10"))
        request.strategy = "momentum"

        with patch.object(self.service.validator, "validate_position_request"):
            self.service.open_position(self.portfolio, request)

        assert self.portfolio.increment_version.call_count == 1

        # Setup for close
        position = Mock(spec=Position)
        position.quantity = Quantity(Decimal("10"))
        position.average_entry_price = Price(Decimal("150"))
        position.realized_pnl = Money(Decimal("100"))
        position.close_position = Mock()
        position.mark_as_closed = Mock()

        self.portfolio.get_position.return_value = position

        # Close position
        with patch.object(self.service.validator, "can_close_position", return_value=(True, None)):
            self.service.close_position(
                self.portfolio, "AAPL", Price(Decimal("160")), Money(Decimal("10"))
            )

        assert self.portfolio.increment_version.call_count == 2

    def test_service_methods_are_stateless(self):
        """Test that service methods don't maintain state between calls."""
        # First call
        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("10000"))
        ):
            value1 = self.service.get_portfolio_metrics(self.portfolio)

        # Modify portfolio
        self.portfolio.cash_balance = Money(Decimal("20000"))

        # Second call should reflect new state
        with patch.object(
            self.service.calculator, "get_total_value", return_value=Money(Decimal("20000"))
        ):
            value2 = self.service.get_portfolio_metrics(self.portfolio)

        # Values should be different
        assert value1["value"]["cash"] == 10000.0
        assert value2["value"]["cash"] == 20000.0
