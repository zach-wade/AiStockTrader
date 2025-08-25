"""Extended comprehensive tests for RiskCalculator domain service.

This test suite provides extensive coverage for the RiskCalculator service,
testing all risk metrics, VAR calculations, portfolio risk assessment, and edge cases.
"""

from decimal import Decimal

import pytest

from src.domain.entities.portfolio import Portfolio
from src.domain.entities.position import Position
from src.domain.services.risk_calculator import RiskCalculator


class TestRiskCalculatorBasics:
    """Test basic RiskCalculator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        self.position = Position.open_position(
            symbol="AAPL", quantity=100, entry_price=Decimal("150.00")
        )
        self.portfolio = Portfolio(account_id="test_account", initial_capital=Decimal("100000"))

    def test_initialization(self):
        """Test RiskCalculator initialization."""
        assert isinstance(self.calculator, RiskCalculator)
        assert hasattr(self.calculator, "calculate_position_risk")
        assert hasattr(self.calculator, "calculate_portfolio_risk")

    def test_calculate_position_risk_long(self):
        """Test risk calculation for long position."""
        risk = self.calculator.calculate_position_risk(
            self.position, current_price=Decimal("145.00")
        )

        assert "value_at_risk" in risk
        assert "max_loss" in risk
        assert "current_exposure" in risk
        assert "unrealized_pnl" in risk

        assert risk["unrealized_pnl"] == Decimal("-500.00")  # Loss of $5 per share
        assert risk["current_exposure"] == Decimal("14500.00")  # 100 * 145

    def test_calculate_position_risk_short(self):
        """Test risk calculation for short position."""
        short_position = Position.open_position(
            symbol="AAPL", quantity=-100, entry_price=Decimal("150.00")
        )

        risk = self.calculator.calculate_position_risk(
            short_position, current_price=Decimal("155.00")
        )

        assert risk["unrealized_pnl"] == Decimal("-500.00")  # Loss on short
        assert risk["current_exposure"] == Decimal("15500.00")  # Absolute exposure

    def test_calculate_position_risk_with_stop_loss(self):
        """Test risk calculation with stop loss."""
        self.position.stop_loss = Decimal("145.00")

        risk = self.calculator.calculate_position_risk(
            self.position, current_price=Decimal("148.00")
        )

        assert "stop_loss_risk" in risk
        assert risk["stop_loss_risk"] == Decimal("500.00")  # Max loss if stop hit


class TestValueAtRisk:
    """Test Value at Risk (VaR) calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        self.historical_prices = [
            Decimal("100.00"),
            Decimal("102.00"),
            Decimal("101.00"),
            Decimal("103.00"),
            Decimal("105.00"),
            Decimal("104.00"),
            Decimal("106.00"),
            Decimal("105.50"),
            Decimal("107.00"),
            Decimal("108.00"),
            Decimal("107.50"),
            Decimal("109.00"),
            Decimal("108.50"),
            Decimal("110.00"),
            Decimal("111.00"),
        ]

    def test_calculate_var_historical(self):
        """Test historical VaR calculation."""
        var_95 = self.calculator.calculate_var_historical(
            position_value=Decimal("10000"),
            historical_prices=self.historical_prices,
            confidence_level=Decimal("0.95"),
        )

        assert var_95 is not None
        assert var_95 > 0
        assert var_95 < Decimal("10000")  # VaR should be less than total position

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation."""
        returns = self.calculator._calculate_returns(self.historical_prices)
        volatility = self.calculator._calculate_volatility(returns)

        var_95 = self.calculator.calculate_var_parametric(
            position_value=Decimal("10000"),
            volatility=volatility,
            confidence_level=Decimal("0.95"),
            time_horizon=1,
        )

        assert var_95 is not None
        assert var_95 > 0
        # Parametric VaR at 95% confidence ≈ 1.645 * volatility * position_value
        expected_var = Decimal("1.645") * volatility * Decimal("10000")
        assert abs(var_95 - expected_var) < Decimal("100")  # Allow some tolerance

    def test_calculate_var_monte_carlo(self):
        """Test Monte Carlo VaR calculation."""
        returns = self.calculator._calculate_returns(self.historical_prices)
        mean_return = sum(returns) / len(returns)
        volatility = self.calculator._calculate_volatility(returns)

        var_95 = self.calculator.calculate_var_monte_carlo(
            position_value=Decimal("10000"),
            mean_return=mean_return,
            volatility=volatility,
            confidence_level=Decimal("0.95"),
            time_horizon=1,
            simulations=1000,
        )

        assert var_95 is not None
        assert var_95 > 0
        assert var_95 < Decimal("10000")

    def test_calculate_var_different_confidence_levels(self):
        """Test VaR at different confidence levels."""
        var_90 = self.calculator.calculate_var_historical(
            position_value=Decimal("10000"),
            historical_prices=self.historical_prices,
            confidence_level=Decimal("0.90"),
        )

        var_95 = self.calculator.calculate_var_historical(
            position_value=Decimal("10000"),
            historical_prices=self.historical_prices,
            confidence_level=Decimal("0.95"),
        )

        var_99 = self.calculator.calculate_var_historical(
            position_value=Decimal("10000"),
            historical_prices=self.historical_prices,
            confidence_level=Decimal("0.99"),
        )

        # Higher confidence should mean higher VaR (more conservative)
        assert var_90 < var_95 < var_99

    def test_calculate_var_insufficient_data(self):
        """Test VaR with insufficient historical data."""
        with pytest.raises(ValueError, match="Insufficient historical data"):
            self.calculator.calculate_var_historical(
                position_value=Decimal("10000"),
                historical_prices=[Decimal("100"), Decimal("101")],  # Too few
                confidence_level=Decimal("0.95"),
            )


class TestPortfolioRisk:
    """Test portfolio-level risk calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        self.portfolio = Portfolio(
            account_id="test_account",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("50000"),
        )

        # Add multiple positions
        self.positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("MSFT", 50, Decimal("300.00")),
            Position.open_position("GOOGL", 20, Decimal("2500.00")),
        ]

        for pos in self.positions:
            self.portfolio.positions[pos.symbol] = pos

    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        current_prices = {
            "AAPL": Decimal("155.00"),
            "MSFT": Decimal("310.00"),
            "GOOGL": Decimal("2450.00"),
        }

        risk = self.calculator.calculate_portfolio_risk(self.portfolio, current_prices)

        assert "total_exposure" in risk
        assert "total_value" in risk
        assert "cash_percentage" in risk
        assert "position_concentration" in risk
        assert "largest_position_percentage" in risk
        assert "number_of_positions" in risk

        # Verify calculations
        expected_exposure = (
            100 * Decimal("155.00")  # AAPL
            + 50 * Decimal("310.00")  # MSFT
            + 20 * Decimal("2450.00")  # GOOGL
        )
        assert risk["total_exposure"] == expected_exposure
        assert risk["total_value"] == expected_exposure + Decimal("50000")
        assert risk["number_of_positions"] == 3

    def test_calculate_portfolio_beta(self):
        """Test portfolio beta calculation."""
        position_betas = {"AAPL": Decimal("1.2"), "MSFT": Decimal("1.1"), "GOOGL": Decimal("1.0")}

        current_prices = {
            "AAPL": Decimal("155.00"),
            "MSFT": Decimal("310.00"),
            "GOOGL": Decimal("2450.00"),
        }

        portfolio_beta = self.calculator.calculate_portfolio_beta(
            self.portfolio, position_betas, current_prices
        )

        # Weighted average beta
        aapl_value = 100 * Decimal("155.00")
        msft_value = 50 * Decimal("310.00")
        googl_value = 20 * Decimal("2450.00")
        total_value = aapl_value + msft_value + googl_value

        expected_beta = (
            (aapl_value / total_value) * Decimal("1.2")
            + (msft_value / total_value) * Decimal("1.1")
            + (googl_value / total_value) * Decimal("1.0")
        )

        assert abs(portfolio_beta - expected_beta) < Decimal("0.01")

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.015"),
            Decimal("0.005"),
            Decimal("-0.005"),
            Decimal("0.025"),
            Decimal("0.01"),
            Decimal("-0.002"),
            Decimal("0.018"),
            Decimal("0.012"),
            Decimal("0.008"),
        ]

        sharpe = self.calculator.calculate_sharpe_ratio(
            returns=returns,
            risk_free_rate=Decimal("0.02"),  # 2% annual
        )

        assert sharpe is not None
        # With positive returns above risk-free rate, Sharpe should be positive
        mean_return = sum(returns) / len(returns) * 252  # Annualized
        if mean_return > Decimal("0.02"):
            assert sharpe > 0

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        portfolio_values = [
            Decimal("100000"),
            Decimal("105000"),
            Decimal("110000"),
            Decimal("108000"),
            Decimal("95000"),
            Decimal("98000"),
            Decimal("102000"),
            Decimal("100000"),
            Decimal("104000"),
        ]

        max_dd = self.calculator.calculate_max_drawdown(portfolio_values)

        # Max drawdown from 110000 to 95000
        expected_dd = (Decimal("95000") - Decimal("110000")) / Decimal("110000")
        assert abs(max_dd - abs(expected_dd)) < Decimal("0.01")


class TestRiskMetrics:
    """Test various risk metrics calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("-0.01"),
            Decimal("0.015"),
            Decimal("0.005"),
            Decimal("-0.005"),
            Decimal("0.025"),
            Decimal("0.01"),
            Decimal("-0.002"),
            Decimal("0.018"),
            Decimal("0.012"),
            Decimal("0.008"),
        ]

        sortino = self.calculator.calculate_sortino_ratio(
            returns=returns, target_return=Decimal("0.01")
        )

        assert sortino is not None
        # Sortino focuses on downside deviation
        assert sortino != 0

    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        annual_return = Decimal("0.15")  # 15% annual return
        max_drawdown = Decimal("0.10")  # 10% max drawdown

        calmar = self.calculator.calculate_calmar_ratio(
            annual_return=annual_return, max_drawdown=max_drawdown
        )

        assert calmar == Decimal("1.5")  # 0.15 / 0.10

    def test_calculate_information_ratio(self):
        """Test information ratio calculation."""
        portfolio_returns = [
            Decimal("0.012"),
            Decimal("0.018"),
            Decimal("-0.005"),
            Decimal("0.020"),
            Decimal("0.008"),
            Decimal("-0.003"),
        ]

        benchmark_returns = [
            Decimal("0.010"),
            Decimal("0.015"),
            Decimal("-0.002"),
            Decimal("0.018"),
            Decimal("0.005"),
            Decimal("-0.001"),
        ]

        info_ratio = self.calculator.calculate_information_ratio(
            portfolio_returns=portfolio_returns, benchmark_returns=benchmark_returns
        )

        assert info_ratio is not None


class TestStressTests:
    """Test stress testing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        self.portfolio = Portfolio(
            account_id="test_account",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("30000"),
        )

        # Add positions
        positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("MSFT", 50, Decimal("300.00")),
            Position.open_position("SPY", 200, Decimal("400.00")),  # Index ETF
        ]

        for pos in positions:
            self.portfolio.positions[pos.symbol] = pos

    def test_stress_test_market_crash(self):
        """Test portfolio under market crash scenario."""
        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
            "SPY": Decimal("400.00"),
        }

        # Simulate 20% market crash
        crash_scenario = {
            "AAPL": Decimal("0.80"),  # 20% drop
            "MSFT": Decimal("0.80"),
            "SPY": Decimal("0.80"),
        }

        stressed_values = self.calculator.apply_stress_scenario(
            self.portfolio, current_prices, crash_scenario
        )

        assert "stressed_portfolio_value" in stressed_values
        assert "loss_amount" in stressed_values
        assert "loss_percentage" in stressed_values

        # Verify 20% loss on positions
        original_value = (
            sum(
                abs(pos.quantity) * current_prices[pos.symbol]
                for pos in self.portfolio.positions.values()
            )
            + self.portfolio.cash_balance
        )

        assert stressed_values["loss_percentage"] == pytest.approx(Decimal("0.20"), rel=0.1)

    def test_stress_test_sector_specific(self):
        """Test sector-specific stress scenario."""
        current_prices = {
            "AAPL": Decimal("150.00"),
            "MSFT": Decimal("300.00"),
            "SPY": Decimal("400.00"),
        }

        # Tech sector drops more than market
        tech_crash = {
            "AAPL": Decimal("0.70"),  # 30% drop
            "MSFT": Decimal("0.65"),  # 35% drop
            "SPY": Decimal("0.90"),  # 10% drop (market)
        }

        stressed_values = self.calculator.apply_stress_scenario(
            self.portfolio, current_prices, tech_crash
        )

        # Tech-heavy portfolio should show significant loss
        assert stressed_values["loss_percentage"] > Decimal("0.15")


class TestCorrelationAndDiversification:
    """Test correlation and diversification metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()

        # Create price history for multiple assets
        self.price_history = {
            "AAPL": [Decimal(str(p)) for p in [150, 152, 151, 153, 155, 154, 156]],
            "MSFT": [Decimal(str(p)) for p in [300, 302, 301, 303, 305, 304, 306]],
            "GLD": [
                Decimal(str(p)) for p in [180, 179, 181, 180, 179, 182, 181]
            ],  # Gold (uncorrelated)
        }

    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        correlation_matrix = self.calculator.calculate_correlation_matrix(self.price_history)

        assert "AAPL" in correlation_matrix
        assert "MSFT" in correlation_matrix
        assert "GLD" in correlation_matrix

        # Correlation with itself should be 1
        assert correlation_matrix["AAPL"]["AAPL"] == pytest.approx(Decimal("1.0"), rel=0.01)

        # AAPL and MSFT should be highly correlated (similar price movements)
        assert correlation_matrix["AAPL"]["MSFT"] > Decimal("0.9")

        # Gold should have lower correlation with tech stocks
        assert abs(correlation_matrix["AAPL"]["GLD"]) < Decimal("0.5")

    def test_calculate_diversification_ratio(self):
        """Test portfolio diversification ratio."""
        portfolio = Portfolio(account_id="test", initial_capital=Decimal("100000"))

        positions = [
            Position.open_position("AAPL", 100, Decimal("150.00")),
            Position.open_position("MSFT", 50, Decimal("300.00")),
            Position.open_position("GLD", 100, Decimal("180.00")),
        ]

        for pos in positions:
            portfolio.positions[pos.symbol] = pos

        current_prices = {
            "AAPL": Decimal("155.00"),
            "MSFT": Decimal("310.00"),
            "GLD": Decimal("182.00"),
        }

        div_ratio = self.calculator.calculate_diversification_ratio(
            portfolio, self.price_history, current_prices
        )

        # Diversification ratio > 1 indicates benefit from diversification
        assert div_ratio > Decimal("1.0")


class TestLiquidityRisk:
    """Test liquidity risk calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        self.position = Position.open_position(
            symbol="AAPL",
            quantity=10000,
            entry_price=Decimal("150.00"),  # Large position
        )

    def test_calculate_liquidity_risk(self):
        """Test liquidity risk for large positions."""
        avg_daily_volume = 50000000  # 50M shares

        liquidity_risk = self.calculator.calculate_liquidity_risk(
            position=self.position,
            avg_daily_volume=avg_daily_volume,
            current_price=Decimal("150.00"),
        )

        assert "position_size_vs_adv" in liquidity_risk
        assert "estimated_market_impact" in liquidity_risk
        assert "days_to_liquidate" in liquidity_risk

        # Position is 0.02% of daily volume
        assert liquidity_risk["position_size_vs_adv"] == Decimal("0.0002")

        # Small positions should have minimal market impact
        assert liquidity_risk["estimated_market_impact"] < Decimal("0.01")

    def test_calculate_liquidity_risk_large_position(self):
        """Test liquidity risk for very large position."""
        large_position = Position.open_position(
            symbol="SMALLCAP",
            quantity=500000,
            entry_price=Decimal("10.00"),  # Very large position
        )

        avg_daily_volume = 1000000  # Only 1M daily volume

        liquidity_risk = self.calculator.calculate_liquidity_risk(
            position=large_position,
            avg_daily_volume=avg_daily_volume,
            current_price=Decimal("10.00"),
        )

        # Position is 50% of daily volume - high liquidity risk
        assert liquidity_risk["position_size_vs_adv"] == Decimal("0.50")

        # Large positions should show significant market impact
        assert liquidity_risk["estimated_market_impact"] > Decimal("0.05")

        # Should take multiple days to liquidate
        assert liquidity_risk["days_to_liquidate"] > 1


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()

    def test_calculate_risk_empty_portfolio(self):
        """Test risk calculation for empty portfolio."""
        portfolio = Portfolio(
            account_id="test", initial_capital=Decimal("100000"), cash_balance=Decimal("100000")
        )

        risk = self.calculator.calculate_portfolio_risk(portfolio, {})

        assert risk["total_exposure"] == Decimal("0")
        assert risk["number_of_positions"] == 0
        assert risk["cash_percentage"] == Decimal("1.0")

    def test_calculate_var_zero_position_value(self):
        """Test VaR with zero position value."""
        var = self.calculator.calculate_var_parametric(
            position_value=Decimal("0"),
            volatility=Decimal("0.02"),
            confidence_level=Decimal("0.95"),
        )

        assert var == Decimal("0")

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        # All returns are the same
        returns = [Decimal("0.01")] * 10

        sharpe = self.calculator.calculate_sharpe_ratio(
            returns=returns, risk_free_rate=Decimal("0.005")
        )

        # Should handle zero volatility gracefully
        assert sharpe is None or sharpe == float("inf")

    def test_calculate_max_drawdown_increasing_values(self):
        """Test max drawdown with only increasing values."""
        values = [Decimal(str(i * 1000)) for i in range(1, 11)]

        max_dd = self.calculator.calculate_max_drawdown(values)

        assert max_dd == Decimal("0")  # No drawdown

    def test_invalid_confidence_level(self):
        """Test VaR with invalid confidence level."""
        with pytest.raises(ValueError, match="Confidence level must be between"):
            self.calculator.calculate_var_parametric(
                position_value=Decimal("10000"),
                volatility=Decimal("0.02"),
                confidence_level=Decimal("1.5"),  # Invalid
            )

    def test_negative_time_horizon(self):
        """Test VaR with negative time horizon."""
        with pytest.raises(ValueError, match="Time horizon must be positive"):
            self.calculator.calculate_var_parametric(
                position_value=Decimal("10000"),
                volatility=Decimal("0.02"),
                confidence_level=Decimal("0.95"),
                time_horizon=-1,
            )
