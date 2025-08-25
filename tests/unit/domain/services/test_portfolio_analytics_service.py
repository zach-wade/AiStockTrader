"""
Comprehensive unit tests for PortfolioAnalyticsService.

Tests all portfolio analytics functionality including performance metrics,
risk calculations, position weights, and P&L calculations.
"""

from datetime import datetime

import pytest

from src.domain.services.portfolio_analytics_service import (
    PortfolioAnalyticsService,
    PortfolioValue,
    PositionInfo,
    TradeRecord,
)


class TestPortfolioAnalyticsService:
    """Test suite for PortfolioAnalyticsService."""

    @pytest.fixture
    def service(self):
        """Create a PortfolioAnalyticsService instance."""
        return PortfolioAnalyticsService(risk_free_rate=0.02)

    @pytest.fixture
    def sample_portfolio_values(self):
        """Create sample portfolio values for testing."""
        base_time = datetime.now().timestamp()
        return [
            PortfolioValue(
                timestamp=base_time + i * 86400,  # Daily values
                portfolio_id="test_portfolio",
                value=10000 + i * 100,  # Increasing values
            )
            for i in range(10)
        ]

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade records for testing."""
        base_time = datetime.now().timestamp()
        return [
            TradeRecord(
                timestamp=base_time + i * 3600,
                order_id=f"order_{i}",
                symbol=f"STOCK{i % 3}",
                portfolio_id="test_portfolio",
                strategy="test_strategy",
                operation="buy" if i % 2 == 0 else "sell",
                status="completed",
                duration_ms=100.0,
            )
            for i in range(5)
        ]

    @pytest.fixture
    def sample_positions(self):
        """Create sample position info for testing."""
        return {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.00,
                market_value=16000.00,
                unrealized_pnl=1000.00,
                last_update=datetime.now().timestamp(),
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800.00,
                market_value=145000.00,
                unrealized_pnl=-5000.00,
                last_update=datetime.now().timestamp(),
            ),
            "MSFT": PositionInfo(
                symbol="MSFT",
                quantity=75,
                avg_cost=380.00,
                market_value=30000.00,
                unrealized_pnl=1500.00,
                last_update=datetime.now().timestamp(),
            ),
        }

    # Initialization Tests

    def test_initialization_default(self):
        """Test default initialization."""
        service = PortfolioAnalyticsService()
        assert service.risk_free_rate == 0.02
        assert service.trading_days_per_year == 252

    def test_initialization_custom_risk_free_rate(self):
        """Test initialization with custom risk-free rate."""
        service = PortfolioAnalyticsService(risk_free_rate=0.05)
        assert service.risk_free_rate == 0.05
        assert service.trading_days_per_year == 252

    # Performance Metrics Tests

    def test_calculate_performance_metrics_valid(
        self, service, sample_portfolio_values, sample_trades
    ):
        """Test calculation of valid performance metrics."""
        metrics = service.calculate_performance_metrics(
            portfolio_values=sample_portfolio_values,
            trades=sample_trades,
            current_positions_count=3,
            period_days=10,
        )

        assert metrics.portfolio_id == "test_portfolio"
        assert metrics.period_days == 10
        assert metrics.start_value == 10000
        assert metrics.end_value == 10900
        assert metrics.total_return_percent == 9.0
        assert metrics.total_trades == 5
        assert metrics.current_positions == 3
        assert metrics.data_points == 10
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown_percent >= 0
        assert metrics.annualized_volatility_percent >= 0

    def test_calculate_performance_metrics_no_values(self, service):
        """Test calculation with no portfolio values."""
        with pytest.raises(ValueError, match="No portfolio values provided"):
            service.calculate_performance_metrics(
                portfolio_values=[], trades=[], current_positions_count=0, period_days=1
            )

    def test_calculate_performance_metrics_insufficient_data(self, service):
        """Test calculation with insufficient data points."""
        single_value = [
            PortfolioValue(timestamp=datetime.now().timestamp(), portfolio_id="test", value=10000)
        ]

        with pytest.raises(ValueError, match="Insufficient data points"):
            service.calculate_performance_metrics(
                portfolio_values=single_value, trades=[], current_positions_count=0, period_days=1
            )

    def test_calculate_performance_metrics_invalid_start_value(self, service):
        """Test calculation with invalid start value."""
        invalid_values = [
            PortfolioValue(timestamp=1.0, portfolio_id="test", value=0),
            PortfolioValue(timestamp=2.0, portfolio_id="test", value=1000),
        ]

        with pytest.raises(ValueError, match="Invalid start value"):
            service.calculate_performance_metrics(
                portfolio_values=invalid_values, trades=[], current_positions_count=0, period_days=1
            )

    def test_calculate_performance_metrics_unsorted_values(self, service, sample_portfolio_values):
        """Test that values are sorted by timestamp."""
        # Reverse the order
        reversed_values = list(reversed(sample_portfolio_values))

        metrics = service.calculate_performance_metrics(
            portfolio_values=reversed_values, trades=[], current_positions_count=0, period_days=10
        )

        # Should still calculate correctly after sorting
        assert metrics.start_value == 10000
        assert metrics.end_value == 10900

    def test_calculate_performance_metrics_no_trades(self, service, sample_portfolio_values):
        """Test calculation with no trades."""
        metrics = service.calculate_performance_metrics(
            portfolio_values=sample_portfolio_values,
            trades=None,
            current_positions_count=0,
            period_days=10,
        )

        assert metrics.total_trades == 0

    # Total Return Tests

    def test_calculate_total_return_positive(self, service):
        """Test calculation of positive total return."""
        return_pct = service._calculate_total_return(10000, 11000)
        assert return_pct == 10.0

    def test_calculate_total_return_negative(self, service):
        """Test calculation of negative total return."""
        return_pct = service._calculate_total_return(10000, 9000)
        assert return_pct == -10.0

    def test_calculate_total_return_zero_start(self, service):
        """Test calculation with zero start value."""
        return_pct = service._calculate_total_return(0, 1000)
        assert return_pct == 0.0

    def test_calculate_total_return_negative_start(self, service):
        """Test calculation with negative start value."""
        return_pct = service._calculate_total_return(-1000, 1000)
        assert return_pct == 0.0

    def test_calculate_total_return_no_change(self, service):
        """Test calculation with no change in value."""
        return_pct = service._calculate_total_return(10000, 10000)
        assert return_pct == 0.0

    # Daily Returns Tests

    def test_calculate_daily_returns_normal(self, service):
        """Test calculation of normal daily returns."""
        values = [100, 110, 105, 115, 120]
        returns = service._calculate_daily_returns(values)

        assert len(returns) == 4
        assert returns[0] == pytest.approx(0.1)  # 10% gain
        assert returns[1] == pytest.approx(-0.0454545, rel=1e-5)  # ~4.5% loss
        assert returns[2] == pytest.approx(0.0952381, rel=1e-5)  # ~9.5% gain
        assert returns[3] == pytest.approx(0.0434783, rel=1e-5)  # ~4.3% gain

    def test_calculate_daily_returns_with_zero(self, service):
        """Test calculation with zero values."""
        values = [100, 0, 50, 100]
        returns = service._calculate_daily_returns(values)

        assert len(returns) == 3
        assert returns[0] == -1.0  # 100% loss
        assert returns[1] == 0.0  # Can't calculate from 0
        assert returns[2] == 1.0  # 100% gain

    def test_calculate_daily_returns_single_value(self, service):
        """Test calculation with single value."""
        values = [100]
        returns = service._calculate_daily_returns(values)

        assert len(returns) == 0

    def test_calculate_daily_returns_empty(self, service):
        """Test calculation with empty values."""
        returns = service._calculate_daily_returns([])
        assert returns == []

    # Volatility Tests

    def test_calculate_annualized_volatility_normal(self, service):
        """Test calculation of normal annualized volatility."""
        daily_returns = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008]
        volatility = service._calculate_annualized_volatility(daily_returns)

        # Should be positive and reasonable
        assert volatility > 0
        assert volatility < 100  # Less than 100% annually

    def test_calculate_annualized_volatility_no_variation(self, service):
        """Test calculation with no variation."""
        daily_returns = [0.01, 0.01, 0.01, 0.01]
        volatility = service._calculate_annualized_volatility(daily_returns)

        assert volatility == 0.0

    def test_calculate_annualized_volatility_single_return(self, service):
        """Test calculation with single return."""
        daily_returns = [0.05]
        volatility = service._calculate_annualized_volatility(daily_returns)

        assert volatility == 0.0

    def test_calculate_annualized_volatility_empty(self, service):
        """Test calculation with empty returns."""
        volatility = service._calculate_annualized_volatility([])
        assert volatility == 0.0

    def test_calculate_annualized_volatility_high_variance(self, service):
        """Test calculation with high variance returns."""
        daily_returns = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
        volatility = service._calculate_annualized_volatility(daily_returns)

        # Should be high due to large swings
        assert volatility > 100  # More than 100% annually

    # Max Drawdown Tests

    def test_calculate_max_drawdown_normal(self, service):
        """Test calculation of normal max drawdown."""
        values = [100, 110, 105, 95, 100, 120, 110, 115]
        drawdown = service._calculate_max_drawdown(values)

        # Max drawdown from 110 to 95 = (110-95)/110 = 13.6%
        assert drawdown == pytest.approx(13.636363, rel=1e-5)

    def test_calculate_max_drawdown_no_drawdown(self, service):
        """Test calculation with no drawdown (always increasing)."""
        values = [100, 110, 120, 130, 140, 150]
        drawdown = service._calculate_max_drawdown(values)

        assert drawdown == 0.0

    def test_calculate_max_drawdown_complete_loss(self, service):
        """Test calculation with complete loss."""
        values = [100, 110, 50, 0]
        drawdown = service._calculate_max_drawdown(values)

        assert drawdown == 100.0

    def test_calculate_max_drawdown_empty(self, service):
        """Test calculation with empty values."""
        drawdown = service._calculate_max_drawdown([])
        assert drawdown == 0.0

    def test_calculate_max_drawdown_single_value(self, service):
        """Test calculation with single value."""
        drawdown = service._calculate_max_drawdown([100])
        assert drawdown == 0.0

    def test_calculate_max_drawdown_recovery(self, service):
        """Test calculation with drawdown and recovery."""
        values = [100, 120, 80, 90, 130, 125]
        drawdown = service._calculate_max_drawdown(values)

        # Max drawdown from 120 to 80 = (120-80)/120 = 33.3%
        assert drawdown == pytest.approx(33.333333, rel=1e-5)

    # Sharpe Ratio Tests

    def test_calculate_sharpe_ratio_positive(self, service):
        """Test calculation of positive Sharpe ratio."""
        sharpe = service._calculate_sharpe_ratio(15.0, 20.0)

        # (0.15 - 0.02) / 0.20 = 0.65
        assert sharpe == pytest.approx(0.65, rel=1e-5)

    def test_calculate_sharpe_ratio_negative(self, service):
        """Test calculation of negative Sharpe ratio."""
        sharpe = service._calculate_sharpe_ratio(-5.0, 10.0)

        # (-0.05 - 0.02) / 0.10 = -0.7
        assert sharpe == pytest.approx(-0.7, rel=1e-5)

    def test_calculate_sharpe_ratio_zero_volatility(self, service):
        """Test calculation with zero volatility."""
        sharpe = service._calculate_sharpe_ratio(10.0, 0.0)
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_high_return_low_vol(self, service):
        """Test calculation with high return and low volatility."""
        sharpe = service._calculate_sharpe_ratio(50.0, 5.0)

        # (0.50 - 0.02) / 0.05 = 9.6
        assert sharpe == pytest.approx(9.6, rel=1e-5)

    def test_calculate_sharpe_ratio_custom_risk_free_rate(self):
        """Test Sharpe ratio with custom risk-free rate."""
        service = PortfolioAnalyticsService(risk_free_rate=0.05)
        sharpe = service._calculate_sharpe_ratio(20.0, 15.0)

        # (0.20 - 0.05) / 0.15 = 1.0
        assert sharpe == pytest.approx(1.0, rel=1e-5)

    # Position Weights Tests

    def test_calculate_position_weights_normal(self, service, sample_positions):
        """Test calculation of normal position weights."""
        weights = service.calculate_position_weights(sample_positions)

        assert len(weights) == 3

        # Total value = 16000 + 145000 + 30000 = 191000
        assert weights["AAPL"]["weight_percent"] == pytest.approx(8.377, rel=0.01)
        assert weights["GOOGL"]["weight_percent"] == pytest.approx(75.916, rel=0.01)
        assert weights["MSFT"]["weight_percent"] == pytest.approx(15.707, rel=0.01)

        # Check other fields
        assert weights["AAPL"]["market_value"] == 16000
        assert weights["AAPL"]["quantity"] == 100
        assert weights["AAPL"]["avg_cost"] == 150
        assert weights["AAPL"]["unrealized_pnl"] == 1000

    def test_calculate_position_weights_empty(self, service):
        """Test calculation with no positions."""
        weights = service.calculate_position_weights({})
        assert weights == {}

    def test_calculate_position_weights_single_position(self, service):
        """Test calculation with single position."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.00,
                market_value=16000.00,
                unrealized_pnl=1000.00,
                last_update=datetime.now().timestamp(),
            )
        }

        weights = service.calculate_position_weights(positions)

        assert len(weights) == 1
        assert weights["AAPL"]["weight_percent"] == 100.0

    def test_calculate_position_weights_zero_total_value(self, service):
        """Test calculation with zero total value."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=100,
                avg_cost=150.00,
                market_value=0,
                unrealized_pnl=-15000.00,
                last_update=datetime.now().timestamp(),
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800.00,
                market_value=0,
                unrealized_pnl=-140000.00,
                last_update=datetime.now().timestamp(),
            ),
        }

        weights = service.calculate_position_weights(positions)
        assert weights == {}

    def test_calculate_position_weights_negative_values(self, service):
        """Test calculation with negative market values (short positions)."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=-100,
                avg_cost=150.00,
                market_value=-16000.00,
                unrealized_pnl=1000.00,
                last_update=datetime.now().timestamp(),
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800.00,
                market_value=140000.00,
                unrealized_pnl=0,
                last_update=datetime.now().timestamp(),
            ),
        }

        weights = service.calculate_position_weights(positions)

        # Total = -16000 + 140000 = 124000
        assert weights["AAPL"]["weight_percent"] == pytest.approx(-12.903, rel=0.01)
        assert weights["GOOGL"]["weight_percent"] == pytest.approx(112.903, rel=0.01)

    # Portfolio P&L Tests

    def test_calculate_portfolio_pnl_normal(self, service, sample_positions):
        """Test calculation of normal portfolio P&L."""
        pnl = service.calculate_portfolio_pnl(sample_positions, realized_pnl=5000.0)

        assert pnl["realized_pnl"] == 5000.0
        assert pnl["unrealized_pnl"] == -2500.0  # 1000 - 5000 + 1500
        assert pnl["total_pnl"] == 2500.0

    def test_calculate_portfolio_pnl_no_realized(self, service, sample_positions):
        """Test calculation with no realized P&L."""
        pnl = service.calculate_portfolio_pnl(sample_positions)

        assert pnl["realized_pnl"] == 0.0
        assert pnl["unrealized_pnl"] == -2500.0
        assert pnl["total_pnl"] == -2500.0

    def test_calculate_portfolio_pnl_no_positions(self, service):
        """Test calculation with no positions."""
        pnl = service.calculate_portfolio_pnl({}, realized_pnl=1000.0)

        assert pnl["realized_pnl"] == 1000.0
        assert pnl["unrealized_pnl"] == 0.0
        assert pnl["total_pnl"] == 1000.0

    def test_calculate_portfolio_pnl_all_profitable(self, service):
        """Test calculation with all profitable positions."""
        positions = {
            "AAPL": PositionInfo(
                symbol="AAPL",
                quantity=100,
                avg_cost=150,
                market_value=20000,
                unrealized_pnl=5000,
                last_update=datetime.now().timestamp(),
            ),
            "GOOGL": PositionInfo(
                symbol="GOOGL",
                quantity=50,
                avg_cost=2800,
                market_value=150000,
                unrealized_pnl=10000,
                last_update=datetime.now().timestamp(),
            ),
        }

        pnl = service.calculate_portfolio_pnl(positions, realized_pnl=3000.0)

        assert pnl["realized_pnl"] == 3000.0
        assert pnl["unrealized_pnl"] == 15000.0
        assert pnl["total_pnl"] == 18000.0

    # Risk-Adjusted Metrics Tests

    def test_calculate_risk_adjusted_metrics_full(self, service):
        """Test calculation of all risk-adjusted metrics."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.015, -0.01, 0.025, -0.015]
        benchmark_returns = [0.01, 0.005, 0.015, -0.01, 0.008, -0.003, 0.01, -0.005, 0.012, -0.008]

        metrics = service.calculate_risk_adjusted_metrics(returns, benchmark_returns)

        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "information_ratio" in metrics

        # All metrics should be finite numbers
        for value in metrics.values():
            assert isinstance(value, (int, float))
            assert value != float("inf")
            assert value != float("-inf")

    def test_calculate_risk_adjusted_metrics_no_benchmark(self, service):
        """Test calculation without benchmark returns."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.015]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "information_ratio" not in metrics

    def test_calculate_risk_adjusted_metrics_insufficient_data(self, service):
        """Test calculation with insufficient data."""
        returns = [0.01]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        assert len(metrics) == 0

    def test_calculate_risk_adjusted_metrics_no_negative_returns(self, service):
        """Test Sortino ratio with no negative returns."""
        returns = [0.01, 0.02, 0.015, 0.03, 0.025]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        # Should still calculate but Sortino won't have downside deviation
        assert "calmar_ratio" in metrics

    def test_calculate_risk_adjusted_metrics_all_negative(self, service):
        """Test calculation with all negative returns."""
        returns = [-0.01, -0.02, -0.015, -0.03, -0.025]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        assert "sortino_ratio" in metrics
        assert metrics["sortino_ratio"] < 0  # Should be negative

    def test_calculate_risk_adjusted_metrics_mismatched_benchmark(self, service):
        """Test calculation with mismatched benchmark length."""
        returns = [0.01, 0.02, 0.015]
        benchmark_returns = [0.005, 0.01]  # Different length

        metrics = service.calculate_risk_adjusted_metrics(returns, benchmark_returns)

        # Information ratio should not be calculated
        assert "information_ratio" not in metrics

    def test_calculate_risk_adjusted_sortino_ratio(self, service):
        """Test Sortino ratio calculation specifically."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.005]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        assert "sortino_ratio" in metrics
        # Sortino ratio calculation depends on risk-free rate
        # With the given returns, it could be negative or positive
        assert isinstance(metrics["sortino_ratio"], (int, float))

    def test_calculate_risk_adjusted_calmar_ratio(self, service):
        """Test Calmar ratio calculation specifically."""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.015]

        metrics = service.calculate_risk_adjusted_metrics(returns)

        assert "calmar_ratio" in metrics
        # Should have a reasonable value
        assert -10 < metrics["calmar_ratio"] < 10

    def test_calculate_risk_adjusted_information_ratio(self, service):
        """Test Information ratio calculation specifically."""
        returns = [0.02, 0.01, 0.03, 0.02, 0.01]
        benchmark = [0.01, 0.005, 0.015, 0.01, 0.008]

        metrics = service.calculate_risk_adjusted_metrics(returns, benchmark)

        assert "information_ratio" in metrics
        # Portfolio outperforms benchmark
        assert metrics["information_ratio"] > 0

    # Edge Cases and Integration Tests

    def test_performance_metrics_with_volatility_edge_cases(self, service):
        """Test performance metrics with edge case volatility scenarios."""
        # Create portfolio with zero volatility (constant value)
        constant_values = [
            PortfolioValue(timestamp=float(i), portfolio_id="test", value=10000) for i in range(10)
        ]

        metrics = service.calculate_performance_metrics(
            portfolio_values=constant_values, trades=[], current_positions_count=0, period_days=10
        )

        assert metrics.total_return_percent == 0.0
        assert metrics.annualized_volatility_percent == 0.0
        assert metrics.sharpe_ratio == 0.0  # No excess return

    def test_performance_metrics_with_extreme_values(self, service):
        """Test performance metrics with extreme value changes."""
        extreme_values = [
            PortfolioValue(timestamp=1.0, portfolio_id="test", value=100),
            PortfolioValue(timestamp=2.0, portfolio_id="test", value=10000),
            PortfolioValue(timestamp=3.0, portfolio_id="test", value=50),
            PortfolioValue(timestamp=4.0, portfolio_id="test", value=20000),
        ]

        metrics = service.calculate_performance_metrics(
            portfolio_values=extreme_values, trades=[], current_positions_count=0, period_days=4
        )

        assert metrics.total_return_percent == 19900.0  # 100 to 20000
        assert metrics.max_drawdown_percent > 99  # Nearly 100% drawdown

    def test_position_weights_rounding(self, service):
        """Test that position weights sum to approximately 100%."""
        positions = {
            f"STOCK{i}": PositionInfo(
                symbol=f"STOCK{i}",
                quantity=100,
                avg_cost=100,
                market_value=10000 + i,  # Slightly different values
                unrealized_pnl=i * 10,
                last_update=datetime.now().timestamp(),
            )
            for i in range(10)
        }

        weights = service.calculate_position_weights(positions)

        total_weight = sum(w["weight_percent"] for w in weights.values())
        assert total_weight == pytest.approx(100.0, rel=0.01)

    def test_concurrent_calculations(self, service, sample_portfolio_values, sample_trades):
        """Test that service can handle concurrent calculations."""
        # This tests thread safety of calculations
        import concurrent.futures

        def calculate():
            return service.calculate_performance_metrics(
                portfolio_values=sample_portfolio_values,
                trades=sample_trades,
                current_positions_count=3,
                period_days=10,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(calculate) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.total_return_percent == first_result.total_return_percent
            assert result.sharpe_ratio == first_result.sharpe_ratio

    def test_large_dataset_performance(self, service):
        """Test performance with large dataset."""
        # Create 1000 daily values
        large_values = [
            PortfolioValue(
                timestamp=float(i),
                portfolio_id="test",
                value=10000 + (i * 10) + (i % 7 * 100),  # Some variation
            )
            for i in range(1000)
        ]

        # Should complete without error
        metrics = service.calculate_performance_metrics(
            portfolio_values=large_values, trades=[], current_positions_count=0, period_days=1000
        )

        assert metrics.data_points == 1000
        assert metrics.period_days == 1000

    def test_portfolio_values_with_metadata(self, service):
        """Test that portfolio values with metadata are handled correctly."""
        values_with_metadata = [
            PortfolioValue(
                timestamp=float(i),
                portfolio_id="test",
                value=10000 + i * 100,
                metadata={"source": "test", "version": i},
            )
            for i in range(5)
        ]

        metrics = service.calculate_performance_metrics(
            portfolio_values=values_with_metadata,
            trades=[],
            current_positions_count=0,
            period_days=5,
        )

        # Metadata should not affect calculations
        assert metrics.start_value == 10000
        assert metrics.end_value == 10400

    def test_trades_with_optional_fields(self, service, sample_portfolio_values):
        """Test that trades with various optional fields are handled."""
        trades_with_optionals = [
            TradeRecord(
                timestamp=1.0,
                order_id="1",
                symbol="AAPL",
                portfolio_id="test",
                strategy="strat1",
                operation="buy",
                status="filled",
                duration_ms=100,
                context={"test": "data"},
                submit_time=0.9,
            ),
            TradeRecord(
                timestamp=2.0,
                order_id="2",
                symbol="GOOGL",
                portfolio_id="test",
                # Minimal fields only
            ),
        ]

        metrics = service.calculate_performance_metrics(
            portfolio_values=sample_portfolio_values,
            trades=trades_with_optionals,
            current_positions_count=2,
            period_days=10,
        )

        assert metrics.total_trades == 2

    def test_custom_trading_days_per_year(self, service):
        """Test that trading days per year affects volatility calculation."""
        # Modify trading days
        service.trading_days_per_year = 365

        daily_returns = [0.01, -0.02, 0.015, -0.005, 0.02]
        volatility = service._calculate_annualized_volatility(daily_returns)

        # Should be different from default 252 days
        service.trading_days_per_year = 252
        volatility_252 = service._calculate_annualized_volatility(daily_returns)

        assert volatility != volatility_252
