"""
Comprehensive unit tests for StrategyAnalyticsService.

Tests all strategy analytics functionality including performance metrics,
win rates, profit factors, risk-reward ratios, and strategy comparisons.
"""

from datetime import datetime

import pytest

from src.domain.services.strategy_analytics_service import (
    StrategyAnalyticsService,
    StrategyTradeRecord,
)


class TestStrategyAnalyticsService:
    """Test suite for StrategyAnalyticsService."""

    @pytest.fixture
    def service(self):
        """Create a StrategyAnalyticsService instance."""
        return StrategyAnalyticsService()

    @pytest.fixture
    def sample_winning_trades(self):
        """Create sample winning trade records."""
        base_time = datetime.now().timestamp()
        return [
            StrategyTradeRecord(
                timestamp=base_time + i * 3600,
                order_id=f"win_{i}",
                symbol=f"STOCK{i % 3}",
                strategy="test_strategy",
                pnl=100 + i * 50,  # Increasing profits
                duration_seconds=300 + i * 60,
                side="buy" if i % 2 == 0 else "sell",
                quantity=100,
                price=150 + i * 5,
            )
            for i in range(5)
        ]

    @pytest.fixture
    def sample_losing_trades(self):
        """Create sample losing trade records."""
        base_time = datetime.now().timestamp()
        return [
            StrategyTradeRecord(
                timestamp=base_time + i * 3600,
                order_id=f"loss_{i}",
                symbol=f"STOCK{i % 2}",
                strategy="test_strategy",
                pnl=-(50 + i * 25),  # Increasing losses
                duration_seconds=200 + i * 40,
                side="sell" if i % 2 == 0 else "buy",
                quantity=50,
                price=140 + i * 3,
            )
            for i in range(3)
        ]

    @pytest.fixture
    def sample_mixed_trades(self, sample_winning_trades, sample_losing_trades):
        """Create mixed winning and losing trades."""
        return sample_winning_trades + sample_losing_trades

    @pytest.fixture
    def sample_strategies_data(self):
        """Create sample data for multiple strategies."""
        base_time = datetime.now().timestamp()

        return {
            "strategy_a": [
                StrategyTradeRecord(
                    timestamp=base_time + i * 3600,
                    order_id=f"a_{i}",
                    symbol="AAPL",
                    strategy="strategy_a",
                    pnl=100 if i % 2 == 0 else -50,
                    duration_seconds=300,
                )
                for i in range(10)
            ],
            "strategy_b": [
                StrategyTradeRecord(
                    timestamp=base_time + i * 3600,
                    order_id=f"b_{i}",
                    symbol="GOOGL",
                    strategy="strategy_b",
                    pnl=200 if i % 3 == 0 else -75,
                    duration_seconds=450,
                )
                for i in range(15)
            ],
            "strategy_c": [
                StrategyTradeRecord(
                    timestamp=base_time + i * 3600,
                    order_id=f"c_{i}",
                    symbol="MSFT",
                    strategy="strategy_c",
                    pnl=50 if i % 4 == 0 else -25,
                    duration_seconds=200,
                )
                for i in range(8)
            ],
        }

    # Initialization Tests

    def test_initialization_default(self):
        """Test default initialization."""
        service = StrategyAnalyticsService()
        assert service.min_trades_for_statistics == 10

    # Performance Metrics Tests

    def test_calculate_strategy_performance_with_trades(self, service, sample_mixed_trades):
        """Test calculation of performance metrics with mixed trades."""
        metrics = service.calculate_strategy_performance(
            strategy_name="test_strategy", trades=sample_mixed_trades
        )

        # Calculate expected total P&L
        # Winning trades: 100, 150, 200, 250, 300 = 1000
        # Losing trades: -50, -75, -100 = -225
        # Total: 1000 - 225 = 775

        assert metrics.strategy_name == "test_strategy"
        assert metrics.total_trades == 8
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 3
        assert metrics.win_rate_percent == 62.5
        assert metrics.total_pnl == 775  # Corrected sum of all P&L
        assert metrics.avg_pnl_per_trade == pytest.approx(96.875, rel=0.01)
        assert metrics.symbols_traded == 3
        assert metrics.profit_factor > 0
        assert metrics.expectancy != 0
        assert metrics.max_drawdown >= 0

    def test_calculate_strategy_performance_no_trades(self, service):
        """Test calculation with no trades."""
        metrics = service.calculate_strategy_performance(strategy_name="empty_strategy", trades=[])

        assert metrics.strategy_name == "empty_strategy"
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate_percent == 0
        assert metrics.total_pnl == 0
        assert metrics.avg_pnl_per_trade == 0
        assert metrics.profit_factor == 0
        assert metrics.last_trade_time is None

    def test_calculate_strategy_performance_all_winning(self, service, sample_winning_trades):
        """Test calculation with all winning trades."""
        metrics = service.calculate_strategy_performance(
            strategy_name="winning_strategy", trades=sample_winning_trades
        )

        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 0
        assert metrics.win_rate_percent == 100.0
        assert metrics.total_pnl > 0
        assert metrics.avg_loss == 0
        assert metrics.profit_factor == float("inf")

    def test_calculate_strategy_performance_all_losing(self, service, sample_losing_trades):
        """Test calculation with all losing trades."""
        metrics = service.calculate_strategy_performance(
            strategy_name="losing_strategy", trades=sample_losing_trades
        )

        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 3
        assert metrics.win_rate_percent == 0.0
        assert metrics.total_pnl < 0
        assert metrics.avg_win == 0
        assert metrics.profit_factor == 0

    def test_calculate_strategy_performance_neutral_trades(self, service):
        """Test calculation with neutral (zero P&L) trades."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i),
                order_id=f"neutral_{i}",
                symbol="AAPL",
                strategy="neutral",
                pnl=0.0,
                duration_seconds=300,
            )
            for i in range(5)
        ]

        metrics = service.calculate_strategy_performance(
            strategy_name="neutral_strategy", trades=trades
        )

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate_percent == 0
        assert metrics.total_pnl == 0

    def test_calculate_strategy_performance_with_none_pnl(self, service):
        """Test calculation with trades having None P&L."""
        trades = [
            StrategyTradeRecord(
                timestamp=1.0, order_id="1", symbol="AAPL", strategy="test", pnl=100
            ),
            StrategyTradeRecord(
                timestamp=2.0,
                order_id="2",
                symbol="GOOGL",
                strategy="test",
                pnl=None,  # None P&L
            ),
            StrategyTradeRecord(
                timestamp=3.0, order_id="3", symbol="MSFT", strategy="test", pnl=-50
            ),
        ]

        metrics = service.calculate_strategy_performance(strategy_name="test", trades=trades)

        assert metrics.total_trades == 3
        assert metrics.total_pnl == 50  # 100 - 50, None ignored

    # Max Consecutive Tests

    def test_calculate_max_consecutive_wins(self, service):
        """Test calculation of maximum consecutive wins."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
            )
            for i, pnl in enumerate([100, 50, 75, -30, 40, 60, 80, -20, -10, 30])
        ]

        max_wins = service._calculate_max_consecutive(trades, lambda t: t.pnl and t.pnl > 0)

        assert max_wins == 3  # Three consecutive wins (40, 60, 80)

    def test_calculate_max_consecutive_losses(self, service):
        """Test calculation of maximum consecutive losses."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
            )
            for i, pnl in enumerate([100, -50, -75, -30, 40, -60, -80, -20, 10])
        ]

        max_losses = service._calculate_max_consecutive(trades, lambda t: t.pnl and t.pnl < 0)

        assert max_losses == 3  # Three consecutive losses (-60, -80, -20)

    def test_calculate_max_consecutive_no_matches(self, service):
        """Test max consecutive with no matching trades."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=100
            )
            for i in range(5)
        ]

        max_losses = service._calculate_max_consecutive(trades, lambda t: t.pnl and t.pnl < 0)

        assert max_losses == 0

    # Expectancy Tests

    def test_calculate_expectancy_positive(self, service):
        """Test calculation of positive expectancy."""
        expectancy = service._calculate_expectancy(0.6, 100, 50)

        # (0.6 * 100) - (0.4 * 50) = 60 - 20 = 40
        assert expectancy == 40

    def test_calculate_expectancy_negative(self, service):
        """Test calculation of negative expectancy."""
        expectancy = service._calculate_expectancy(0.3, 100, 150)

        # (0.3 * 100) - (0.7 * 150) = 30 - 105 = -75
        assert expectancy == -75

    def test_calculate_expectancy_zero_loss(self, service):
        """Test expectancy with zero average loss."""
        expectancy = service._calculate_expectancy(0.8, 100, 0)

        # (0.8 * 100) - (0.2 * 0) = 80
        assert expectancy == 80

    def test_calculate_expectancy_breakeven(self, service):
        """Test expectancy at breakeven."""
        expectancy = service._calculate_expectancy(0.5, 100, 100)

        # (0.5 * 100) - (0.5 * 100) = 0
        assert expectancy == 0

    # Strategy Drawdown Tests

    def test_calculate_strategy_drawdown_normal(self, service):
        """Test calculation of normal strategy drawdown."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
            )
            for i, pnl in enumerate([100, 200, -150, -100, 150, -50, 100])
        ]

        drawdown = service._calculate_strategy_drawdown(trades)

        # Peak at 300, drawdown to 50 = 250
        assert drawdown == 250

    def test_calculate_strategy_drawdown_no_drawdown(self, service):
        """Test drawdown with only winning trades."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=100
            )
            for i in range(5)
        ]

        drawdown = service._calculate_strategy_drawdown(trades)
        assert drawdown == 0

    def test_calculate_strategy_drawdown_empty(self, service):
        """Test drawdown with no trades."""
        drawdown = service._calculate_strategy_drawdown([])
        assert drawdown == 0

    def test_calculate_strategy_drawdown_recovery(self, service):
        """Test drawdown with recovery."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
            )
            for i, pnl in enumerate([100, 200, -300, 150, 250, 100])
        ]

        drawdown = service._calculate_strategy_drawdown(trades)

        # Peak at 300, drawdown to 0 = 300
        assert drawdown == 300

    def test_calculate_strategy_drawdown_unsorted(self, service):
        """Test that trades are sorted before drawdown calculation."""
        trades = [
            StrategyTradeRecord(
                timestamp=3.0, order_id="3", symbol="TEST", strategy="test", pnl=-100
            ),
            StrategyTradeRecord(
                timestamp=1.0, order_id="1", symbol="TEST", strategy="test", pnl=100
            ),
            StrategyTradeRecord(
                timestamp=2.0, order_id="2", symbol="TEST", strategy="test", pnl=200
            ),
        ]

        drawdown = service._calculate_strategy_drawdown(trades)

        # After sorting: 100, 200, -100
        # Peak at 300, ends at 200 = drawdown of 100
        assert drawdown == 100

    # Risk-Reward Ratio Tests

    def test_calculate_risk_reward_ratio_normal(self, service, sample_mixed_trades):
        """Test calculation of normal risk-reward ratio."""
        ratio = service.calculate_risk_reward_ratio(sample_mixed_trades)

        # Should be positive (avg win / avg loss)
        assert ratio > 0

    def test_calculate_risk_reward_ratio_no_winners(self, service, sample_losing_trades):
        """Test risk-reward ratio with no winning trades."""
        ratio = service.calculate_risk_reward_ratio(sample_losing_trades)
        assert ratio == 0

    def test_calculate_risk_reward_ratio_no_losers(self, service, sample_winning_trades):
        """Test risk-reward ratio with no losing trades."""
        ratio = service.calculate_risk_reward_ratio(sample_winning_trades)
        assert ratio == 0

    def test_calculate_risk_reward_ratio_empty(self, service):
        """Test risk-reward ratio with no trades."""
        ratio = service.calculate_risk_reward_ratio([])
        assert ratio == 0

    def test_calculate_risk_reward_ratio_equal(self, service):
        """Test risk-reward ratio with equal wins and losses."""
        trades = [
            StrategyTradeRecord(
                timestamp=1.0, order_id="1", symbol="TEST", strategy="test", pnl=100
            ),
            StrategyTradeRecord(
                timestamp=2.0, order_id="2", symbol="TEST", strategy="test", pnl=-100
            ),
        ]

        ratio = service.calculate_risk_reward_ratio(trades)
        assert ratio == 1.0

    # Kelly Criterion Tests

    def test_calculate_kelly_criterion_positive(self, service):
        """Test Kelly criterion with positive expectancy."""
        kelly = service.calculate_kelly_criterion(60, 100, 50)

        # p=0.6, q=0.4, b=2
        # kelly = (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        # But capped at 0.25
        assert kelly == 0.25

    def test_calculate_kelly_criterion_negative(self, service):
        """Test Kelly criterion with negative expectancy."""
        kelly = service.calculate_kelly_criterion(30, 100, 150)

        # Negative expectancy should return 0
        assert kelly == 0

    def test_calculate_kelly_criterion_zero_loss(self, service):
        """Test Kelly criterion with zero average loss."""
        kelly = service.calculate_kelly_criterion(60, 100, 0)
        assert kelly == 0

    def test_calculate_kelly_criterion_breakeven(self, service):
        """Test Kelly criterion at breakeven."""
        kelly = service.calculate_kelly_criterion(50, 100, 100)

        # p=0.5, q=0.5, b=1
        # kelly = (0.5 * 1 - 0.5) / 1 = 0
        assert kelly == 0

    def test_calculate_kelly_criterion_high_win_rate(self, service):
        """Test Kelly criterion with very high win rate."""
        kelly = service.calculate_kelly_criterion(90, 100, 100)

        # p=0.9, q=0.1, b=1
        # kelly = (0.9 * 1 - 0.1) / 1 = 0.8
        # But capped at 0.25
        assert kelly == 0.25

    def test_calculate_kelly_criterion_edge_cases(self, service):
        """Test Kelly criterion edge cases."""
        # Win rate > 100 (should handle gracefully)
        kelly = service.calculate_kelly_criterion(110, 100, 50)
        assert kelly == 0.25  # Capped

        # Negative win rate
        kelly = service.calculate_kelly_criterion(-10, 100, 50)
        assert kelly == 0

    # Trade Distribution Analysis Tests

    def test_analyze_trade_distribution_normal(self, service, sample_mixed_trades):
        """Test analysis of normal trade distribution."""
        analysis = service.analyze_trade_distribution(sample_mixed_trades)

        assert "count" in analysis
        assert "mean" in analysis
        assert "median" in analysis
        assert "std_dev" in analysis
        assert "min" in analysis
        assert "max" in analysis
        assert "skewness" in analysis
        assert "kurtosis" in analysis
        assert "percentiles" in analysis

        assert analysis["count"] == 8
        assert analysis["min"] < 0  # Has losses
        assert analysis["max"] > 0  # Has wins

    def test_analyze_trade_distribution_empty(self, service):
        """Test analysis with no trades."""
        analysis = service.analyze_trade_distribution([])
        assert analysis == {}

    def test_analyze_trade_distribution_no_pnl(self, service):
        """Test analysis with trades having no P&L."""
        trades = [
            StrategyTradeRecord(
                timestamp=1.0, order_id="1", symbol="TEST", strategy="test", pnl=None
            )
            for _ in range(5)
        ]

        analysis = service.analyze_trade_distribution(trades)
        assert analysis == {}

    def test_analyze_trade_distribution_single_trade(self, service):
        """Test analysis with single trade."""
        trades = [
            StrategyTradeRecord(
                timestamp=1.0, order_id="1", symbol="TEST", strategy="test", pnl=100
            )
        ]

        analysis = service.analyze_trade_distribution(trades)

        assert analysis["count"] == 1
        assert analysis["mean"] == 100
        assert analysis["median"] == 100
        assert analysis["std_dev"] == 0
        assert analysis["min"] == 100
        assert analysis["max"] == 100

    def test_analyze_trade_distribution_percentiles(self, service):
        """Test percentile calculations in distribution analysis."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=i * 10
            )
            for i in range(100)
        ]

        analysis = service.analyze_trade_distribution(trades)
        percentiles = analysis["percentiles"]

        assert percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"]
        assert percentiles["p75"] <= percentiles["p95"]

    # Skewness and Kurtosis Tests

    def test_calculate_skewness_normal(self, service):
        """Test skewness calculation for normal distribution."""
        # Symmetric distribution should have skewness near 0
        values = [-2, -1, 0, 1, 2]
        skewness = service._calculate_skewness(values)
        assert abs(skewness) < 0.1

    def test_calculate_skewness_positive(self, service):
        """Test skewness calculation for positively skewed distribution."""
        # Right-skewed distribution
        values = [1, 2, 3, 4, 5, 10, 20, 30]
        skewness = service._calculate_skewness(values)
        assert skewness > 0

    def test_calculate_skewness_negative(self, service):
        """Test skewness calculation for negatively skewed distribution."""
        # Left-skewed distribution
        values = [-30, -20, -10, 1, 2, 3, 4, 5]
        skewness = service._calculate_skewness(values)
        assert skewness < 0

    def test_calculate_skewness_insufficient_data(self, service):
        """Test skewness with insufficient data."""
        values = [1, 2]
        skewness = service._calculate_skewness(values)
        assert skewness == 0

    def test_calculate_kurtosis_normal(self, service):
        """Test kurtosis calculation."""
        # More values for meaningful kurtosis
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        kurtosis = service._calculate_kurtosis(values)
        assert isinstance(kurtosis, float)

    def test_calculate_kurtosis_insufficient_data(self, service):
        """Test kurtosis with insufficient data."""
        values = [1, 2, 3]
        kurtosis = service._calculate_kurtosis(values)
        assert kurtosis == 0

    def test_calculate_kurtosis_zero_std(self, service):
        """Test kurtosis with zero standard deviation."""
        values = [5, 5, 5, 5, 5]
        kurtosis = service._calculate_kurtosis(values)
        assert kurtosis == 0

    # Strategy Comparison Tests

    def test_compare_strategies_normal(self, service, sample_strategies_data):
        """Test normal strategy comparison."""
        comparison = service.compare_strategies(sample_strategies_data)

        assert len(comparison.strategies) == 3
        assert comparison.best_performer in sample_strategies_data.keys()
        assert comparison.worst_performer in sample_strategies_data.keys()
        assert comparison.most_active in sample_strategies_data.keys()
        assert comparison.most_consistent in sample_strategies_data.keys()
        assert comparison.highest_win_rate in sample_strategies_data.keys()
        assert comparison.highest_profit_factor in sample_strategies_data.keys()

    def test_compare_strategies_empty(self, service):
        """Test comparison with no strategies."""
        with pytest.raises(ValueError, match="No strategies to compare"):
            service.compare_strategies({})

    def test_compare_strategies_single(self, service):
        """Test comparison with single strategy."""
        single_strategy = {
            "only_strategy": [
                StrategyTradeRecord(
                    timestamp=1.0, order_id="1", symbol="TEST", strategy="only", pnl=100
                )
            ]
        }

        comparison = service.compare_strategies(single_strategy)

        # All metrics should point to the only strategy
        assert comparison.best_performer == "only_strategy"
        assert comparison.worst_performer == "only_strategy"
        assert comparison.most_active == "only_strategy"

    def test_compare_strategies_with_period(self, service, sample_strategies_data):
        """Test comparison with specific time period."""
        base_time = datetime.now().timestamp()
        period = (base_time, base_time + 5 * 3600)  # First 5 hours

        comparison = service.compare_strategies(sample_strategies_data, period)

        assert comparison.comparison_period == period
        assert len(comparison.strategies) == 3

    def test_compare_strategies_no_trades_in_period(self, service):
        """Test comparison when no trades fall in period."""
        future_time = datetime.now().timestamp() + 1000000
        period = (future_time, future_time + 3600)

        strategies = {
            "strategy_a": [
                StrategyTradeRecord(
                    timestamp=1.0, order_id="1", symbol="TEST", strategy="a", pnl=100
                )
            ]
        }

        comparison = service.compare_strategies(strategies, period)

        # Should still work but with empty metrics
        assert comparison.comparison_period == period

    def test_compare_strategies_consistency_calculation(self, service):
        """Test consistency score calculation in comparison."""
        strategies = {
            "consistent": [
                StrategyTradeRecord(
                    timestamp=float(i),
                    order_id=str(i),
                    symbol="TEST",
                    strategy="consistent",
                    pnl=10,  # Steady profits
                )
                for i in range(10)
            ],
            "volatile": [
                StrategyTradeRecord(
                    timestamp=float(i),
                    order_id=str(i),
                    symbol="TEST",
                    strategy="volatile",
                    pnl=100 if i % 2 == 0 else -90,  # High variance
                )
                for i in range(10)
            ],
        }

        comparison = service.compare_strategies(strategies)

        # Consistent strategy should be marked as most consistent
        assert comparison.most_consistent == "consistent"

    def test_compare_strategies_infinite_profit_factor(self, service):
        """Test comparison handling infinite profit factor."""
        strategies = {
            "all_wins": [
                StrategyTradeRecord(
                    timestamp=float(i), order_id=str(i), symbol="TEST", strategy="all_wins", pnl=100
                )
                for i in range(5)
            ],
            "mixed": [
                StrategyTradeRecord(
                    timestamp=float(i),
                    order_id=str(i),
                    symbol="TEST",
                    strategy="mixed",
                    pnl=100 if i % 2 == 0 else -50,
                )
                for i in range(5)
            ],
        }

        comparison = service.compare_strategies(strategies)

        # Should handle infinite profit factor gracefully
        assert comparison.highest_profit_factor == "mixed"  # Not infinite

    # Edge Cases and Integration Tests

    def test_performance_metrics_with_all_fields(self, service):
        """Test performance metrics with all optional fields populated."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i),
                order_id=f"order_{i}",
                symbol=f"SYMBOL_{i}",
                strategy="complete",
                pnl=100 if i % 2 == 0 else -50,
                duration_seconds=300.5,
                side="buy" if i % 2 == 0 else "sell",
                quantity=100.5,
                price=150.75,
            )
            for i in range(20)
        ]

        metrics = service.calculate_strategy_performance("complete", trades)

        assert metrics.total_trades == 20
        assert metrics.avg_trade_duration_seconds == 300.5
        assert metrics.symbols_traded == 20  # Each trade has unique symbol

    def test_performance_metrics_recovery_factor(self, service):
        """Test recovery factor calculation."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
            )
            for i, pnl in enumerate([100, 200, -300, 150, 250])  # Net: 400, max DD: 300
        ]

        metrics = service.calculate_strategy_performance("test", trades)

        # Recovery factor = total_pnl / max_drawdown = 400 / 300
        assert metrics.recovery_factor == pytest.approx(1.333, rel=0.01)

    def test_performance_metrics_zero_drawdown_recovery(self, service):
        """Test recovery factor with zero drawdown."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=100
            )
            for i in range(5)
        ]

        metrics = service.calculate_strategy_performance("test", trades)

        # No drawdown means recovery factor = 0
        assert metrics.recovery_factor == 0

    def test_large_dataset_performance(self, service):
        """Test performance with large dataset."""
        # Create 10000 trades
        large_trades = [
            StrategyTradeRecord(
                timestamp=float(i),
                order_id=str(i),
                symbol=f"SYMBOL_{i % 100}",
                strategy="large",
                pnl=(i % 7 - 3) * 50,  # Varying P&L
                duration_seconds=i % 1000,
            )
            for i in range(10000)
        ]

        # Should complete without error
        metrics = service.calculate_strategy_performance("large", large_trades)

        assert metrics.total_trades == 10000
        assert metrics.symbols_traded == 100

    def test_mixed_none_values(self, service):
        """Test handling of mixed None values in trade fields."""
        trades = [
            StrategyTradeRecord(
                timestamp=1.0,
                order_id="1",
                symbol="TEST",
                strategy="test",
                pnl=100,
                duration_seconds=None,
            ),
            StrategyTradeRecord(
                timestamp=2.0,
                order_id="2",
                symbol="TEST",
                strategy="test",
                pnl=None,
                duration_seconds=200,
            ),
            StrategyTradeRecord(
                timestamp=3.0,
                order_id="3",
                symbol=None,
                strategy="test",
                pnl=50,
                duration_seconds=300,
            ),
        ]

        metrics = service.calculate_strategy_performance("test", trades)

        # Should handle None values gracefully
        assert metrics.total_trades == 3
        assert metrics.avg_trade_duration_seconds == 250  # (200 + 300) / 2

    def test_concurrent_calculations(self, service, sample_mixed_trades):
        """Test thread safety of calculations."""
        import concurrent.futures

        def calculate():
            return service.calculate_strategy_performance("concurrent_test", sample_mixed_trades)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(calculate) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.total_pnl == first_result.total_pnl
            assert result.win_rate_percent == first_result.win_rate_percent

    def test_floating_point_precision(self, service):
        """Test handling of floating point precision issues."""
        trades = [
            StrategyTradeRecord(
                timestamp=float(i),
                order_id=str(i),
                symbol="TEST",
                strategy="test",
                pnl=0.1 + 0.2 if i % 2 == 0 else -(0.1 + 0.2),
            )
            for i in range(100)
        ]

        metrics = service.calculate_strategy_performance("test", trades)

        # Should handle floating point arithmetic correctly
        assert abs(metrics.total_pnl) < 0.01  # Should be near zero

    def test_strategy_name_preservation(self, service, sample_mixed_trades):
        """Test that strategy names are preserved correctly."""
        special_name = "Strategy-123_v2.0 (test)"
        metrics = service.calculate_strategy_performance(special_name, sample_mixed_trades)

        assert metrics.strategy_name == special_name

    def test_timestamp_ordering_importance(self, service):
        """Test that timestamp ordering affects consecutive calculations."""
        # Create trades with specific win/loss pattern
        trades = []
        patterns = [100, 100, -50, 100, 100, 100, -50, -50]

        for i, pnl in enumerate(patterns):
            trades.append(
                StrategyTradeRecord(
                    timestamp=float(i), order_id=str(i), symbol="TEST", strategy="test", pnl=pnl
                )
            )

        metrics = service.calculate_strategy_performance("test", trades)
        assert metrics.max_consecutive_wins == 3  # Middle sequence of 3 wins
        assert metrics.max_consecutive_losses == 2  # End sequence of 2 losses
