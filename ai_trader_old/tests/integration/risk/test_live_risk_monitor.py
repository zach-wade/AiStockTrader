"""
Integration tests for Live Risk Monitor.

Tests real-time position monitoring, risk limit checks,
and circuit breaker activation in realistic scenarios.
"""

# Standard library imports
import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

# Third-party imports
import pytest

# Local imports
from main.events.types import RiskEvent
from main.models.common import Order, OrderSide, OrderType, Position
from main.risk_management.circuit_breakers import CircuitBreaker, CircuitBreakerType
from main.risk_management.live_risk_monitor import LiveRiskMonitor


@pytest.fixture
async def risk_monitor():
    """Create risk monitor instance."""
    config = {
        "risk": {
            "max_position_size": 10000,
            "max_portfolio_value": 100000,
            "max_daily_loss": 5000,
            "max_drawdown": 0.10,
            "position_limits": {
                "max_positions": 10,
                "max_sector_concentration": 0.30,
                "max_single_position": 0.20,
            },
        }
    }

    monitor = LiveRiskMonitor(config)
    await monitor.start()
    yield monitor
    await monitor.stop()


@pytest.fixture
def mock_portfolio():
    """Create mock portfolio with positions."""
    positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
            market_value=15500.0,
            cost_basis=15000.0,
            unrealized_pnl=500.0,
            unrealized_pnl_pct=3.33,
            realized_pnl=0.0,
            side="long",
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            quantity=50,
            avg_entry_price=2500.0,
            current_price=2450.0,
            market_value=122500.0,
            cost_basis=125000.0,
            unrealized_pnl=-2500.0,
            unrealized_pnl_pct=-2.0,
            realized_pnl=0.0,
            side="long",
        ),
        "MSFT": Position(
            symbol="MSFT",
            quantity=-30,  # Short position
            avg_entry_price=300.0,
            current_price=310.0,
            market_value=-9300.0,
            cost_basis=-9000.0,
            unrealized_pnl=-300.0,
            unrealized_pnl_pct=-3.33,
            realized_pnl=0.0,
            side="short",
        ),
    }

    portfolio = MagicMock()
    portfolio.get_positions = MagicMock(return_value=positions)
    portfolio.get_total_value = MagicMock(return_value=95000.0)
    portfolio.get_daily_pnl = MagicMock(return_value=-1500.0)
    portfolio.get_max_drawdown = MagicMock(return_value=-0.05)

    return portfolio


class TestLiveRiskMonitorIntegration:
    """Test live risk monitoring functionality."""

    @pytest.mark.asyncio
    async def test_position_limit_monitoring(self, risk_monitor, mock_portfolio):
        """Test monitoring of position limits."""
        risk_monitor.portfolio = mock_portfolio
        violations = []

        # Set up violation handler
        async def violation_handler(event: RiskEvent):
            violations.append(event)

        risk_monitor.add_violation_handler(violation_handler)

        # Add position that exceeds limit
        large_position = Position(
            symbol="TSLA",
            quantity=100,
            avg_entry_price=800.0,
            current_price=850.0,
            market_value=85000.0,  # 89% of portfolio!
            cost_basis=80000.0,
            unrealized_pnl=5000.0,
            unrealized_pnl_pct=6.25,
            realized_pnl=0.0,
            side="long",
        )

        # Check position
        is_valid = await risk_monitor.check_position_limits(large_position)

        assert not is_valid
        assert len(violations) > 0
        assert violations[0].severity == "critical"
        assert "position size" in violations[0].message.lower()

    @pytest.mark.asyncio
    async def test_real_time_pnl_monitoring(self, risk_monitor, mock_portfolio):
        """Test real-time P&L monitoring and alerts."""
        risk_monitor.portfolio = mock_portfolio
        pnl_updates = []

        # Track P&L updates
        async def pnl_handler(pnl_data: dict[str, Any]):
            pnl_updates.append(pnl_data)

        risk_monitor.add_pnl_handler(pnl_handler)

        # Start monitoring
        monitor_task = asyncio.create_task(risk_monitor.monitor_pnl(interval=0.1))

        # Simulate P&L changes
        pnl_values = [-1000, -2000, -3000, -4000, -5500]  # Exceeds daily limit

        for pnl in pnl_values:
            mock_portfolio.get_daily_pnl = MagicMock(return_value=pnl)
            await asyncio.sleep(0.15)

        # Stop monitoring
        monitor_task.cancel()

        # Verify P&L tracking
        assert len(pnl_updates) >= 4

        # Check if daily loss limit triggered
        limit_exceeded = any(update.get("daily_loss_exceeded", False) for update in pnl_updates)
        assert limit_exceeded

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, risk_monitor):
        """Test circuit breaker activation on risk events."""
        # Create circuit breakers
        breakers = {
            "drawdown": CircuitBreaker(
                name="drawdown",
                breaker_type=CircuitBreakerType.PERCENTAGE,
                threshold=0.08,  # 8% drawdown
                cooldown_seconds=300,
            ),
            "daily_loss": CircuitBreaker(
                name="daily_loss",
                breaker_type=CircuitBreakerType.ABSOLUTE,
                threshold=4000,  # $4000 daily loss
                cooldown_seconds=86400,  # 24 hours
            ),
            "volatility": CircuitBreaker(
                name="volatility",
                breaker_type=CircuitBreakerType.VOLATILITY,
                threshold=0.50,  # 50% annualized vol
                cooldown_seconds=3600,
            ),
        }

        risk_monitor.circuit_breakers = breakers

        # Track breaker trips
        trips = []

        async def trip_handler(breaker_name: str, data: dict):
            trips.append({"breaker": breaker_name, "timestamp": datetime.now(), "data": data})

        risk_monitor.add_circuit_breaker_handler(trip_handler)

        # Trigger drawdown breaker
        await risk_monitor.check_drawdown(current_drawdown=-0.09)

        # Trigger daily loss breaker
        await risk_monitor.check_daily_loss(current_loss=-4500)

        # Verify breakers tripped
        assert len(trips) == 2
        assert any(t["breaker"] == "drawdown" for t in trips)
        assert any(t["breaker"] == "daily_loss" for t in trips)

        # Verify trading is halted
        assert risk_monitor.is_trading_halted()

    @pytest.mark.asyncio
    async def test_position_concentration_monitoring(self, risk_monitor, mock_portfolio):
        """Test sector and position concentration limits."""
        # Add sector metadata to positions
        positions_with_sectors = {
            "AAPL": {"position": mock_portfolio.get_positions()["AAPL"], "sector": "Technology"},
            "GOOGL": {"position": mock_portfolio.get_positions()["GOOGL"], "sector": "Technology"},
            "MSFT": {"position": mock_portfolio.get_positions()["MSFT"], "sector": "Technology"},
            "JPM": {
                "position": Position(
                    symbol="JPM",
                    quantity=200,
                    avg_entry_price=150.0,
                    current_price=155.0,
                    market_value=31000.0,
                    cost_basis=30000.0,
                    unrealized_pnl=1000.0,
                    unrealized_pnl_pct=3.33,
                    realized_pnl=0.0,
                    side="long",
                ),
                "sector": "Financial",
            },
        }

        # Calculate sector concentration
        sector_exposure = await risk_monitor.calculate_sector_concentration(positions_with_sectors)

        # Tech sector should be over-concentrated (3 positions)
        assert sector_exposure["Technology"] > 0.60  # Over 60% in tech

        # Check concentration limits
        violations = await risk_monitor.check_concentration_limits(sector_exposure)

        assert len(violations) > 0
        assert any("Technology" in v["message"] for v in violations)

    @pytest.mark.asyncio
    async def test_correlated_risk_monitoring(self, risk_monitor):
        """Test monitoring of correlated positions."""
        # Create correlated positions
        correlated_positions = {
            "SPY": Position(
                symbol="SPY",
                quantity=100,
                avg_entry_price=400.0,
                current_price=410.0,
                market_value=41000.0,
                cost_basis=40000.0,
                unrealized_pnl=1000.0,
                unrealized_pnl_pct=2.5,
                realized_pnl=0.0,
                side="long",
            ),
            "QQQ": Position(
                symbol="QQQ",
                quantity=150,
                avg_entry_price=300.0,
                current_price=310.0,
                market_value=46500.0,
                cost_basis=45000.0,
                unrealized_pnl=1500.0,
                unrealized_pnl_pct=3.33,
                realized_pnl=0.0,
                side="long",
            ),
            "IWM": Position(
                symbol="IWM",
                quantity=200,
                avg_entry_price=200.0,
                current_price=195.0,
                market_value=39000.0,
                cost_basis=40000.0,
                unrealized_pnl=-1000.0,
                unrealized_pnl_pct=-2.5,
                realized_pnl=0.0,
                side="long",
            ),
        }

        # Mock correlation matrix
        correlation_matrix = {
            ("SPY", "QQQ"): 0.95,  # Highly correlated
            ("SPY", "IWM"): 0.85,
            ("QQQ", "IWM"): 0.80,
        }

        risk_monitor.get_correlation = MagicMock(
            side_effect=lambda s1, s2: correlation_matrix.get((s1, s2), 0)
        )

        # Check correlated risk
        correlated_risk = await risk_monitor.calculate_correlated_risk(correlated_positions)

        # Should identify high correlation risk
        assert correlated_risk["max_correlation"] > 0.9
        assert correlated_risk["risk_score"] > 0.8  # High risk score

        # Check if warning generated
        warnings = await risk_monitor.check_correlation_limits(correlated_positions)

        assert len(warnings) > 0
        assert any("correlation" in w["message"].lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_stress_test_monitoring(self, risk_monitor, mock_portfolio):
        """Test stress testing and scenario analysis."""
        # Define stress scenarios
        scenarios = [
            {
                "name": "Market Crash",
                "market_move": -0.20,  # 20% down
                "volatility_spike": 2.0,  # Double volatility
                "correlation_increase": 0.3,
            },
            {
                "name": "Flash Crash",
                "market_move": -0.10,  # 10% down
                "volatility_spike": 3.0,  # Triple volatility
                "correlation_increase": 0.5,
            },
            {
                "name": "Sector Rotation",
                "market_move": 0.0,
                "sector_moves": {"Technology": -0.15, "Financial": 0.10, "Energy": 0.20},
            },
        ]

        risk_monitor.portfolio = mock_portfolio
        stress_results = []

        # Run stress tests
        for scenario in scenarios:
            result = await risk_monitor.run_stress_test(scenario, mock_portfolio.get_positions())
            stress_results.append(result)

        # Verify stress test results
        assert len(stress_results) == 3

        # Market crash should show significant losses
        market_crash_result = stress_results[0]
        assert market_crash_result["expected_loss"] < -10000  # Significant loss
        assert market_crash_result["var_95"] < -5000

        # Check if any scenario breaches risk limits
        breaches = [r for r in stress_results if r["breaches_limits"]]
        assert len(breaches) >= 1  # At least one scenario should breach limits

    @pytest.mark.asyncio
    async def test_order_risk_validation(self, risk_monitor, mock_portfolio):
        """Test pre-trade risk validation for orders."""
        risk_monitor.portfolio = mock_portfolio

        # Test various orders
        test_orders = [
            # Normal order - should pass
            Order(
                order_id="ORD1",
                symbol="IBM",
                quantity=50,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                limit_price=150.0,
            ),
            # Large order - should fail
            Order(
                order_id="ORD2",
                symbol="AMZN",
                quantity=1000,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                limit_price=None,
            ),
            # Order that would exceed position limit - should fail
            Order(
                order_id="ORD3",
                symbol="AAPL",  # Already have position
                quantity=500,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                limit_price=155.0,
            ),
        ]

        validation_results = []

        for order in test_orders:
            result = await risk_monitor.validate_order(order)
            validation_results.append(
                {
                    "order_id": order.order_id,
                    "valid": result["is_valid"],
                    "reasons": result.get("rejection_reasons", []),
                }
            )

        # Check validation results
        assert validation_results[0]["valid"] == True  # Normal order
        assert validation_results[1]["valid"] == False  # Large order
        assert validation_results[2]["valid"] == False  # Position limit

        # Verify rejection reasons
        assert any("size" in r.lower() for r in validation_results[1]["reasons"])
        assert any("position" in r.lower() for r in validation_results[2]["reasons"])

    @pytest.mark.asyncio
    async def test_risk_metrics_aggregation(self, risk_monitor, mock_portfolio):
        """Test aggregation and reporting of risk metrics."""
        risk_monitor.portfolio = mock_portfolio

        # Start metrics collection
        collection_task = asyncio.create_task(risk_monitor.collect_risk_metrics(interval=0.1))

        # Run for a short time
        await asyncio.sleep(0.5)

        # Stop collection
        collection_task.cancel()

        # Get aggregated metrics
        metrics = risk_monitor.get_risk_metrics_summary()

        # Verify metrics collected
        assert "portfolio_var" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "position_count" in metrics
        assert "total_exposure" in metrics
        assert "risk_score" in metrics

        # Check risk score calculation
        assert 0 <= metrics["risk_score"] <= 1

        # Generate risk report
        report = risk_monitor.generate_risk_report()

        assert "summary" in report
        assert "positions" in report
        assert "alerts" in report
        assert "recommendations" in report
