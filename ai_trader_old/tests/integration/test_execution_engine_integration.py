"""
Integration tests for execution engine
"""

# Standard library imports
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest

# Local imports
from main.models.common import Order, OrderSide, OrderStatus, OrderType, Position
from main.trading_engine.core.execution_engine import (
    ExecutionEngine,
    ExecutionEngineStatus,
    ExecutionMode,
)
from main.trading_engine.core.trading_system import TradingMode


class MockBroker:
    """Mock broker for testing"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.orders = {}
        self.positions = {}
        self.is_connected = False

    async def connect(self) -> bool:
        self.is_connected = True
        return True

    async def disconnect(self):
        self.is_connected = False

    async def place_order(self, order: Order) -> str:
        order_id = f"TEST_{datetime.now().timestamp()}"
        self.orders[order_id] = order
        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_positions(self) -> dict[str, Position]:
        return self.positions

    async def get_account_info(self) -> dict[str, Any]:
        return {
            "account_id": "TEST_ACCOUNT",
            "buying_power": 100000.0,
            "cash": 100000.0,
            "portfolio_value": 100000.0,
        }


@pytest.mark.integration
@pytest.mark.asyncio
class TestExecutionEngineIntegration:
    """Test execution engine integration with all components"""

    @pytest.fixture
    async def test_config(self):
        """Test configuration"""
        return {
            "brokers": {
                "test_broker": {
                    "enabled": True,
                    "type": "mock",
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                }
            },
            "execution": {
                "fast_path_enabled": True,
                "max_order_size": 10000,
                "max_position_size": 50000,
            },
            "risk_management": {
                "max_drawdown": 0.1,
                "max_daily_loss": 0.05,
                "circuit_breaker": {"enabled": True, "max_loss_threshold": 0.02},
            },
        }

    @pytest.fixture
    async def mock_broker_interface(self):
        """Create mock broker interface"""
        return MockBroker(config={})

    @pytest.fixture
    async def execution_engine(self, test_config):
        """Create execution engine instance"""
        engine = ExecutionEngine(
            config=test_config,
            trading_mode=TradingMode.PAPER,
            execution_mode=ExecutionMode.SEMI_AUTO,
        )
        return engine

    async def test_execution_engine_initialization(self, execution_engine, test_config):
        """Test execution engine initialization"""
        # Mock broker creation
        with patch.object(execution_engine, "_create_broker_interface") as mock_create_broker:
            mock_broker = MockBroker(config={})
            mock_create_broker.return_value = mock_broker

            # Initialize engine
            success = await execution_engine.initialize()

            assert success is True
            assert execution_engine.status == ExecutionEngineStatus.READY
            assert execution_engine.session_start_time is not None
            assert len(execution_engine.trading_systems) > 0

    async def test_start_stop_trading(self, execution_engine):
        """Test starting and stopping trading operations"""
        # Mock initialization
        execution_engine.status = ExecutionEngineStatus.READY
        execution_engine.trading_systems = {
            "test_broker": Mock(
                enable_trading=AsyncMock(return_value=True),
                disable_trading=AsyncMock(return_value=True),
            )
        }

        # Start trading
        success = await execution_engine.start_trading()
        assert success is True
        assert execution_engine.status == ExecutionEngineStatus.ACTIVE

        # Pause trading
        await execution_engine.pause_trading()
        assert execution_engine.status == ExecutionEngineStatus.PAUSED

        # Resume trading
        await execution_engine.resume_trading()
        assert execution_engine.status == ExecutionEngineStatus.ACTIVE

    async def test_cross_system_order_submission(self, execution_engine):
        """Test cross-system order submission"""
        # Setup mock trading system
        mock_trading_system = Mock()
        mock_trading_system.submit_order = AsyncMock(return_value="TEST_ORDER_123")
        mock_trading_system.get_system_status = AsyncMock(
            return_value={"status": "running", "trading_enabled": True}
        )

        execution_engine.trading_systems = {"test_broker": mock_trading_system}
        execution_engine.active_brokers = {"test_broker"}

        # Create test order
        test_order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        )

        # Submit order
        order_id = await execution_engine.submit_cross_system_order(test_order)

        assert order_id == "TEST_ORDER_123"
        assert execution_engine.session_metrics["total_orders_submitted"] == 1
        mock_trading_system.submit_order.assert_called_once()

    async def test_emergency_stop(self, execution_engine):
        """Test emergency stop functionality"""
        # Setup mock trading systems
        mock_system1 = Mock(disable_trading=AsyncMock())
        mock_system2 = Mock(disable_trading=AsyncMock())

        execution_engine.trading_systems = {"broker1": mock_system1, "broker2": mock_system2}

        # Execute emergency stop
        await execution_engine.emergency_stop()

        assert execution_engine.emergency_stop_active is True
        assert execution_engine.status == ExecutionEngineStatus.EMERGENCY
        mock_system1.disable_trading.assert_called_once()
        mock_system2.disable_trading.assert_called_once()

    async def test_emergency_liquidation(self, execution_engine):
        """Test emergency liquidation functionality"""
        # Setup mock position
        mock_position = Position(symbol="AAPL", quantity=100, avg_price=150.0, current_price=155.0)

        mock_trading_system = Mock()
        mock_trading_system.position_manager = Mock()
        mock_trading_system.position_manager.get_all_positions = AsyncMock(
            return_value=[mock_position]
        )

        execution_engine.trading_systems = {"test_broker": mock_trading_system}
        execution_engine.submit_cross_system_order = AsyncMock(return_value="LIQUIDATION_ORDER")

        # Execute emergency liquidation
        await execution_engine.emergency_liquidate_all()

        assert execution_engine.emergency_liquidation_active is True
        assert execution_engine.status == ExecutionEngineStatus.EMERGENCY
        execution_engine.submit_cross_system_order.assert_called_once()

    async def test_position_synchronization(self, execution_engine):
        """Test position synchronization across systems"""
        # Setup mock trading systems with positions
        mock_system1 = Mock()
        mock_system1.position_manager = Mock()
        mock_system1.position_manager.get_position_summary = AsyncMock(
            return_value={"position_count": 3}
        )

        mock_system2 = Mock()
        mock_system2.position_manager = Mock()
        mock_system2.position_manager.get_position_summary = AsyncMock(
            return_value={"position_count": 2}
        )

        execution_engine.trading_systems = {"broker1": mock_system1, "broker2": mock_system2}

        # Synchronize positions
        await execution_engine._synchronize_positions()

        assert execution_engine.session_metrics["active_positions"] == 5

    async def test_comprehensive_status(self, execution_engine):
        """Test comprehensive status reporting"""
        # Setup mock data
        execution_engine.status = ExecutionEngineStatus.ACTIVE
        execution_engine.session_start_time = datetime.now(UTC)
        execution_engine.active_brokers = {"broker1", "broker2"}
        execution_engine.session_metrics = {
            "total_orders_submitted": 10,
            "total_orders_filled": 8,
            "total_orders_cancelled": 2,
            "active_positions": 5,
            "total_realized_pnl": 1500.0,
        }

        mock_system = Mock()
        mock_system.get_system_status = AsyncMock(
            return_value={"status": "running", "trading_enabled": True, "active_orders": 2}
        )

        execution_engine.trading_systems = {"broker1": mock_system}

        # Get status
        status = await execution_engine.get_comprehensive_status()

        assert status["engine_status"] == "active"
        assert status["trading_mode"] == "paper"
        assert "broker1" in status["active_brokers"]
        assert status["session_metrics"]["total_orders_submitted"] == 10
        assert "broker1" in status["trading_systems"]

    async def test_performance_metrics_update(self, execution_engine):
        """Test performance metrics update"""
        # Setup mock trading system
        mock_system = Mock()
        mock_system.get_system_status = AsyncMock(
            return_value={
                "status": "running",
                "metrics": {"filled_orders": 10, "cancelled_orders": 2},
                "position_summary": {"total_unrealized_pnl": 2500.0},
            }
        )

        execution_engine.trading_systems = {"test_broker": mock_system}

        # Update metrics
        await execution_engine._update_performance_metrics()

        assert execution_engine.session_metrics["total_unrealized_pnl"] == 2500.0

    async def test_broker_selection_logic(self, execution_engine):
        """Test optimal broker selection for orders"""
        # Setup multiple brokers with different statuses
        mock_system1 = Mock()
        mock_system1.get_system_status = AsyncMock(
            return_value={"status": "running", "trading_enabled": True}
        )

        mock_system2 = Mock()
        mock_system2.get_system_status = AsyncMock(
            return_value={"status": "running", "trading_enabled": False}  # Not available
        )

        execution_engine.trading_systems = {"broker1": mock_system1, "broker2": mock_system2}
        execution_engine.active_brokers = {"broker1", "broker2"}

        # Create test order
        test_order = Order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100, order_type=OrderType.MARKET
        )

        # Test with preferred broker that's not available
        selected = await execution_engine._select_optimal_broker(test_order, "broker2")
        assert selected == "broker1"  # Should fall back to available broker

        # Test with available preferred broker
        selected = await execution_engine._select_optimal_broker(test_order, "broker1")
        assert selected == "broker1"

    async def test_system_health_monitoring(self, execution_engine):
        """Test system health monitoring functionality"""
        # Setup mock trading systems
        mock_system1 = Mock()
        mock_system1.get_system_status = AsyncMock(return_value={"status": "running"})

        mock_system2 = Mock()
        mock_system2.get_system_status = AsyncMock(return_value={"status": "error"})

        execution_engine.trading_systems = {"broker1": mock_system1, "broker2": mock_system2}

        # Perform health check
        health_ok = await execution_engine._perform_health_checks()

        assert health_ok is True  # At least one system is running


@pytest.mark.integration
@pytest.mark.asyncio
class TestExecutionManagerIntegration:
    """Test execution manager integration"""

    @pytest.fixture
    async def mock_orchestrator(self, test_config):
        """Create mock orchestrator"""
        orchestrator = Mock()
        orchestrator.config = test_config
        orchestrator.mode = TradingMode.PAPER
        return orchestrator

    @pytest.fixture
    async def execution_manager(self, mock_orchestrator):
        """Create execution manager instance"""
        # Local imports
        from main.orchestration.managers.execution_manager import ExecutionManager

        return ExecutionManager(mock_orchestrator)

    async def test_signal_processing_workflow(self, execution_manager):
        """Test complete signal processing workflow"""
        # Mock execution engine
        execution_manager.execution_engine = Mock()
        execution_manager.execution_engine.submit_order = AsyncMock(return_value="ORDER_123")

        # Mock portfolio manager
        execution_manager.portfolio_manager = Mock()

        # Enable order acceptance
        execution_manager.accepting_orders = True
        execution_manager.status = ExecutionStatus.READY

        # Create test signal
        test_signal = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "strategy": "momentum",
            "confidence": 0.85,
        }

        # Process signal
        order_id = await execution_manager.process_signal(test_signal)

        assert order_id == "ORDER_123"
        assert execution_manager.metrics.total_orders == 1
        execution_manager.execution_engine.submit_order.assert_called_once()

    async def test_position_management(self, execution_manager):
        """Test position management functionality"""
        # Mock portfolio manager
        mock_portfolio = Mock()
        mock_portfolio.get_position = AsyncMock(
            return_value={
                "symbol": "AAPL",
                "quantity": 100,
                "avg_price": 150.0,
                "unrealized_pnl": 500.0,
            }
        )
        mock_portfolio.get_all_positions = AsyncMock(
            return_value={"AAPL": {"quantity": 100}, "GOOGL": {"quantity": 50}}
        )

        execution_manager.portfolio_manager = mock_portfolio

        # Get single position
        position = await execution_manager.get_position("AAPL")
        assert position["symbol"] == "AAPL"
        assert position["quantity"] == 100

        # Get all positions
        positions = await execution_manager.get_all_positions()
        assert len(positions) == 2
        assert "AAPL" in positions

    async def test_risk_integration(self, execution_manager):
        """Test risk management integration"""
        # Mock risk components
        mock_circuit_breaker = Mock()
        mock_circuit_breaker.check_order_allowed = AsyncMock(return_value=False)

        execution_manager.circuit_breaker = mock_circuit_breaker
        execution_manager.accepting_orders = True
        execution_manager.status = ExecutionStatus.READY

        # Create test signal
        test_signal = {"symbol": "AAPL", "side": "buy", "quantity": 1000}  # Large order

        # Process signal - should be rejected by risk
        order_id = await execution_manager.process_signal(test_signal)

        assert order_id is None
        mock_circuit_breaker.check_order_allowed.assert_called_once()
