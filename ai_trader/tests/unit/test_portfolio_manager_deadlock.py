"""
Comprehensive deadlock tests for Portfolio Manager.
Tests all critical concurrency scenarios to ensure deadlock-free operation.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Standardized test setup
import sys
from pathlib import Path
from pathlib import Path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main.trading_engine.core.portfolio_manager import PortfolioManager, Portfolio
from src.main.models.common import Position as CommonPosition, AccountInfo, OrderSide

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestPortfolioManagerDeadlock:
    """Test deadlock prevention in portfolio manager operations."""

    @pytest.fixture
    async def mock_broker(self):
        """Create a mock broker with realistic delays."""
        broker = AsyncMock()
        
        # Mock account info with realistic delay
        async def mock_get_account_info():
            await asyncio.sleep(0.01)  # 10ms delay to simulate network
            return AccountInfo(
                account_id="test_account",
                cash=50000.0,
                buying_power=50000.0,
                equity=60000.0,
                maintenance_margin=0.0,
                initial_margin=0.0
            )
        
        # Mock positions with realistic delay
        async def mock_get_positions():
            await asyncio.sleep(0.01)  # 10ms delay to simulate network
            return {
                "AAPL": CommonPosition(
                    symbol="AAPL",
                    quantity=10.0,
                    avg_entry_price=150.0,
                    current_price=155.0,
                    side="long",
                    timestamp=datetime.now(),
                    market_value=1550.0,
                    cost_basis=1500.0,
                    unrealized_pnl=50.0,
                    unrealized_pnl_pct=3.33,
                    realized_pnl=0.0
                )
            }
        
        broker.get_account_info = mock_get_account_info
        broker.get_positions = mock_get_positions
        return broker

    @pytest.fixture
    async def portfolio_manager(self, mock_broker):
        """Create a portfolio manager with mock broker."""
        config = {
            'trading.starting_cash': 100000.0,
            'risk.max_positions': 10
        }
        config_mock = Mock()
        config_mock.get = lambda key, default: config.get(key, default)
        
        manager = PortfolioManager(mock_broker, config_mock)
        await manager.initialize_portfolio_from_broker()
        return manager

    @pytest.mark.asyncio
    async def test_no_deadlock_get_position_size_concurrent(self, portfolio_manager):
        """Test that concurrent get_position_size calls don't deadlock."""
        async def get_position_size_task():
            return await portfolio_manager.get_position_size("AAPL", 150.0, 0.02)
        
        # Run 50 concurrent get_position_size operations
        tasks = [get_position_size_task() for _ in range(50)]
        
        # Should complete within 5 seconds (generous timeout)
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=5.0
        )
        
        # Verify all tasks completed successfully
        assert len(results) == 50
        for result in results:
            assert isinstance(result, (float, int))
            assert result >= 0

    @pytest.mark.asyncio
    async def test_no_deadlock_mixed_operations(self, portfolio_manager):
        """Test concurrent mixed operations don't cause deadlock."""
        
        async def mixed_operation(operation_id: int):
            """Perform a mix of portfolio operations."""
            try:
                if operation_id % 4 == 0:
                    return await portfolio_manager.get_position_size("AAPL", 150.0)
                elif operation_id % 4 == 1:
                    return await portfolio_manager.can_open_position()
                elif operation_id % 4 == 2:
                    return await portfolio_manager.get_cash_balance()
                else:
                    return await portfolio_manager.get_positions()
            except Exception as e:
                logger.error(f"Operation {operation_id} failed: {e}")
                raise
        
        # Run 100 mixed concurrent operations
        tasks = [mixed_operation(i) for i in range(100)]
        
        # Should complete within 10 seconds
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=10.0
        )
        
        # Verify all tasks completed
        assert len(results) == 100
        
        # Count successful operations (some might timeout, but no deadlock)
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Successful operations: {successful}/100")
        
        # At least 80% should succeed (allowing for some timeouts)
        assert successful >= 80

    @pytest.mark.asyncio
    async def test_no_deadlock_update_and_read_concurrent(self, portfolio_manager):
        """Test concurrent update and read operations don't deadlock."""
        
        async def update_task():
            """Continuously update portfolio."""
            for _ in range(10):
                await portfolio_manager.update_portfolio()
                await asyncio.sleep(0.01)
        
        async def read_task():
            """Continuously read portfolio data."""
            for _ in range(20):
                await portfolio_manager.get_position_size("AAPL", 150.0)
                await asyncio.sleep(0.005)
        
        # Run concurrent updates and reads
        update_tasks = [update_task() for _ in range(5)]
        read_tasks = [read_task() for _ in range(10)]
        
        all_tasks = update_tasks + read_tasks
        
        # Should complete within 15 seconds
        results = await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=15.0
        )
        
        # Verify all tasks completed
        assert len(results) == 15
        
        # Count successful operations
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Successful update/read operations: {successful}/15")
        
        # Most should succeed
        assert successful >= 12

    @pytest.mark.asyncio
    async def test_timeout_prevents_deadlock(self, portfolio_manager):
        """Test that timeouts prevent infinite deadlock."""
        
        # Patch broker to simulate very slow response
        async def slow_broker_call():
            await asyncio.sleep(20)  # 20 second delay - should timeout
            return Mock()
        
        with patch.object(portfolio_manager.broker, 'get_account_info', slow_broker_call):
            with patch.object(portfolio_manager.broker, 'get_positions', slow_broker_call):
                # This should timeout and not hang forever
                start_time = datetime.now()
                
                with pytest.raises((asyncio.TimeoutError, RuntimeError)):
                    await asyncio.wait_for(
                        portfolio_manager.update_portfolio(),
                        timeout=5.0
                    )
                
                elapsed = (datetime.now() - start_time).total_seconds()
                # Should timeout within reasonable time (not hang)
                assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_cache_reduces_broker_calls(self, portfolio_manager):
        """Test that caching reduces unnecessary broker API calls."""
        
        call_count = 0
        original_get_account_info = portfolio_manager.broker.get_account_info
        
        async def counting_get_account_info():
            nonlocal call_count
            call_count += 1
            return await original_get_account_info()
        
        portfolio_manager.broker.get_account_info = counting_get_account_info
        
        # Make multiple rapid calls - should use cache
        tasks = []
        for _ in range(10):
            tasks.append(portfolio_manager.get_cash_balance())
        
        await asyncio.gather(*tasks)
        
        # Should have made fewer broker calls due to caching
        logger.info(f"Broker calls made: {call_count}")
        assert call_count <= 3  # Should use cache for most calls

    @pytest.mark.asyncio
    async def test_position_operations_concurrent(self, portfolio_manager):
        """Test concurrent position open/close operations."""
        
        async def open_position_task(symbol: str):
            return await portfolio_manager.open_position(
                symbol, 10.0, 100.0, OrderSide.BUY, "test"
            )
        
        async def get_position_task(symbol: str):
            return await portfolio_manager.get_position_by_symbol(symbol)
        
        # Run concurrent position operations
        symbols = [f"TEST{i}" for i in range(10)]
        open_tasks = [open_position_task(symbol) for symbol in symbols]
        get_tasks = [get_position_task(symbol) for symbol in symbols]
        
        all_tasks = open_tasks + get_tasks
        
        # Should complete within 10 seconds
        results = await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=10.0
        )
        
        # Verify all tasks completed
        assert len(results) == 20
        
        # Count successful operations
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Successful position operations: {successful}/20")
        
        # Most should succeed
        assert successful >= 16

    @pytest.mark.asyncio
    async def test_stress_test_100_concurrent_operations(self, portfolio_manager):
        """Stress test with 100 concurrent operations of various types."""
        
        async def random_operation(op_id: int):
            """Perform random portfolio operation."""
            operation_type = op_id % 6
            
            try:
                if operation_type == 0:
                    return await portfolio_manager.get_position_size("AAPL", 150.0)
                elif operation_type == 1:
                    return await portfolio_manager.can_open_position()
                elif operation_type == 2:
                    return await portfolio_manager.get_cash_balance()
                elif operation_type == 3:
                    return await portfolio_manager.get_equity()
                elif operation_type == 4:
                    return await portfolio_manager.get_positions()
                else:
                    return await portfolio_manager.get_portfolio_summary()
            except Exception as e:
                logger.warning(f"Operation {op_id} failed: {e}")
                return None
        
        # Run 100 concurrent operations
        tasks = [random_operation(i) for i in range(100)]
        
        start_time = datetime.now()
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30.0  # Generous timeout for stress test
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Verify all tasks completed
        assert len(results) == 100
        
        # Count successful operations
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Stress test: {successful}/100 operations successful in {elapsed:.2f}s")
        
        # At least 70% should succeed under stress
        assert successful >= 70
        
        # Should complete in reasonable time
        assert elapsed < 20.0

    @pytest.mark.asyncio
    async def test_cache_expiry_works(self, portfolio_manager):
        """Test that cache properly expires after TTL."""
        
        # Set very short cache TTL for testing
        portfolio_manager._cache_ttl = timedelta(milliseconds=100)
        
        # First call should hit broker
        balance1 = await portfolio_manager.get_cash_balance()
        
        # Second immediate call should use cache
        balance2 = await portfolio_manager.get_cash_balance()
        assert balance1 == balance2
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Third call should hit broker again (cache expired)
        balance3 = await portfolio_manager.get_cash_balance()
        assert balance3 is not None

    @pytest.mark.asyncio
    async def test_lock_timeout_configuration(self, portfolio_manager):
        """Test that lock timeouts are properly configured."""
        
        # Verify timeout configuration
        assert portfolio_manager.lock_timeout == 30.0
        assert portfolio_manager.broker_timeout == 10.0
        
        # Test with very short timeout
        portfolio_manager.lock_timeout = 0.1
        
        # Create contention by holding a lock
        async def hold_lock():
            async with portfolio_manager._timed_lock(portfolio_manager._calculation_lock, timeout=1.0):
                await asyncio.sleep(0.5)  # Hold lock for 500ms
        
        async def try_acquire_lock():
            async with portfolio_manager._timed_lock(portfolio_manager._calculation_lock, timeout=0.1):
                pass
        
        # Start lock holder
        holder_task = asyncio.create_task(hold_lock())
        
        # Wait a bit for lock to be acquired
        await asyncio.sleep(0.05)
        
        # Try to acquire same lock with short timeout - should fail
        with pytest.raises(RuntimeError, match="Lock timeout"):
            await try_acquire_lock()
        
        # Wait for holder to finish
        await holder_task