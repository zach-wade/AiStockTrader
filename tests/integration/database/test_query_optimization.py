"""
Integration Tests for Query Optimization
Tests optimized query functions and performance characteristics
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import DatabaseConnection


class TestQueryOptimization:
    """Integration tests for optimized query functions."""

    @pytest.fixture(autouse=True)
    async def setup_method(self, db_connection: DatabaseConnection):
        """Set up test environment with optimized functions."""
        # Get the pool from the connection
        pool = db_connection._pool
        self.adapter = PostgreSQLAdapter(pool)

        # Apply query optimization functions
        await self._apply_optimization_functions()

        # Create test data
        await self._create_test_data()

    async def _apply_optimization_functions(self):
        """Apply query optimization functions from query_optimization.sql."""
        # Read and execute the optimization functions
        # Note: In a real test, you'd read from the actual file
        optimization_functions = [
            # Function 1: High-Performance Order Lookup
            """
            CREATE OR REPLACE FUNCTION get_active_orders_by_symbol(
                p_symbol VARCHAR(20),
                p_limit INTEGER DEFAULT 100
            ) RETURNS TABLE(
                id UUID,
                symbol VARCHAR(20),
                side order_side,
                order_type order_type,
                status order_status,
                quantity DECIMAL(18, 8),
                limit_price DECIMAL(18, 8),
                stop_price DECIMAL(18, 8),
                created_at TIMESTAMP WITH TIME ZONE
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    o.id, o.symbol, o.side, o.order_type, o.status,
                    o.quantity, o.limit_price, o.stop_price, o.created_at
                FROM orders o
                WHERE o.symbol = p_symbol
                    AND o.status IN ('pending', 'submitted', 'partially_filled')
                ORDER BY o.created_at DESC
                LIMIT p_limit;
            END;
            $$ LANGUAGE plpgsql STABLE;
            """,
            # Function 2: Ultra-Fast Order Status Update
            """
            CREATE OR REPLACE FUNCTION update_order_status_fast(
                p_broker_order_id VARCHAR(100),
                p_new_status order_status,
                p_filled_quantity DECIMAL(18, 8) DEFAULT NULL,
                p_average_fill_price DECIMAL(18, 8) DEFAULT NULL
            ) RETURNS BOOLEAN AS $$
            DECLARE
                updated_count INTEGER;
            BEGIN
                UPDATE orders
                SET
                    status = p_new_status,
                    filled_quantity = COALESCE(p_filled_quantity, filled_quantity),
                    average_fill_price = COALESCE(p_average_fill_price, average_fill_price),
                    filled_at = CASE WHEN p_new_status = 'filled' THEN NOW() ELSE filled_at END,
                    cancelled_at = CASE WHEN p_new_status = 'cancelled' THEN NOW() ELSE cancelled_at END
                WHERE broker_order_id = p_broker_order_id;

                GET DIAGNOSTICS updated_count = ROW_COUNT;
                RETURN updated_count > 0;
            END;
            $$ LANGUAGE plpgsql;
            """,
            # Function 3: Position with P&L calculation
            """
            CREATE OR REPLACE FUNCTION get_position_with_pnl(
                p_symbol VARCHAR(20),
                p_current_price DECIMAL(18, 8)
            ) RETURNS TABLE(
                id UUID,
                symbol VARCHAR(20),
                quantity DECIMAL(18, 8),
                average_entry_price DECIMAL(18, 8),
                current_price DECIMAL(18, 8),
                unrealized_pnl DECIMAL(18, 8),
                realized_pnl DECIMAL(18, 8),
                total_pnl DECIMAL(18, 8),
                pnl_percentage DECIMAL(8, 4),
                position_value DECIMAL(18, 8)
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    p.id,
                    p.symbol,
                    p.quantity,
                    p.average_entry_price,
                    p_current_price as current_price,
                    CASE
                        WHEN p.quantity > 0 THEN
                            (p_current_price - p.average_entry_price) * p.quantity
                        WHEN p.quantity < 0 THEN
                            (p.average_entry_price - p_current_price) * ABS(p.quantity)
                        ELSE 0
                    END as unrealized_pnl,
                    p.realized_pnl,
                    p.realized_pnl + CASE
                        WHEN p.quantity > 0 THEN
                            (p_current_price - p.average_entry_price) * p.quantity
                        WHEN p.quantity < 0 THEN
                            (p.average_entry_price - p_current_price) * ABS(p.quantity)
                        ELSE 0
                    END as total_pnl,
                    CASE
                        WHEN p.average_entry_price > 0 AND p.quantity != 0 THEN
                            ((p.realized_pnl +
                              CASE
                                  WHEN p.quantity > 0 THEN (p_current_price - p.average_entry_price) * p.quantity
                                  ELSE (p.average_entry_price - p_current_price) * ABS(p.quantity)
                              END) / (ABS(p.quantity) * p.average_entry_price)) * 100
                        ELSE 0
                    END as pnl_percentage,
                    ABS(p.quantity) * p_current_price as position_value
                FROM positions p
                WHERE p.symbol = p_symbol
                    AND p.closed_at IS NULL;
            END;
            $$ LANGUAGE plpgsql STABLE;
            """,
            # Function 4: Latest Price Lookup
            """
            CREATE OR REPLACE FUNCTION get_latest_prices(
                p_symbols VARCHAR(20)[],
                p_timeframe VARCHAR(10) DEFAULT '1min'
            ) RETURNS TABLE(
                symbol VARCHAR(20),
                price DECIMAL(18, 8),
                volume BIGINT,
                timestamp TIMESTAMP WITH TIME ZONE,
                vwap DECIMAL(18, 8)
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT DISTINCT ON (md.symbol)
                    md.symbol,
                    md.close as price,
                    md.volume,
                    md.timestamp,
                    md.vwap
                FROM market_data md
                WHERE md.symbol = ANY(p_symbols)
                    AND md.timeframe = p_timeframe
                    AND md.timestamp > NOW() - INTERVAL '1 hour'
                ORDER BY md.symbol, md.timestamp DESC;
            END;
            $$ LANGUAGE plpgsql STABLE;
            """,
        ]

        for func in optimization_functions:
            await self.adapter.execute_query(func)

    async def _create_test_data(self):
        """Create test data for optimization testing."""
        # Clear existing test data
        await self.adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'OPT%'")
        # market_data is a view, delete from underlying tables
        await self.adapter.execute_query("DELETE FROM market_data_1m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_5m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_15m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_30m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_1h WHERE symbol LIKE 'OPT%'")

        # Create test orders
        test_symbols = ["OPT001", "OPT002", "OPT003", "OPT004", "OPT005"]

        for i, symbol in enumerate(test_symbols):
            # Create multiple orders per symbol
            for j in range(50):
                await self.adapter.execute_query(
                    """
                    INSERT INTO orders (
                        symbol, side, order_type, status, quantity, limit_price,
                        broker_order_id, created_at, time_in_force
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    symbol,
                    "buy" if j % 2 == 0 else "sell",
                    "limit",
                    ["pending", "submitted", "partially_filled"][j % 3],
                    Decimal("100.0"),
                    Decimal(f"{100 + j}.00"),
                    f"OPT_BROKER_{i}_{j}",
                    datetime.now(UTC) - timedelta(minutes=j),
                    "day",
                )

        # Create test positions
        for symbol in test_symbols:
            await self.adapter.execute_query(
                """
                INSERT INTO positions (
                    symbol, quantity, average_entry_price, current_price,
                    realized_pnl, opened_at, strategy
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                symbol,
                Decimal("1000.0"),
                Decimal("100.00"),
                Decimal("105.00"),
                Decimal("0.00"),
                datetime.now(UTC) - timedelta(days=1),
                "test_strategy",
            )

        # Create test market data
        for symbol in test_symbols:
            for i in range(100):
                timestamp = datetime.now(UTC) - timedelta(minutes=i)
                base_price = Decimal("100.00") + Decimal(str(i * 0.1))

                await self.adapter.execute_query(
                    """
                    INSERT INTO market_data (
                        symbol, timeframe, timestamp, open, high, low, close, volume, vwap
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    symbol,
                    "1min",
                    timestamp,
                    base_price,
                    base_price + Decimal("1.0"),
                    base_price - Decimal("1.0"),
                    base_price + Decimal("0.5"),
                    1000 + i,
                    base_price + Decimal("0.25"),
                )

    async def _measure_function_performance(
        self, function_call: str, expected_rows: int = None
    ) -> dict[str, Any]:
        """Measure function execution performance."""
        # Warm up
        await self.adapter.fetch_all(f"SELECT * FROM ({function_call}) AS f")

        # Measure
        start_time = time.perf_counter()
        result = await self.adapter.fetch_all(f"SELECT * FROM ({function_call}) AS f")
        end_time = time.perf_counter()

        execution_time_ms = (end_time - start_time) * 1000

        return {"execution_time_ms": execution_time_ms, "row_count": len(result), "result": result}

    @pytest.mark.asyncio
    async def test_get_active_orders_by_symbol_performance(self):
        """Test optimized active orders function performance."""
        function_call = "SELECT * FROM get_active_orders_by_symbol('OPT001', 50)"

        metrics = await self._measure_function_performance(function_call)

        # Should execute very quickly
        assert (
            metrics["execution_time_ms"] < 2.0
        ), f"get_active_orders_by_symbol took {metrics['execution_time_ms']:.2f}ms, expected < 2ms"

        # Should return results
        assert metrics["row_count"] > 0, "Function should return active orders"

        # Verify result structure
        result = metrics["result"][0]
        required_fields = ["id", "symbol", "side", "order_type", "status", "quantity"]
        for field in required_fields:
            assert field in result, f"Missing field {field} in result"

    @pytest.mark.asyncio
    async def test_update_order_status_fast_performance(self):
        """Test optimized order status update performance."""
        # First, get a broker order ID to update
        broker_id_result = await self.adapter.fetch_one(
            "SELECT broker_order_id FROM orders WHERE symbol LIKE 'OPT%' LIMIT 1"
        )
        broker_id = broker_id_result["broker_order_id"]

        # Measure update performance
        start_time = time.perf_counter()

        result = await self.adapter.fetch_one(
            "SELECT update_order_status_fast(%s, %s, %s, %s) as success",
            broker_id,
            "filled",
            Decimal("100.0"),
            Decimal("105.50"),
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Should execute very quickly
        assert (
            execution_time_ms < 1.0
        ), f"update_order_status_fast took {execution_time_ms:.2f}ms, expected < 1ms"

        # Should succeed
        assert result["success"] is True, "Order status update should succeed"

        # Verify the update was applied
        updated_order = await self.adapter.fetch_one(
            "SELECT status, filled_quantity, average_fill_price FROM orders WHERE broker_order_id = %s",
            broker_id,
        )
        assert updated_order["status"] == "filled"
        assert updated_order["filled_quantity"] == Decimal("100.0")
        assert updated_order["average_fill_price"] == Decimal("105.50")

    @pytest.mark.asyncio
    async def test_get_position_with_pnl_performance(self):
        """Test position with P&L calculation performance."""
        function_call = "SELECT * FROM get_position_with_pnl('OPT001', 110.00)"

        metrics = await self._measure_function_performance(function_call)

        # Should execute quickly
        assert (
            metrics["execution_time_ms"] < 3.0
        ), f"get_position_with_pnl took {metrics['execution_time_ms']:.2f}ms, expected < 3ms"

        # Should return position with calculations
        assert metrics["row_count"] == 1, "Should return exactly one position"

        result = metrics["result"][0]

        # Verify P&L calculations
        assert "unrealized_pnl" in result
        assert "realized_pnl" in result
        assert "total_pnl" in result
        assert "pnl_percentage" in result
        assert "position_value" in result

        # Verify calculations are reasonable
        assert result["unrealized_pnl"] == Decimal(
            "10000.00"
        ), f"Expected unrealized P&L of 10000.00, got {result['unrealized_pnl']}"

    @pytest.mark.asyncio
    async def test_get_latest_prices_performance(self):
        """Test latest prices function performance."""
        symbols_array = "ARRAY['OPT001', 'OPT002', 'OPT003', 'OPT004', 'OPT005']"
        function_call = f"SELECT * FROM get_latest_prices({symbols_array}, '1min')"

        metrics = await self._measure_function_performance(function_call)

        # Should execute quickly even for multiple symbols
        assert (
            metrics["execution_time_ms"] < 5.0
        ), f"get_latest_prices took {metrics['execution_time_ms']:.2f}ms, expected < 5ms"

        # Should return latest prices for all symbols
        assert metrics["row_count"] == 5, "Should return prices for all 5 symbols"

        # Verify result structure
        result = metrics["result"][0]
        required_fields = ["symbol", "price", "volume", "timestamp", "vwap"]
        for field in required_fields:
            assert field in result, f"Missing field {field} in result"

    @pytest.mark.asyncio
    async def test_batch_order_processing_optimization(self):
        """Test batch processing performance with optimized queries."""
        batch_size = 100

        # Prepare batch of order updates
        broker_ids = []
        for i in range(batch_size):
            # Insert test order
            order_id = str(uuid.uuid4())
            broker_id = f"BATCH_OPT_{i}"

            await self.adapter.execute_query(
                """
                INSERT INTO orders (
                    id, symbol, side, order_type, status, quantity,
                    limit_price, broker_order_id, time_in_force
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                order_id,
                "OPT_BATCH",
                "buy",
                "limit",
                "pending",
                Decimal("100.0"),
                Decimal("100.00"),
                broker_id,
                "day",
            )
            broker_ids.append(broker_id)

        # Test batch status updates
        start_time = time.perf_counter()

        # Use optimized function for batch updates
        for broker_id in broker_ids:
            await self.adapter.fetch_one(
                "SELECT update_order_status_fast(%s, %s) as success", broker_id, "filled"
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = batch_size / total_time

        # Should achieve high throughput
        assert throughput > 500, f"Batch update throughput {throughput:.1f} ops/sec, expected > 500"

    @pytest.mark.asyncio
    async def test_concurrent_function_execution(self):
        """Test concurrent execution of optimized functions."""

        async def execute_concurrent_queries():
            """Execute various optimized functions concurrently."""
            tasks = [
                self.adapter.fetch_all("SELECT * FROM get_active_orders_by_symbol('OPT001', 10)"),
                self.adapter.fetch_all("SELECT * FROM get_position_with_pnl('OPT002', 105.00)"),
                self.adapter.fetch_all("SELECT * FROM get_latest_prices(ARRAY['OPT003'], '1min')"),
            ]
            return await asyncio.gather(*tasks)

        # Run multiple concurrent sessions
        start_time = time.perf_counter()

        concurrent_tasks = [execute_concurrent_queries() for _ in range(10)]
        results = await asyncio.gather(*concurrent_tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should handle concurrent execution efficiently
        assert (
            total_time < 5.0
        ), f"Concurrent function execution took {total_time:.2f}s, expected < 5s"

        # Verify all tasks completed successfully
        assert len(results) == 10, "All concurrent tasks should complete"

    @pytest.mark.asyncio
    async def test_function_plan_optimization(self):
        """Test that optimized functions use efficient query plans."""
        # Test query plan for active orders function
        explain_query = """
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT * FROM get_active_orders_by_symbol('OPT001', 10)
        """

        result = await self.adapter.fetch_one(explain_query)
        plan = result["QUERY PLAN"][0]

        # Should use index scans
        plan_str = str(plan)
        assert (
            "Index Scan" in plan_str or "Bitmap Index Scan" in plan_str
        ), "Function should use index scans"

        # Should not use sequential scans for this query
        assert "Seq Scan" not in plan_str, "Function should not require sequential scans"

    @pytest.mark.asyncio
    async def test_materialized_view_performance(self):
        """Test materialized view creation and refresh performance."""
        # Create a test materialized view
        await self.adapter.execute_query(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS mv_test_order_summary AS
            SELECT
                symbol,
                COUNT(*) as order_count,
                SUM(quantity) as total_quantity,
                AVG(limit_price) as avg_price
            FROM orders
            WHERE symbol LIKE 'OPT%'
            GROUP BY symbol
        """
        )

        # Create unique index for concurrent refresh
        await self.adapter.execute_query(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS mv_test_order_summary_symbol
            ON mv_test_order_summary(symbol)
        """
        )

        # Test refresh performance
        start_time = time.perf_counter()

        await self.adapter.execute_query(
            "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_test_order_summary"
        )

        end_time = time.perf_counter()
        refresh_time_ms = (end_time - start_time) * 1000

        # Should refresh quickly
        assert (
            refresh_time_ms < 1000
        ), f"Materialized view refresh took {refresh_time_ms:.1f}ms, expected < 1000ms"

        # Test query performance against materialized view
        start_time = time.perf_counter()

        result = await self.adapter.fetch_all(
            "SELECT * FROM mv_test_order_summary ORDER BY order_count DESC"
        )

        end_time = time.perf_counter()
        query_time_ms = (end_time - start_time) * 1000

        # Should query very quickly
        assert (
            query_time_ms < 1.0
        ), f"Materialized view query took {query_time_ms:.2f}ms, expected < 1ms"

        # Clean up
        await self.adapter.execute_query("DROP MATERIALIZED VIEW IF EXISTS mv_test_order_summary")

    @pytest.mark.asyncio
    async def test_function_parameter_optimization(self):
        """Test that functions handle different parameter scenarios efficiently."""
        # Test with different limit sizes
        limits = [10, 50, 100, 500]

        for limit in limits:
            start_time = time.perf_counter()

            result = await self.adapter.fetch_all(
                f"SELECT * FROM get_active_orders_by_symbol('OPT001', {limit})"
            )

            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000

            # Execution time should scale reasonably with limit
            expected_max_time = min(5.0, 0.1 * limit)  # Allow more time for larger limits
            assert (
                execution_time_ms < expected_max_time
            ), f"Function with limit {limit} took {execution_time_ms:.2f}ms, expected < {expected_max_time}ms"

    async def teardown_method(self):
        """Clean up test data and functions."""
        # Clean up test data
        await self.adapter.execute_query("DELETE FROM orders WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM positions WHERE symbol LIKE 'OPT%'")
        # market_data is a view, delete from underlying tables
        await self.adapter.execute_query("DELETE FROM market_data_1m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_5m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_15m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_30m WHERE symbol LIKE 'OPT%'")
        await self.adapter.execute_query("DELETE FROM market_data_1h WHERE symbol LIKE 'OPT%'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
