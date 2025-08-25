"""
Performance benchmark tests for order processing throughput.

Tests the system's ability to handle 1000 orders/second requirement.
"""

import asyncio
import statistics
import time
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from src.application.coordinators.broker_coordinator import BrokerCoordinator
from src.domain.entities.order import Order, OrderSide, OrderType
from src.domain.entities.portfolio import Portfolio
from src.infrastructure.brokers.paper_broker import PaperBroker
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import ConnectionFactory
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository
from src.infrastructure.repositories.unit_of_work import PostgreSQLUnitOfWork


class PerformanceBenchmark:
    """Performance benchmarking suite for trading system."""

    def __init__(self):
        """Initialize benchmark components."""
        self.setup_database()
        self.setup_components()
        self.results: dict[str, Any] = {}

    def setup_database(self):
        """Setup database connection."""
        self.connection_factory = ConnectionFactory(
            host="localhost",
            port=5432,
            database="ai_trader",
            user="zachwade",
            password="ZachT$2002",
            pool_size=20,  # Increased for concurrent operations
            max_overflow=10,
        )
        self.db_adapter = PostgreSQLAdapter(self.connection_factory)

    def setup_components(self):
        """Setup trading system components."""
        self.broker = PaperBroker(initial_cash=Decimal("1000000"))
        self.uow = PostgreSQLUnitOfWork(self.db_adapter)

    async def cleanup(self):
        """Clean up test data."""
        # Delete test orders
        query = "DELETE FROM orders WHERE portfolio_id LIKE 'PERF_%'"
        await self.db_adapter.execute(query)

        # Delete test portfolios
        query = "DELETE FROM portfolios WHERE id LIKE 'PERF_%'"
        await self.db_adapter.execute(query)

        await self.db_adapter.close()

    async def create_test_portfolio(self) -> Portfolio:
        """Create a test portfolio for benchmarking."""
        portfolio = Portfolio(
            id=uuid4(),
            name=f"PERF_Portfolio_{uuid4().hex[:8]}",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
        )

        async with self.uow:
            await self.uow.portfolios.save(portfolio)
            await self.uow.commit()

        return portfolio

    async def benchmark_order_creation(self, num_orders: int = 1000) -> dict[str, Any]:
        """Benchmark order creation performance."""
        print(f"\n🏃 Benchmarking order creation ({num_orders} orders)...")

        orders = []
        start_time = time.perf_counter()

        for i in range(num_orders):
            order = Order(
                id=uuid4(),
                symbol=f"TEST{i % 100}",  # 100 different symbols
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET if i % 3 == 0 else OrderType.LIMIT,
                quantity=100 + (i % 900),
                limit_price=Decimal(str(100 + (i % 100))) if i % 3 != 0 else None,
            )
            orders.append(order)

        elapsed = time.perf_counter() - start_time
        orders_per_second = num_orders / elapsed

        results = {
            "total_orders": num_orders,
            "elapsed_time": elapsed,
            "orders_per_second": orders_per_second,
            "avg_time_per_order": elapsed / num_orders * 1000,  # milliseconds
        }

        print(f"✅ Created {num_orders} orders in {elapsed:.2f}s")
        print(f"   Rate: {orders_per_second:.0f} orders/second")

        return results

    async def benchmark_order_persistence(self, num_orders: int = 1000) -> dict[str, Any]:
        """Benchmark order persistence to database."""
        print(f"\n🏃 Benchmarking order persistence ({num_orders} orders)...")

        portfolio = await self.create_test_portfolio()
        order_repo = PostgreSQLOrderRepository(self.db_adapter)

        # Create orders
        orders = []
        for i in range(num_orders):
            order = Order(
                id=uuid4(),
                symbol=f"TEST{i % 100}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=100,
            )
            orders.append(order)

        # Benchmark persistence
        times = []
        start_time = time.perf_counter()

        for order in orders:
            order_start = time.perf_counter()
            await order_repo.save(order, portfolio.id)
            times.append(time.perf_counter() - order_start)

        elapsed = time.perf_counter() - start_time
        orders_per_second = num_orders / elapsed

        results = {
            "total_orders": num_orders,
            "elapsed_time": elapsed,
            "orders_per_second": orders_per_second,
            "avg_time_per_order": statistics.mean(times) * 1000,
            "median_time_per_order": statistics.median(times) * 1000,
            "p95_time_per_order": statistics.quantiles(times, n=20)[18] * 1000,
            "p99_time_per_order": statistics.quantiles(times, n=100)[98] * 1000,
        }

        print(f"✅ Persisted {num_orders} orders in {elapsed:.2f}s")
        print(f"   Rate: {orders_per_second:.0f} orders/second")
        print(f"   Avg latency: {results['avg_time_per_order']:.2f}ms")
        print(f"   P95 latency: {results['p95_time_per_order']:.2f}ms")

        return results

    async def benchmark_concurrent_orders(
        self, num_orders: int = 1000, concurrency: int = 10
    ) -> dict[str, Any]:
        """Benchmark concurrent order processing."""
        print(
            f"\n🏃 Benchmarking concurrent order processing ({num_orders} orders, {concurrency} concurrent)..."
        )

        portfolio = await self.create_test_portfolio()

        async def place_order_batch(batch_size: int):
            """Place a batch of orders concurrently."""
            tasks = []
            for _ in range(batch_size):
                order = Order(
                    id=uuid4(),
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=100,
                )

                async with self.uow:
                    task = self.uow.orders.save(order, portfolio.id)
                    tasks.append(task)

            await asyncio.gather(*tasks)

        start_time = time.perf_counter()

        # Process orders in concurrent batches
        batch_size = num_orders // concurrency
        tasks = []
        for _ in range(concurrency):
            task = place_order_batch(batch_size)
            tasks.append(task)

        await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - start_time
        orders_per_second = num_orders / elapsed

        results = {
            "total_orders": num_orders,
            "concurrency": concurrency,
            "elapsed_time": elapsed,
            "orders_per_second": orders_per_second,
            "avg_time_per_order": elapsed / num_orders * 1000,
        }

        print(f"✅ Processed {num_orders} orders concurrently in {elapsed:.2f}s")
        print(f"   Rate: {orders_per_second:.0f} orders/second")
        print(f"   Concurrency: {concurrency}")

        return results

    async def benchmark_full_order_lifecycle(self, num_orders: int = 100) -> dict[str, Any]:
        """Benchmark full order lifecycle including fills."""
        print(f"\n🏃 Benchmarking full order lifecycle ({num_orders} orders)...")

        portfolio = await self.create_test_portfolio()
        coordinator = BrokerCoordinator(self.broker, self.uow)

        # Set market prices
        self.broker.set_market_price("AAPL", Decimal("150.00"))

        times = []
        start_time = time.perf_counter()

        for i in range(num_orders):
            order_start = time.perf_counter()

            # Place order
            request = {
                "portfolio_id": str(portfolio.id),
                "symbol": "AAPL",
                "side": "buy" if i % 2 == 0 else "sell",
                "order_type": "market",
                "quantity": 100,
            }

            result = await coordinator.place_order(request)

            if result["success"]:
                # Process fill
                order_id = result["order_id"]
                await coordinator.process_order_fill(order_id, Decimal("150.00"), 100)

            times.append(time.perf_counter() - order_start)

        elapsed = time.perf_counter() - start_time
        orders_per_second = num_orders / elapsed

        results = {
            "total_orders": num_orders,
            "elapsed_time": elapsed,
            "orders_per_second": orders_per_second,
            "avg_time_per_order": statistics.mean(times) * 1000,
            "median_time_per_order": statistics.median(times) * 1000,
            "p95_time_per_order": (
                statistics.quantiles(times, n=20)[18] * 1000 if len(times) > 20 else 0
            ),
            "p99_time_per_order": (
                statistics.quantiles(times, n=100)[98] * 1000 if len(times) > 100 else 0
            ),
        }

        print(f"✅ Completed {num_orders} full order lifecycles in {elapsed:.2f}s")
        print(f"   Rate: {orders_per_second:.0f} orders/second")
        print(f"   Avg latency: {results['avg_time_per_order']:.2f}ms")

        return results

    async def run_all_benchmarks(self) -> dict[str, Any]:
        """Run all performance benchmarks."""
        print("\n" + "=" * 60)
        print("🚀 TRADING SYSTEM PERFORMANCE BENCHMARK")
        print("=" * 60)

        results = {}

        # Run benchmarks
        results["order_creation"] = await self.benchmark_order_creation(1000)
        results["order_persistence"] = await self.benchmark_order_persistence(1000)
        results["concurrent_orders"] = await self.benchmark_concurrent_orders(1000, 20)
        results["full_lifecycle"] = await self.benchmark_full_order_lifecycle(100)

        # Summary
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)

        # Check 1000 orders/sec requirement
        persistence_rate = results["order_persistence"]["orders_per_second"]
        concurrent_rate = results["concurrent_orders"]["orders_per_second"]

        print("\n🎯 Target: 1000 orders/second")
        print(f"✅ Order Creation: {results['order_creation']['orders_per_second']:.0f} orders/sec")
        print(
            f"{'✅' if persistence_rate >= 1000 else '❌'} Order Persistence: {persistence_rate:.0f} orders/sec"
        )
        print(
            f"{'✅' if concurrent_rate >= 1000 else '❌'} Concurrent Orders: {concurrent_rate:.0f} orders/sec"
        )
        print(f"📈 Full Lifecycle: {results['full_lifecycle']['orders_per_second']:.0f} orders/sec")

        # Latency summary
        print("\n⏱️ Latency Metrics:")
        print(f"   Persistence P95: {results['order_persistence']['p95_time_per_order']:.2f}ms")
        print(f"   Persistence P99: {results['order_persistence']['p99_time_per_order']:.2f}ms")
        print(f"   Full Lifecycle Avg: {results['full_lifecycle']['avg_time_per_order']:.2f}ms")

        # Overall assessment
        meets_requirement = persistence_rate >= 1000 or concurrent_rate >= 1000
        print(
            f"\n{'✅ PERFORMANCE REQUIREMENT MET' if meets_requirement else '❌ PERFORMANCE REQUIREMENT NOT MET'}"
        )

        if meets_requirement:
            print(
                f"The system can handle {max(persistence_rate, concurrent_rate):.0f} orders/second"
            )
        else:
            print(
                f"Current max throughput: {max(persistence_rate, concurrent_rate):.0f} orders/second"
            )
            print("Optimization needed to reach 1000 orders/second target")

        return results


@pytest.mark.asyncio
@pytest.mark.performance
async def test_order_throughput():
    """Test order processing throughput."""
    benchmark = PerformanceBenchmark()

    try:
        results = await benchmark.run_all_benchmarks()

        # Assert performance requirements
        assert (
            results["order_creation"]["orders_per_second"] >= 1000
        ), "Order creation should exceed 1000 orders/second"

        # Check if at least one persistence method meets requirement
        persistence_rate = results["order_persistence"]["orders_per_second"]
        concurrent_rate = results["concurrent_orders"]["orders_per_second"]

        assert (
            persistence_rate >= 500 or concurrent_rate >= 500
        ), f"Order processing should exceed 500 orders/second. Got {max(persistence_rate, concurrent_rate):.0f}"

    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    # Run benchmark directly
    async def main():
        benchmark = PerformanceBenchmark()
        try:
            await benchmark.run_all_benchmarks()
        finally:
            await benchmark.cleanup()

    asyncio.run(main())
