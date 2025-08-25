"""
Performance benchmark suite for order processing.

Validates the system can handle 1000 orders/second as required.
"""

import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from src.domain.entities.order import Order
from src.domain.services.order_processor import OrderProcessor
from src.domain.value_objects.quantity import Quantity
from src.infrastructure.brokers.paper_broker import PaperBroker
from src.infrastructure.database.adapter import DatabaseAdapter
from src.infrastructure.database.connection import DatabaseConnection
from src.infrastructure.repositories.order_repository import PostgreSQLOrderRepository
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository


class OrderProcessingBenchmark:
    """Benchmark suite for order processing performance."""

    def __init__(self):
        self.results: dict[str, Any] = {"benchmarks": [], "summary": {}}

    async def setup_infrastructure(self) -> tuple:
        """Set up infrastructure for benchmarking."""
        # Database connection
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "ai_trader",
            "user": "zachwade",
            "password": "ZachT$2002",
            "min_pool_size": 10,
            "max_pool_size": 50,  # Increased for high throughput
        }

        connection = DatabaseConnection(**db_config)
        await connection.connect()

        adapter = DatabaseAdapter(connection)

        # Repositories
        order_repo = PostgreSQLOrderRepository(adapter)
        portfolio_repo = PostgreSQLPortfolioRepository(adapter)

        # Broker
        broker = PaperBroker()

        # Order processor
        processor = OrderProcessor(
            broker=broker, order_repository=order_repo, portfolio_repository=portfolio_repo
        )

        return processor, connection

    def create_test_order(self, index: int) -> Order:
        """Create a test order for benchmarking."""
        return Order.create_market_order(
            symbol=f"TEST{index % 100:03d}",  # Distribute across 100 symbols
            quantity=Quantity(Decimal("100")),
            side="buy",
        )

    async def benchmark_single_order(self, processor: OrderProcessor) -> float:
        """Benchmark single order processing time."""
        order = self.create_test_order(1)

        start_time = time.perf_counter()
        await processor.submit_order(order)
        end_time = time.perf_counter()

        return (end_time - start_time) * 1000  # Convert to milliseconds

    async def benchmark_batch_orders(
        self, processor: OrderProcessor, batch_size: int
    ) -> dict[str, float]:
        """Benchmark batch order processing."""
        orders = [self.create_test_order(i) for i in range(batch_size)]

        start_time = time.perf_counter()

        # Submit orders concurrently
        tasks = [processor.submit_order(order) for order in orders]
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        total_time = end_time - start_time

        return {
            "batch_size": batch_size,
            "total_time_seconds": total_time,
            "orders_per_second": batch_size / total_time,
            "avg_time_per_order_ms": (total_time / batch_size) * 1000,
        }

    async def benchmark_sustained_throughput(
        self, processor: OrderProcessor, duration_seconds: int = 10, target_rate: int = 1000
    ) -> dict[str, Any]:
        """Benchmark sustained order throughput."""
        orders_submitted = 0
        errors = 0
        latencies = []

        interval = 1.0 / target_rate  # Time between orders
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            order = self.create_test_order(orders_submitted)

            order_start = time.perf_counter()
            try:
                await processor.submit_order(order)
                order_end = time.perf_counter()
                latencies.append((order_end - order_start) * 1000)
                orders_submitted += 1
            except Exception:
                errors += 1

            # Rate limiting
            await asyncio.sleep(interval)

        actual_duration = time.perf_counter() - start_time

        return {
            "duration_seconds": actual_duration,
            "orders_submitted": orders_submitted,
            "errors": errors,
            "actual_rate": orders_submitted / actual_duration,
            "target_rate": target_rate,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p50_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": self._percentile(latencies, 99) if latencies else 0,
        }

    async def benchmark_concurrent_load(
        self, processor: OrderProcessor, concurrent_clients: int = 10, orders_per_client: int = 100
    ) -> dict[str, Any]:
        """Benchmark with multiple concurrent clients."""

        async def client_workload(client_id: int) -> list[float]:
            """Simulate a single client submitting orders."""
            latencies = []
            for i in range(orders_per_client):
                order = self.create_test_order(client_id * 1000 + i)

                start = time.perf_counter()
                await processor.submit_order(order)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)

            return latencies

        start_time = time.perf_counter()

        # Run all clients concurrently
        tasks = [client_workload(i) for i in range(concurrent_clients)]
        all_latencies = await asyncio.gather(*tasks)

        end_time = time.perf_counter()

        # Flatten latencies
        flat_latencies = [lat for client_lats in all_latencies for lat in client_lats]

        total_orders = concurrent_clients * orders_per_client
        total_time = end_time - start_time

        return {
            "concurrent_clients": concurrent_clients,
            "orders_per_client": orders_per_client,
            "total_orders": total_orders,
            "total_time_seconds": total_time,
            "orders_per_second": total_orders / total_time,
            "avg_latency_ms": statistics.mean(flat_latencies),
            "p50_latency_ms": statistics.median(flat_latencies),
            "p95_latency_ms": self._percentile(flat_latencies, 95),
            "p99_latency_ms": self._percentile(flat_latencies, 99),
        }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        size = len(data)
        sorted_data = sorted(data)
        index = int(size * percentile / 100)
        if index >= size:
            index = size - 1
        return sorted_data[index]

    async def run_all_benchmarks(self) -> None:
        """Run all performance benchmarks."""
        print("Setting up infrastructure...")
        processor, connection = await self.setup_infrastructure()

        try:
            # 1. Single order latency
            print("\n1. Benchmarking single order latency...")
            single_latency = await self.benchmark_single_order(processor)
            self.results["benchmarks"].append(
                {"name": "Single Order Latency", "result": f"{single_latency:.2f} ms"}
            )
            print(f"   Single order latency: {single_latency:.2f} ms")

            # 2. Small batch performance
            print("\n2. Benchmarking small batch (100 orders)...")
            small_batch = await self.benchmark_batch_orders(processor, 100)
            self.results["benchmarks"].append(
                {"name": "Small Batch (100 orders)", "result": small_batch}
            )
            print(f"   Orders/sec: {small_batch['orders_per_second']:.1f}")
            print(f"   Avg latency: {small_batch['avg_time_per_order_ms']:.2f} ms")

            # 3. Large batch performance
            print("\n3. Benchmarking large batch (1000 orders)...")
            large_batch = await self.benchmark_batch_orders(processor, 1000)
            self.results["benchmarks"].append(
                {"name": "Large Batch (1000 orders)", "result": large_batch}
            )
            print(f"   Orders/sec: {large_batch['orders_per_second']:.1f}")
            print(f"   Avg latency: {large_batch['avg_time_per_order_ms']:.2f} ms")

            # 4. Sustained throughput test
            print("\n4. Benchmarking sustained throughput (10 seconds at 100 orders/sec)...")
            sustained = await self.benchmark_sustained_throughput(processor, 10, 100)
            self.results["benchmarks"].append({"name": "Sustained Throughput", "result": sustained})
            print(f"   Actual rate: {sustained['actual_rate']:.1f} orders/sec")
            print(f"   P95 latency: {sustained['p95_latency_ms']:.2f} ms")
            print(f"   Errors: {sustained['errors']}")

            # 5. Concurrent load test
            print("\n5. Benchmarking concurrent load (10 clients, 100 orders each)...")
            concurrent = await self.benchmark_concurrent_load(processor, 10, 100)
            self.results["benchmarks"].append({"name": "Concurrent Load", "result": concurrent})
            print(f"   Orders/sec: {concurrent['orders_per_second']:.1f}")
            print(f"   P95 latency: {concurrent['p95_latency_ms']:.2f} ms")

            # Calculate summary
            self._calculate_summary()

            # Save results
            self._save_results()

        finally:
            await connection.disconnect()

    def _calculate_summary(self) -> None:
        """Calculate summary statistics."""
        # Find max throughput
        max_throughput = 0
        for benchmark in self.results["benchmarks"]:
            if isinstance(benchmark["result"], dict):
                if "orders_per_second" in benchmark["result"]:
                    max_throughput = max(max_throughput, benchmark["result"]["orders_per_second"])

        self.results["summary"] = {
            "max_throughput_observed": max_throughput,
            "target_throughput": 1000,
            "meets_target": max_throughput >= 1000,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def _save_results(self) -> None:
        """Save benchmark results to file."""
        with open("performance_benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(
            f"Max throughput observed: {self.results['summary']['max_throughput_observed']:.1f} orders/sec"
        )
        print(f"Target throughput: {self.results['summary']['target_throughput']} orders/sec")
        print(f"Meets target: {'✅ YES' if self.results['summary']['meets_target'] else '❌ NO'}")
        print("\nDetailed results saved to: performance_benchmark_results.json")


async def main():
    """Run the benchmark suite."""
    benchmark = OrderProcessingBenchmark()
    await benchmark.run_all_benchmarks()


if __name__ == "__main__":
    print("=" * 60)
    print("ORDER PROCESSING PERFORMANCE BENCHMARK")
    print("=" * 60)
    asyncio.run(main())
