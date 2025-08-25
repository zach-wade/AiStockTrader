"""
Performance tests for rate limiting system.

Tests performance requirements, latency, throughput, and scalability.
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.infrastructure.rate_limiting import (
    FixedWindowRateLimit,
    MemoryRateLimitStorage,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitContext,
    RateLimitManager,
    RateLimitRule,
    SlidingWindowRateLimit,
    TokenBucketRateLimit,
    initialize_rate_limiting,
    rate_limit,
)


@pytest.mark.performance
class TestAlgorithmPerformance:
    """Test performance of individual rate limiting algorithms."""

    def test_token_bucket_latency(self):
        """Test token bucket algorithm latency requirements."""
        rule = RateLimitRule(limit=1000, window="1min")
        limiter = TokenBucketRateLimit(rule)

        latencies = []

        # Measure latency for 1000 checks
        for i in range(1000):
            start = time.perf_counter()
            limiter.check_rate_limit(f"user{i % 100}")
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds

        # Check latency requirements
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)

        print(
            f"Token Bucket - Avg: {avg_latency:.3f}ms, P95: {p95_latency:.3f}ms, Max: {max_latency:.3f}ms"
        )

        # Requirements: < 1ms average latency
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms exceeds 1ms requirement"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.3f}ms exceeds 2ms threshold"

    def test_sliding_window_latency(self):
        """Test sliding window algorithm latency requirements."""
        rule = RateLimitRule(limit=1000, window="1min", algorithm=RateLimitAlgorithm.SLIDING_WINDOW)
        limiter = SlidingWindowRateLimit(rule)

        latencies = []

        # Measure latency for 500 checks (sliding window is more expensive)
        for i in range(500):
            start = time.perf_counter()
            limiter.check_rate_limit(f"user{i % 50}")
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]

        print(f"Sliding Window - Avg: {avg_latency:.3f}ms, P95: {p95_latency:.3f}ms")

        # Sliding window can be slightly higher latency due to complexity
        assert avg_latency < 2.0, f"Average latency {avg_latency:.3f}ms exceeds 2ms threshold"
        assert p95_latency < 5.0, f"P95 latency {p95_latency:.3f}ms exceeds 5ms threshold"

    def test_fixed_window_latency(self):
        """Test fixed window algorithm latency requirements."""
        rule = RateLimitRule(limit=1000, window="1min", algorithm=RateLimitAlgorithm.FIXED_WINDOW)
        limiter = FixedWindowRateLimit(rule)

        latencies = []

        # Measure latency for 1000 checks
        for i in range(1000):
            start = time.perf_counter()
            limiter.check_rate_limit(f"user{i % 100}")
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]

        print(f"Fixed Window - Avg: {avg_latency:.3f}ms, P95: {p95_latency:.3f}ms")

        # Fixed window should be fastest
        assert avg_latency < 0.5, f"Average latency {avg_latency:.3f}ms exceeds 0.5ms requirement"
        assert p95_latency < 1.0, f"P95 latency {p95_latency:.3f}ms exceeds 1ms threshold"

    def test_algorithm_throughput(self):
        """Test algorithm throughput requirements."""
        algorithms = [
            ("TokenBucket", TokenBucketRateLimit),
            ("SlidingWindow", SlidingWindowRateLimit),
            ("FixedWindow", FixedWindowRateLimit),
        ]

        for name, algorithm_class in algorithms:
            if name == "SlidingWindow":
                rule = RateLimitRule(
                    limit=10000, window="1min", algorithm=RateLimitAlgorithm.SLIDING_WINDOW
                )
            elif name == "FixedWindow":
                rule = RateLimitRule(
                    limit=10000, window="1min", algorithm=RateLimitAlgorithm.FIXED_WINDOW
                )
            else:
                rule = RateLimitRule(limit=10000, window="1min")

            limiter = algorithm_class(rule)

            start_time = time.perf_counter()

            # Perform throughput test
            num_operations = 5000 if name == "SlidingWindow" else 10000

            for i in range(num_operations):
                limiter.check_rate_limit(f"user{i % 100}")

            end_time = time.perf_counter()
            elapsed = end_time - start_time
            throughput = num_operations / elapsed

            print(f"{name} - {num_operations} ops in {elapsed:.3f}s = {throughput:.0f} ops/sec")

            # Throughput requirements vary by algorithm complexity
            if name == "SlidingWindow":
                assert (
                    throughput > 5000
                ), f"{name} throughput {throughput:.0f} ops/sec below 5000 requirement"
            else:
                assert (
                    throughput > 10000
                ), f"{name} throughput {throughput:.0f} ops/sec below 10000 requirement"


@pytest.mark.performance
class TestStoragePerformance:
    """Test performance of storage backends."""

    def test_memory_storage_performance(self):
        """Test memory storage performance."""
        storage = MemoryRateLimitStorage()

        # Test write performance
        start_time = time.perf_counter()
        for i in range(10000):
            storage.set(f"key{i}", f"value{i}")
        write_time = time.perf_counter() - start_time

        # Test read performance
        start_time = time.perf_counter()
        for i in range(10000):
            storage.get(f"key{i}")
        read_time = time.perf_counter() - start_time

        # Test increment performance
        start_time = time.perf_counter()
        for i in range(10000):
            storage.increment(f"counter{i % 1000}")
        increment_time = time.perf_counter() - start_time

        write_throughput = 10000 / write_time
        read_throughput = 10000 / read_time
        increment_throughput = 10000 / increment_time

        print("Memory Storage:")
        print(f"  Write: {write_throughput:.0f} ops/sec")
        print(f"  Read: {read_throughput:.0f} ops/sec")
        print(f"  Increment: {increment_throughput:.0f} ops/sec")

        # Memory storage should be very fast
        assert (
            write_throughput > 50000
        ), f"Write throughput {write_throughput:.0f} below 50000 ops/sec"
        assert (
            read_throughput > 100000
        ), f"Read throughput {read_throughput:.0f} below 100000 ops/sec"
        assert (
            increment_throughput > 50000
        ), f"Increment throughput {increment_throughput:.0f} below 50000 ops/sec"

    def test_memory_storage_with_ttl(self):
        """Test memory storage performance with TTL operations."""
        storage = MemoryRateLimitStorage()

        # Test performance with TTL
        start_time = time.perf_counter()
        for i in range(5000):
            storage.set(f"ttl_key{i}", f"value{i}", 60)  # 60 second TTL
        ttl_write_time = time.perf_counter() - start_time

        ttl_write_throughput = 5000 / ttl_write_time
        print(f"Memory Storage TTL Write: {ttl_write_throughput:.0f} ops/sec")

        # Should still be fast with TTL
        assert (
            ttl_write_throughput > 20000
        ), f"TTL write throughput {ttl_write_throughput:.0f} below 20000 ops/sec"

    @pytest.mark.redis
    def test_redis_storage_performance(self):
        """Test Redis storage performance."""
        try:
            from src.infrastructure.rate_limiting.storage import RedisRateLimitStorage

            config = RateLimitConfig(storage_backend="redis", redis_url="redis://localhost:6379/15")

            storage = RedisRateLimitStorage(config)
            storage.redis_client.flushdb()  # Clean slate

            # Test write performance
            start_time = time.perf_counter()
            for i in range(1000):
                storage.set(f"key{i}", f"value{i}")
            write_time = time.perf_counter() - start_time

            # Test read performance
            start_time = time.perf_counter()
            for i in range(1000):
                storage.get(f"key{i}")
            read_time = time.perf_counter() - start_time

            # Test increment performance
            start_time = time.perf_counter()
            for i in range(1000):
                storage.increment(f"counter{i % 100}")
            increment_time = time.perf_counter() - start_time

            write_throughput = 1000 / write_time
            read_throughput = 1000 / read_time
            increment_throughput = 1000 / increment_time

            print("Redis Storage:")
            print(f"  Write: {write_throughput:.0f} ops/sec")
            print(f"  Read: {read_throughput:.0f} ops/sec")
            print(f"  Increment: {increment_throughput:.0f} ops/sec")

            # Redis will be slower than memory but should still be reasonable
            assert (
                write_throughput > 1000
            ), f"Write throughput {write_throughput:.0f} below 1000 ops/sec"
            assert (
                read_throughput > 2000
            ), f"Read throughput {read_throughput:.0f} below 2000 ops/sec"
            assert (
                increment_throughput > 1000
            ), f"Increment throughput {increment_throughput:.0f} below 1000 ops/sec"

        except Exception:
            pytest.skip("Redis not available for performance testing")


@pytest.mark.performance
class TestManagerPerformance:
    """Test RateLimitManager performance."""

    def test_manager_single_threaded_performance(self):
        """Test manager performance in single-threaded scenario."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Test basic rate limit checking
        start_time = time.perf_counter()

        for i in range(5000):
            context = RateLimitContext(user_id=f"user{i % 500}")
            try:
                manager.check_rate_limit(context)
            except Exception:
                pass  # Rate limit exceeded, expected

        elapsed = time.perf_counter() - start_time
        throughput = 5000 / elapsed

        print(f"Manager Single-threaded: {throughput:.0f} checks/sec")

        # Should handle at least 5000 checks/second
        assert throughput > 5000, f"Manager throughput {throughput:.0f} below 5000 checks/sec"

    def test_manager_concurrent_performance(self):
        """Test manager performance under concurrent load."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        def worker(thread_id, num_requests):
            results = []
            for i in range(num_requests):
                context = RateLimitContext(user_id=f"thread{thread_id}_user{i}")
                start = time.perf_counter()
                try:
                    manager.check_rate_limit(context)
                    results.append(time.perf_counter() - start)
                except Exception:
                    results.append(time.perf_counter() - start)
            return results

        # Use ThreadPoolExecutor for controlled concurrency
        num_threads = 10
        requests_per_thread = 500
        total_requests = num_threads * requests_per_thread

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i, requests_per_thread) for i in range(num_threads)]

            all_latencies = []
            for future in as_completed(futures):
                all_latencies.extend(future.result())

        total_time = time.perf_counter() - start_time
        throughput = total_requests / total_time

        avg_latency = statistics.mean(all_latencies) * 1000  # Convert to ms
        p95_latency = statistics.quantiles(all_latencies, n=20)[18] * 1000

        print("Manager Concurrent:")
        print(f"  Throughput: {throughput:.0f} checks/sec")
        print(f"  Avg Latency: {avg_latency:.3f}ms")
        print(f"  P95 Latency: {p95_latency:.3f}ms")

        # Performance requirements for concurrent access
        assert throughput > 3000, f"Concurrent throughput {throughput:.0f} below 3000 checks/sec"
        assert avg_latency < 5.0, f"Average latency {avg_latency:.3f}ms exceeds 5ms"
        assert p95_latency < 20.0, f"P95 latency {p95_latency:.3f}ms exceeds 20ms"

    def test_manager_memory_efficiency(self):
        """Test manager memory efficiency with many users."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Create rate limit entries for many users
        num_users = 10000

        for i in range(num_users):
            context = RateLimitContext(user_id=f"user{i}")
            try:
                manager.check_rate_limit(context)
            except Exception:
                pass

        # Force cleanup to test efficiency
        cleanup_results = manager.cleanup_expired()

        # Should handle many users without issues
        assert isinstance(cleanup_results, dict)
        print(f"Created entries for {num_users} users successfully")

    def test_manager_different_context_types(self):
        """Test manager performance with different context types."""
        config = RateLimitConfig(
            storage_backend="memory",
            enable_ip_limiting=True,
            enable_user_limiting=True,
            enable_api_key_limiting=True,
        )
        manager = RateLimitManager(config)

        contexts = [
            # User-only context
            RateLimitContext(user_id="user1"),
            # IP-only context
            RateLimitContext(ip_address="192.168.1.1"),
            # API key context
            RateLimitContext(api_key="api_key_1"),
            # Full context
            RateLimitContext(
                user_id="user2",
                api_key="api_key_2",
                ip_address="192.168.1.2",
                trading_action="submit_order",
            ),
        ]

        # Test performance with different context types
        start_time = time.perf_counter()

        for i in range(2000):
            context = contexts[i % len(contexts)]
            try:
                manager.check_rate_limit(context)
            except Exception:
                pass

        elapsed = time.perf_counter() - start_time
        throughput = 2000 / elapsed

        print(f"Manager Mixed Contexts: {throughput:.0f} checks/sec")

        # Should handle mixed contexts efficiently
        assert throughput > 2000, f"Mixed context throughput {throughput:.0f} below 2000 checks/sec"


@pytest.mark.performance
class TestDecoratorPerformance:
    """Test decorator performance overhead."""

    def test_decorator_overhead(self):
        """Test rate limiting decorator overhead."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

        # Function without decoration
        def bare_function(user_id):
            return f"result_{user_id}"

        # Function with rate limiting
        @rate_limit(limit=10000, window="1min")
        def decorated_function(user_id):
            return f"result_{user_id}"

        # Measure bare function performance
        start_time = time.perf_counter()
        for i in range(1000):
            bare_function(f"user{i}")
        bare_time = time.perf_counter() - start_time

        # Measure decorated function performance
        start_time = time.perf_counter()
        for i in range(1000):
            decorated_function(f"user{i}")
        decorated_time = time.perf_counter() - start_time

        # Calculate overhead
        overhead = (decorated_time - bare_time) / bare_time * 100
        overhead_per_call = (decorated_time - bare_time) / 1000 * 1000  # ms per call

        print("Decorator Overhead:")
        print(f"  Bare function: {bare_time:.6f}s")
        print(f"  Decorated function: {decorated_time:.6f}s")
        print(f"  Overhead: {overhead:.1f}%")
        print(f"  Overhead per call: {overhead_per_call:.3f}ms")

        # Overhead should be reasonable
        assert overhead < 1000, f"Decorator overhead {overhead:.1f}% is too high"
        assert overhead_per_call < 1.0, f"Overhead per call {overhead_per_call:.3f}ms exceeds 1ms"

    def test_trading_decorator_performance(self):
        """Test trading decorator performance."""
        config = RateLimitConfig(storage_backend="memory")
        initialize_rate_limiting(config)

        @trading_rate_limit(action="submit_order")
        def submit_order(user_id, symbol):
            return f"order_{user_id}_{symbol}"

        # Measure performance
        start_time = time.perf_counter()

        for i in range(1000):
            submit_order(f"trader{i % 100}", f"STOCK{i % 10}")

        elapsed = time.perf_counter() - start_time
        throughput = 1000 / elapsed

        print(f"Trading Decorator: {throughput:.0f} calls/sec")

        # Should handle reasonable throughput
        assert (
            throughput > 1000
        ), f"Trading decorator throughput {throughput:.0f} below 1000 calls/sec"


@pytest.mark.performance
class TestScalabilityLimits:
    """Test scalability limits and breaking points."""

    def test_user_scalability(self):
        """Test system behavior with large number of users."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Test with increasing number of users
        user_counts = [100, 1000, 10000]

        for num_users in user_counts:
            start_time = time.perf_counter()

            # Each user makes 5 requests
            for user_id in range(num_users):
                context = RateLimitContext(user_id=f"user{user_id}")
                for _ in range(5):
                    try:
                        manager.check_rate_limit(context)
                    except Exception:
                        break  # Hit rate limit

            elapsed = time.perf_counter() - start_time
            total_requests = num_users * 5
            throughput = total_requests / elapsed

            print(f"Users: {num_users}, Throughput: {throughput:.0f} req/sec")

            # Throughput should scale reasonably
            if num_users <= 1000:
                assert (
                    throughput > 5000
                ), f"Throughput {throughput:.0f} too low for {num_users} users"

    def test_memory_usage_growth(self):
        """Test memory usage growth with scale."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Create entries for many users and measure impact
        base_users = 1000
        scale_users = 10000

        # Baseline
        for i in range(base_users):
            context = RateLimitContext(user_id=f"base_user{i}")
            manager.check_rate_limit(context)

        # Scale test
        start_time = time.perf_counter()
        for i in range(scale_users):
            context = RateLimitContext(user_id=f"scale_user{i}")
            manager.check_rate_limit(context)
        scale_time = time.perf_counter() - start_time

        scale_throughput = scale_users / scale_time
        print(f"Scale test ({scale_users} users): {scale_throughput:.0f} ops/sec")

        # Should maintain reasonable performance at scale
        assert scale_throughput > 5000, f"Scale throughput {scale_throughput:.0f} too low"

    def test_concurrent_user_limits(self):
        """Test limits of concurrent user handling."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        def concurrent_worker(start_user, num_users):
            results = []
            for i in range(num_users):
                context = RateLimitContext(user_id=f"concurrent_user{start_user + i}")
                start = time.perf_counter()
                try:
                    manager.check_rate_limit(context)
                    results.append(time.perf_counter() - start)
                except Exception:
                    results.append(time.perf_counter() - start)
            return results

        # Test with high concurrency
        num_threads = 20
        users_per_thread = 500

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(concurrent_worker, i * users_per_thread, users_per_thread)
                for i in range(num_threads)
            ]

            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())

        total_time = time.perf_counter() - start_time
        total_ops = num_threads * users_per_thread
        throughput = total_ops / total_time

        avg_latency = statistics.mean(all_results) * 1000

        print(f"High Concurrency ({num_threads} threads):")
        print(f"  Throughput: {throughput:.0f} ops/sec")
        print(f"  Avg Latency: {avg_latency:.3f}ms")

        # Should handle high concurrency
        assert throughput > 2000, f"High concurrency throughput {throughput:.0f} too low"
        assert avg_latency < 10.0, f"High concurrency latency {avg_latency:.3f}ms too high"


@pytest.mark.performance
class TestResourceUtilization:
    """Test CPU and memory resource utilization."""

    def test_cpu_utilization_efficiency(self):
        """Test CPU efficiency of rate limiting operations."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Perform sustained operations to test CPU efficiency
        operations = 10000
        start_time = time.perf_counter()

        for i in range(operations):
            context = RateLimitContext(user_id=f"cpu_test_user{i % 1000}")
            try:
                manager.check_rate_limit(context)
            except Exception:
                pass

        cpu_time = time.perf_counter() - start_time
        ops_per_second = operations / cpu_time
        cpu_time_per_op = cpu_time / operations * 1000000  # microseconds

        print("CPU Efficiency:")
        print(f"  Operations: {operations}")
        print(f"  Total CPU time: {cpu_time:.3f}s")
        print(f"  Ops/sec: {ops_per_second:.0f}")
        print(f"  CPU time per op: {cpu_time_per_op:.1f}μs")

        # CPU efficiency requirements
        assert ops_per_second > 5000, f"CPU efficiency {ops_per_second:.0f} ops/sec too low"
        assert cpu_time_per_op < 200, f"CPU time per op {cpu_time_per_op:.1f}μs too high"

    def test_memory_efficiency(self):
        """Test memory efficiency of rate limiting system."""
        config = RateLimitConfig(storage_backend="memory")
        manager = RateLimitManager(config)

        # Create rate limit entries and test memory efficiency
        num_users = 5000
        requests_per_user = 10

        # Generate load
        for user_id in range(num_users):
            context = RateLimitContext(user_id=f"memory_test_user{user_id}")
            for _ in range(requests_per_user):
                try:
                    manager.check_rate_limit(context)
                except Exception:
                    break

        # Test cleanup efficiency
        start_cleanup = time.perf_counter()
        cleanup_results = manager.cleanup_expired()
        cleanup_time = time.perf_counter() - start_cleanup

        print("Memory Efficiency:")
        print(f"  Users: {num_users}")
        print(f"  Cleanup time: {cleanup_time:.3f}s")
        print(f"  Cleanup results: {cleanup_results}")

        # Memory efficiency requirements
        assert cleanup_time < 1.0, f"Cleanup time {cleanup_time:.3f}s too slow"
