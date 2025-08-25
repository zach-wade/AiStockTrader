"""
Comprehensive Redis Cache Layer Usage Example

This example demonstrates how to use the Redis caching layer in the trading system,
including basic operations, decorators, and trading-specific functionality.
"""

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

# Import cache components
from src.infrastructure.cache import (
    CacheConfig,
    CacheManager,
    cache_market_data,
    cache_portfolio_calculation,
    cache_result,
    invalidate_portfolio_cache,
)


async def basic_cache_operations_example():
    """Demonstrate basic cache operations."""
    print("=== Basic Cache Operations Example ===")

    # Initialize cache manager
    cache_manager = CacheManager()

    try:
        await cache_manager.connect()

        # Basic set and get
        test_data = {"symbol": "AAPL", "price": Decimal("150.25"), "timestamp": datetime.now()}

        print("Setting cache value...")
        await cache_manager.set("example_key", test_data, ttl=300)

        print("Getting cache value...")
        result = await cache_manager.get("example_key")
        print(f"Retrieved: {result}")

        # Check if key exists
        exists = await cache_manager._redis_cache.exists("example_key")
        print(f"Key exists: {exists}")

        # Delete key
        deleted = await cache_manager.delete("example_key")
        print(f"Key deleted: {deleted}")

    finally:
        await cache_manager.disconnect()


async def trading_specific_cache_example():
    """Demonstrate trading-specific cache operations."""
    print("\n=== Trading-Specific Cache Example ===")

    cache_manager = CacheManager()

    try:
        await cache_manager.connect()

        # Market data caching
        print("Caching market data...")
        market_data = {
            "price": Decimal("150.00"),
            "bid": Decimal("149.98"),
            "ask": Decimal("150.02"),
            "volume": 1000000,
            "timestamp": "2023-01-01T10:00:00Z",
        }

        await cache_manager.cache_market_data("AAPL", market_data, "quote")

        # Retrieve market data
        print("Retrieving market data...")
        retrieved_data = await cache_manager.get_market_data("AAPL", "quote")
        print(f"Market data: {retrieved_data}")

        # Portfolio calculation caching
        print("Caching portfolio calculation...")
        portfolio_result = {
            "total_value": Decimal("100000.00"),
            "total_pnl": Decimal("5000.00"),
            "cash": Decimal("10000.00"),
            "positions_count": 15,
        }

        await cache_manager.cache_portfolio_calculation("portfolio_123", "value", portfolio_result)

        # Risk calculation caching
        print("Caching risk calculation...")
        risk_metrics = {
            "var_95": Decimal("1000.00"),
            "var_99": Decimal("2000.00"),
            "beta": Decimal("1.2"),
            "sharpe_ratio": Decimal("1.5"),
            "max_drawdown": Decimal("0.05"),
        }

        calculation_params = {"lookback_days": 30, "confidence_level": 0.95, "method": "historical"}

        await cache_manager.cache_risk_calculation(
            "portfolio_123", "var", risk_metrics, calculation_params
        )

        # Retrieve risk calculation
        print("Retrieving risk calculation...")
        risk_data = await cache_manager.get_risk_calculation("portfolio_123", "var")
        print(f"Risk data: {risk_data}")

        # Session caching
        print("Caching user session...")
        session_data = {
            "user_id": "user_456",
            "login_time": datetime.now().isoformat(),
            "permissions": ["read", "write", "admin"],
            "preferences": {"theme": "dark", "notifications": True, "risk_tolerance": "moderate"},
        }

        await cache_manager.cache_user_session("session_789", session_data)

        # Retrieve session
        retrieved_session = await cache_manager.get_user_session("session_789")
        print(f"Session data: {retrieved_session}")

    finally:
        await cache_manager.disconnect()


async def cache_decorator_example():
    """Demonstrate cache decorators in action."""
    print("\n=== Cache Decorator Example ===")

    # Simulate expensive market data fetch
    @cache_market_data(ttl=60)
    async def fetch_stock_price(symbol: str) -> dict[str, Any]:
        """Simulate fetching stock price from external API."""
        print(f"Fetching price for {symbol} from external API...")
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            "symbol": symbol,
            "price": Decimal("150.00") + Decimal(str(hash(symbol) % 100)),
            "timestamp": datetime.now(),
            "source": "external_api",
        }

    # Simulate expensive portfolio calculation
    @cache_portfolio_calculation(ttl=300)
    async def calculate_portfolio_metrics(portfolio_id: str) -> dict[str, Any]:
        """Simulate complex portfolio calculation."""
        print(f"Calculating metrics for portfolio {portfolio_id}...")
        await asyncio.sleep(0.2)  # Simulate computation time

        return {
            "portfolio_id": portfolio_id,
            "total_value": Decimal("100000.00"),
            "daily_pnl": Decimal("1500.00"),
            "ytd_return": Decimal("0.15"),
            "sharpe_ratio": Decimal("1.8"),
            "calculated_at": datetime.now(),
        }

    # Function that invalidates portfolio cache
    @invalidate_portfolio_cache()
    async def update_portfolio_position(portfolio_id: str, symbol: str, quantity: int):
        """Simulate updating a portfolio position."""
        print(f"Updating {symbol} position to {quantity} shares in {portfolio_id}")
        await asyncio.sleep(0.05)  # Simulate database update
        return f"Updated {symbol} position successfully"

    # Test the decorated functions
    print("First call to fetch_stock_price (should fetch from API):")
    start_time = time.time()
    result1 = await fetch_stock_price("AAPL")
    time1 = time.time() - start_time
    print(f"Result: {result1['price']}, Time: {time1:.3f}s")

    print("\nSecond call to fetch_stock_price (should use cache):")
    start_time = time.time()
    result2 = await fetch_stock_price("AAPL")
    time2 = time.time() - start_time
    print(f"Result: {result2['price']}, Time: {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x faster")

    print("\nFirst call to calculate_portfolio_metrics:")
    start_time = time.time()
    portfolio_result1 = await calculate_portfolio_metrics("portfolio_123")
    time1 = time.time() - start_time
    print(f"Total value: {portfolio_result1['total_value']}, Time: {time1:.3f}s")

    print("\nSecond call to calculate_portfolio_metrics (should use cache):")
    start_time = time.time()
    portfolio_result2 = await calculate_portfolio_metrics("portfolio_123")
    time2 = time.time() - start_time
    print(f"Total value: {portfolio_result2['total_value']}, Time: {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x faster")

    print("\nUpdating portfolio (will invalidate cache):")
    await update_portfolio_position("portfolio_123", "AAPL", 150)

    print("\nCalling calculate_portfolio_metrics after invalidation:")
    start_time = time.time()
    portfolio_result3 = await calculate_portfolio_metrics("portfolio_123")
    time3 = time.time() - start_time
    print(f"Time: {time3:.3f}s (should be slower due to cache invalidation)")


async def batch_operations_example():
    """Demonstrate batch cache operations."""
    print("\n=== Batch Operations Example ===")

    cache_manager = CacheManager()

    try:
        await cache_manager.connect()

        # Prepare batch data
        batch_data = {}
        for i in range(10):
            batch_data[f"stock_{i}"] = {
                "symbol": f"STOCK{i:02d}",
                "price": Decimal(f"{100 + i * 10}"),
                "volume": 1000000 + i * 100000,
            }

        print("Setting multiple keys...")
        start_time = time.time()
        await cache_manager.set_many(batch_data, ttl=300)
        set_time = time.time() - start_time
        print(f"Set {len(batch_data)} keys in {set_time:.3f}s")

        print("Getting multiple keys...")
        start_time = time.time()
        keys = list(batch_data.keys())
        results = await cache_manager.get_many(keys)
        get_time = time.time() - start_time
        print(f"Retrieved {len(results)} keys in {get_time:.3f}s")

        print("Clearing namespace...")
        deleted_count = await cache_manager.clear_namespace("test")
        print(f"Cleared {deleted_count} keys from test namespace")

    finally:
        await cache_manager.disconnect()


async def performance_metrics_example():
    """Demonstrate cache performance monitoring."""
    print("\n=== Performance Metrics Example ===")

    cache_manager = CacheManager()

    try:
        await cache_manager.connect()

        # Perform various operations to generate metrics
        print("Performing operations to generate metrics...")

        # Cache hits and misses
        await cache_manager.set("perf_test_1", {"value": 1})
        await cache_manager.set("perf_test_2", {"value": 2})

        # Generate cache hits
        for i in range(5):
            await cache_manager.get("perf_test_1")

        # Generate cache misses
        for i in range(3):
            await cache_manager.get("perf_test_missing", "default")

        # Get comprehensive stats
        stats = await cache_manager.get_stats()

        print("\nCache Performance Statistics:")
        print("-" * 40)

        manager_stats = stats["cache_manager"]
        print(f"Total Operations: {manager_stats['total_operations']}")
        print(f"Cache Hits: {manager_stats['hit_count']}")
        print(f"Cache Misses: {manager_stats['miss_count']}")
        print(f"Hit Rate: {manager_stats['hit_rate']:.2%}")
        print(f"Error Rate: {manager_stats['error_rate']:.2%}")
        print(f"Average Latency: {manager_stats['avg_latency_ms']:.2f}ms")

        if "redis" in stats:
            redis_stats = stats["redis"]
            print("\nRedis Statistics:")
            print(f"Connected Clients: {redis_stats.get('connected_clients', 'N/A')}")
            print(f"Used Memory: {redis_stats.get('used_memory_human', 'N/A')}")
            if "keyspace_hits" in redis_stats and "keyspace_misses" in redis_stats:
                total_ops = redis_stats["keyspace_hits"] + redis_stats["keyspace_misses"]
                if total_ops > 0:
                    hit_rate = redis_stats["keyspace_hits"] / total_ops
                    print(f"Redis Hit Rate: {hit_rate:.2%}")

        # Namespace-specific stats
        if stats.get("namespace_stats"):
            print("\nNamespace Statistics:")
            for namespace, ns_stats in stats["namespace_stats"].items():
                print(f"  {namespace}:")
                print(f"    Operations: {ns_stats['total_operations']}")
                print(f"    Hit Rate: {ns_stats['hit_rate']:.2%}")
                print(f"    Avg Latency: {ns_stats['avg_latency_ms']:.2f}ms")

    finally:
        await cache_manager.disconnect()


async def error_handling_example():
    """Demonstrate cache error handling and resilience."""
    print("\n=== Error Handling Example ===")

    # Test with invalid configuration (should handle gracefully)
    invalid_config = CacheConfig()
    invalid_config.redis.host = "invalid_host"
    invalid_config.redis.port = 9999

    cache_manager = CacheManager(invalid_config)

    print("Testing with invalid Redis configuration...")
    try:
        await cache_manager.connect()
        print("Connected successfully (unexpected)")
    except Exception as e:
        print(f"Connection failed as expected: {type(e).__name__}")

    # Test decorator fallback behavior
    print("\nTesting decorator fallback behavior...")

    @cache_result(ttl=300, namespace="test")
    async def fallback_function(x: int) -> str:
        return f"computed_result_{x}"

    try:
        # This may work with fallback behavior or raise an exception
        result = await fallback_function(42)
        print(f"Function executed with result: {result}")
    except Exception as e:
        print(f"Function failed with cache unavailable: {type(e).__name__}")

    await cache_manager.disconnect()


async def main():
    """Run all cache examples."""
    print("Redis Cache Layer Comprehensive Example")
    print("=" * 50)

    try:
        await basic_cache_operations_example()
        await trading_specific_cache_example()
        await cache_decorator_example()
        await batch_operations_example()
        await performance_metrics_example()
        await error_handling_example()

        print("\n" + "=" * 50)
        print("All cache examples completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("- Sub-millisecond cache operations")
        print("- Automatic serialization of complex trading objects")
        print("- Decorator-based caching with minimal code changes")
        print("- Trading-specific cache operations")
        print("- Batch operations for high throughput")
        print("- Comprehensive performance monitoring")
        print("- Graceful error handling and fallback behavior")
        print("- Cache invalidation patterns")

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        print("This is likely due to Redis not being available.")
        print("To run this example:")
        print("1. Install Redis: pip install redis")
        print("2. Start Redis server: redis-server")
        print("3. Run this example again")


if __name__ == "__main__":
    asyncio.run(main())
