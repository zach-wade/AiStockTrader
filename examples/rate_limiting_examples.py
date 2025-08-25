"""
Rate Limiting Examples for the AI Trading System.

Demonstrates various usage patterns and integration scenarios
for the comprehensive rate limiting system.
"""

import asyncio
import threading
import time
from datetime import datetime
from typing import Any

# Import rate limiting components
from src.infrastructure.rate_limiting import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitContext,
    RateLimitExceeded,
    RateLimitManager,
    RateLimitRule,
    RateLimitTier,
    TradingRateLimitExceeded,
    TradingRateLimits,
    api_rate_limit,
    initialize_rate_limiting,
    ip_rate_limit,
    rate_limit,
    trading_rate_limit,
)


def example_basic_usage():
    """Basic rate limiting usage example."""
    print("=== Basic Rate Limiting Usage ===")

    # Configure rate limiting
    config = RateLimitConfig(storage_backend="memory", enable_monitoring=True)

    # Initialize the system
    initialize_rate_limiting(config)
    manager = RateLimitManager(config)

    # Create a context for rate limiting
    context = RateLimitContext(
        user_id="example_user", api_key="example_api_key", ip_address="192.168.1.1"
    )

    try:
        # Check rate limits (will apply default limits)
        statuses = manager.check_rate_limit(context)
        print(f"Rate limit check passed: {len(statuses)} limiters checked")

        for status in statuses:
            print(f"  {status.rule_id}: {status.remaining}/{status.limit} remaining")

    except RateLimitExceeded as e:
        print(f"Rate limit exceeded: {e.message}")
        print(f"Retry after: {e.retry_after} seconds")


def example_decorator_usage():
    """Rate limiting decorators usage example."""
    print("\n=== Rate Limiting Decorators ===")

    # Initialize rate limiting
    config = RateLimitConfig(storage_backend="memory")
    initialize_rate_limiting(config)

    # Basic rate limiting decorator
    @rate_limit(limit=5, window="1min")
    def get_user_data(user_id: str) -> dict[str, Any]:
        return {"user_id": user_id, "data": "user information"}

    # Trading-specific rate limiting
    @trading_rate_limit(action="submit_order")
    def submit_order(user_id: str, symbol: str, quantity: int) -> dict[str, Any]:
        return {
            "order_id": f"order_{int(time.time())}",
            "user_id": user_id,
            "symbol": symbol,
            "quantity": quantity,
            "status": "submitted",
        }

    # API endpoint rate limiting
    @api_rate_limit(tier=RateLimitTier.PREMIUM)
    def premium_api_endpoint(user_id: str) -> dict[str, Any]:
        return {"premium_data": "exclusive information"}

    # IP-based rate limiting
    @ip_rate_limit(limit=10, window="1min")
    def public_endpoint(ip_address: str) -> str:
        return f"Public data for IP {ip_address}"

    # Test the decorated functions
    try:
        # Test basic rate limiting
        for i in range(3):
            result = get_user_data("user123")
            print(f"User data call {i+1}: {result['user_id']}")

        # Test trading rate limiting
        order_result = submit_order("trader1", "AAPL", 100)
        print(f"Order submitted: {order_result['order_id']}")

        # Test premium API
        premium_result = premium_api_endpoint("premium_user")
        print(f"Premium API result: {premium_result['premium_data']}")

        # Test IP limiting
        public_result = public_endpoint("192.168.1.1")
        print(f"Public endpoint: {public_result}")

    except RateLimitExceeded as e:
        print(f"Rate limit exceeded in decorator example: {e.message}")


def example_trading_platform():
    """Complete trading platform rate limiting example."""
    print("\n=== Trading Platform Example ===")

    # Configure trading-specific rate limits
    trading_limits = TradingRateLimits()
    trading_limits.order_submission = RateLimitRule(
        limit=100,  # 100 orders per minute
        window="1min",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_allowance=20,  # Allow short bursts
    )
    trading_limits.market_data_requests = RateLimitRule(
        limit=1000,  # 1000 market data requests per minute
        window="1min",
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
    )
    trading_limits.portfolio_queries = RateLimitRule(
        limit=50,
        window="1min",  # 50 portfolio queries per minute
    )

    config = RateLimitConfig(
        storage_backend="memory", trading_limits=trading_limits, enable_monitoring=True
    )

    initialize_rate_limiting(config)

    # Define trading functions with rate limiting
    @trading_rate_limit(action="submit_order")
    def submit_order(user_id: str, symbol: str, quantity: int, order_type: str = "market"):
        return {
            "order_id": f"ORD_{int(time.time())}_{symbol}",
            "user_id": user_id,
            "symbol": symbol,
            "quantity": quantity,
            "order_type": order_type,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "submitted",
        }

    @trading_rate_limit(action="get_market_data")
    def get_market_data(user_id: str, symbol: str):
        return {
            "symbol": symbol,
            "price": 150.0 + (hash(symbol + str(time.time())) % 100),
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @trading_rate_limit(action="get_portfolio")
    def get_portfolio(user_id: str):
        return {
            "user_id": user_id,
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "avg_price": 150.0},
                {"symbol": "GOOGL", "quantity": 50, "avg_price": 2800.0},
            ],
            "cash_balance": 10000.0,
            "total_value": 265000.0,
        }

    # Simulate trading activity
    trader_id = "professional_trader"
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    try:
        print(f"Trader {trader_id} starting trading session...")

        # Get portfolio
        portfolio = get_portfolio(trader_id)
        print(f"Portfolio value: ${portfolio['total_value']:,.2f}")

        # Get market data for watchlist
        for symbol in symbols:
            market_data = get_market_data(trader_id, symbol)
            print(f"{symbol}: ${market_data['price']:.2f}")

        # Submit some orders
        orders = [("AAPL", 100, "market"), ("GOOGL", 10, "limit"), ("MSFT", 50, "market")]

        for symbol, quantity, order_type in orders:
            order = submit_order(trader_id, symbol, quantity, order_type)
            print(f"Order submitted: {order['order_id']} - {quantity} shares of {symbol}")

        print("Trading session completed successfully!")

    except TradingRateLimitExceeded as e:
        print(f"Trading rate limit exceeded: {e.message}")
        print(f"Action: {e.trading_action}, User: {e.user_id}")
        print(f"Retry after: {e.retry_after} seconds")


def example_api_gateway():
    """API Gateway rate limiting example."""
    print("\n=== API Gateway Example ===")

    # Configure tiered rate limits
    config = RateLimitConfig(
        storage_backend="memory",
        enable_ip_limiting=True,
        enable_user_limiting=True,
        enable_api_key_limiting=True,
    )

    # Customize limits for different tiers
    config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
        limit=100, window="1min"
    )
    config.default_limits[RateLimitTier.PREMIUM]["api_requests"] = RateLimitRule(
        limit=500, window="1min"
    )
    config.default_limits[RateLimitTier.ENTERPRISE]["api_requests"] = RateLimitRule(
        limit=2000, window="1min"
    )

    manager = RateLimitManager(config)

    # Simulate different types of users
    users = [
        {
            "user_id": "basic_user_1",
            "api_key": "basic_key_1",
            "tier": RateLimitTier.BASIC,
            "ip": "192.168.1.100",
        },
        {
            "user_id": "premium_user_1",
            "api_key": "premium_key_1",
            "tier": RateLimitTier.PREMIUM,
            "ip": "192.168.1.101",
        },
        {
            "user_id": "enterprise_user_1",
            "api_key": "enterprise_key_1",
            "tier": RateLimitTier.ENTERPRISE,
            "ip": "192.168.1.102",
        },
    ]

    # Test rate limiting for different tiers
    for user in users:
        print(f"\nTesting {user['tier'].value} user: {user['user_id']}")

        context = RateLimitContext(
            user_id=user["user_id"],
            api_key=user["api_key"],
            ip_address=user["ip"],
            user_tier=user["tier"],
            endpoint="/api/v1/data",
            method="GET",
        )

        try:
            # Make several API calls
            for i in range(5):
                statuses = manager.check_rate_limit(context)
                remaining = min(s.remaining for s in statuses)
                print(f"  Request {i+1}: Success (min remaining: {remaining})")

        except RateLimitExceeded as e:
            print(f"  Rate limit exceeded: {e.message}")

        # Get current status
        status = manager.get_status(context)
        for rule_type, rule_status in status.items():
            if "remaining" in rule_status:
                print(f"  {rule_type}: {rule_status['remaining']}/{rule_status['limit']} remaining")


def example_multi_algorithm_comparison():
    """Compare different rate limiting algorithms."""
    print("\n=== Algorithm Comparison ===")

    algorithms = [
        RateLimitAlgorithm.TOKEN_BUCKET,
        RateLimitAlgorithm.SLIDING_WINDOW,
        RateLimitAlgorithm.FIXED_WINDOW,
    ]

    for algorithm in algorithms:
        print(f"\nTesting {algorithm.value} algorithm:")

        rule = RateLimitRule(limit=5, window="10s", algorithm=algorithm)

        config = RateLimitConfig(storage_backend="memory")
        config.default_limits[RateLimitTier.BASIC]["test"] = rule

        manager = RateLimitManager(config)
        context = RateLimitContext(user_id="test_user")

        # Test burst behavior
        print("  Burst test (5 rapid requests):")
        for i in range(5):
            try:
                statuses = manager.check_rate_limit(context, rule_types=["test"])
                print(f"    Request {i+1}: Allowed")
            except RateLimitExceeded:
                print(f"    Request {i+1}: Denied")

        # Test 6th request
        try:
            manager.check_rate_limit(context, rule_types=["test"])
            print("    Request 6: Allowed")
        except RateLimitExceeded:
            print("    Request 6: Denied")


def example_concurrent_access():
    """Demonstrate rate limiting under concurrent access."""
    print("\n=== Concurrent Access Example ===")

    config = RateLimitConfig(storage_backend="memory")
    config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
        limit=100, window="1min"
    )

    manager = RateLimitManager(config)
    results = []

    def worker(thread_id: int, num_requests: int):
        """Worker function for concurrent testing."""
        thread_results = []

        for i in range(num_requests):
            context = RateLimitContext(user_id=f"user_{thread_id}_{i}")

            try:
                statuses = manager.check_rate_limit(context)
                thread_results.append("allowed")
            except RateLimitExceeded:
                thread_results.append("denied")

        results.extend(thread_results)

    # Start multiple threads
    threads = []
    num_threads = 5
    requests_per_thread = 20

    print(f"Starting {num_threads} threads with {requests_per_thread} requests each...")

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i, requests_per_thread))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Analyze results
    total_requests = len(results)
    allowed_requests = results.count("allowed")
    denied_requests = results.count("denied")

    print("Results:")
    print(f"  Total requests: {total_requests}")
    print(f"  Allowed: {allowed_requests}")
    print(f"  Denied: {denied_requests}")
    print(f"  Success rate: {allowed_requests/total_requests:.1%}")


def example_monitoring_and_metrics():
    """Demonstrate monitoring and metrics collection."""
    print("\n=== Monitoring and Metrics Example ===")

    from src.infrastructure.rate_limiting.monitoring import initialize_monitoring

    # Configure with monitoring enabled
    config = RateLimitConfig(
        storage_backend="memory",
        enable_monitoring=True,
        alert_threshold=0.8,  # Alert at 80% utilization
    )

    # Initialize monitoring
    monitor = initialize_monitoring(config)
    manager = RateLimitManager(config)

    # Override some limits for demo
    config.default_limits[RateLimitTier.BASIC]["api_requests"] = RateLimitRule(
        limit=10, window="1min"
    )

    context = RateLimitContext(user_id="monitored_user", api_key="monitored_key")

    print("Generating rate limit activity for monitoring...")

    # Generate some activity
    for i in range(15):  # More than the limit
        try:
            start_time = time.perf_counter()
            statuses = manager.check_rate_limit(context)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record the check in monitoring
            for status in statuses:
                monitor.record_rate_limit_check(
                    status.rule_id,
                    status.identifier,
                    status.allowed,
                    status.current_count,
                    status.limit,
                    duration_ms,
                )

            print(f"  Request {i+1}: Allowed")

        except RateLimitExceeded as e:
            print(f"  Request {i+1}: Denied - {e.message}")

    # Get dashboard data
    dashboard_data = monitor.get_dashboard_data()

    print("\nMonitoring Dashboard Data:")
    print(f"  Total requests (5min): {dashboard_data['metrics']['health']['total_requests_5min']}")
    print(f"  Error rate (5min): {dashboard_data['metrics']['health']['error_rate_5min']:.1%}")
    print(f"  System healthy: {dashboard_data['metrics']['health']['healthy']}")

    # Get health status
    health = monitor.get_health_status()
    print(f"\nHealth Status: {health['status']}")
    print(f"  Healthy: {health['healthy']}")
    print(f"  Error rate: {health['checks']['error_rate']:.1%}")


async def example_async_usage():
    """Example of rate limiting in async context."""
    print("\n=== Async Usage Example ===")

    # Note: This is a conceptual example
    # The rate limiting system is thread-safe and can be used with async code

    config = RateLimitConfig(storage_backend="memory")
    manager = RateLimitManager(config)

    async def async_api_call(user_id: str, request_id: int):
        """Simulate an async API call with rate limiting."""
        context = RateLimitContext(user_id=user_id)

        try:
            # Rate limit check (synchronous but fast)
            statuses = manager.check_rate_limit(context)

            # Simulate async work
            await asyncio.sleep(0.1)

            return f"Request {request_id} completed for {user_id}"

        except RateLimitExceeded:
            return f"Request {request_id} rate limited for {user_id}"

    # Create multiple concurrent async calls
    tasks = []
    for i in range(10):
        task = async_api_call("async_user", i)
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"  {result}")


def main():
    """Run all examples."""
    print("AI Trading System - Rate Limiting Examples")
    print("=" * 50)

    # Run synchronous examples
    example_basic_usage()
    example_decorator_usage()
    example_trading_platform()
    example_api_gateway()
    example_multi_algorithm_comparison()
    example_concurrent_access()
    example_monitoring_and_metrics()

    # Run async example
    print("\nRunning async example...")
    asyncio.run(example_async_usage())

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
