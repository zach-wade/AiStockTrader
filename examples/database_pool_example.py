"""
Example: High-Performance Database Connection Pool Usage

Demonstrates the enhanced database connection pooling with:
- 100+ max connections for high throughput
- Exponential backoff retry logic
- Connection validation
- Real-time metrics monitoring
"""

import asyncio
import logging
from datetime import datetime

from src.infrastructure.database.connection import (
    ConnectionFactory,
    DatabaseConfig,
    DatabaseConnection,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def simulate_high_throughput_operations(
    connection: DatabaseConnection, num_operations: int = 100
):
    """
    Simulate high-throughput database operations.

    Args:
        connection: Database connection instance
        num_operations: Number of concurrent operations to simulate
    """

    async def execute_query(query_id: int):
        """Execute a single database query."""
        try:
            async with connection.acquire() as conn:
                async with conn.cursor() as cur:
                    # Simulate a query
                    await cur.execute(
                        "SELECT pg_sleep(0.01), %s as query_id",
                        (query_id,),  # 10ms query
                    )
                    result = await cur.fetchone()
                    logger.debug(f"Query {query_id} completed")
                    return result
        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}")
            raise

    # Execute operations concurrently
    tasks = [execute_query(i) for i in range(num_operations)]
    start_time = datetime.now()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = (datetime.now() - start_time).total_seconds()
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = sum(1 for r in results if isinstance(r, Exception))

    logger.info(
        f"Completed {num_operations} operations in {elapsed:.2f}s "
        f"(throughput: {num_operations/elapsed:.0f} ops/sec, "
        f"successful: {successful}, failed: {failed})"
    )

    return results


async def monitor_pool_health(connection: DatabaseConnection, duration: int = 60):
    """
    Monitor connection pool health for a specified duration.

    Args:
        connection: Database connection instance
        duration: Monitoring duration in seconds
    """
    logger.info(f"Starting pool health monitoring for {duration} seconds...")

    end_time = datetime.now().timestamp() + duration

    while datetime.now().timestamp() < end_time:
        stats = await connection.get_pool_stats()

        if stats.get("status") == "connected":
            current = stats.get("current_state", {})
            performance = stats.get("performance_metrics", {})
            health = stats.get("health_metrics", {})

            logger.info(
                f"Pool Status - Active: {current.get('active_connections', 0)}, "
                f"Idle: {current.get('idle_connections', 0)}, "
                f"Waiting: {current.get('waiting_requests', 0)}, "
                f"Avg Acquisition: {performance.get('avg_acquisition_time_ms', 0):.2f}ms, "
                f"Errors: {health.get('connection_errors', 0)}, "
                f"Exhausted: {health.get('pool_exhausted_count', 0)}"
            )

            # Alert if pool is near exhaustion
            pool_config = stats.get("pool_config", {})
            max_size = pool_config.get("max_size", 100)
            active = current.get("active_connections", 0)

            if active > max_size * 0.8:
                logger.warning(f"Pool near exhaustion: {active}/{max_size} connections active")

        await asyncio.sleep(5)  # Check every 5 seconds


async def demonstrate_retry_logic():
    """Demonstrate connection retry logic with simulated failures."""
    # Create config with invalid host to trigger retry
    config = DatabaseConfig(
        host="invalid-host-12345",  # This will fail
        port=5432,
        database="test_db",
        user="test_user",
        password="test_pass",
        max_pool_size=100,
        max_retry_attempts=3,
        initial_retry_delay=1.0,
        retry_backoff_multiplier=2.0,
    )

    logger.info("Testing retry logic with invalid host...")

    try:
        connection = DatabaseConnection(config)
        await connection.connect()
    except Exception as e:
        logger.info(f"Connection failed as expected after retries: {e}")

    logger.info("Retry demonstration complete")


async def main():
    """Main demonstration function."""
    logger.info("=== Database Connection Pool Enhancement Demo ===")

    # Configuration for high-throughput operations
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="ai_trader",
        user="postgres",
        password="postgres",  # Use environment variables in production
        # Pool configuration for high throughput
        min_pool_size=10,
        max_pool_size=100,  # Support 1000+ ops/sec
        max_idle_time=300.0,
        max_lifetime=3600.0,
        # Connection validation
        validate_on_checkout=True,
        validation_query="SELECT 1",
        max_validation_failures=3,
        # Retry configuration
        max_retry_attempts=5,
        initial_retry_delay=0.5,
        max_retry_delay=30.0,
        retry_backoff_multiplier=2.0,
        retry_jitter=True,
        # Health monitoring
        health_check_interval=10.0,
        health_check_timeout=5.0,
        enable_pool_metrics=True,
        metrics_collection_interval=5.0,
    )

    try:
        # Create connection with retry logic
        logger.info("Creating database connection with enhanced configuration...")
        connection = await ConnectionFactory.create_connection(config)

        # Display initial pool stats
        stats = await connection.get_pool_stats()
        logger.info(f"Initial pool configuration: {stats.get('pool_config', {})}")

        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_pool_health(connection, duration=30))

        # Test 1: Normal load
        logger.info("\n--- Test 1: Normal Load (50 concurrent queries) ---")
        await simulate_high_throughput_operations(connection, num_operations=50)

        # Test 2: High load
        logger.info("\n--- Test 2: High Load (200 concurrent queries) ---")
        await simulate_high_throughput_operations(connection, num_operations=200)

        # Test 3: Stress test
        logger.info("\n--- Test 3: Stress Test (500 concurrent queries) ---")
        await simulate_high_throughput_operations(connection, num_operations=500)

        # Wait for monitoring to complete
        await monitor_task

        # Final stats
        final_stats = await connection.get_pool_stats()
        logger.info("\n=== Final Pool Statistics ===")
        logger.info(f"Performance Metrics: {final_stats.get('performance_metrics', {})}")
        logger.info(f"Health Metrics: {final_stats.get('health_metrics', {})}")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

    finally:
        # Clean up
        await ConnectionFactory.close_all()
        logger.info("Database connections closed")

    # Demonstrate retry logic
    logger.info("\n=== Retry Logic Demonstration ===")
    await demonstrate_retry_logic()

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
