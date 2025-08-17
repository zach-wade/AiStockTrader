#!/usr/bin/env python3
"""
Test script to verify the psycopg3 conversion works correctly.

This script tests basic database connectivity and operations
using the converted infrastructure.
"""

# Standard library imports
import asyncio
import logging

# Local imports
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import ConnectionFactory, DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_connection():
    """Test basic database connection."""
    try:
        # Create database configuration
        config = DatabaseConfig.from_env()
        logger.info(f"Testing connection to {config.host}:{config.port}/{config.database}")

        # Create connection factory and connect
        factory = ConnectionFactory()
        connection = await factory.create_connection(config)

        logger.info("✅ Database connection successful")

        # Test connection pool
        pool = await connection.connect()
        logger.info(f"✅ Connection pool created with max_size={pool.max_size}")

        # Test adapter
        adapter = PostgreSQLAdapter(pool)

        # Test health check
        health = await adapter.health_check()
        logger.info(f"✅ Health check: {'passed' if health else 'failed'}")

        # Test basic query
        result = await adapter.fetch_one("SELECT 1 as test_value")
        logger.info(f"✅ Basic query successful: {result}")

        # Test connection info
        info = await adapter.get_connection_info()
        logger.info(f"✅ Connection info: {info}")

        # Cleanup
        await factory.close_all()
        logger.info("✅ Connection closed successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_transactions():
    """Test transaction functionality."""
    try:
        config = DatabaseConfig.from_env()
        factory = ConnectionFactory()
        connection = await factory.create_connection(config)
        pool = await connection.connect()
        adapter = PostgreSQLAdapter(pool)

        # Test transaction
        await adapter.begin_transaction()
        logger.info("✅ Transaction started")

        # Test query within transaction
        result = await adapter.fetch_one("SELECT 'transaction_test' as result")
        logger.info(f"✅ Query in transaction: {result}")

        # Test rollback
        await adapter.rollback_transaction()
        logger.info("✅ Transaction rolled back")

        # Cleanup
        await factory.close_all()

        return True

    except Exception as e:
        logger.error(f"❌ Transaction test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🔄 Starting psycopg3 conversion tests...")

    tests = [
        ("Basic Connection", test_connection),
        ("Transactions", test_transactions),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        result = await test_func()
        results.append((test_name, result))

    logger.info("\n📊 Test Results:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 All tests passed! psycopg3 conversion successful.")
    else:
        logger.error("\n💥 Some tests failed. Check the logs above.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
