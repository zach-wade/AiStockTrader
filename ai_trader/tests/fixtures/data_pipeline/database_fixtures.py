"""
Database fixtures for data pipeline testing with REAL components.

Provides database setup and teardown for integration testing.
Uses real AsyncDatabaseAdapter with test database configuration.
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator, Any
from datetime import datetime, timezone

from main.data_pipeline.storage.database_adapter import AsyncDatabaseAdapter
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.database_models import Base


@pytest.fixture
async def test_database_config():
    """Provide test database configuration."""
    return {
        'database': {
            'host': os.getenv('TEST_DB_HOST', 'localhost'),
            'port': int(os.getenv('TEST_DB_PORT', '5432')),
            'name': os.getenv('TEST_DB_NAME', 'ai_trader_test'),
            'user': os.getenv('TEST_DB_USER', 'zachwade'),
            'password': os.getenv('TEST_DB_PASSWORD', '')
        }
    }


@pytest.fixture
async def real_database(test_database_config) -> AsyncGenerator[AsyncDatabaseAdapter, None]:
    """
    Provide a real AsyncDatabaseAdapter for integration testing.
    
    This creates a real database connection for testing, not a mock.
    """
    db = AsyncDatabaseAdapter(config=test_database_config)
    
    try:
        await db.initialize()
        yield db
    finally:
        await db.close()


@pytest.fixture
async def database_factory():
    """Provide a real DatabaseFactory."""
    return DatabaseFactory()


# Test database utilities
async def create_test_database_if_needed():
    """Create test database if it doesn't exist."""
    # This would connect to postgres and create the test database
    # Implementation depends on your setup
    pass


async def clean_test_database(db: AsyncDatabaseAdapter):
    """Clean all test data from database."""
    # Implementation to clean test data between tests
    pass


# Real component fixtures (no mocks)
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests using real components."""
    return {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'ai_trader_test',
            'user': 'zachwade',
            'password': ''
        },
        'storage': {
            'archive': {
                'storage_type': 'local',
                'local_path': '/tmp/ai_trader_test_archive'
            }
        },
        'data_pipeline': {
            'storage': {
                'archive': {
                    'storage_type': 'local',
                    'local_path': '/tmp/ai_trader_test_archive'
                }
            }
        }
    }