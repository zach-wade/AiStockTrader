"""
Unit tests for DatabaseFactory.
"""

# Standard library imports
from unittest.mock import Mock, patch

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.interfaces.database import IAsyncDatabase


class TestDatabaseFactory:
    """Test DatabaseFactory functionality."""

    def test_create_async_database(self):
        """Test creating an async database instance."""
        # Create test config
        config = DictConfig(
            {
                "database": {
                    "url": "postgresql://test:test@localhost:5432/testdb",
                    "pool_size": 5,
                    "max_overflow": 10,
                    "echo": False,
                }
            }
        )

        factory = DatabaseFactory()

        # Mock the AsyncDatabaseAdapter import
        with patch(
            "main.data_pipeline.storage.database_factory.AsyncDatabaseAdapter"
        ) as mock_adapter_class:
            mock_adapter_instance = Mock(spec=IAsyncDatabase)
            mock_adapter_class.return_value = mock_adapter_instance

            # Create database
            db = factory.create_async_database(config)

            # Verify
            assert db is mock_adapter_instance
            mock_adapter_class.assert_called_once_with(config)

    def test_create_async_database_with_dict_config(self):
        """Test creating an async database with plain dict config."""
        # Create test config as plain dict
        config = {"database": {"url": "postgresql://test:test@localhost:5432/testdb"}}

        factory = DatabaseFactory()

        with patch(
            "main.data_pipeline.storage.database_factory.AsyncDatabaseAdapter"
        ) as mock_adapter_class:
            mock_adapter_instance = Mock(spec=IAsyncDatabase)
            mock_adapter_class.return_value = mock_adapter_instance

            # Create database
            db = factory.create_async_database(config)

            # Verify
            assert db is mock_adapter_instance
            # The factory should have converted dict to DictConfig
            call_args = mock_adapter_class.call_args[0][0]
            assert isinstance(call_args, DictConfig)

    def test_get_database_with_async_type(self):
        """Test get_database with async type."""
        config = DictConfig({"database": {"url": "postgresql://test:test@localhost:5432/testdb"}})

        factory = DatabaseFactory()

        with patch(
            "main.data_pipeline.storage.database_factory.AsyncDatabaseAdapter"
        ) as mock_adapter_class:
            mock_adapter_instance = Mock(spec=IAsyncDatabase)
            mock_adapter_class.return_value = mock_adapter_instance

            # Get database with async type
            db = factory.get_database("async", config)

            # Verify
            assert db is mock_adapter_instance
            mock_adapter_class.assert_called_once_with(config)

    def test_factory_methods_are_static(self):
        """Test that factory methods can be called without instance."""
        config = DictConfig({"database": {"url": "postgresql://test:test@localhost:5432/testdb"}})

        # Should be able to call without creating instance
        with patch(
            "main.data_pipeline.storage.database_factory.AsyncDatabaseAdapter"
        ) as mock_adapter_class:
            mock_adapter_instance = Mock(spec=IAsyncDatabase)
            mock_adapter_class.return_value = mock_adapter_instance

            factory = DatabaseFactory()
            db = factory.create_async_database(config)

            assert db is mock_adapter_instance
