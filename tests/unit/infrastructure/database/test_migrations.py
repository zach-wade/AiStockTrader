"""
Unit tests for Database Migration System.

Tests migration management, schema creation, and version tracking functionality.
"""

# Standard library imports
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, mock_open, patch

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import RepositoryError
from src.infrastructure.database.migrations import (
    Migration,
    MigrationError,
    MigrationManager,
    SchemaManager,
)


@pytest.fixture
def sample_migration():
    """Sample migration for testing."""
    return Migration(
        version="001",
        name="create_orders_table",
        description="Create orders table with all required fields",
        up_sql="CREATE TABLE orders (id UUID PRIMARY KEY);",
        down_sql="DROP TABLE orders;",
        checksum="abc123def456",
    )


@pytest.fixture
def mock_adapter():
    """Mock database adapter for migration tests."""
    adapter = AsyncMock()
    adapter.execute_query.return_value = "EXECUTE 1"
    adapter.fetch_one.return_value = None
    adapter.fetch_all.return_value = []
    return adapter


@pytest.fixture
def migration_manager(mock_adapter):
    """Migration manager with mocked adapter."""
    return MigrationManager(mock_adapter, migration_dir=Path("/test/migrations"))


@pytest.mark.unit
class TestMigration:
    """Test Migration dataclass."""

    def test_migration_creation(self):
        """Test migration creation with all fields."""
        migration = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        assert migration.version == "001"
        assert migration.name == "create_users"
        assert migration.description == "Create users table"
        assert migration.up_sql == "CREATE TABLE users (id SERIAL);"
        assert migration.down_sql == "DROP TABLE users;"
        assert migration.checksum == "hash123"

    def test_migration_file_name_property(self):
        """Test migration file name generation."""
        migration = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        assert migration.file_name == "001_create_users.sql"

    def test_migration_full_name_property(self):
        """Test migration full name generation."""
        migration = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        assert migration.full_name == "001_create_users"

    def test_migration_equality(self):
        """Test migration equality comparison."""
        migration1 = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        migration2 = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        migration3 = Migration(
            version="002",
            name="create_orders",
            description="Create orders table",
            up_sql="CREATE TABLE orders (id SERIAL);",
            down_sql="DROP TABLE orders;",
            checksum="hash456",
        )

        assert migration1 == migration2
        assert migration1 != migration3

    def test_migration_string_representation(self):
        """Test migration string representation."""
        migration = Migration(
            version="001",
            name="create_users",
            description="Create users table",
            up_sql="CREATE TABLE users (id SERIAL);",
            down_sql="DROP TABLE users;",
            checksum="hash123",
        )

        result = str(migration)
        assert "001_create_users" in result
        assert "Create users table" in result


@pytest.mark.unit
class TestMigrationManager:
    """Test MigrationManager functionality."""

    def test_manager_initialization(self, mock_adapter):
        """Test manager initialization."""
        migration_dir = Path("/test/migrations")
        manager = MigrationManager(mock_adapter, migration_dir)

        assert manager.adapter == mock_adapter
        assert manager.migration_dir == migration_dir
        assert manager._applied_migrations == []

    @pytest.mark.asyncio
    async def test_ensure_migration_table_not_exists(self, migration_manager, mock_adapter):
        """Test migration table creation when it doesn't exist."""
        mock_adapter.fetch_one.return_value = None  # Table doesn't exist

        await migration_manager._ensure_migration_table()

        # Should execute CREATE TABLE command
        mock_adapter.execute_query.assert_called()
        create_call = mock_adapter.execute_query.call_args_list[0]
        assert "CREATE TABLE" in create_call[0][0]
        assert "schema_migrations" in create_call[0][0]

    @pytest.mark.asyncio
    async def test_ensure_migration_table_exists(self, migration_manager, mock_adapter):
        """Test migration table check when it already exists."""
        mock_adapter.fetch_one.return_value = {"table_name": "schema_migrations"}

        await migration_manager._ensure_migration_table()

        # Should only check existence, not create table
        assert mock_adapter.execute_query.call_count == 0

    @pytest.mark.asyncio
    async def test_load_applied_migrations(self, migration_manager, mock_adapter):
        """Test loading applied migrations from database."""
        mock_adapter.fetch_all.return_value = [
            {
                "version": "001",
                "name": "create_users",
                "checksum": "hash123",
                "applied_at": datetime.now(UTC),
            },
            {
                "version": "002",
                "name": "create_orders",
                "checksum": "hash456",
                "applied_at": datetime.now(UTC),
            },
        ]

        await migration_manager._load_applied_migrations()

        assert len(migration_manager._applied_migrations) == 2
        assert migration_manager._applied_migrations[0]["version"] == "001"
        assert migration_manager._applied_migrations[1]["version"] == "002"

    def test_is_migration_applied_true(self, migration_manager):
        """Test checking if migration is applied when it is."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_users", "checksum": "hash123"}
        ]

        result = migration_manager._is_migration_applied("001")
        assert result is True

    def test_is_migration_applied_false(self, migration_manager):
        """Test checking if migration is applied when it isn't."""
        migration_manager._applied_migrations = [
            {"version": "002", "name": "create_orders", "checksum": "hash456"}
        ]

        result = migration_manager._is_migration_applied("001")
        assert result is False

    def test_validate_migration_checksum_valid(self, migration_manager, sample_migration):
        """Test migration checksum validation when valid."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_orders_table", "checksum": "abc123def456"}
        ]

        # Should not raise error
        migration_manager._validate_migration_checksum(sample_migration)

    def test_validate_migration_checksum_invalid(self, migration_manager, sample_migration):
        """Test migration checksum validation when invalid."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_orders_table", "checksum": "different_hash"}
        ]

        with pytest.raises(MigrationError, match="Checksum mismatch"):
            migration_manager._validate_migration_checksum(sample_migration)

    def test_validate_migration_checksum_not_applied(self, migration_manager, sample_migration):
        """Test migration checksum validation when migration not applied."""
        migration_manager._applied_migrations = []

        # Should not raise error for non-applied migration
        migration_manager._validate_migration_checksum(sample_migration)

    @pytest.mark.asyncio
    async def test_record_migration(self, migration_manager, mock_adapter, sample_migration):
        """Test recording applied migration."""
        await migration_manager._record_migration(sample_migration)

        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args

        # Check INSERT statement
        assert "INSERT INTO schema_migrations" in call_args[0][0]
        assert call_args[0][1] == sample_migration.version
        assert call_args[0][2] == sample_migration.name
        assert call_args[0][3] == sample_migration.checksum

    @pytest.mark.asyncio
    async def test_remove_migration_record(self, migration_manager, mock_adapter):
        """Test removing migration record."""
        await migration_manager._remove_migration_record("001")

        mock_adapter.execute_query.assert_called_once()
        call_args = mock_adapter.execute_query.call_args

        # Check DELETE statement
        assert "DELETE FROM schema_migrations" in call_args[0][0]
        assert call_args[0][1] == "001"

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_discover_migrations(self, mock_glob, mock_exists, migration_manager):
        """Test discovering migration files."""
        mock_exists.return_value = True
        mock_files = [
            Path("/test/migrations/001_create_users.sql"),
            Path("/test/migrations/002_create_orders.sql"),
            Path("/test/migrations/003_add_indexes.sql"),
        ]
        mock_glob.return_value = mock_files

        with patch.object(migration_manager, "_load_migration_from_file") as mock_load:
            mock_load.side_effect = [
                Migration("001", "create_users", "Create users", "CREATE...", "DROP...", "hash1"),
                Migration("002", "create_orders", "Create orders", "CREATE...", "DROP...", "hash2"),
                Migration("003", "add_indexes", "Add indexes", "CREATE...", "DROP...", "hash3"),
            ]

            migrations = migration_manager._discover_migrations()

            assert len(migrations) == 3
            assert migrations[0].version == "001"
            assert migrations[1].version == "002"
            assert migrations[2].version == "003"

    @patch("pathlib.Path.exists")
    def test_discover_migrations_no_directory(self, mock_exists, migration_manager):
        """Test discovering migrations when directory doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(MigrationError, match="Migration directory does not exist"):
            migration_manager._discover_migrations()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="-- Migration: create_users\n-- Description: Create users table\n-- Up:\nCREATE TABLE users (id SERIAL);\n-- Down:\nDROP TABLE users;",
    )
    def test_load_migration_from_file(self, mock_file, migration_manager):
        """Test loading migration from file."""
        file_path = Path("/test/migrations/001_create_users.sql")

        migration = migration_manager._load_migration_from_file(file_path)

        assert migration.version == "001"
        assert migration.name == "create_users"
        assert migration.description == "Create users table"
        assert "CREATE TABLE users" in migration.up_sql
        assert "DROP TABLE users" in migration.down_sql

    @patch("builtins.open", new_callable=mock_open, read_data="Invalid migration file content")
    def test_load_migration_from_file_invalid(self, mock_file, migration_manager):
        """Test loading migration from invalid file."""
        file_path = Path("/test/migrations/001_invalid.sql")

        with pytest.raises(MigrationError, match="Invalid migration file format"):
            migration_manager._load_migration_from_file(file_path)

    @pytest.mark.asyncio
    async def test_apply_migration_success(self, migration_manager, mock_adapter, sample_migration):
        """Test successful migration application."""
        migration_manager._applied_migrations = []

        await migration_manager._apply_migration(sample_migration)

        # Should execute migration SQL and record it
        assert mock_adapter.execute_query.call_count == 2  # Migration SQL + record

        # Check migration SQL was executed
        migration_call = mock_adapter.execute_query.call_args_list[0]
        assert sample_migration.up_sql in migration_call[0][0]

    @pytest.mark.asyncio
    async def test_apply_migration_already_applied(self, migration_manager, sample_migration):
        """Test applying migration that's already applied."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_orders_table", "checksum": "abc123def456"}
        ]

        with pytest.raises(MigrationError, match="Migration .* is already applied"):
            await migration_manager._apply_migration(sample_migration)

    @pytest.mark.asyncio
    async def test_apply_migration_execution_error(
        self, migration_manager, mock_adapter, sample_migration
    ):
        """Test migration application with execution error."""
        migration_manager._applied_migrations = []
        mock_adapter.execute_query.side_effect = [
            RepositoryError("SQL execution failed"),
            "EXECUTE 1",  # For potential rollback
        ]

        with pytest.raises(MigrationError, match="Failed to apply migration"):
            await migration_manager._apply_migration(sample_migration)

    @pytest.mark.asyncio
    async def test_revert_migration_success(
        self, migration_manager, mock_adapter, sample_migration
    ):
        """Test successful migration reversion."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_orders_table", "checksum": "abc123def456"}
        ]

        await migration_manager._revert_migration(sample_migration)

        # Should execute down SQL and remove record
        assert mock_adapter.execute_query.call_count == 2  # Down SQL + remove record

        # Check down SQL was executed
        revert_call = mock_adapter.execute_query.call_args_list[0]
        assert sample_migration.down_sql in revert_call[0][0]

    @pytest.mark.asyncio
    async def test_revert_migration_not_applied(self, migration_manager, sample_migration):
        """Test reverting migration that's not applied."""
        migration_manager._applied_migrations = []

        with pytest.raises(MigrationError, match="Migration .* is not applied"):
            await migration_manager._revert_migration(sample_migration)

    @pytest.mark.asyncio
    async def test_get_current_version_with_migrations(self, migration_manager):
        """Test getting current version when migrations exist."""
        migration_manager._applied_migrations = [
            {"version": "001", "applied_at": datetime(2023, 1, 1, tzinfo=UTC)},
            {"version": "003", "applied_at": datetime(2023, 1, 3, tzinfo=UTC)},
            {"version": "002", "applied_at": datetime(2023, 1, 2, tzinfo=UTC)},
        ]

        version = await migration_manager.get_current_version()
        assert version == "003"  # Latest by applied_at

    @pytest.mark.asyncio
    async def test_get_current_version_no_migrations(self, migration_manager):
        """Test getting current version when no migrations applied."""
        migration_manager._applied_migrations = []

        version = await migration_manager.get_current_version()
        assert version is None

    @pytest.mark.asyncio
    async def test_get_pending_migrations(self, migration_manager):
        """Test getting pending migrations."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_users", "checksum": "hash1"}
        ]

        all_migrations = [
            Migration("001", "create_users", "Create users", "CREATE...", "DROP...", "hash1"),
            Migration("002", "create_orders", "Create orders", "CREATE...", "DROP...", "hash2"),
            Migration("003", "add_indexes", "Add indexes", "CREATE...", "DROP...", "hash3"),
        ]

        with patch.object(migration_manager, "_discover_migrations", return_value=all_migrations):
            pending = await migration_manager.get_pending_migrations()

            assert len(pending) == 2
            assert pending[0].version == "002"
            assert pending[1].version == "003"

    @pytest.mark.asyncio
    async def test_migrate_to_latest(self, migration_manager, mock_adapter):
        """Test migrating to latest version."""
        migration_manager._applied_migrations = []

        all_migrations = [
            Migration(
                "001",
                "create_users",
                "Create users",
                "CREATE TABLE users;",
                "DROP TABLE users;",
                "hash1",
            ),
            Migration(
                "002",
                "create_orders",
                "Create orders",
                "CREATE TABLE orders;",
                "DROP TABLE orders;",
                "hash2",
            ),
        ]

        with (
            patch.object(migration_manager, "_discover_migrations", return_value=all_migrations),
            patch.object(migration_manager, "_apply_migration") as mock_apply,
        ):
            result = await migration_manager.migrate_to_latest()

            assert result == 2  # Two migrations applied
            assert mock_apply.call_count == 2

    @pytest.mark.asyncio
    async def test_migrate_to_version(self, migration_manager):
        """Test migrating to specific version."""
        migration_manager._applied_migrations = []

        all_migrations = [
            Migration("001", "create_users", "Create users", "CREATE...", "DROP...", "hash1"),
            Migration("002", "create_orders", "Create orders", "CREATE...", "DROP...", "hash2"),
            Migration("003", "add_indexes", "Add indexes", "CREATE...", "DROP...", "hash3"),
        ]

        with (
            patch.object(migration_manager, "_discover_migrations", return_value=all_migrations),
            patch.object(migration_manager, "_apply_migration") as mock_apply,
        ):
            result = await migration_manager.migrate_to_version("002")

            assert result == 2  # Two migrations applied (001 and 002)
            assert mock_apply.call_count == 2

    @pytest.mark.asyncio
    async def test_migrate_to_invalid_version(self, migration_manager):
        """Test migrating to invalid version."""
        all_migrations = [
            Migration("001", "create_users", "Create users", "CREATE...", "DROP...", "hash1")
        ]

        with (
            patch.object(migration_manager, "_discover_migrations", return_value=all_migrations),
            pytest.raises(MigrationError, match="Migration version .* not found"),
        ):
            await migration_manager.migrate_to_version("999")

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, migration_manager):
        """Test rolling back to specific version."""
        migration_manager._applied_migrations = [
            {"version": "001", "name": "create_users", "checksum": "hash1"},
            {"version": "002", "name": "create_orders", "checksum": "hash2"},
            {"version": "003", "name": "add_indexes", "checksum": "hash3"},
        ]

        all_migrations = [
            Migration("001", "create_users", "Create users", "CREATE...", "DROP...", "hash1"),
            Migration("002", "create_orders", "Create orders", "CREATE...", "DROP...", "hash2"),
            Migration("003", "add_indexes", "Add indexes", "CREATE...", "DROP...", "hash3"),
        ]

        with (
            patch.object(migration_manager, "_discover_migrations", return_value=all_migrations),
            patch.object(migration_manager, "_revert_migration") as mock_revert,
        ):
            result = await migration_manager.rollback_to_version("001")

            assert result == 2  # Two migrations reverted (003 and 002)
            assert mock_revert.call_count == 2


@pytest.mark.unit
class TestSchemaManager:
    """Test SchemaManager functionality."""

    @pytest.fixture
    def schema_manager(self, mock_adapter):
        return SchemaManager(mock_adapter)

    @pytest.mark.asyncio
    async def test_create_tables_success(self, schema_manager, mock_adapter):
        """Test successful table creation."""
        with patch("builtins.open", mock_open(read_data="CREATE TABLE test (id SERIAL);")):
            await schema_manager.create_tables()

            mock_adapter.execute_query.assert_called()
            call_args = mock_adapter.execute_query.call_args
            assert "CREATE TABLE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_drop_tables_success(self, schema_manager, mock_adapter):
        """Test successful table dropping."""
        mock_adapter.fetch_values.return_value = ["orders", "positions", "portfolios"]

        await schema_manager.drop_tables()

        # Should execute DROP TABLE for each table
        assert mock_adapter.execute_query.call_count == 3

    @pytest.mark.asyncio
    async def test_table_exists_true(self, schema_manager, mock_adapter):
        """Test table existence check when table exists."""
        mock_adapter.fetch_one.return_value = {"table_name": "orders"}

        exists = await schema_manager.table_exists("orders")

        assert exists is True

    @pytest.mark.asyncio
    async def test_table_exists_false(self, schema_manager, mock_adapter):
        """Test table existence check when table doesn't exist."""
        mock_adapter.fetch_one.return_value = None

        exists = await schema_manager.table_exists("nonexistent")

        assert exists is False

    @pytest.mark.asyncio
    async def test_get_table_list(self, schema_manager, mock_adapter):
        """Test getting list of tables."""
        mock_adapter.fetch_values.return_value = ["orders", "positions", "portfolios"]

        tables = await schema_manager.get_table_list()

        assert tables == ["orders", "positions", "portfolios"]

    @pytest.mark.asyncio
    async def test_validate_schema_success(self, schema_manager, mock_adapter):
        """Test successful schema validation."""
        mock_adapter.fetch_values.return_value = [
            "orders",
            "positions",
            "portfolios",
            "schema_migrations",
        ]

        # Should not raise error
        await schema_manager.validate_schema()

    @pytest.mark.asyncio
    async def test_validate_schema_missing_tables(self, schema_manager, mock_adapter):
        """Test schema validation with missing tables."""
        mock_adapter.fetch_values.return_value = ["orders"]  # Missing tables

        with pytest.raises(MigrationError, match="Missing required tables"):
            await schema_manager.validate_schema()

    @pytest.mark.asyncio
    async def test_backup_schema(self, schema_manager, mock_adapter):
        """Test schema backup creation."""
        mock_adapter.fetch_values.return_value = ["orders", "positions"]

        with patch("builtins.open", mock_open()) as mock_file:
            backup_path = await schema_manager.backup_schema()

            assert backup_path.suffix == ".sql"
            mock_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_schema(self, schema_manager, mock_adapter):
        """Test schema restoration from backup."""
        backup_content = "CREATE TABLE orders (id UUID);\nCREATE TABLE positions (id UUID);"

        with patch("builtins.open", mock_open(read_data=backup_content)):
            await schema_manager.restore_schema(Path("/test/backup.sql"))

            # Should execute the backup SQL
            mock_adapter.execute_query.assert_called()


@pytest.mark.unit
class TestMigrationError:
    """Test MigrationError exception."""

    def test_migration_error_creation(self):
        """Test MigrationError creation."""
        error = MigrationError("Migration failed")

        assert str(error) == "Migration failed"
        assert isinstance(error, Exception)

    def test_migration_error_with_cause(self):
        """Test MigrationError with underlying cause."""
        cause = ValueError("Invalid SQL")
        try:
            raise MigrationError("Migration failed") from cause
        except MigrationError as error:
            assert str(error) == "Migration failed"
            assert error.__cause__ == cause
