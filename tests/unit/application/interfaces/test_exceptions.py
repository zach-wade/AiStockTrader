"""
Tests for application interface exceptions.

Tests all exception classes defined in the interfaces module to ensure
proper initialization and attribute handling.
"""

from uuid import uuid4

import pytest

from src.application.interfaces.exceptions import (
    ConcurrencyError,
    ConfigurationError,
    ConnectionError,
    DuplicateEntityError,
    EntityNotFoundError,
    FactoryError,
    IntegrityError,
    OrderNotFoundError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    RepositoryError,
    TimeoutError,
    TransactionAlreadyActiveError,
    TransactionCommitError,
    TransactionError,
    TransactionNotActiveError,
    TransactionRollbackError,
    ValidationError,
)


class TestRepositoryError:
    """Test the base RepositoryError class."""

    def test_init_with_message_only(self):
        """Test creating RepositoryError with message only."""
        message = "Something went wrong"
        error = RepositoryError(message)

        assert str(error) == message
        assert error.cause is None

    def test_init_with_message_and_cause(self):
        """Test creating RepositoryError with message and cause."""
        message = "Something went wrong"
        cause = ValueError("Original error")
        error = RepositoryError(message, cause)

        assert str(error) == message
        assert error.cause == cause

    def test_inheritance(self):
        """Test that RepositoryError inherits from Exception."""
        error = RepositoryError("test")
        assert isinstance(error, Exception)


class TestEntityNotFoundError:
    """Test the EntityNotFoundError class."""

    def test_init_with_uuid(self):
        """Test creating EntityNotFoundError with UUID identifier."""
        entity_type = "Order"
        identifier = uuid4()
        error = EntityNotFoundError(entity_type, identifier)

        expected_message = f"{entity_type} with identifier '{identifier}' not found"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_init_with_string(self):
        """Test creating EntityNotFoundError with string identifier."""
        entity_type = "Symbol"
        identifier = "AAPL"
        error = EntityNotFoundError(entity_type, identifier)

        expected_message = f"{entity_type} with identifier '{identifier}' not found"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_inheritance(self):
        """Test that EntityNotFoundError inherits from RepositoryError."""
        error = EntityNotFoundError("Test", "123")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestOrderNotFoundError:
    """Test the OrderNotFoundError class."""

    def test_init(self):
        """Test creating OrderNotFoundError."""
        order_id = uuid4()
        error = OrderNotFoundError(order_id)

        expected_message = f"Order with identifier '{order_id}' not found"
        assert str(error) == expected_message
        assert error.entity_type == "Order"
        assert error.identifier == order_id
        assert error.order_id == order_id

    def test_inheritance(self):
        """Test that OrderNotFoundError inherits from EntityNotFoundError."""
        error = OrderNotFoundError(uuid4())
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestPositionNotFoundError:
    """Test the PositionNotFoundError class."""

    def test_init(self):
        """Test creating PositionNotFoundError."""
        position_id = uuid4()
        error = PositionNotFoundError(position_id)

        expected_message = f"Position with identifier '{position_id}' not found"
        assert str(error) == expected_message
        assert error.entity_type == "Position"
        assert error.identifier == position_id
        assert error.position_id == position_id

    def test_inheritance(self):
        """Test that PositionNotFoundError inherits from EntityNotFoundError."""
        error = PositionNotFoundError(uuid4())
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestPortfolioNotFoundError:
    """Test the PortfolioNotFoundError class."""

    def test_init(self):
        """Test creating PortfolioNotFoundError."""
        portfolio_id = uuid4()
        error = PortfolioNotFoundError(portfolio_id)

        expected_message = f"Portfolio with identifier '{portfolio_id}' not found"
        assert str(error) == expected_message
        assert error.entity_type == "Portfolio"
        assert error.identifier == portfolio_id
        assert error.portfolio_id == portfolio_id

    def test_inheritance(self):
        """Test that PortfolioNotFoundError inherits from EntityNotFoundError."""
        error = PortfolioNotFoundError(uuid4())
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestDuplicateEntityError:
    """Test the DuplicateEntityError class."""

    def test_init_with_uuid(self):
        """Test creating DuplicateEntityError with UUID identifier."""
        entity_type = "Order"
        identifier = uuid4()
        error = DuplicateEntityError(entity_type, identifier)

        expected_message = f"{entity_type} with identifier '{identifier}' already exists"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_init_with_string(self):
        """Test creating DuplicateEntityError with string identifier."""
        entity_type = "Symbol"
        identifier = "AAPL"
        error = DuplicateEntityError(entity_type, identifier)

        expected_message = f"{entity_type} with identifier '{identifier}' already exists"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_inheritance(self):
        """Test that DuplicateEntityError inherits from RepositoryError."""
        error = DuplicateEntityError("Test", "123")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestConcurrencyError:
    """Test the ConcurrencyError class."""

    def test_init_with_uuid(self):
        """Test creating ConcurrencyError with UUID identifier."""
        entity_type = "Order"
        identifier = uuid4()
        error = ConcurrencyError(entity_type, identifier)

        expected_message = (
            f"{entity_type} with identifier '{identifier}' was modified by another process"
        )
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_init_with_string(self):
        """Test creating ConcurrencyError with string identifier."""
        entity_type = "Position"
        identifier = "POS123"
        error = ConcurrencyError(entity_type, identifier)

        expected_message = (
            f"{entity_type} with identifier '{identifier}' was modified by another process"
        )
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.identifier == identifier

    def test_inheritance(self):
        """Test that ConcurrencyError inherits from RepositoryError."""
        error = ConcurrencyError("Test", "123")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestValidationError:
    """Test the ValidationError class."""

    def test_init(self):
        """Test creating ValidationError."""
        entity_type = "Order"
        field = "quantity"
        value = -100
        message = "Quantity must be positive"
        error = ValidationError(entity_type, field, value, message)

        expected_message = f"{entity_type}.{field} validation failed: {message}"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.field == field
        assert error.value == value

    def test_init_with_none_value(self):
        """Test creating ValidationError with None value."""
        entity_type = "Position"
        field = "entry_price"
        value = None
        message = "Entry price is required"
        error = ValidationError(entity_type, field, value, message)

        expected_message = f"{entity_type}.{field} validation failed: {message}"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.field == field
        assert error.value is None

    def test_init_with_complex_value(self):
        """Test creating ValidationError with complex value."""
        entity_type = "Portfolio"
        field = "positions"
        value = {"AAPL": 100, "GOOGL": 50}
        message = "Too many positions"
        error = ValidationError(entity_type, field, value, message)

        expected_message = f"{entity_type}.{field} validation failed: {message}"
        assert str(error) == expected_message
        assert error.entity_type == entity_type
        assert error.field == field
        assert error.value == value

    def test_inheritance(self):
        """Test that ValidationError inherits from RepositoryError."""
        error = ValidationError("Test", "field", "value", "message")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTransactionError:
    """Test the base TransactionError class."""

    def test_init(self):
        """Test creating TransactionError."""
        message = "Transaction failed"
        error = TransactionError(message)

        assert str(error) == message

    def test_inheritance(self):
        """Test that TransactionError inherits from RepositoryError."""
        error = TransactionError("test")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTransactionNotActiveError:
    """Test the TransactionNotActiveError class."""

    def test_init(self):
        """Test creating TransactionNotActiveError."""
        error = TransactionNotActiveError()

        assert str(error) == "No active transaction"

    def test_inheritance(self):
        """Test that TransactionNotActiveError inherits from TransactionError."""
        error = TransactionNotActiveError()
        assert isinstance(error, TransactionError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTransactionAlreadyActiveError:
    """Test the TransactionAlreadyActiveError class."""

    def test_init(self):
        """Test creating TransactionAlreadyActiveError."""
        error = TransactionAlreadyActiveError()

        assert str(error) == "Transaction is already active"

    def test_inheritance(self):
        """Test that TransactionAlreadyActiveError inherits from TransactionError."""
        error = TransactionAlreadyActiveError()
        assert isinstance(error, TransactionError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTransactionCommitError:
    """Test the TransactionCommitError class."""

    def test_init_without_cause(self):
        """Test creating TransactionCommitError without cause."""
        error = TransactionCommitError()

        assert str(error) == "Transaction commit failed"
        assert error.cause is None

    def test_init_with_cause(self):
        """Test creating TransactionCommitError with cause."""
        cause = ValueError("Constraint violation")
        error = TransactionCommitError(cause)

        assert str(error) == "Transaction commit failed"
        assert error.cause == cause

    def test_inheritance(self):
        """Test that TransactionCommitError inherits from TransactionError."""
        error = TransactionCommitError()
        assert isinstance(error, TransactionError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTransactionRollbackError:
    """Test the TransactionRollbackError class."""

    def test_init_without_cause(self):
        """Test creating TransactionRollbackError without cause."""
        error = TransactionRollbackError()

        assert str(error) == "Transaction rollback failed"
        assert error.cause is None

    def test_init_with_cause(self):
        """Test creating TransactionRollbackError with cause."""
        cause = RuntimeError("Connection lost")
        error = TransactionRollbackError(cause)

        assert str(error) == "Transaction rollback failed"
        assert error.cause == cause

    def test_inheritance(self):
        """Test that TransactionRollbackError inherits from TransactionError."""
        error = TransactionRollbackError()
        assert isinstance(error, TransactionError)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestConnectionError:
    """Test the ConnectionError class."""

    def test_init_default_message(self):
        """Test creating ConnectionError with default message."""
        error = ConnectionError()

        assert str(error) == "Database connection failed"

    def test_init_custom_message(self):
        """Test creating ConnectionError with custom message."""
        message = "Connection to database server failed"
        error = ConnectionError(message)

        assert str(error) == message

    def test_inheritance(self):
        """Test that ConnectionError inherits from RepositoryError."""
        error = ConnectionError()
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestTimeoutError:
    """Test the TimeoutError class."""

    def test_init(self):
        """Test creating TimeoutError."""
        operation = "query_orders"
        timeout_seconds = 30.5
        error = TimeoutError(operation, timeout_seconds)

        expected_message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        assert str(error) == expected_message
        assert error.operation == operation
        assert error.timeout_seconds == timeout_seconds

    def test_init_integer_timeout(self):
        """Test creating TimeoutError with integer timeout."""
        operation = "save_position"
        timeout_seconds = 10
        error = TimeoutError(operation, timeout_seconds)

        expected_message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        assert str(error) == expected_message
        assert error.operation == operation
        assert error.timeout_seconds == timeout_seconds

    def test_inheritance(self):
        """Test that TimeoutError inherits from RepositoryError."""
        error = TimeoutError("test", 1.0)
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestIntegrityError:
    """Test the IntegrityError class."""

    def test_init_constraint_only(self):
        """Test creating IntegrityError with constraint only."""
        constraint = "unique_order_id"
        error = IntegrityError(constraint)

        expected_message = f"Integrity constraint '{constraint}' violated"
        assert str(error) == expected_message
        assert error.constraint == constraint

    def test_init_constraint_and_message(self):
        """Test creating IntegrityError with constraint and message."""
        constraint = "foreign_key_portfolio_id"
        message = "Referenced portfolio does not exist"
        error = IntegrityError(constraint, message)

        expected_message = f"Integrity constraint '{constraint}' violated: {message}"
        assert str(error) == expected_message
        assert error.constraint == constraint

    def test_init_empty_message(self):
        """Test creating IntegrityError with empty message."""
        constraint = "check_positive_quantity"
        message = ""
        error = IntegrityError(constraint, message)

        expected_message = f"Integrity constraint '{constraint}' violated"
        assert str(error) == expected_message
        assert error.constraint == constraint

    def test_inheritance(self):
        """Test that IntegrityError inherits from RepositoryError."""
        error = IntegrityError("test")
        assert isinstance(error, RepositoryError)
        assert isinstance(error, Exception)


class TestFactoryError:
    """Test the FactoryError class."""

    def test_init(self):
        """Test creating FactoryError."""
        factory_type = "BrokerFactory"
        message = "Unknown broker type 'invalid'"
        error = FactoryError(factory_type, message)

        expected_message = f"{factory_type} factory error: {message}"
        assert str(error) == expected_message
        assert error.factory_type == factory_type

    def test_init_repository_factory(self):
        """Test creating FactoryError for repository factory."""
        factory_type = "RepositoryFactory"
        message = "Database connection not available"
        error = FactoryError(factory_type, message)

        expected_message = f"{factory_type} factory error: {message}"
        assert str(error) == expected_message
        assert error.factory_type == factory_type

    def test_inheritance(self):
        """Test that FactoryError inherits from Exception."""
        error = FactoryError("Test", "message")
        assert isinstance(error, Exception)
        # Note: FactoryError does NOT inherit from RepositoryError


class TestConfigurationError:
    """Test the ConfigurationError class."""

    def test_init_message_only(self):
        """Test creating ConfigurationError with message only."""
        message = "Invalid configuration parameter"
        error = ConfigurationError(message)

        assert str(error) == message
        assert error.cause is None

    def test_init_message_and_cause(self):
        """Test creating ConfigurationError with message and cause."""
        message = "Failed to load configuration file"
        cause = FileNotFoundError("config.yaml not found")
        error = ConfigurationError(message, cause)

        assert str(error) == message
        assert error.cause == cause

    def test_inheritance(self):
        """Test that ConfigurationError inherits from Exception."""
        error = ConfigurationError("test")
        assert isinstance(error, Exception)
        # Note: ConfigurationError does NOT inherit from RepositoryError


class TestExceptionRaisingAndCatching:
    """Test that exceptions can be properly raised and caught."""

    def test_raise_and_catch_repository_error(self):
        """Test raising and catching RepositoryError."""
        with pytest.raises(RepositoryError) as exc_info:
            raise RepositoryError("test error")

        assert str(exc_info.value) == "test error"

    def test_raise_and_catch_order_not_found_error(self):
        """Test raising and catching OrderNotFoundError."""
        order_id = uuid4()

        with pytest.raises(OrderNotFoundError) as exc_info:
            raise OrderNotFoundError(order_id)

        assert exc_info.value.order_id == order_id

    def test_catch_specific_exception_as_base(self):
        """Test catching specific exception as base exception."""
        order_id = uuid4()

        with pytest.raises(RepositoryError) as exc_info:
            raise OrderNotFoundError(order_id)

        # Should be able to catch OrderNotFoundError as RepositoryError
        assert isinstance(exc_info.value, OrderNotFoundError)
        assert isinstance(exc_info.value, RepositoryError)

    def test_catch_transaction_error_as_repository_error(self):
        """Test catching TransactionError as RepositoryError."""
        with pytest.raises(RepositoryError) as exc_info:
            raise TransactionNotActiveError()

        # Should be able to catch TransactionNotActiveError as RepositoryError
        assert isinstance(exc_info.value, TransactionNotActiveError)
        assert isinstance(exc_info.value, TransactionError)
        assert isinstance(exc_info.value, RepositoryError)

    def test_factory_error_not_repository_error(self):
        """Test that FactoryError cannot be caught as RepositoryError."""
        with pytest.raises(FactoryError):
            raise FactoryError("TestFactory", "test message")

        # Should NOT be catchable as RepositoryError
        with pytest.raises(FactoryError):
            try:
                raise FactoryError("TestFactory", "test message")
            except RepositoryError:
                pytest.fail("FactoryError should not be caught as RepositoryError")

    def test_configuration_error_not_repository_error(self):
        """Test that ConfigurationError cannot be caught as RepositoryError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("test message")

        # Should NOT be catchable as RepositoryError
        with pytest.raises(ConfigurationError):
            try:
                raise ConfigurationError("test message")
            except RepositoryError:
                pytest.fail("ConfigurationError should not be caught as RepositoryError")
