"""
Unit tests for base use case classes.

Tests the foundation use case patterns including request/response handling,
validation, error handling, and transactional behavior.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from src.application.use_cases.base import (
    TransactionalUseCase,
    UseCase,
    UseCaseRequest,
    UseCaseResponse,
)


# Test Request/Response Classes
@dataclass
class TestRequest:
    """Test request for use case testing."""

    test_field: str
    optional_field: str | None = None
    request_id: UUID | None = None
    correlation_id: UUID | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize request with defaults."""
        if self.request_id is None:
            self.request_id = uuid4()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestResponse(UseCaseResponse):
    """Test response for use case testing."""

    result: str | None = None


# Concrete Test Use Cases
class TestUseCase(UseCase[TestRequest, TestResponse]):
    """Concrete use case for testing."""

    def __init__(self, name: str | None = None):
        super().__init__(name)
        self.validate_called = False
        self.process_called = False
        self.validation_error: str | None = None
        self.process_result: TestResponse | None = None
        self.process_error: Exception | None = None

    async def validate(self, request: TestRequest) -> str | None:
        """Validate the test request."""
        self.validate_called = True
        return self.validation_error

    async def process(self, request: TestRequest) -> TestResponse:
        """Process the test request."""
        self.process_called = True
        if self.process_error:
            raise self.process_error
        return self.process_result or TestResponse(
            success=True,
            result=f"Processed: {getattr(request, 'test_field', 'default')}",
            request_id=getattr(request, "request_id", uuid4()),
        )


class TestTransactionalUseCase(TransactionalUseCase[TestRequest, TestResponse]):
    """Concrete transactional use case for testing."""

    def __init__(self, unit_of_work: AsyncMock, name: str | None = None):
        super().__init__(unit_of_work, name)
        self.validate_called = False
        self.process_called = False
        self.validation_error: str | None = None
        self.process_result: TestResponse | None = None
        self.process_error: Exception | None = None

    async def validate(self, request: TestRequest) -> str | None:
        """Validate the test request."""
        self.validate_called = True
        return self.validation_error

    async def process(self, request: TestRequest) -> TestResponse:
        """Process the test request."""
        self.process_called = True
        if self.process_error:
            raise self.process_error
        return self.process_result or TestResponse(
            success=True,
            result=f"Processed: {getattr(request, 'test_field', 'default')}",
            request_id=getattr(request, "request_id", uuid4()),
        )


# Test Fixtures
@pytest.fixture
def mock_unit_of_work():
    """Create a mock unit of work."""
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    return uow


# Test Classes
class TestUseCaseRequest:
    """Test the UseCaseRequest base class."""

    def test_request_initialization_with_defaults(self):
        """Test request initialization with default values."""
        request = UseCaseRequest()

        assert request.request_id is not None
        assert isinstance(request.request_id, UUID)
        assert request.correlation_id is None
        assert request.metadata == {}

    def test_request_initialization_with_values(self):
        """Test request initialization with provided values."""
        request_id = uuid4()
        correlation_id = uuid4()
        metadata = {"key": "value"}

        request = UseCaseRequest(
            request_id=request_id, correlation_id=correlation_id, metadata=metadata
        )

        assert request.request_id == request_id
        assert request.correlation_id == correlation_id
        assert request.metadata == metadata

    def test_request_post_init(self):
        """Test request post initialization."""
        request = UseCaseRequest(request_id=None, metadata=None)

        assert request.request_id is not None
        assert request.metadata == {}


class TestUseCaseResponse:
    """Test the UseCaseResponse base class."""

    def test_response_initialization(self):
        """Test response initialization."""
        response = UseCaseResponse(success=True, data="test_data", error=None, request_id=uuid4())

        assert response.success is True
        assert response.data == "test_data"
        assert response.error is None
        assert response.request_id is not None

    def test_success_response(self):
        """Test creating a success response."""
        request_id = uuid4()
        data = {"result": "success"}

        response = UseCaseResponse.success_response(data, request_id)

        assert response.success is True
        assert response.data == data
        assert response.error is None
        assert response.request_id == request_id

    def test_error_response(self):
        """Test creating an error response."""
        request_id = uuid4()
        error = "Test error message"

        response = UseCaseResponse.error_response(error, request_id)

        assert response.success is False
        assert response.data is None
        assert response.error == error
        assert response.request_id == request_id


class TestUseCaseBase:
    """Test the base UseCase class."""

    @pytest.mark.asyncio
    async def test_use_case_initialization_with_default_name(self):
        """Test use case initialization with default name."""
        use_case = TestUseCase()

        assert use_case.name == "TestUseCase"
        assert use_case.logger is not None

    @pytest.mark.asyncio
    async def test_use_case_initialization_with_custom_name(self):
        """Test use case initialization with custom name."""
        use_case = TestUseCase(name="CustomName")

        assert use_case.name == "CustomName"
        assert use_case.logger is not None

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful use case execution."""
        use_case = TestUseCase()
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert use_case.validate_called is True
        assert use_case.process_called is True
        assert response.success is True
        assert response.result == "Processed: test_value"
        assert response.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test use case execution with validation error."""
        use_case = TestUseCase()
        use_case.validation_error = "Validation failed"
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert use_case.validate_called is True
        assert use_case.process_called is False
        assert response.success is False
        assert response.error == "Validation failed"

    @pytest.mark.asyncio
    async def test_execution_with_process_exception(self):
        """Test use case execution with process exception."""
        use_case = TestUseCase()
        use_case.process_error = ValueError("Process error")
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert use_case.validate_called is True
        assert use_case.process_called is True
        assert response.success is False
        assert response.error == "Process error"

    @pytest.mark.asyncio
    async def test_execution_with_request_without_id(self):
        """Test execution with request that doesn't have request_id."""
        use_case = TestUseCase()

        # Create a simple request without request_id attribute
        class SimpleRequest:
            test_field = "test"

        request = SimpleRequest()
        response = await use_case.execute(request)

        # Should still execute successfully
        assert response.request_id is not None

    @pytest.mark.asyncio
    async def test_logging_during_execution(self):
        """Test that proper logging occurs during execution."""
        use_case = TestUseCase()
        request = TestRequest(test_field="test_value")

        with patch.object(use_case.logger, "info") as mock_info:
            with patch.object(use_case.logger, "warning") as mock_warning:
                with patch.object(use_case.logger, "error") as mock_error:
                    response = await use_case.execute(request)

                    # Should log execution start and success
                    assert mock_info.call_count == 2
                    assert not mock_warning.called
                    assert not mock_error.called

    @pytest.mark.asyncio
    async def test_logging_validation_failure(self):
        """Test logging when validation fails."""
        use_case = TestUseCase()
        use_case.validation_error = "Invalid input"
        request = TestRequest(test_field="test_value")

        with patch.object(use_case.logger, "warning") as mock_warning:
            response = await use_case.execute(request)

            mock_warning.assert_called_once()
            assert "Validation failed" in str(mock_warning.call_args)

    @pytest.mark.asyncio
    async def test_logging_process_error(self):
        """Test logging when process raises exception."""
        use_case = TestUseCase()
        use_case.process_error = RuntimeError("Process failed")
        request = TestRequest(test_field="test_value")

        with patch.object(use_case.logger, "error") as mock_error:
            response = await use_case.execute(request)

            mock_error.assert_called_once()
            assert "Error executing" in str(mock_error.call_args)

    @pytest.mark.asyncio
    async def test_create_error_response(self):
        """Test the _create_error_response method."""
        use_case = TestUseCase()
        request_id = uuid4()
        error_message = "Test error"

        response = use_case._create_error_response(error_message, request_id)

        assert response.success is False
        assert response.error == error_message
        assert response.request_id == request_id


class TestTransactionalUseCaseClass:
    """Test the TransactionalUseCase class."""

    @pytest.mark.asyncio
    async def test_transactional_use_case_initialization(self, mock_unit_of_work):
        """Test transactional use case initialization."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)

        assert use_case.unit_of_work == mock_unit_of_work
        assert use_case.name == "TestTransactionalUseCase"

    @pytest.mark.asyncio
    async def test_successful_transaction_commit(self, mock_unit_of_work):
        """Test successful transaction with commit."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert response.success is True
        mock_unit_of_work.__aenter__.assert_called_once()
        mock_unit_of_work.__aexit__.assert_called_once()
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_transaction_rollback(self, mock_unit_of_work):
        """Test failed transaction with rollback."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        use_case.process_result = TestResponse(
            success=False, error="Process failed", request_id=uuid4()
        )
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert response.success is False
        mock_unit_of_work.__aenter__.assert_called_once()
        mock_unit_of_work.__aexit__.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()
        mock_unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_during_transaction(self, mock_unit_of_work):
        """Test exception during transaction causes rollback."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        use_case.process_error = RuntimeError("Transaction error")
        request = TestRequest(test_field="test_value")

        # The base UseCase.execute catches exceptions and returns error response
        # But the TransactionalUseCase wraps it in a transaction context
        response = await use_case.execute(request)

        # The exception is caught by base UseCase.execute and converted to error response
        assert response.success is False
        assert response.error == "Transaction error"
        mock_unit_of_work.__aenter__.assert_called_once()
        # Rollback is called because response.success is False
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_error_in_transaction(self, mock_unit_of_work):
        """Test validation error in transactional use case."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        use_case.validation_error = "Invalid request"
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        assert response.success is False
        assert response.error == "Invalid request"
        mock_unit_of_work.rollback.assert_called_once()
        mock_unit_of_work.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_transaction_logging(self, mock_unit_of_work):
        """Test logging in transactional use case."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        request = TestRequest(test_field="test_value")

        with patch.object(use_case.logger, "info") as mock_info:
            response = await use_case.execute(request)

            # Should log transaction start, execution, and commit
            assert mock_info.call_count >= 3
            calls_str = str(mock_info.call_args_list)
            assert "Starting transaction" in calls_str
            assert "Transaction committed" in calls_str

    @pytest.mark.asyncio
    async def test_transaction_rollback_logging(self, mock_unit_of_work):
        """Test logging when transaction is rolled back."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)
        use_case.process_result = TestResponse(
            success=False, error="Process failed", request_id=uuid4()
        )
        request = TestRequest(test_field="test_value")

        with patch.object(use_case.logger, "info") as mock_info:
            response = await use_case.execute(request)

            # Check that rollback was logged
            calls_str = str(mock_info.call_args_list)
            assert "Transaction rolled back" in calls_str

    @pytest.mark.asyncio
    async def test_response_without_success_attribute(self, mock_unit_of_work):
        """Test handling response without success attribute."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)

        # Create a response without success attribute
        class CustomResponse:
            def __init__(self):
                self.data = "test"

        use_case.process_result = CustomResponse()
        request = TestRequest(test_field="test_value")

        response = await use_case.execute(request)

        # Should default to treating as successful
        mock_unit_of_work.commit.assert_called_once()
        mock_unit_of_work.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_request_without_request_id_in_transaction(self, mock_unit_of_work):
        """Test transactional use case with request without request_id."""
        use_case = TestTransactionalUseCase(mock_unit_of_work)

        class SimpleRequest:
            test_field = "test"

        request = SimpleRequest()
        response = await use_case.execute(request)

        # Should still execute successfully
        mock_unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_exception_during_context(self, mock_unit_of_work):
        """Test exception that occurs during transaction context management."""
        # Make the super().execute raise an exception
        use_case = TestTransactionalUseCase(mock_unit_of_work)

        # Mock super().execute to raise exception
        with patch.object(UseCase, "execute", side_effect=RuntimeError("Context error")):
            request = TestRequest(test_field="test_value")

            with pytest.raises(RuntimeError, match="Context error"):
                await use_case.execute(request)

            # Rollback should have been called
            mock_unit_of_work.rollback.assert_called_once()
            mock_unit_of_work.commit.assert_not_called()
