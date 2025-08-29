"""
Unit tests for Unit of Work interface contracts.

Tests the Unit of Work interfaces to ensure they define proper contracts
for transaction management and atomic operations.
"""

# Standard library imports
from typing import Any
from unittest.mock import AsyncMock

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.exceptions import FactoryError, TransactionError
from src.application.interfaces.unit_of_work import (
    ITransactionManager,
    IUnitOfWork,
    IUnitOfWorkFactory,
)


class MockUnitOfWork:
    """Mock implementation of IUnitOfWork for interface testing."""

    def __init__(self):
        self.orders = AsyncMock()
        self.positions = AsyncMock()
        self.portfolios = AsyncMock()
        self.call_log = []
        self._active = False
        self._committed = False
        self._rolled_back = False

    async def begin_transaction(self) -> None:
        self.call_log.append("begin_transaction")
        if self._active:
            raise TransactionError("Transaction is already active")
        self._active = True

    async def commit(self) -> None:
        self.call_log.append("commit")
        if not self._active:
            raise TransactionError("No active transaction to commit")
        self._committed = True
        self._active = False

    async def rollback(self) -> None:
        self.call_log.append("rollback")
        if not self._active:
            return  # Allow rollback when no active transaction
        self._rolled_back = True
        self._active = False

    async def is_active(self) -> bool:
        self.call_log.append("is_active")
        return self._active

    async def flush(self) -> None:
        self.call_log.append("flush")
        if not self._active:
            raise TransactionError("Cannot flush without active transaction")

    async def refresh(self, entity) -> None:
        self.call_log.append(("refresh", entity))
        if not self._active:
            raise TransactionError("Cannot refresh without active transaction")

    async def __aenter__(self):
        self.call_log.append("__aenter__")
        await self.begin_transaction()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.call_log.append(("__aexit__", exc_type, exc_val, exc_tb))
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()


class MockUnitOfWorkFactory:
    """Mock implementation of IUnitOfWorkFactory for interface testing."""

    def __init__(self):
        self.call_log = []
        self.should_fail = False

    def create_unit_of_work(self) -> IUnitOfWork:
        self.call_log.append("create_unit_of_work")
        if self.should_fail:
            raise FactoryError("UnitOfWork", "Failed to create unit of work")
        return MockUnitOfWork()


class MockTransactionManager:
    """Mock implementation of ITransactionManager for interface testing."""

    def __init__(self):
        self.call_log = []
        self.uow_factory = MockUnitOfWorkFactory()

    async def execute_in_transaction(self, operation) -> Any:
        self.call_log.append(("execute_in_transaction", operation))
        async with self.uow_factory.create_unit_of_work() as uow:
            return await operation(uow)

    async def execute_with_retry(
        self, operation, max_retries: int = 3, backoff_factor: float = 1.0
    ) -> Any:
        self.call_log.append(("execute_with_retry", operation, max_retries, backoff_factor))

        for attempt in range(max_retries + 1):
            try:
                async with self.uow_factory.create_unit_of_work() as uow:
                    return await operation(uow)
            except Exception:
                if attempt == max_retries:
                    raise
                # In real implementation, would sleep with backoff
                continue

    async def execute_batch(self, operations: list) -> list[Any]:
        self.call_log.append(("execute_batch", operations))
        results = []
        async with self.uow_factory.create_unit_of_work() as uow:
            for operation in operations:
                result = await operation(uow)
                results.append(result)
        return results


@pytest.mark.unit
class TestUnitOfWorkInterface:
    """Test IUnitOfWork interface contract."""

    @pytest.fixture
    def uow(self):
        return MockUnitOfWork()

    @pytest.mark.asyncio
    async def test_transaction_lifecycle(self, uow):
        """Test basic transaction lifecycle."""
        # Begin transaction
        await uow.begin_transaction()
        assert await uow.is_active()
        assert "begin_transaction" in uow.call_log

        # Commit transaction
        await uow.commit()
        assert not await uow.is_active()
        assert "commit" in uow.call_log
        assert uow._committed

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, uow):
        """Test transaction rollback."""
        await uow.begin_transaction()
        assert await uow.is_active()

        await uow.rollback()
        assert not await uow.is_active()
        assert "rollback" in uow.call_log
        assert uow._rolled_back

    @pytest.mark.asyncio
    async def test_double_begin_transaction_error(self, uow):
        """Test that beginning a transaction twice raises error."""
        await uow.begin_transaction()

        with pytest.raises(TransactionError, match="already active"):
            await uow.begin_transaction()

    @pytest.mark.asyncio
    async def test_commit_without_transaction_error(self, uow):
        """Test that committing without active transaction raises error."""
        with pytest.raises(TransactionError, match="No active transaction"):
            await uow.commit()

    @pytest.mark.asyncio
    async def test_flush_without_transaction_error(self, uow):
        """Test that flushing without active transaction raises error."""
        with pytest.raises(TransactionError, match="Cannot flush without active transaction"):
            await uow.flush()

    @pytest.mark.asyncio
    async def test_refresh_without_transaction_error(self, uow):
        """Test that refreshing without active transaction raises error."""
        with pytest.raises(TransactionError, match="Cannot refresh without active transaction"):
            await uow.refresh("some_entity")

    @pytest.mark.asyncio
    async def test_flush_with_active_transaction(self, uow):
        """Test flush with active transaction."""
        await uow.begin_transaction()
        await uow.flush()

        assert "flush" in uow.call_log

    @pytest.mark.asyncio
    async def test_refresh_with_active_transaction(self, uow):
        """Test refresh with active transaction."""
        await uow.begin_transaction()
        entity = "test_entity"
        await uow.refresh(entity)

        assert ("refresh", entity) in uow.call_log

    @pytest.mark.asyncio
    async def test_repository_access(self, uow):
        """Test that UoW provides repository access."""
        assert hasattr(uow, "orders")
        assert hasattr(uow, "positions")
        assert hasattr(uow, "portfolios")

    @pytest.mark.asyncio
    async def test_context_manager_success(self, uow):
        """Test context manager with successful operation."""
        async with uow as context_uow:
            assert context_uow is uow
            assert await uow.is_active()

        assert "__aenter__" in uow.call_log
        assert any("__aexit__" in str(entry) for entry in uow.call_log)
        assert uow._committed
        assert not uow._rolled_back

    @pytest.mark.asyncio
    async def test_context_manager_exception(self, uow):
        """Test context manager with exception causing rollback."""
        try:
            async with uow:
                assert await uow.is_active()
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert "__aenter__" in uow.call_log
        assert any("__aexit__" in str(entry) for entry in uow.call_log)
        assert not uow._committed
        assert uow._rolled_back

    @pytest.mark.asyncio
    async def test_rollback_when_no_active_transaction(self, uow):
        """Test that rollback doesn't raise error when no active transaction."""
        # Should not raise error
        await uow.rollback()
        assert "rollback" in uow.call_log


@pytest.mark.unit
class TestUnitOfWorkFactoryInterface:
    """Test IUnitOfWorkFactory interface contract."""

    @pytest.fixture
    def factory(self):
        return MockUnitOfWorkFactory()

    def test_create_unit_of_work_success(self, factory):
        """Test successful unit of work creation."""
        uow = factory.create_unit_of_work()

        assert uow is not None
        assert hasattr(uow, "orders")
        assert hasattr(uow, "positions")
        assert hasattr(uow, "portfolios")
        assert "create_unit_of_work" in factory.call_log

    def test_create_unit_of_work_failure(self, factory):
        """Test unit of work creation failure."""
        factory.should_fail = True

        with pytest.raises(FactoryError, match="Failed to create unit of work"):
            factory.create_unit_of_work()

    def test_factory_creates_different_instances(self, factory):
        """Test that factory creates different instances."""
        uow1 = factory.create_unit_of_work()
        uow2 = factory.create_unit_of_work()

        assert uow1 is not uow2
        assert len(factory.call_log) == 2


@pytest.mark.unit
class TestTransactionManagerInterface:
    """Test ITransactionManager interface contract."""

    @pytest.fixture
    def manager(self):
        return MockTransactionManager()

    @pytest.mark.asyncio
    async def test_execute_in_transaction_success(self, manager):
        """Test successful operation execution in transaction."""

        @pytest.mark.asyncio
        async def test_operation(uow):
            return "success"

        result = await manager.execute_in_transaction(test_operation)

        assert result == "success"
        assert any("execute_in_transaction" in str(entry) for entry in manager.call_log)

    @pytest.mark.asyncio
    async def test_execute_in_transaction_failure(self, manager):
        """Test operation failure in transaction."""

        async def failing_operation(uow):
            raise ValueError("Operation failed")

        with pytest.raises(ValueError, match="Operation failed"):
            await manager.execute_in_transaction(failing_operation)

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, manager):
        """Test successful operation with retry mechanism."""

        @pytest.mark.asyncio
        async def test_operation(uow):
            return "success"

        result = await manager.execute_with_retry(test_operation, max_retries=3, backoff_factor=1.5)

        assert result == "success"
        assert any("execute_with_retry" in str(entry) for entry in manager.call_log)

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, manager):
        """Test operation that succeeds after retries."""
        call_count = 0

        async def eventually_successful_operation(uow):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await manager.execute_with_retry(eventually_successful_operation, max_retries=3)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_max_retries_exceeded(self, manager):
        """Test operation that fails after max retries."""

        async def always_failing_operation(uow):
            raise ValueError("Persistent failure")

        with pytest.raises(ValueError, match="Persistent failure"):
            await manager.execute_with_retry(always_failing_operation, max_retries=2)

    @pytest.mark.asyncio
    async def test_execute_batch_success(self, manager):
        """Test successful batch operation execution."""

        async def operation1(uow):
            return "result1"

        async def operation2(uow):
            return "result2"

        operations = [operation1, operation2]
        results = await manager.execute_batch(operations)

        assert results == ["result1", "result2"]
        assert any("execute_batch" in str(entry) for entry in manager.call_log)

    @pytest.mark.asyncio
    async def test_execute_batch_partial_failure(self, manager):
        """Test batch operation with one failing operation."""

        async def successful_operation(uow):
            return "success"

        async def failing_operation(uow):
            raise ValueError("Operation failed")

        operations = [successful_operation, failing_operation]

        with pytest.raises(ValueError, match="Operation failed"):
            await manager.execute_batch(operations)


@pytest.mark.unit
class TestTransactionErrorHandling:
    """Test transaction-related error handling."""

    def test_transaction_error_creation(self):
        """Test TransactionError creation and inheritance."""
        error = TransactionError("Transaction failed")

        assert str(error) == "Transaction failed"
        assert isinstance(error, Exception)

    def test_factory_error_creation(self):
        """Test FactoryError creation and inheritance."""
        error = FactoryError("UnitOfWork", "Factory failed")

        assert str(error) == "UnitOfWork factory error: Factory failed"
        assert isinstance(error, Exception)
        assert error.factory_type == "UnitOfWork"


@pytest.mark.unit
class TestInterfaceTypeAnnotations:
    """Test interface type annotations and protocol compliance."""

    def test_unit_of_work_protocol_methods(self):
        """Test IUnitOfWork protocol has expected methods."""
        # Local imports
        from src.application.interfaces.unit_of_work import IUnitOfWork

        expected_methods = [
            "begin_transaction",
            "commit",
            "rollback",
            "is_active",
            "flush",
            "refresh",
            "__aenter__",
            "__aexit__",
        ]

        for method in expected_methods:
            assert hasattr(IUnitOfWork, method)

        # Check repository properties via annotations (Protocol classes don't have actual attributes)
        expected_properties = ["orders", "positions", "portfolios"]
        for prop in expected_properties:
            assert prop in IUnitOfWork.__annotations__

    def test_unit_of_work_factory_protocol_methods(self):
        """Test IUnitOfWorkFactory protocol has expected methods."""

        expected_methods = ["create_unit_of_work"]

        for method in expected_methods:
            assert hasattr(IUnitOfWorkFactory, method)

    def test_transaction_manager_protocol_methods(self):
        """Test ITransactionManager protocol has expected methods."""

        expected_methods = ["execute_in_transaction", "execute_with_retry", "execute_batch"]

        for method in expected_methods:
            assert hasattr(ITransactionManager, method)


@pytest.mark.unit
class TestConcurrentTransactionBehavior:
    """Test concurrent transaction behavior patterns."""

    @pytest.mark.asyncio
    async def test_nested_transaction_context_managers(self):
        """Test that nested context managers work properly."""
        uow = MockUnitOfWork()

        # Outer transaction
        async with uow as outer:
            assert await outer.is_active()

            # Inner operations (would be separate UoW in real implementation)
            inner_uow = MockUnitOfWork()
            async with inner_uow as inner:
                assert await inner.is_active()

            # Outer transaction still active
            assert await outer.is_active()

        # Both transactions committed
        assert outer._committed
        assert inner._committed

    @pytest.mark.asyncio
    async def test_transaction_isolation(self):
        """Test transaction isolation between different UoW instances."""
        uow1 = MockUnitOfWork()
        uow2 = MockUnitOfWork()

        await uow1.begin_transaction()
        await uow2.begin_transaction()

        assert await uow1.is_active()
        assert await uow2.is_active()

        # Commit one, rollback other
        await uow1.commit()
        await uow2.rollback()

        assert uow1._committed
        assert uow2._rolled_back
        assert not uow1._rolled_back
        assert not uow2._committed
