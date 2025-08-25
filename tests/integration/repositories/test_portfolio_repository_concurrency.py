"""
Integration tests for Portfolio repository concurrency features.

Tests optimistic locking, pessimistic locking, and concurrent database operations.
"""

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio

from src.application.interfaces.exceptions import ConcurrencyError, PortfolioNotFoundError
from src.domain.entities.portfolio import Portfolio
from src.domain.exceptions import DeadlockException, OptimisticLockException, StaleDataException
from src.domain.services.concurrency_service import ConcurrencyService
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.repositories.portfolio_repository import PostgreSQLPortfolioRepository


class TestPortfolioRepositoryConcurrency:
    """Test suite for portfolio repository concurrency features."""

    @pytest_asyncio.fixture
    async def mock_adapter(self):
        """Create a mock database adapter for testing."""

        class MockAdapter(PostgreSQLAdapter):
            def __init__(self):
                self.data = {}
                self.locked_rows = set()
                self.transaction_active = False
                self.version_sequence = {}

            async def fetch_one(self, query: str, *params):
                if "FOR UPDATE NOWAIT" in query:
                    # Simulate lock acquisition
                    if params and params[0] in self.locked_rows:
                        raise Exception("could not obtain lock")
                    if params:
                        self.locked_rows.add(params[0])

                if "SELECT" in query and "portfolios" in query:
                    if params and params[0] in self.data:
                        portfolio_data = self.data[params[0]].copy()
                        return portfolio_data
                    return None

                return {"version": 1}

            async def execute_query(self, query: str, *params):
                if query.strip().upper() == "BEGIN":
                    self.transaction_active = True
                    return "BEGIN"

                if query.strip().upper() == "COMMIT":
                    self.transaction_active = False
                    self.locked_rows.clear()
                    return "COMMIT"

                if query.strip().upper() == "ROLLBACK":
                    self.transaction_active = False
                    self.locked_rows.clear()
                    return "ROLLBACK"

                if "UPDATE" in query and "portfolios" in query:
                    # Simulate optimistic locking
                    portfolio_id = None
                    expected_version = None

                    # Extract portfolio ID and version from params
                    if params:
                        # Assume portfolio ID is second to last param and version is last
                        portfolio_id = params[-2]
                        expected_version = params[-1]

                    if portfolio_id in self.data:
                        current_version = self.data[portfolio_id]["version"]

                        if current_version != expected_version:
                            return "UPDATE 0"

                        # Update successful
                        self.data[portfolio_id]["version"] = current_version + 1
                        return "UPDATE 1"

                    return "UPDATE 0"

                if "INSERT" in query:
                    return "INSERT 0 1"

                return "OK"

            async def fetch_all(self, query: str, *params):
                return []

        return MockAdapter()

    @pytest_asyncio.fixture
    async def repository(self, mock_adapter):
        """Create a portfolio repository for testing."""
        concurrency_service = ConcurrencyService(max_retries=3)
        return PostgreSQLPortfolioRepository(mock_adapter, concurrency_service=concurrency_service)

    @pytest_asyncio.fixture
    async def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        return Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            version=1,
        )

    @pytest.mark.asyncio
    async def test_optimistic_locking_success(self, repository, sample_portfolio, mock_adapter):
        """Test successful optimistic locking update."""
        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "version": sample_portfolio.version,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # Update portfolio
        sample_portfolio.cash_balance = Decimal("95000")

        updated = await repository.update_portfolio(sample_portfolio)

        # Version should be incremented
        assert updated.version > sample_portfolio.version
        assert mock_adapter.data[sample_portfolio.id]["version"] == updated.version

    @pytest.mark.asyncio
    async def test_optimistic_locking_version_conflict(
        self, repository, sample_portfolio, mock_adapter
    ):
        """Test version conflict in optimistic locking."""
        # Add portfolio to mock data with different version
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version + 1,  # Higher version
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # This should trigger retry logic due to version mismatch
        with pytest.raises(OptimisticLockException):
            await repository.update_portfolio(sample_portfolio)

    @pytest.mark.asyncio
    async def test_pessimistic_locking_success(self, repository, sample_portfolio, mock_adapter):
        """Test successful pessimistic locking."""
        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # Should successfully acquire lock
        locked_portfolio = await repository.get_portfolio_for_update(sample_portfolio.id)

        assert locked_portfolio is not None
        assert locked_portfolio.id == sample_portfolio.id
        assert sample_portfolio.id in mock_adapter.locked_rows

    @pytest.mark.asyncio
    async def test_pessimistic_locking_conflict(self, repository, sample_portfolio, mock_adapter):
        """Test pessimistic locking conflict."""
        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # Simulate another transaction holding the lock
        mock_adapter.locked_rows.add(sample_portfolio.id)

        with pytest.raises(ConcurrencyError):
            await repository.get_portfolio_for_update(sample_portfolio.id)

    @pytest.mark.asyncio
    async def test_atomic_update_success(self, repository, sample_portfolio, mock_adapter):
        """Test successful atomic update."""
        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        updates = {
            "cash_balance": Decimal("95000"),
            "trades_count": 1,
        }

        updated = await repository.update_portfolio_atomic(
            sample_portfolio.id, updates, expected_version=sample_portfolio.version
        )

        assert updated.version == sample_portfolio.version + 1
        assert mock_adapter.data[sample_portfolio.id]["version"] == updated.version

    @pytest.mark.asyncio
    async def test_atomic_update_version_mismatch(self, repository, sample_portfolio, mock_adapter):
        """Test atomic update with version mismatch."""
        # Add portfolio to mock data with different version
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version + 1,  # Different version
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        updates = {"cash_balance": Decimal("95000")}

        with pytest.raises(StaleDataException):
            await repository.update_portfolio_atomic(
                sample_portfolio.id, updates, expected_version=sample_portfolio.version
            )

    @pytest.mark.asyncio
    async def test_atomic_update_portfolio_not_found(self, repository, sample_portfolio):
        """Test atomic update with non-existent portfolio."""
        updates = {"cash_balance": Decimal("95000")}

        with pytest.raises(PortfolioNotFoundError):
            await repository.update_portfolio_atomic(sample_portfolio.id, updates)

    @pytest.mark.asyncio
    async def test_concurrent_updates_simulation(self, repository, sample_portfolio, mock_adapter):
        """Test simulation of concurrent updates."""
        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # Simulate multiple concurrent updates
        async def update_worker(worker_id: int):
            updates = {
                "trades_count": worker_id,
                "name": f"Updated by worker {worker_id}",
            }

            try:
                current_version = mock_adapter.data[sample_portfolio.id]["version"]
                await repository.update_portfolio_atomic(
                    sample_portfolio.id, updates, expected_version=current_version
                )
                return f"Worker {worker_id} succeeded"
            except (StaleDataException, OptimisticLockException):
                return f"Worker {worker_id} failed due to version conflict"
            except Exception as e:
                return f"Worker {worker_id} failed with error: {e}"

        # Run multiple concurrent workers
        tasks = [update_worker(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least one should succeed, others may fail due to conflicts
        success_count = sum(1 for result in results if "succeeded" in str(result))
        assert success_count >= 1

    def test_concurrency_service_integration(self, repository):
        """Test integration with concurrency service."""
        assert repository.concurrency_service is not None
        assert repository.concurrency_service.max_retries == 3

        # Test metrics
        metrics = repository.concurrency_service.get_metrics()
        assert isinstance(metrics, dict)
        assert "version_conflicts" in metrics

    @pytest.mark.asyncio
    async def test_repository_with_custom_concurrency_service(self, mock_adapter):
        """Test repository with custom concurrency service configuration."""
        custom_service = ConcurrencyService(
            max_retries=5,
            base_delay=0.05,
            backoff_factor=1.5,
        )

        repository = PostgreSQLPortfolioRepository(mock_adapter, concurrency_service=custom_service)

        assert repository.concurrency_service.max_retries == 5
        assert repository.concurrency_service.base_delay == 0.05
        assert repository.concurrency_service.backoff_factor == 1.5

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_handling(
        self, repository, sample_portfolio, mock_adapter
    ):
        """Test deadlock detection and handling."""
        # Modify mock adapter to raise deadlock exception
        original_execute = mock_adapter.execute_query

        async def deadlock_execute(query: str, *params):
            if "UPDATE" in query and "portfolios" in query:
                raise Exception("deadlock detected")
            return await original_execute(query, *params)

        mock_adapter.execute_query = deadlock_execute

        # Add portfolio to mock data
        mock_adapter.data[sample_portfolio.id] = {
            "id": sample_portfolio.id,
            "version": sample_portfolio.version,
            "name": sample_portfolio.name,
            "initial_capital": sample_portfolio.initial_capital,
            "cash_balance": sample_portfolio.cash_balance,
            "created_at": sample_portfolio.created_at,
            "last_updated": sample_portfolio.last_updated,
            "max_position_size": sample_portfolio.max_position_size,
            "max_portfolio_risk": sample_portfolio.max_portfolio_risk,
            "max_positions": sample_portfolio.max_positions,
            "max_leverage": sample_portfolio.max_leverage,
            "total_realized_pnl": sample_portfolio.total_realized_pnl,
            "total_commission_paid": sample_portfolio.total_commission_paid,
            "trades_count": sample_portfolio.trades_count,
            "winning_trades": sample_portfolio.winning_trades,
            "losing_trades": sample_portfolio.losing_trades,
            "strategy": sample_portfolio.strategy,
            "tags": sample_portfolio.tags,
        }

        # Should detect deadlock and raise appropriate exception
        with pytest.raises(DeadlockException):
            await repository.update_portfolio(sample_portfolio)
