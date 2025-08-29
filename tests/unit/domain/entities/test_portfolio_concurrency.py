"""
Unit tests for Portfolio entity thread safety and concurrency features.

Tests optimistic locking, version increment behavior, and concurrent access patterns.
"""

import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.domain.entities.portfolio import Portfolio, PositionRequest
from src.domain.exceptions import StaleDataException
from src.domain.value_objects import Money, Price, Quantity


class TestPortfolioConcurrency:
    """Test suite for portfolio concurrency and thread safety."""

    @pytest.fixture
    def sample_portfolio(self) -> Portfolio:
        """Create a sample portfolio for testing."""
        return Portfolio(
            id=uuid4(),
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000")),
            cash_balance=Money(Decimal("100000")),
            max_position_size=Money(Decimal("20000")),  # Increase limit to allow test positions
            max_portfolio_risk=Decimal("0.25"),  # Increase risk limit to 25%
            version=1,
        )

    def test_version_initialization(self, sample_portfolio: Portfolio) -> None:
        """Test that portfolio version is properly initialized."""
        assert sample_portfolio.version == 1

    def test_version_increment_on_position_open(self, sample_portfolio: Portfolio) -> None:
        """Test that version is incremented when opening a position."""
        initial_version = sample_portfolio.version

        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )

        sample_portfolio.open_position(request)

        assert sample_portfolio.version == initial_version + 1
        assert sample_portfolio.last_updated is not None

    def test_version_increment_on_position_close(self, sample_portfolio: Portfolio) -> None:
        """Test that version is incremented when closing a position."""
        # First open a position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
            commission=Money(Decimal("1.00")),
        )
        sample_portfolio.open_position(request)
        initial_version = sample_portfolio.version

        # Then close it
        sample_portfolio.close_position("AAPL", Price(Decimal("160.00")), Money(Decimal("1.00")))

        assert sample_portfolio.version == initial_version + 1

    def test_version_increment_on_price_update(self, sample_portfolio: Portfolio) -> None:
        """Test that version is incremented when updating position price."""
        # First open a position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
        )
        sample_portfolio.open_position(request)
        initial_version = sample_portfolio.version

        # Update price
        sample_portfolio.update_position_price("AAPL", Price(Decimal("155.00")))

        assert sample_portfolio.version == initial_version + 1

    def test_version_increment_on_multiple_price_update(self, sample_portfolio: Portfolio) -> None:
        """Test that version is incremented only once for batch price updates."""
        # First open multiple positions
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            request = PositionRequest(
                symbol=symbol,
                quantity=Quantity(Decimal("50")),
                entry_price=Price(Decimal("150.00")),
            )
            sample_portfolio.open_position(request)

        initial_version = sample_portfolio.version

        # Update all prices
        prices = {
            "AAPL": Price(Decimal("155.00")),
            "GOOGL": Price(Decimal("2800.00")),
            "MSFT": Price(Decimal("300.00")),
        }
        sample_portfolio.update_all_prices(prices)

        # Should increment only once for batch update
        assert sample_portfolio.version == initial_version + 1

    def test_no_version_increment_on_empty_price_update(self, sample_portfolio: Portfolio) -> None:
        """Test that version is not incremented for empty price updates."""
        initial_version = sample_portfolio.version

        # Update with empty prices
        sample_portfolio.update_all_prices({})

        # Version should not change
        assert sample_portfolio.version == initial_version

    def test_version_check_success(self, sample_portfolio: Portfolio) -> None:
        """Test version checking with correct version."""
        expected_version = sample_portfolio.version

        # Should not raise exception
        sample_portfolio._check_version(expected_version)

    def test_version_check_failure(self, sample_portfolio: Portfolio) -> None:
        """Test version checking with incorrect version."""
        wrong_version = sample_portfolio.version + 1

        with pytest.raises(StaleDataException) as exc_info:
            sample_portfolio._check_version(wrong_version)

        assert exc_info.value.entity_type == "Portfolio"
        assert exc_info.value.entity_id == sample_portfolio.id
        assert exc_info.value.expected_version == wrong_version
        assert exc_info.value.actual_version == sample_portfolio.version

    def test_concurrent_position_operations(self, sample_portfolio: Portfolio) -> None:
        """Test concurrent operations on the same portfolio."""
        results = []
        errors = []

        def worker(thread_id: int) -> None:
            try:
                request = PositionRequest(
                    symbol=f"STOCK{thread_id}",
                    quantity=Quantity(Decimal("100")),
                    entry_price=Price(Decimal("100.00")),
                    commission=Money(Decimal("1.00")),
                )

                # Add small delay to increase chance of contention
                time.sleep(0.01)

                sample_portfolio.open_position(request)
                results.append(f"Thread {thread_id} succeeded")

            except Exception as e:
                errors.append(f"Thread {thread_id} failed: {e}")

        # Start multiple threads
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should succeed (no version conflicts at entity level)
        assert len(results) == num_threads
        assert len(errors) == 0

        # Version should be incremented for each operation
        # Initial version + num_threads operations
        assert sample_portfolio.version == 1 + num_threads

    def test_version_consistency_after_operations(self, sample_portfolio: Portfolio) -> None:
        """Test that version remains consistent after various operations."""
        initial_version = sample_portfolio.version
        operations_count = 0

        # Open position
        request = PositionRequest(
            symbol="AAPL",
            quantity=Quantity(Decimal("100")),
            entry_price=Price(Decimal("150.00")),
        )
        sample_portfolio.open_position(request)
        operations_count += 1

        # Update price
        sample_portfolio.update_position_price("AAPL", Price(Decimal("155.00")))
        operations_count += 1

        # Close position
        sample_portfolio.close_position("AAPL", Price(Decimal("160.00")))
        operations_count += 1

        expected_version = initial_version + operations_count
        assert sample_portfolio.version == expected_version

    def test_manual_version_increment(self, sample_portfolio: Portfolio) -> None:
        """Test manual version increment method."""
        initial_version = sample_portfolio.version
        old_timestamp = sample_portfolio.last_updated or datetime.now(UTC)

        sample_portfolio._increment_version()

        assert sample_portfolio.version == initial_version + 1
        assert sample_portfolio.last_updated is not None
        if old_timestamp:
            assert sample_portfolio.last_updated >= old_timestamp

    def test_portfolio_immutability_after_version_check_failure(
        self, sample_portfolio: Portfolio
    ) -> None:
        """Test that portfolio state remains unchanged after version check failure."""
        initial_state = {
            "version": sample_portfolio.version,
            "cash_balance": sample_portfolio.cash_balance,
            "trades_count": sample_portfolio.trades_count,
        }

        # Try version check with wrong version
        try:
            sample_portfolio._check_version(sample_portfolio.version + 1)
        except StaleDataException:
            pass

        # State should be unchanged
        assert sample_portfolio.version == initial_state["version"]
        assert sample_portfolio.cash_balance == initial_state["cash_balance"]
        assert sample_portfolio.trades_count == initial_state["trades_count"]

    def test_version_increment_thread_safety(self, sample_portfolio: Portfolio) -> None:
        """Test that version increment is thread-safe."""
        initial_version = sample_portfolio.version
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def increment_version() -> None:
            # Wait for all threads to be ready
            barrier.wait()
            # All threads increment simultaneously
            sample_portfolio._increment_version()

        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_version)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All increments should have been applied
        # Note: This tests the entity itself, not repository-level locking
        expected_version = initial_version + num_threads
        assert sample_portfolio.version == expected_version

    def test_version_post_init_handling(self) -> None:
        """Test that version is properly handled in __post_init__."""
        # Create portfolio without version
        portfolio = Portfolio(
            name="Test",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
        )

        # Version should be set to 1
        assert portfolio.version == 1

    def test_version_post_init_with_existing_version(self) -> None:
        """Test that existing version is preserved in __post_init__."""
        portfolio = Portfolio(
            name="Test",
            initial_capital=Decimal("100000"),
            cash_balance=Decimal("100000"),
            version=5,
        )

        # Existing version should be preserved
        assert portfolio.version == 5

    def test_stale_data_exception_details(self, sample_portfolio: Portfolio) -> None:
        """Test that StaleDataException contains proper details."""
        expected_version = 999

        with pytest.raises(StaleDataException) as exc_info:
            sample_portfolio._check_version(expected_version)

        exception = exc_info.value

        # Check all details are populated
        assert exception.entity_type == "Portfolio"
        assert exception.entity_id == sample_portfolio.id
        assert exception.expected_version == expected_version
        assert exception.actual_version == sample_portfolio.version
        assert exception.details["entity_type"] == "Portfolio"
        assert exception.details["entity_id"] == str(sample_portfolio.id)
        assert exception.details["expected_version"] == expected_version
        assert exception.details["actual_version"] == sample_portfolio.version

    def test_version_increment_preserves_other_timestamps(
        self, sample_portfolio: Portfolio
    ) -> None:
        """Test that version increment doesn't interfere with created_at."""
        created_at = sample_portfolio.created_at

        sample_portfolio._increment_version()

        # created_at should be unchanged
        assert sample_portfolio.created_at == created_at
        # last_updated should be updated
        assert sample_portfolio.last_updated is not None
        assert sample_portfolio.last_updated >= created_at
