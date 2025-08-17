"""
Integration tests for Market Data Repository

Tests the PostgreSQL implementation of the market data repository
with a real database connection.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta

# Third-party imports
import pytest

# Local imports
from src.application.interfaces.market_data import Bar
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import ConnectionFactory, DatabaseConfig
from src.infrastructure.repositories.market_data_repository import MarketDataRepository


@pytest.fixture
async def db_adapter():
    """Create a database adapter for testing."""
    # Use test database configuration
    # Use environment variables for sensitive data in real tests
    # Standard library imports
    import os

    config = DatabaseConfig(
        host=os.getenv("TEST_DB_HOST", "localhost"),
        port=int(os.getenv("TEST_DB_PORT", "5432")),
        database=os.getenv("TEST_DB_NAME", "trading_test"),
        user=os.getenv("TEST_DB_USER", "trading_user"),
        password=os.getenv("TEST_DB_PASSWORD", "test_password"),  # - test credential
        min_pool_size=1,
        max_pool_size=5,
    )

    # Create connection using the factory
    connection = await ConnectionFactory.create_connection(config, force_new=True)

    # Get the pool from the connection
    pool = await connection.connect()

    adapter = PostgreSQLAdapter(pool)

    try:
        # Clean up any existing test data
        await adapter.execute_query("TRUNCATE TABLE market_data CASCADE")
        yield adapter
    finally:
        # Clean up after tests
        await adapter.execute_query("TRUNCATE TABLE market_data CASCADE")
        await connection.disconnect()
        ConnectionFactory.reset()  # Reset factory for clean test state


@pytest.fixture
def repository(db_adapter):
    """Create a market data repository instance."""
    return MarketDataRepository(db_adapter)


@pytest.fixture
def sample_bar():
    """Create a sample bar for testing."""
    return Bar(
        symbol=Symbol("AAPL"),
        timestamp=datetime.now(UTC),
        open=Price("150.00"),
        high=Price("152.00"),
        low=Price("149.50"),
        close=Price("151.50"),
        volume=1000000,
        vwap=Price("151.00"),
        trade_count=5000,
        timeframe="1min",
    )


@pytest.fixture
def sample_bars():
    """Create multiple sample bars for testing."""
    base_time = datetime.now(UTC).replace(microsecond=0)
    bars = []

    for i in range(10):
        timestamp = base_time - timedelta(minutes=i)
        bars.append(
            Bar(
                symbol=Symbol("AAPL"),
                timestamp=timestamp,
                open=Price(f"{150.00 + i * 0.1:.2f}"),
                high=Price(f"{152.00 + i * 0.1:.2f}"),
                low=Price(f"{149.00 + i * 0.1:.2f}"),
                close=Price(f"{151.00 + i * 0.1:.2f}"),
                volume=1000000 + i * 10000,
                vwap=Price(f"{150.50 + i * 0.1:.2f}"),
                trade_count=5000 + i * 100,
                timeframe="1min",
            )
        )

    return bars


class TestMarketDataRepository:
    """Test suite for MarketDataRepository."""

    @pytest.mark.asyncio
    async def test_save_bar(self, repository, sample_bar):
        """Test saving a single bar."""
        # Save the bar
        await repository.save_bar(sample_bar)

        # Retrieve and verify
        latest = await repository.get_latest_bar(sample_bar.symbol.value, sample_bar.timeframe)

        assert latest is not None
        assert latest.symbol == sample_bar.symbol
        assert latest.open == sample_bar.open
        assert latest.high == sample_bar.high
        assert latest.low == sample_bar.low
        assert latest.close == sample_bar.close
        assert latest.volume == sample_bar.volume
        assert latest.vwap == sample_bar.vwap
        assert latest.trade_count == sample_bar.trade_count

    @pytest.mark.asyncio
    async def test_save_bar_duplicate_handling(self, repository, sample_bar):
        """Test that duplicate bars are handled correctly."""
        # Save the same bar twice
        await repository.save_bar(sample_bar)

        # Modify the bar slightly
        modified_bar = Bar(
            symbol=sample_bar.symbol,
            timestamp=sample_bar.timestamp,
            open=Price("155.00"),  # Different price
            high=sample_bar.high,
            low=sample_bar.low,
            close=sample_bar.close,
            volume=sample_bar.volume,
            timeframe=sample_bar.timeframe,
        )

        # Save again - should update due to ON CONFLICT
        await repository.save_bar(modified_bar)

        # Verify the bar was updated
        latest = await repository.get_latest_bar(sample_bar.symbol.value, sample_bar.timeframe)

        assert latest is not None
        assert latest.open == Price("155.00")  # Should be updated

    @pytest.mark.asyncio
    async def test_save_bars_batch(self, repository, sample_bars):
        """Test batch saving of bars."""
        # Save multiple bars
        await repository.save_bars(sample_bars)

        # Verify all bars were saved
        symbols = await repository.get_symbols_with_data()
        assert "AAPL" in symbols

        # Get all bars
        start = sample_bars[-1].timestamp - timedelta(minutes=1)
        end = sample_bars[0].timestamp + timedelta(minutes=1)

        retrieved = await repository.get_bars("AAPL", start, end, "1min")

        assert len(retrieved) == len(sample_bars)

    @pytest.mark.asyncio
    async def test_get_latest_bar(self, repository, sample_bars):
        """Test retrieving the latest bar."""
        await repository.save_bars(sample_bars)

        latest = await repository.get_latest_bar("AAPL", "1min")

        assert latest is not None
        # The latest should be the first in our list (most recent timestamp)
        assert latest.timestamp == sample_bars[0].timestamp
        assert latest.close == sample_bars[0].close

    @pytest.mark.asyncio
    async def test_get_latest_bar_not_found(self, repository):
        """Test retrieving latest bar when none exists."""
        latest = await repository.get_latest_bar("NONEXISTENT", "1min")
        assert latest is None

    @pytest.mark.asyncio
    async def test_get_bars_by_date_range(self, repository, sample_bars):
        """Test retrieving bars within a date range."""
        await repository.save_bars(sample_bars)

        # Get bars from middle of range
        start = sample_bars[7].timestamp
        end = sample_bars[3].timestamp

        retrieved = await repository.get_bars("AAPL", start, end, "1min")

        # Should get bars 3-7 (inclusive), ordered ascending
        assert len(retrieved) == 5
        assert retrieved[0].timestamp == sample_bars[7].timestamp
        assert retrieved[-1].timestamp == sample_bars[3].timestamp

    @pytest.mark.asyncio
    async def test_get_bars_empty_range(self, repository, sample_bars):
        """Test retrieving bars with no data in range."""
        await repository.save_bars(sample_bars)

        # Request bars from future
        future_start = datetime.now(UTC) + timedelta(days=1)
        future_end = future_start + timedelta(hours=1)

        retrieved = await repository.get_bars("AAPL", future_start, future_end, "1min")

        assert len(retrieved) == 0

    @pytest.mark.asyncio
    async def test_get_bars_by_count(self, repository, sample_bars):
        """Test retrieving a specific number of recent bars."""
        await repository.save_bars(sample_bars)

        # Get 5 most recent bars
        retrieved = await repository.get_bars_by_count("AAPL", 5, timeframe="1min")

        assert len(retrieved) == 5
        # Should be ordered ascending
        for i in range(1, len(retrieved)):
            assert retrieved[i].timestamp > retrieved[i - 1].timestamp
        # Most recent in retrieved should match most recent in sample
        assert retrieved[-1].timestamp == sample_bars[0].timestamp

    @pytest.mark.asyncio
    async def test_get_bars_by_count_with_end(self, repository, sample_bars):
        """Test retrieving bars by count with end date."""
        await repository.save_bars(sample_bars)

        # Get 3 bars ending at bar[5]
        end_time = sample_bars[5].timestamp
        retrieved = await repository.get_bars_by_count("AAPL", 3, end=end_time, timeframe="1min")

        assert len(retrieved) == 3
        assert retrieved[-1].timestamp == sample_bars[5].timestamp
        assert retrieved[0].timestamp == sample_bars[7].timestamp

    @pytest.mark.asyncio
    async def test_delete_old_bars(self, repository, sample_bars):
        """Test deleting old bars."""
        await repository.save_bars(sample_bars)

        # Delete bars older than 5 minutes ago
        cutoff = sample_bars[5].timestamp
        deleted_count = await repository.delete_old_bars(cutoff)

        # Should have deleted bars 6-9 (4 bars)
        assert deleted_count == 4

        # Verify remaining bars
        all_bars = await repository.get_bars(
            "AAPL",
            sample_bars[-1].timestamp - timedelta(minutes=1),
            sample_bars[0].timestamp + timedelta(minutes=1),
            "1min",
        )

        assert len(all_bars) == 6  # Bars 0-5 remain

    @pytest.mark.asyncio
    async def test_get_symbols_with_data(self, repository):
        """Test retrieving list of symbols with data."""
        # Create bars for multiple symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        base_time = datetime.now(UTC)

        for symbol in symbols:
            bar = Bar(
                symbol=Symbol(symbol),
                timestamp=base_time,
                open=Price("100.00"),
                high=Price("101.00"),
                low=Price("99.00"),
                close=Price("100.50"),
                volume=100000,
                timeframe="1min",
            )
            await repository.save_bar(bar)

        # Get symbols
        retrieved_symbols = await repository.get_symbols_with_data()

        assert len(retrieved_symbols) == 3
        assert set(retrieved_symbols) == set(symbols)

    @pytest.mark.asyncio
    async def test_get_data_range(self, repository, sample_bars):
        """Test retrieving data range for a symbol."""
        await repository.save_bars(sample_bars)

        # Get data range
        data_range = await repository.get_data_range("AAPL")

        assert data_range is not None
        earliest, latest = data_range

        # Earliest should be the oldest bar (last in list)
        assert earliest == sample_bars[-1].timestamp
        # Latest should be the newest bar (first in list)
        assert latest == sample_bars[0].timestamp

    @pytest.mark.asyncio
    async def test_get_data_range_no_data(self, repository):
        """Test retrieving data range when no data exists."""
        data_range = await repository.get_data_range("NONEXISTENT")
        assert data_range is None

    @pytest.mark.asyncio
    async def test_timezone_handling(self, repository):
        """Test that timezones are properly handled."""
        # Create bars with different timezone representations
        base_time = datetime.now(UTC)

        # UTC bar
        utc_bar = Bar(
            symbol=Symbol("TZ_TEST"),
            timestamp=base_time,
            open=Price("100.00"),
            high=Price("101.00"),
            low=Price("99.00"),
            close=Price("100.50"),
            volume=100000,
            timeframe="1min",
        )

        # Naive datetime (should be treated as UTC)
        naive_bar = Bar(
            symbol=Symbol("TZ_TEST"),
            timestamp=base_time.replace(tzinfo=None) + timedelta(minutes=1),
            open=Price("101.00"),
            high=Price("102.00"),
            low=Price("100.00"),
            close=Price("101.50"),
            volume=100000,
            timeframe="1min",
        )

        # Save both
        await repository.save_bar(utc_bar)
        await repository.save_bar(naive_bar)

        # Retrieve and verify
        bars = await repository.get_bars(
            "TZ_TEST", base_time - timedelta(minutes=1), base_time + timedelta(minutes=2), "1min"
        )

        assert len(bars) == 2
        # All retrieved bars should have timezone info
        for bar in bars:
            assert bar.timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_different_timeframes(self, repository):
        """Test storing bars with different timeframes."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        timeframes = ["1min", "5min", "1hour", "1day"]

        for tf in timeframes:
            bar = Bar(
                symbol=Symbol("MULTI_TF"),
                timestamp=base_time,
                open=Price("100.00"),
                high=Price("101.00"),
                low=Price("99.00"),
                close=Price("100.50"),
                volume=100000,
                timeframe=tf,
            )
            await repository.save_bar(bar)

        # Each timeframe should have its own bar
        for tf in timeframes:
            bar = await repository.get_latest_bar("MULTI_TF", tf)
            assert bar is not None
            assert bar.timeframe == tf

    @pytest.mark.asyncio
    async def test_large_batch_insert(self, repository):
        """Test inserting a large batch of bars."""
        base_time = datetime.now(UTC).replace(microsecond=0)

        # Create 1000 bars
        large_batch = []
        for i in range(1000):
            timestamp = base_time - timedelta(minutes=i)
            large_batch.append(
                Bar(
                    symbol=Symbol("LARGE_BATCH"),
                    timestamp=timestamp,
                    open=Price(f"{100.00 + (i % 10) * 0.1:.2f}"),
                    high=Price(f"{101.00 + (i % 10) * 0.1:.2f}"),
                    low=Price(f"{99.00 + (i % 10) * 0.1:.2f}"),
                    close=Price(f"{100.50 + (i % 10) * 0.1:.2f}"),
                    volume=100000 + i * 100,
                    timeframe="1min",
                )
            )

        # Save batch
        await repository.save_bars(large_batch)

        # Verify count
        bars = await repository.get_bars(
            "LARGE_BATCH",
            base_time - timedelta(minutes=1001),
            base_time + timedelta(minutes=1),
            "1min",
        )

        assert len(bars) == 1000
