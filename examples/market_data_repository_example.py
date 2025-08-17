"""
Market Data Repository Usage Example

Demonstrates how to use the MarketDataRepository to store and retrieve
market data bars in PostgreSQL.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta

# Local imports
from src.application.interfaces.market_data import Bar
from src.domain.value_objects.price import Price
from src.domain.value_objects.symbol import Symbol
from src.infrastructure.database.adapter import PostgreSQLAdapter
from src.infrastructure.database.connection import create_connection_pool
from src.infrastructure.repositories.market_data_repository import MarketDataRepository


async def main():
    """Demonstrate market data repository usage."""

    # Create database connection pool
    pool = await create_connection_pool(
        host="localhost",
        port=5432,
        database="trading_db",
        user="trading_user",
        password="trading_password",
        min_size=2,
        max_size=10,
    )

    # Create adapter and repository
    adapter = PostgreSQLAdapter(pool)
    repository = MarketDataRepository(adapter)

    try:
        # 1. Save a single bar
        print("1. Saving a single bar...")
        bar = Bar(
            symbol=Symbol("AAPL"),
            timestamp=datetime.now(UTC),
            open=Price("150.00"),
            high=Price("152.50"),
            low=Price("149.75"),
            close=Price("151.25"),
            volume=10_000_000,
            vwap=Price("151.00"),
            trade_count=50_000,
            timeframe="1min",
        )
        await repository.save_bar(bar)
        print(f"   Saved bar for {bar.symbol.value} at {bar.timestamp}")

        # 2. Save multiple bars in batch
        print("\n2. Saving multiple bars in batch...")
        base_time = datetime.now(UTC).replace(microsecond=0)
        bars = []

        for i in range(5):
            timestamp = base_time - timedelta(minutes=i)
            bars.append(
                Bar(
                    symbol=Symbol("GOOGL"),
                    timestamp=timestamp,
                    open=Price(f"{2800.00 + i * 10:.2f}"),
                    high=Price(f"{2810.00 + i * 10:.2f}"),
                    low=Price(f"{2795.00 + i * 10:.2f}"),
                    close=Price(f"{2805.00 + i * 10:.2f}"),
                    volume=1_000_000 + i * 100_000,
                    vwap=Price(f"{2802.50 + i * 10:.2f}"),
                    trade_count=10_000 + i * 1000,
                    timeframe="5min",
                )
            )

        await repository.save_bars(bars)
        print(f"   Saved {len(bars)} bars for GOOGL")

        # 3. Get the latest bar
        print("\n3. Getting latest bar...")
        latest_aapl = await repository.get_latest_bar("AAPL", "1min")
        if latest_aapl:
            print(
                f"   Latest AAPL bar: Close={latest_aapl.close.value}, Volume={latest_aapl.volume}"
            )

        latest_googl = await repository.get_latest_bar("GOOGL", "5min")
        if latest_googl:
            print(
                f"   Latest GOOGL bar: Close={latest_googl.close.value}, Volume={latest_googl.volume}"
            )

        # 4. Get bars by date range
        print("\n4. Getting bars by date range...")
        start = base_time - timedelta(minutes=10)
        end = base_time

        googl_bars = await repository.get_bars("GOOGL", start, end, "5min")
        print(f"   Found {len(googl_bars)} GOOGL bars between {start} and {end}")

        # 5. Get bars by count
        print("\n5. Getting most recent bars by count...")
        recent_bars = await repository.get_bars_by_count("GOOGL", 3, timeframe="5min")
        print(f"   Retrieved {len(recent_bars)} most recent GOOGL bars")
        for i, bar in enumerate(recent_bars):
            print(f"     Bar {i+1}: {bar.timestamp} - Close={bar.close.value}")

        # 6. Get symbols with data
        print("\n6. Getting symbols with data...")
        symbols = await repository.get_symbols_with_data()
        print(f"   Symbols with data: {', '.join(symbols)}")

        # 7. Get data range for a symbol
        print("\n7. Getting data range for symbols...")
        for symbol in symbols:
            data_range = await repository.get_data_range(symbol)
            if data_range:
                earliest, latest = data_range
                print(f"   {symbol}: {earliest} to {latest}")

        # 8. Demonstrate duplicate handling
        print("\n8. Testing duplicate bar handling...")
        # Try to save the same bar again with different values
        duplicate_bar = Bar(
            symbol=Symbol("AAPL"),
            timestamp=bar.timestamp,  # Same timestamp
            open=Price("155.00"),  # Different values
            high=Price("157.00"),
            low=Price("154.00"),
            close=Price("156.00"),
            volume=12_000_000,
            timeframe="1min",
        )
        await repository.save_bar(duplicate_bar)

        # Verify it was updated
        updated = await repository.get_latest_bar("AAPL", "1min")
        if updated and updated.timestamp == bar.timestamp:
            print(f"   Bar updated: New close={updated.close.value} (was {bar.close.value})")

        # 9. Store bars with different timeframes
        print("\n9. Storing bars with different timeframes...")
        timeframes = ["1min", "5min", "1hour", "1day"]
        test_time = datetime.now(UTC).replace(microsecond=0)

        for tf in timeframes:
            tf_bar = Bar(
                symbol=Symbol("MSFT"),
                timestamp=test_time,
                open=Price("380.00"),
                high=Price("382.00"),
                low=Price("379.00"),
                close=Price("381.00"),
                volume=5_000_000,
                timeframe=tf,
            )
            await repository.save_bar(tf_bar)

        print(f"   Saved MSFT bars for timeframes: {', '.join(timeframes)}")

        # Verify each timeframe has its own bar
        for tf in timeframes:
            tf_bar = await repository.get_latest_bar("MSFT", tf)
            if tf_bar:
                print(f"   MSFT {tf}: {tf_bar.timestamp}")

        # 10. Delete old bars
        print("\n10. Deleting old bars...")
        cutoff = base_time - timedelta(minutes=3)
        deleted_count = await repository.delete_old_bars(cutoff)
        print(f"   Deleted {deleted_count} bars older than {cutoff}")

        # Verify deletion
        remaining_googl = await repository.get_bars(
            "GOOGL", base_time - timedelta(minutes=10), base_time, "5min"
        )
        print(f"   Remaining GOOGL bars: {len(remaining_googl)}")

        print("\n✓ Market Data Repository example completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
    finally:
        # Clean up
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
