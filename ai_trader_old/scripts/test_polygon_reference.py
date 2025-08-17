#!/usr/bin/env python3
"""
Test script to investigate Polygon's reference data and S&P 500 information.
"""

# Standard library imports
import asyncio
import logging
import os

# Add the parent directory to the path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.ingestion.polygon_reference_client import PolygonReferenceClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_ticker_details():
    """Test fetching ticker details from Polygon."""
    config = get_config()

    # Create the reference client
    ref_client = PolygonReferenceClient(config)

    # Test with known S&P 500 symbols
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("\n" + "=" * 80)
    print("TESTING POLYGON TICKER DETAILS API")
    print("=" * 80)

    for symbol in test_symbols:
        print(f"\n📊 Fetching details for {symbol}...")
        details = await ref_client.get_ticker_details(symbol)

        if details:
            print(f"✅ Found details for {symbol}:")
            print(f"   Name: {details.get('name', 'N/A')}")
            print(f"   Market Cap: {details.get('market_cap', 'N/A')}")
            print(f"   SIC Description: {details.get('sic_description', 'N/A')}")
            print(f"   Primary Exchange: {details.get('primary_exchange', 'N/A')}")
            print(f"   Type: {details.get('type', 'N/A')}")
            print(f"   Active: {details.get('active', 'N/A')}")

            # Check for any fields that might indicate S&P 500 membership
            print("\n   🔍 Checking for S&P 500 indicators:")

            # Check tags
            if "tags" in details:
                print(f"   Tags: {details['tags']}")

            # Check for any fields containing 'sp500', 'index', etc.
            sp500_fields = []
            for key, value in details.items():
                if isinstance(value, str) and any(
                    term in str(value).lower() for term in ["sp500", "s&p 500", "spx"]
                ):
                    sp500_fields.append(f"{key}: {value}")
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and any(
                            term in item.lower() for term in ["sp500", "s&p 500", "spx"]
                        ):
                            sp500_fields.append(f"{key}: {item}")

            if sp500_fields:
                print(f"   Found S&P 500 references in: {sp500_fields}")
            else:
                print("   No S&P 500 references found in ticker details")

            # Print all available fields for investigation
            print("\n   📋 All available fields:")
            for key in sorted(details.keys()):
                if key not in [
                    "name",
                    "market_cap",
                    "sic_description",
                    "primary_exchange",
                    "type",
                    "active",
                ]:
                    print(f"   {key}: {details[key]}")
        else:
            print(f"❌ No details found for {symbol}")

    # Test index ticker
    print("\n" + "=" * 80)
    print("TESTING S&P 500 INDEX TICKER")
    print("=" * 80)

    index_symbols = ["I:SPX", "SPX", "$SPX"]
    for index in index_symbols:
        print(f"\n📊 Testing index ticker: {index}")
        details = await ref_client.get_ticker_details(index)
        if details:
            print(f"✅ Found details for {index}")
            print(f"   Type: {details.get('type', 'N/A')}")
            print(f"   Name: {details.get('name', 'N/A')}")
        else:
            print(f"❌ No details found for {index}")

    # Test get_index_constituents
    print("\n📊 Testing get_index_constituents...")
    constituents = await ref_client.get_index_constituents("I:SPX")
    if constituents:
        print(f"✅ Found {len(constituents)} constituents")
        print(f"   First 10: {constituents[:10]}")
    else:
        print("❌ No constituents found")


async def test_all_tickers_endpoint():
    """Test if we can get S&P 500 info from the list all tickers endpoint."""
    config = get_config()

    print("\n" + "=" * 80)
    print("TESTING POLYGON LIST TICKERS ENDPOINT")
    print("=" * 80)

    # Use the market client to access SDK directly
    # Local imports
    from main.data_pipeline.ingestion.polygon_market_client import PolygonMarketClient

    client = PolygonMarketClient(config)

    # Get a few tickers and check their data
    print("\n📊 Fetching ticker list with enhanced parameters...")

    try:
        # Try to get tickers with market cap > $100B (potential S&P 500 members)
        tickers = await asyncio.to_thread(
            client.sdk_client.list_tickers,
            market="stocks",
            active=True,
            limit=10,
            sort="market_cap",
            order="desc",
        )

        print(f"✅ Retrieved {len(list(tickers))} tickers")

        for ticker in list(tickers)[:5]:
            print(f"\n   Symbol: {ticker.ticker}")
            print(f"   Name: {ticker.name}")
            print(f"   Market Cap: {getattr(ticker, 'market_cap', 'N/A')}")
            print(f"   Primary Exchange: {getattr(ticker, 'primary_exchange', 'N/A')}")

            # Print all attributes
            print(f"   All attributes: {dir(ticker)}")

    except Exception as e:
        print(f"❌ Error fetching tickers: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting Polygon Reference Data Investigation...")

    # Test ticker details
    await test_ticker_details()

    # Test list tickers endpoint
    await test_all_tickers_endpoint()

    print("\n✅ Investigation complete!")


if __name__ == "__main__":
    asyncio.run(main())
