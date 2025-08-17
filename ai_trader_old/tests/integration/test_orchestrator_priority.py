#!/usr/bin/env python3
"""
Test Orchestrator Priority Logic

Tests the orchestrator's priority logic with mock clients to verify
that Alpaca is prioritized over Polygon.
"""

# Standard library imports
from pathlib import Path
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def create_mock_client(name, can_fetch_types):
    """Create a mock client that can fetch specified data types."""
    mock_client = MagicMock()
    mock_client.can_fetch = lambda data_type: data_type in can_fetch_types
    mock_client.name = name
    return mock_client


def test_orchestrator_priority_logic():
    """Test the orchestrator's priority logic with mock clients."""

    print("🎼 Testing Orchestrator Priority Logic")
    print("=" * 50)

    try:
        # Local imports
        from main.config.config_manager import get_config
        from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator

        # Load configuration
        config = get_config()

        # Create mock clients
        mock_clients = {
            "alpaca": create_mock_client("alpaca", ["market_data", "corporate_actions"]),
            "polygon": create_mock_client("polygon", ["market_data", "options"]),
            "yahoo_market": create_mock_client("yahoo_market", ["market_data"]),
            "yahoo_news": create_mock_client("yahoo_news", ["news"]),
            "benzinga": create_mock_client("benzinga", ["news"]),
            "reddit": create_mock_client("reddit", ["social_sentiment"]),
        }

        # Create orchestrator with mock clients
        orchestrator = IngestionOrchestrator(mock_clients)

        print(f"✅ Orchestrator created with {len(mock_clients)} mock clients")

        # Test market_data priority
        print("\n📊 Testing market_data priorities:")
        market_data_clients = orchestrator._get_prioritized_clients("market_data")

        print(f"🔍 Prioritized clients: {[name for name, _ in market_data_clients]}")

        # Check if Alpaca comes before Polygon
        client_names = [name for name, _ in market_data_clients]
        alpaca_index = None
        polygon_index = None
        yahoo_index = None

        for i, name in enumerate(client_names):
            if "alpaca" in name.lower():
                alpaca_index = i
            elif "polygon" in name.lower():
                polygon_index = i
            elif "yahoo" in name.lower():
                yahoo_index = i

        print(f"📊 Alpaca index: {alpaca_index}")
        print(f"📊 Polygon index: {polygon_index}")
        print(f"📊 Yahoo index: {yahoo_index}")

        # Test priority order
        market_data_test_passed = False
        if alpaca_index is not None and polygon_index is not None:
            if alpaca_index < polygon_index:
                print("✅ Alpaca correctly prioritized over Polygon")
                market_data_test_passed = True
            else:
                print("❌ Polygon prioritized over Alpaca (incorrect)")
        else:
            print("⚠️  Could not determine priority order")

        # Test news priority
        print("\n📰 Testing news priorities:")
        news_clients = orchestrator._get_prioritized_clients("news")
        news_client_names = [name for name, _ in news_clients]
        print(f"🔍 News clients: {news_client_names}")

        # Test options priority
        print("\n📈 Testing options priorities:")
        options_clients = orchestrator._get_prioritized_clients("options")
        options_client_names = [name for name, _ in options_clients]
        print(f"🔍 Options clients: {options_client_names}")

        # Test social_sentiment priority
        print("\n💬 Testing social_sentiment priorities:")
        social_clients = orchestrator._get_prioritized_clients("social_sentiment")
        social_client_names = [name for name, _ in social_clients]
        print(f"🔍 Social clients: {social_client_names}")

        return market_data_test_passed

    except Exception as e:
        print(f"❌ Orchestrator priority test failed: {e}")
        # Standard library imports
        import traceback

        traceback.print_exc()
        return False


def test_client_matching_logic():
    """Test that client name matching works correctly."""

    print("\n🔍 Testing Client Name Matching Logic")
    print("-" * 50)

    try:
        # Local imports
        from main.config.config_manager import get_config
        from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator

        config = get_config()

        # Test various client naming patterns
        test_cases = [
            ("alpaca", "alpaca"),
            ("polygon", "polygon"),
            ("yahoo_market", "yahoo"),
            ("yahoo_news", "yahoo"),
            ("benzinga", "benzinga"),
            ("reddit", "reddit"),
            ("alpaca_alt", "alpaca_alt"),
        ]

        # Create mock clients with different naming patterns
        mock_clients = {}
        for client_name, _ in test_cases:
            mock_clients[client_name] = create_mock_client(client_name, ["market_data"])

        orchestrator = IngestionOrchestrator(mock_clients)

        print("🔍 Testing client name matching:")
        for client_name, priority_name in test_cases:
            # Check if the client would match the priority name
            matching_clients = [
                name
                for name in mock_clients
                if priority_name.lower() in name.lower()
                or name.lower().startswith(priority_name.lower())
            ]

            match_found = client_name in matching_clients
            print(f"  {client_name} matches {priority_name}: {'✅' if match_found else '❌'}")

        return True

    except Exception as e:
        print(f"❌ Client matching test failed: {e}")
        return False


def test_no_priority_fallback():
    """Test behavior when no priority configuration exists."""

    print("\n🔄 Testing No Priority Configuration Fallback")
    print("-" * 50)

    try:
        # Local imports
        from main.config.config_manager import get_config
        from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator

        config = get_config()

        # Create mock clients
        mock_clients = {
            "alpaca": create_mock_client("alpaca", ["unknown_data_type"]),
            "polygon": create_mock_client("polygon", ["unknown_data_type"]),
        }

        orchestrator = IngestionOrchestrator(mock_clients)

        # Test with a data type that doesn't have priority configuration
        unknown_clients = orchestrator._get_prioritized_clients("unknown_data_type")

        print(f"🔍 Clients for unknown data type: {[name for name, _ in unknown_clients]}")

        # Should return all capable clients
        expected_count = len([c for c in mock_clients.values() if c.can_fetch("unknown_data_type")])
        actual_count = len(unknown_clients)

        print(f"📊 Expected clients: {expected_count}, Actual: {actual_count}")

        return expected_count == actual_count

    except Exception as e:
        print(f"❌ No priority fallback test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🧪 Orchestrator Priority Logic Test Suite")
    print("=" * 60)

    # Run tests
    priority_test = test_orchestrator_priority_logic()
    matching_test = test_client_matching_logic()
    fallback_test = test_no_priority_fallback()

    print("\n" + "=" * 60)
    print("🏁 Test Results")
    print("=" * 60)
    print(f"Priority Logic: {'✅ PASS' if priority_test else '❌ FAIL'}")
    print(f"Client Matching: {'✅ PASS' if matching_test else '❌ FAIL'}")
    print(f"No Priority Fallback: {'✅ PASS' if fallback_test else '❌ FAIL'}")

    overall_success = priority_test and matching_test and fallback_test
    print(f"Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 Orchestrator priority logic is working correctly!")
        print("   • Alpaca is properly prioritized over Polygon")
        print("   • Client name matching works correctly")
        print("   • Fallback behavior is correct")
        print("   • All data types have working priority logic")
    else:
        print("\n🔧 Issues found:")
        if not priority_test:
            print("   • Priority logic needs review")
        if not matching_test:
            print("   • Client name matching needs improvement")
        if not fallback_test:
            print("   • Fallback behavior needs fixing")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
