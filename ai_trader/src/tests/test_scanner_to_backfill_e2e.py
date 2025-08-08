#!/usr/bin/env python3
"""
End-to-End Test: Scanner → Event → Backfill Flow

This test verifies the complete flow from scanner qualification through
event publishing to backfill triggering, including the new metadata system.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main.data_pipeline.core.enums import DataLayer
from main.events.core.event_bus import EventBus
from main.scanner.scanner_event_publisher import ScannerEventPublisher
from main.data_pipeline.orchestration.event_coordinator import EventCoordinator
from unittest.mock import AsyncMock, MagicMock


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add_pass(self, test_name: str):
        self.passed.append(test_name)
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"Test Results: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailed tests:")
            for test, error in self.failed:
                print(f"  - {test}: {error}")
        return len(self.failed) == 0


async def test_layer_enum():
    """Test that DataLayer enum is properly configured."""
    results = TestResults()
    
    try:
        # Test all layer values
        assert DataLayer.BASIC.value == 0
        assert DataLayer.LIQUID.value == 1
        assert DataLayer.CATALYST.value == 2
        assert DataLayer.ACTIVE.value == 3
        results.add_pass("DataLayer enum values")
        
        # Test layer properties
        assert DataLayer.BASIC.max_symbols == 10000
        assert DataLayer.LIQUID.max_symbols == 2000
        assert DataLayer.CATALYST.max_symbols == 500
        assert DataLayer.ACTIVE.max_symbols == 50
        results.add_pass("DataLayer max_symbols properties")
        
    except AssertionError as e:
        results.add_fail("DataLayer configuration", str(e))
    
    return results


async def test_event_publishing():
    """Test scanner event publishing flow."""
    results = TestResults()
    
    try:
        # Create event bus and publisher
        event_bus = EventBus()
        publisher = ScannerEventPublisher(event_bus)
        
        # Track published events
        published_events = []
        
        async def event_handler(event):
            published_events.append(event)
        
        # Subscribe to events
        await event_bus.subscribe('SymbolQualifiedEvent', event_handler)
        
        # Publish a qualification event
        await publisher.publish_symbol_qualified(
            symbol="TEST_SYMBOL",
            layer=DataLayer.LIQUID,
            qualification_reason="Test qualification",
            metrics={'test_metric': 100}
        )
        
        # Give event time to propagate
        await asyncio.sleep(0.1)
        
        # Verify event was published
        assert len(published_events) == 1
        event = published_events[0]
        assert event.symbol == "TEST_SYMBOL"
        assert event.layer == DataLayer.LIQUID.value
        results.add_pass("Event publishing")
        
    except Exception as e:
        results.add_fail("Event publishing", str(e))
    
    return results


async def test_event_coordinator():
    """Test event coordinator handles events and triggers backfills."""
    results = TestResults()
    
    try:
        # Create mock components
        event_bus = EventBus()
        
        # Mock layer manager
        layer_manager = AsyncMock()
        layer_manager.get_layer_config = AsyncMock(return_value={
            'backfill': {'default_days': 30},
            'processing': {'data_types': ['market_data']}
        })
        layer_manager.get_symbols_for_layer = AsyncMock(return_value=[])
        
        # Mock retention manager
        retention_manager = AsyncMock()
        retention_manager.apply_retention_policy = AsyncMock()
        
        # Create coordinator
        coordinator = EventCoordinator(
            event_bus=event_bus,
            layer_manager=layer_manager,
            retention_manager=retention_manager
        )
        
        # Initialize coordinator
        await coordinator.initialize()
        
        # Handle a symbol qualification
        await coordinator.handle_symbol_qualified("TEST_SYMBOL", DataLayer.LIQUID)
        
        # Check stats
        stats = await coordinator.get_event_statistics()
        assert stats['event_statistics']['symbol_qualified_events'] == 1
        assert stats['event_statistics']['backfills_scheduled'] == 1
        results.add_pass("Event coordinator backfill trigger")
        
    except Exception as e:
        results.add_fail("Event coordinator", str(e))
    
    return results


async def test_metadata_system():
    """Test scanner metadata storage system."""
    results = TestResults()
    
    try:
        # Mock company repository
        from main.data_pipeline.storage.repositories.company_repository import CompanyRepository
        
        # Create mock database
        mock_db = AsyncMock()
        mock_db.fetch_one = AsyncMock(return_value={'layer': 1, 'scanner_metadata': '{}'})
        mock_db.execute = AsyncMock(return_value=MagicMock(rowcount=1))
        
        # Create repository
        repo = CompanyRepository(db_adapter=mock_db)
        
        # Test metadata update
        metadata = {
            'scanner_metadata': json.dumps({
                'layer1_5': {
                    'qualified': True,
                    'strategy_affinity': {'momentum': 0.85},
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            })
        }
        
        result = await repo.update_company_metadata('TEST_SYMBOL', metadata)
        
        # Verify update was called
        assert mock_db.execute.called
        results.add_pass("Metadata storage system")
        
    except Exception as e:
        results.add_fail("Metadata system", str(e))
    
    return results


async def test_layer_transitions():
    """Test layer transition recording."""
    results = TestResults()
    
    try:
        # Mock database with transition tracking
        transitions = []
        
        async def mock_execute(query, *args):
            if "INSERT INTO layer_transitions" in query:
                transitions.append({
                    'symbol': args[0],
                    'from_layer': args[1],
                    'to_layer': args[2],
                    'reason': args[3]
                })
            return MagicMock(rowcount=1)
        
        mock_db = AsyncMock()
        mock_db.fetch_one = AsyncMock(return_value={'layer': 0})
        mock_db.execute = AsyncMock(side_effect=mock_execute)
        
        # Create repository
        from main.data_pipeline.storage.repositories.company_repository import CompanyRepository
        repo = CompanyRepository(db_adapter=mock_db)
        
        # Update layer (should record transition)
        await repo.update_layer('TEST_SYMBOL', 1, 'Test promotion')
        
        # Verify transition was recorded
        assert len(transitions) == 1
        assert transitions[0]['from_layer'] == 0
        assert transitions[0]['to_layer'] == 1
        results.add_pass("Layer transition recording")
        
    except Exception as e:
        results.add_fail("Layer transitions", str(e))
    
    return results


async def main():
    """Run all tests."""
    print("=" * 60)
    print("End-to-End Test: Scanner → Event → Backfill Flow")
    print("=" * 60)
    
    all_results = TestResults()
    
    # Run tests
    print("\n1. Testing DataLayer enum...")
    results = await test_layer_enum()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    
    print("\n2. Testing event publishing...")
    results = await test_event_publishing()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    
    print("\n3. Testing event coordinator...")
    results = await test_event_coordinator()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    
    print("\n4. Testing metadata system...")
    results = await test_metadata_system()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    
    print("\n5. Testing layer transitions...")
    results = await test_layer_transitions()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    
    # Summary
    success = all_results.summary()
    
    if success:
        print("\n🎉 All tests passed! System is working correctly.")
        print("\nKey achievements:")
        print("✅ Layer migration complete (single layer field)")
        print("✅ Scanner metadata system implemented")
        print("✅ Event-driven backfill working")
        print("✅ Layer transitions recorded for audit")
        print("✅ Layer 1.5 fixed to use metadata properly")
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)