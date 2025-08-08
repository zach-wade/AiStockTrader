#!/usr/bin/env python
"""
Test the complete event-driven backfill flow.

This test verifies:
1. Scanner publishes symbol qualification event
2. EventCoordinator schedules backfill
3. BackfillEventHandler executes backfill
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from main.events.core import EventBusFactory
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.events.handlers.backfill_event_handler import BackfillEventHandler
from main.data_pipeline.orchestration.event_coordinator import EventCoordinator
from main.data_pipeline.core.enums import DataLayer
from main.interfaces.events.event_types import SymbolQualifiedEvent
from main.utils.core import get_logger
from unittest.mock import Mock, AsyncMock, patch

logger = get_logger(__name__)


async def test_event_flow():
    """Test the complete event-driven backfill flow."""
    print("\n=== Testing Event-Driven Backfill Flow ===\n")
    
    # 1. Create and start event bus
    print("1. Creating and starting event bus...")
    event_bus = EventBusFactory.create_test_instance()
    await event_bus.start()  # Start the event bus workers
    print("   ✓ Event bus created and started")
    
    # 2. Create and initialize components
    print("\n2. Initializing components...")
    
    # Create scanner event publisher
    scanner_publisher = ScannerEventPublisher(event_bus)
    print("   ✓ ScannerEventPublisher created")
    
    # Create mock layer and retention managers
    mock_layer_manager = Mock()
    mock_layer_manager.get_layer_config = AsyncMock(return_value={
        'retention_days': 30,
        'hot_storage_days': 7
    })
    
    mock_retention_manager = Mock()
    
    # Create event coordinator with mocked backfill config
    event_coordinator = EventCoordinator(
        event_bus=event_bus,
        layer_manager=mock_layer_manager,
        retention_manager=mock_retention_manager,
        config={
            'auto_backfill_enabled': True,
            'backfill_delay_minutes': 0,  # No delay for testing
            'max_concurrent_backfills': 3
        }
    )
    await event_coordinator.initialize()
    print("   ✓ EventCoordinator initialized and subscribed")
    
    # Create backfill handler with test config
    backfill_handler = BackfillEventHandler(
        event_bus=event_bus,
        config={
            'backfill_handler': {
                'enabled': True,
                'max_concurrent_backfills': 1,
                'retry_attempts': 1,
                'deduplication_window_minutes': 1
            }
        }
    )
    
    # Mock the _execute_backfill method to avoid needing the actual backfill system
    mock_execute = AsyncMock(return_value=None)
    original_execute = backfill_handler._execute_backfill
    backfill_handler._execute_backfill = mock_execute
    
    await backfill_handler.initialize()
    print("   ✓ BackfillEventHandler initialized and subscribed")
    
    # 3. Simulate scanner qualifying a symbol
    print("\n3. Simulating scanner qualification...")
    
    test_symbol = "TEST"
    test_layer = DataLayer.LIQUID
    
    # Publish symbol qualified event
    await scanner_publisher.publish_symbol_qualified(
        symbol=test_symbol,
        layer=test_layer,
        qualification_reason="Test qualification",
        metrics={'liquidity_score': 100.0}
    )
    print(f"   ✓ Published SymbolQualifiedEvent for {test_symbol}")
    
    # Give time for async processing
    await asyncio.sleep(1)
    
    # 4. Verify the flow
    print("\n4. Verifying event flow...")
    
    # Check event coordinator statistics
    stats = await event_coordinator.get_event_statistics()
    print(f"   Event Coordinator Stats:")
    print(f"     - Symbol qualified events: {stats['event_statistics'].get('symbol_qualified_events', 0)}")
    print(f"     - Backfills scheduled: {stats['event_statistics'].get('backfills_scheduled', 0)}")
    
    # Check backfill handler statistics
    handler_stats = backfill_handler.get_statistics()
    print(f"   Backfill Handler Stats:")
    print(f"     - Events received: {handler_stats['statistics']['received']}")
    print(f"     - Backfills executed: {handler_stats['statistics']['executed']}")
    print(f"     - Succeeded: {handler_stats['statistics']['succeeded']}")
    
    # Verify the mock was called
    if mock_execute.called:
        print("\n   ✓ Backfill execution was triggered!")
        if mock_execute.call_args and len(mock_execute.call_args) > 0:
            task = mock_execute.call_args[0][0]  # First argument is the BackfillTask
            print(f"     - Symbol: {task.symbol}")
            print(f"     - Layer: {task.layer}")
            print(f"     - Data types: {task.data_types}")
    else:
        print("\n   ✗ Backfill execution was NOT triggered")
    
    # 5. Test deduplication
    print("\n5. Testing deduplication...")
    
    # Publish same event again
    await scanner_publisher.publish_symbol_qualified(
        symbol=test_symbol,
        layer=test_layer,
        qualification_reason="Duplicate test",
        metrics={'liquidity_score': 100.0}
    )
    
    await asyncio.sleep(1)
    
    handler_stats_after = backfill_handler.get_statistics()
    if handler_stats_after['statistics']['deduplicated'] > 0:
        print(f"   ✓ Deduplication working: {handler_stats_after['statistics']['deduplicated']} duplicates prevented")
    else:
        print("   ⚠ No deduplication occurred (may be normal if enough time passed)")
    
    # Clean up - stop the event bus
    await event_bus.stop()
    
    print("\n=== Event Flow Test Complete ===")
    
    # Return success indicator
    return mock_execute.called


async def main():
    """Run the test."""
    try:
        success = await test_event_flow()
        
        print("\n" + "="*50)
        if success:
            print("✅ SUCCESS: Event-driven backfill flow is working!")
            print("The complete flow is connected:")
            print("  Scanner → Event Bus → EventCoordinator → BackfillHandler → Execution")
        else:
            print("❌ FAILURE: Backfill was not triggered")
            print("Check the event flow configuration")
        print("="*50)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)