# Integration Test Updates - Events Module

## Summary
Updated existing integration tests to work with the refactored events module architecture.

## Files Updated

### 1. test_event_bus_integration.py
**Changes Made:**
- Updated `RiskEvent` constructor calls to use new field names:
  - `event_type=EventType.RISK_LIMIT_BREACH` → removed (auto-set in __post_init__)
  - `metadata={}` → `metrics={}` 
  - Added required `risk_type` field
- Updated event type subscriptions:
  - `EventType.RISK_LIMIT_BREACH` → `EventType.RISK_ALERT`
- Fixed event construction patterns to match new dataclass structure

**Key Changes:**
```python
# Before
RiskEvent(
    event_type=EventType.RISK_LIMIT_BREACH,
    severity="warning", 
    message="Position size approaching limit",
    metadata={"position_size": 0.95}
)

# After  
RiskEvent(
    risk_type="position_limit",
    severity="warning",
    message="Position size approaching limit", 
    metrics={"position_size": 0.95}
)
```

### 2. test_event_driven_engine.py
**Changes Made:**
- Updated `RiskEvent` constructor calls throughout
- Changed event type references from `RISK_LIMIT_BREACH` to `RISK_ALERT` 
- Updated event subscription patterns
- Fixed field name changes (`metadata` → `metrics`)

### 3. test_event_feature_integration.py  
**Changes Made:**
- Fixed import statements to properly import from event_types module
- Updated import pattern:
  ```python
  # Before
  from main.events import EventBus, ScannerAlertEvent, FeatureRequestEvent, EventType
  
  # After
  from main.events import EventBus, EventType
  from main.events.event_types import ScannerAlertEvent, FeatureRequestEvent
  ```

### 4. test_scanner_feature_bridge.py
**Changes Made:**
- Updated imports to use event_types module for ScanAlert and AlertType
- Fixed ScanAlert constructor to match new dataclass structure:
  - Removed `timestamp`, `source_scanner` fields (auto-generated)
  - Changed `metadata` → `data`
- Updated AlertType usage to use string values instead of enum

## Architectural Changes Addressed

### Event Type Refactoring
- Event types are now properly centralized in event_types.py
- Events use dataclass __post_init__ pattern for auto-initialization
- Field names standardized (metadata → metrics for RiskEvent)

### Import Structure Changes  
- Events now properly exported from main events module
- Specific event classes imported from event_types submodule
- Fixed circular import issues

### Constructor Pattern Changes
- Events use simpler constructor patterns
- Required fields made explicit
- Auto-generated fields (timestamp, event_id) handled in __post_init__

## Tests Status
✅ **Updated and Compatible**
- test_event_bus_integration.py - Core event bus functionality
- test_event_driven_engine.py - Event-driven workflows  
- test_event_feature_integration.py - Scanner to feature pipeline
- test_scanner_feature_bridge.py - Bridge component integration

## Next Steps
1. Run integration tests to verify all updates work correctly
2. Address any remaining compatibility issues
3. Add new integration test scenarios for refactored components
4. Update test documentation and examples

## Refactoring Benefits Validated
- ✅ Cleaner event construction patterns
- ✅ Better separation of concerns in event types
- ✅ Consistent field naming across event types
- ✅ Improved import structure and module organization
- ✅ Maintained backward compatibility where possible