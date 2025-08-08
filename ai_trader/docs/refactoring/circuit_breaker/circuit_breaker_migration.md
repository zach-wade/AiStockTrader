# Circuit Breaker Migration Guide

## Overview

This guide provides comprehensive migration instructions for transitioning from the monolithic circuit breaker system to the new modular architecture. The migration is designed to be gradual and non-disruptive.

## Migration Strategy

### Phase 1: Immediate (Zero Changes Required)
The facade pattern ensures 100% backward compatibility. No code changes are required for existing implementations.

### Phase 2: Gradual Migration
Gradually adopt individual breaker components and new features as needed.

### Phase 3: Full Modernization
Leverage the complete modular architecture with advanced features and analytics.

## Pre-Migration Checklist

### 1. Backup Current Implementation
```bash
# Backup existing circuit breaker usage
cp -r src/main/risk_management/real_time/circuit_breaker.py \
      src/main/risk_management/real_time/circuit_breaker.py.backup
```

### 2. Verify Current Usage
```python
# Audit current circuit breaker usage
grep -r "CircuitBreaker" src/ --include="*.py"
grep -r "circuit_breaker" src/ --include="*.py"
```

### 3. Test Current Functionality
```python
# Ensure existing tests pass
python -m pytest tests/risk_management/test_circuit_breaker.py -v
```

## Migration Scenarios

### Scenario 1: Basic Usage (No Changes Required)

**Before (Monolithic)**
```python
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

config = {
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08,
    'loss_rate_threshold': 0.03,
    'max_positions': 20
}

circuit_breaker = CircuitBreaker(config)

# Check all breakers
results = await circuit_breaker.check_all_breakers(
    portfolio_value, positions, market_conditions
)

# Status checking
is_allowed = circuit_breaker.is_trading_allowed()
status = circuit_breaker.get_breaker_status()
```

**After (Modular - No Changes)**
```python
# Import remains the same - facade provides compatibility
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

config = {
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08,
    'loss_rate_threshold': 0.03,
    'max_positions': 20
}

circuit_breaker = CircuitBreaker(config)

# All existing methods work exactly the same
results = await circuit_breaker.check_all_breakers(
    portfolio_value, positions, market_conditions
)

is_allowed = circuit_breaker.is_trading_allowed()
status = circuit_breaker.get_breaker_status()
```

### Scenario 2: Enhanced Configuration

**Before (Limited Configuration)**
```python
config = {
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08
}
```

**After (Enhanced Configuration)**
```python
config = {
    # Basic thresholds (unchanged)
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08,
    'loss_rate_threshold': 0.03,
    'max_positions': 20,
    
    # New enhanced configuration
    'max_sector_concentration': 0.30,
    'max_long_exposure': 1.0,
    'max_short_exposure': 0.5,
    'anomaly_detection_threshold': 3.0,
    'data_quality_threshold': 95.0,
    
    # Time windows
    'loss_rate_window_minutes': 5,
    'volatility_window_minutes': 30,
    'cooldown_period_minutes': 15,
    
    # Breaker-specific configurations
    'breaker_configs': {
        'volatility': {
            'enabled': True,
            'threshold': 0.05,
            'cooldown_minutes': 15
        },
        'drawdown': {
            'enabled': True,
            'threshold': 0.08,
            'cooldown_minutes': 30
        }
    }
}
```

### Scenario 3: Individual Breaker Usage

**Before (Monolithic Access)**
```python
# Had to use entire circuit breaker for specific checks
circuit_breaker = CircuitBreaker(config)
results = await circuit_breaker.check_all_breakers(
    portfolio_value, positions, market_conditions
)
volatility_tripped = results.get('volatility', False)
```

**After (Individual Breaker Access)**
```python
from ai_trader.risk_management.real_time.circuit_breaker.breakers import (
    VolatilityBreaker, DrawdownBreaker
)
from ai_trader.risk_management.real_time.circuit_breaker.config import BreakerConfig

config = BreakerConfig(config_dict)

# Use individual breakers
volatility_breaker = VolatilityBreaker(BreakerType.VOLATILITY, config)
drawdown_breaker = DrawdownBreaker(BreakerType.DRAWDOWN, config)

# Check specific breakers
vol_tripped = await volatility_breaker.check(
    portfolio_value, positions, market_conditions
)

dd_tripped = await drawdown_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get detailed metrics
vol_stats = volatility_breaker.get_volatility_statistics()
dd_stats = drawdown_breaker.get_drawdown_statistics()
```

### Scenario 4: Event-Driven Architecture

**Before (Polling-Based)**
```python
# Manual status checking
while trading_active:
    results = await circuit_breaker.check_all_breakers(
        portfolio_value, positions, market_conditions
    )
    
    if not circuit_breaker.is_trading_allowed():
        handle_trading_halt()
    
    await asyncio.sleep(1)
```

**After (Event-Driven)**
```python
from ai_trader.risk_management.real_time.circuit_breaker.events import BreakerEventManager

# Set up event handling
event_manager = BreakerEventManager()

def handle_breaker_event(event):
    if event.status == BreakerStatus.TRIPPED:
        logger.warning(f"Breaker tripped: {event.breaker_type.value}")
        handle_trading_halt(event.breaker_type)
    elif event.status == BreakerStatus.WARNING:
        logger.warning(f"Breaker warning: {event.breaker_type.value}")
        send_alert(event.message)

event_manager.register_callback(handle_breaker_event)

# Circuit breaker now automatically triggers events
circuit_breaker = CircuitBreaker(config)
circuit_breaker.event_manager = event_manager
```

## Migration Best Practices

### 1. Gradual Migration Approach

**Step 1: Verify Current Functionality**
```python
# Test current implementation
def test_current_implementation():
    circuit_breaker = CircuitBreaker(config)
    # Run existing tests
    assert circuit_breaker.is_trading_allowed() == True
    # Verify all methods work
```

**Step 2: Add Enhanced Configuration**
```python
# Gradually add new configuration options
config.update({
    'max_sector_concentration': 0.30,
    'data_quality_threshold': 95.0
})
```

**Step 3: Implement Event Handling**
```python
# Add event handling for better monitoring
def setup_event_handling():
    event_manager = BreakerEventManager()
    event_manager.register_callback(log_breaker_events)
    event_manager.register_callback(send_risk_alerts)
```

### 2. Testing Strategy

**Unit Tests**
```python
import pytest
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

def test_backward_compatibility():
    """Test that existing code continues to work."""
    config = {'volatility_threshold': 0.05}
    circuit_breaker = CircuitBreaker(config)
    
    # All original methods should work
    assert hasattr(circuit_breaker, 'check_all_breakers')
    assert hasattr(circuit_breaker, 'is_trading_allowed')
    assert hasattr(circuit_breaker, 'get_breaker_status')

def test_new_features():
    """Test new modular features."""
    config = BreakerConfig(config_dict)
    volatility_breaker = VolatilityBreaker(BreakerType.VOLATILITY, config)
    
    # New methods should be available
    assert hasattr(volatility_breaker, 'get_volatility_statistics')
    assert hasattr(volatility_breaker, 'check_warning_conditions')
```

**Integration Tests**
```python
async def test_migration_integration():
    """Test that migrated code works with existing systems."""
    # Test facade compatibility
    circuit_breaker = CircuitBreaker(config)
    results = await circuit_breaker.check_all_breakers(
        portfolio_value, positions, market_conditions
    )
    
    # Test individual breaker usage
    volatility_breaker = VolatilityBreaker(BreakerType.VOLATILITY, config)
    vol_result = await volatility_breaker.check(
        portfolio_value, positions, market_conditions
    )
    
    # Results should be consistent
    assert results[BreakerType.VOLATILITY] == vol_result
```

### 3. Performance Optimization

**Memory Usage**
```python
# Configure appropriate history limits
config = {
    'volatility_window_minutes': 30,  # Reasonable window
    'max_events': 1000,  # Event history limit
    'max_position_history': 100  # Position history limit
}
```

**Selective Execution**
```python
# Enable only needed breakers
config = {
    'breaker_configs': {
        'volatility': {'enabled': True},
        'drawdown': {'enabled': True},
        'loss_rate': {'enabled': False},  # Disable if not needed
        'position_limit': {'enabled': True}
    }
}
```

## Common Migration Issues and Solutions

### Issue 1: Import Errors

**Problem**
```python
# Old import that might cause issues
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker
```

**Solution**
```python
# Use the new import path (backward compatible)
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

# Or use the facade explicitly
from ai_trader.risk_management.real_time.circuit_breaker.facade import CircuitBreakerFacade
CircuitBreaker = CircuitBreakerFacade
```

### Issue 2: Configuration Conflicts

**Problem**
```python
# Old configuration might not have all required fields
config = {'volatility_threshold': 0.05}
```

**Solution**
```python
# Use configuration validation
from ai_trader.risk_management.real_time.circuit_breaker.config import BreakerConfig

try:
    breaker_config = BreakerConfig(config)
    warnings = breaker_config.validate_limits()
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")
except Exception as e:
    logger.error(f"Configuration error: {e}")
    # Use default configuration
    breaker_config = BreakerConfig({})
```

### Issue 3: Callback Registration

**Problem**
```python
# Old callback registration method
circuit_breaker.callbacks.append(my_callback)
```

**Solution**
```python
# Use the event manager for callbacks
circuit_breaker.register_callback(my_callback)

# Or use the event manager directly
event_manager = circuit_breaker.event_manager
event_manager.register_callback(my_callback)
```

## Validation and Testing

### Migration Validation Script
```python
#!/usr/bin/env python3
"""
Migration validation script to ensure circuit breaker migration is successful.
"""

import asyncio
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

async def validate_migration():
    """Validate that migration is successful."""
    config = {
        'volatility_threshold': 0.05,
        'max_drawdown': 0.08,
        'loss_rate_threshold': 0.03,
        'max_positions': 20
    }
    
    # Test backward compatibility
    circuit_breaker = CircuitBreaker(config)
    
    # Verify all methods exist
    assert hasattr(circuit_breaker, 'check_all_breakers')
    assert hasattr(circuit_breaker, 'is_trading_allowed')
    assert hasattr(circuit_breaker, 'get_breaker_status')
    
    # Test functionality
    is_allowed = circuit_breaker.is_trading_allowed()
    assert isinstance(is_allowed, bool)
    
    status = circuit_breaker.get_breaker_status()
    assert isinstance(status, dict)
    assert 'is_trading_allowed' in status
    
    print("✅ Migration validation successful!")

if __name__ == "__main__":
    asyncio.run(validate_migration())
```

## Post-Migration Optimization

### 1. Enable New Features
```python
# Take advantage of new analytics
volatility_stats = circuit_breaker.get_volatility_statistics()
drawdown_stats = circuit_breaker.get_drawdown_statistics()

# Use enhanced monitoring
enhanced_status = circuit_breaker.get_enhanced_status()
```

### 2. Implement Advanced Configuration
```python
# Use breaker-specific configuration
config = {
    'breaker_configs': {
        'volatility': {
            'threshold': 0.04,  # Stricter for volatility
            'cooldown_minutes': 10
        },
        'drawdown': {
            'threshold': 0.06,  # Stricter for drawdown
            'cooldown_minutes': 20
        }
    }
}
```

### 3. Monitor Performance
```python
# Add performance monitoring
import time

start_time = time.time()
results = await circuit_breaker.check_all_breakers(
    portfolio_value, positions, market_conditions
)
execution_time = time.time() - start_time

logger.info(f"Circuit breaker check took {execution_time:.3f} seconds")
```

## Rollback Strategy

If issues arise, you can quickly rollback:

### 1. Temporary Rollback
```python
# Use the original facade behavior
circuit_breaker = CircuitBreaker(config)
# Continue with original API
```

### 2. Configuration Rollback
```python
# Revert to minimal configuration
config = {
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08
}
```

### 3. Full Rollback
```python
# Temporarily disable problematic breakers
config = {
    'breaker_configs': {
        'volatility': {'enabled': False},
        'drawdown': {'enabled': True}
    }
}
```

## Support and Troubleshooting

### Common Issues
1. **Import Errors**: Check import paths and module structure
2. **Configuration Errors**: Validate configuration with BreakerConfig
3. **Performance Issues**: Check history limits and selective execution
4. **Callback Issues**: Use event manager for callback registration

### Getting Help
- Review the component documentation
- Check the architecture documentation
- Run the validation script
- Enable debug logging for detailed information

### Debug Logging
```python
import logging
logging.getLogger('ai_trader.risk_management.real_time.circuit_breaker').setLevel(logging.DEBUG)
```

---
*Migration Guide*  
*Version: 2.0*  
*Last Updated: 2025-07-15*  
*Author: AI Trading System Development Team*