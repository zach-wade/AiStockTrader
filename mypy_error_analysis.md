# MyPy Type Error Analysis - AI Trading System

## Executive Summary

The project has **476 MyPy type errors** across 42 files. The errors fall into four main categories that, when fixed, will resolve the majority of issues:

1. **Datetime/Timezone Protocol Violations** (26 errors in market_hours_service.py)
2. **Value Object Operator Overloads** (43 operator errors)
3. **Type Conversions** (137 arg-type errors)
4. **Protocol/Interface Mismatches** (65 assignment errors)

## Error Distribution by Type

| Error Code | Count | Description |
|------------|-------|-------------|
| [arg-type] | 137 | Incorrect argument types passed to functions |
| [no-untyped-def] | 97 | Functions missing type annotations |
| [assignment] | 65 | Incompatible type assignments |
| [operator] | 43 | Missing operator overloads |
| [attr-defined] | 43 | Accessing undefined attributes |

## Root Cause Analysis

### 1. TimezoneInfo Protocol Mismatch (Critical - 26+ errors)

**Problem**: The `TimezoneInfo` protocol doesn't match Python's `tzinfo` interface.

```python
# Current protocol (incomplete)
class TimezoneInfo(Protocol):
    def __str__(self) -> str: ...

# Python expects tzinfo interface with:
# - utcoffset(dt)
# - tzname(dt)
# - dst(dt)
```

**Impact**:

- `datetime.now(timezone)` fails because TimezoneInfo != tzinfo
- `.localize()` method calls fail (pytz-specific, not in protocol)
- `.astimezone()` calls fail

### 2. LocalizedDatetime Protocol Issues (Critical - Multiple cascading errors)

**Problem**: The `LocalizedDatetime` protocol conflicts with standard `datetime`:

```python
# Protocol expects:
def __add__(self, other: Any) -> LocalizedDatetime
def __ge__(self, other: LocalizedDatetime) -> bool

# But datetime has:
def __add__(self, other: timedelta) -> datetime
def __ge__(self, other: datetime) -> bool
```

### 3. Value Object Missing Operators (43 operator errors)

**Problem**: Value objects lack comparison with primitives:

```python
# Current implementation only compares with same type
if order.quantity <= 0:  # FAILS: Quantity can't compare with int
if order.limit_price <= 0:  # FAILS: Price can't compare with int
order.quantity * execution_price.value  # FAILS: No __mul__ for Decimal
```

### 4. Money/Decimal Incompatibilities (Multiple errors)

**Problem**: Creating Money from arithmetic operations:

```python
Money(order.quantity * execution_price.value)  # quantity is Quantity, not Decimal
Money(required_capital.amount > portfolio.cash_balance)  # comparing Money.amount with Decimal
```

## Prioritized Fix List

### Priority 1: Fix Timezone/Datetime Protocols (Fixes ~50+ errors)

**File**: `/Users/zachwade/StockMonitoring/src/domain/interfaces/time_service.py`

```python
from datetime import tzinfo as TzInfo
from typing import Protocol, runtime_checkable

@runtime_checkable
class TimezoneInfo(Protocol):
    """Protocol matching Python's tzinfo interface."""

    def tzname(self, dt: datetime | None) -> str | None: ...
    def utcoffset(self, dt: datetime | None) -> timedelta | None: ...
    def dst(self, dt: datetime | None) -> timedelta | None: ...
    def __str__(self) -> str: ...

@runtime_checkable
class LocalizedDatetime(Protocol):
    """Protocol for timezone-aware datetime."""

    # Match datetime's actual signatures
    def __add__(self, other: timedelta) -> "LocalizedDatetime": ...
    def __sub__(self, other: "LocalizedDatetime" | timedelta) -> timedelta | "LocalizedDatetime": ...
    def __lt__(self, other: "LocalizedDatetime" | datetime) -> bool: ...
    def __le__(self, other: "LocalizedDatetime" | datetime) -> bool: ...
    def __gt__(self, other: "LocalizedDatetime" | datetime) -> bool: ...
    def __ge__(self, other: "LocalizedDatetime" | datetime) -> bool: ...

    # Keep domain-specific methods
    def as_datetime(self) -> datetime: ...
    tzinfo: TimezoneInfo | None
```

### Priority 2: Add Value Object Operator Overloads (Fixes ~60+ errors)

**Files**:

- `/Users/zachwade/StockMonitoring/src/domain/value_objects/quantity.py`
- `/Users/zachwade/StockMonitoring/src/domain/value_objects/price.py`
- `/Users/zachwade/StockMonitoring/src/domain/value_objects/money.py`

Add these methods to each value object:

```python
class Quantity:
    # Add comparison with numeric types
    def __le__(self, other: "Quantity | Decimal | int | float") -> bool:
        if isinstance(other, Quantity):
            return self._value <= other._value
        return self._value <= Decimal(str(other))

    def __lt__(self, other: "Quantity | Decimal | int | float") -> bool:
        if isinstance(other, Quantity):
            return self._value < other._value
        return self._value < Decimal(str(other))

    def __ge__(self, other: "Quantity | Decimal | int | float") -> bool:
        if isinstance(other, Quantity):
            return self._value >= other._value
        return self._value >= Decimal(str(other))

    def __gt__(self, other: "Quantity | Decimal | int | float") -> bool:
        if isinstance(other, Quantity):
            return self._value > other._value
        return self._value > Decimal(str(other))

    # Add arithmetic operators
    def __mul__(self, other: "Quantity | Decimal | int | float") -> "Quantity":
        if isinstance(other, Quantity):
            return Quantity(self._value * other._value)
        return Quantity(self._value * Decimal(str(other)))

    def __rmul__(self, other: Decimal | int | float) -> "Quantity":
        return Quantity(Decimal(str(other)) * self._value)

    def __add__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot add Quantity and {type(other)}")
        return Quantity(self._value + other._value)

    def __sub__(self, other: "Quantity") -> "Quantity":
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot subtract {type(other)} from Quantity")
        return Quantity(self._value - other._value)
```

Similar additions needed for `Price` and `Money` classes.

### Priority 3: Fix MarketHoursService Implementation (Fixes 26 errors)

**File**: `/Users/zachwade/StockMonitoring/src/domain/services/market_hours_service.py`

Replace direct timezone usage with TimeService:

```python
def get_next_market_open(self, from_time: datetime | None = None) -> LocalizedDatetime | None:
    if from_time is None:
        current = self.time_service.get_current_time(self.timezone)
    elif not self.time_service.is_timezone_aware(from_time):
        current = self.time_service.localize_naive_datetime(from_time, self.timezone)
    else:
        current = self.time_service.convert_timezone(
            self.time_service.create_adapter(from_time), self.timezone
        )
    # Rest of implementation...
```

### Priority 4: Fix Type Conversions in Services (Fixes ~100+ errors)

**Files**:

- `/Users/zachwade/StockMonitoring/src/domain/services/order_validator.py`
- `/Users/zachwade/StockMonitoring/src/domain/services/order_processor.py`

Use proper value object methods:

```python
# Instead of:
Money(order.quantity * execution_price.value)

# Use:
Money(order.quantity.value * execution_price.value)

# Or add a calculate_value method:
def calculate_order_value(quantity: Quantity, price: Price) -> Money:
    return Money(quantity.value * price.value)
```

### Priority 5: Fix Adapter Implementations (Fixes remaining errors)

**File**: `/Users/zachwade/StockMonitoring/src/infrastructure/time/timezone_service.py`

```python
class TimezoneInfoAdapter:
    """Adapter wrapping tzinfo to match TimezoneInfo protocol."""

    def __init__(self, tz: tzinfo, name: str):
        self._tz = tz
        self._name = name

    def tzname(self, dt: datetime | None) -> str | None:
        return self._tz.tzname(dt)

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return self._tz.utcoffset(dt)

    def dst(self, dt: datetime | None) -> timedelta | None:
        return self._tz.dst(dt)

    def __str__(self) -> str:
        return self._name
```

## Implementation Strategy

### Phase 1: Protocol Fixes (1-2 hours)

1. Update TimezoneInfo and LocalizedDatetime protocols
2. Fix adapter implementations
3. Run MyPy to verify datetime errors resolved

### Phase 2: Value Object Operators (2-3 hours)

1. Add numeric comparison operators to all value objects
2. Add arithmetic operators (**mul**, **add**, **sub**)
3. Add type conversions where needed
4. Run tests to ensure no behavioral changes

### Phase 3: Service Layer Fixes (2-3 hours)

1. Update MarketHoursService to use TimeService properly
2. Fix type conversions in order_validator.py
3. Fix type conversions in order_processor.py
4. Update position_manager.py

### Phase 4: Clean Up (1 hour)

1. Fix remaining arg-type errors in examples/
2. Add missing type annotations
3. Remove unnecessary type ignores

## Expected Results

After implementing these fixes:

- **Datetime/timezone errors**: 50+ errors resolved
- **Operator errors**: 43 errors resolved
- **Type conversion errors**: ~200 errors resolved
- **Assignment errors**: ~65 errors resolved

**Total expected resolution**: ~350-400 of 476 errors (75-85%)

The remaining errors will be minor issues in example files and missing type annotations that can be addressed incrementally.

## Testing Strategy

1. **Unit Tests**: Run existing value object tests after adding operators
2. **Integration Tests**: Verify market hours service still works correctly
3. **Type Checking**: Run `mypy --strict` incrementally on fixed modules
4. **Regression Testing**: Ensure no behavioral changes in order processing

## Notes

- The root cause is attempting to create domain-specific protocols that don't properly extend Python's built-in protocols
- Value objects need to support operations with primitive types for practical use
- The TimeService abstraction needs to properly wrap timezone implementations
- Consider using `@runtime_checkable` for protocols to catch issues earlier
