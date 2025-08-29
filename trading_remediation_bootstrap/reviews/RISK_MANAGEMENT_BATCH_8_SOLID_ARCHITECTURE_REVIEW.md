# SOLID Principles and Architecture Integrity Review - Risk Management Batch 8

## Executive Summary

This review analyzes 5 files from the risk_management pre_trade unified_limit_checker module for SOLID principle compliance and architectural integrity. The review identifies **17 critical violations** affecting system maintainability, extensibility, and architectural coherence.

### Architectural Impact Assessment: **HIGH**

**Justification:** Multiple severe SOLID violations, including SRP breaches in core components, dependency inversion issues, and architectural boundary violations that significantly impact system maintainability and evolution.

## Files Reviewed

1. `checkers/__init__.py` (14 lines)
2. `checkers/drawdown.py` (486 lines)
3. `checkers/position_size.py` (120 lines)
4. `checkers/simple_threshold.py` (130 lines)
5. `events.py` (410 lines)

## Pattern Compliance Checklist

| Principle | Status | Critical Issues |
|-----------|---------|-----------------|
| **Single Responsibility (SRP)** | ❌ | Multiple classes with 3+ responsibilities |
| **Open/Closed (OCP)** | ❌ | Hard-coded configurations, no extension points |
| **Liskov Substitution (LSP)** | ❌ | Inconsistent method signatures across implementations |
| **Interface Segregation (ISP)** | ❌ | Fat interfaces forcing unnecessary dependencies |
| **Dependency Inversion (DIP)** | ❌ | Direct coupling to concrete implementations |
| **Dependency Management** | ❌ | Circular dependency risks, tight coupling |
| **Abstraction Levels** | ❌ | Mixed abstraction levels within classes |
| **Pattern Consistency** | ✅ | Consistent use of checker pattern |

## Critical Violations Found

### ISSUE-3113: Single Responsibility Principle Violation - DrawdownChecker

**File:** `/checkers/drawdown.py`
**Class:** `DrawdownChecker` (lines 53-487)
**Principle:** Single Responsibility (SRP)
**Severity:** HIGH

**Violation:**
The `DrawdownChecker` class has at least 6 distinct responsibilities:

1. Limit checking logic (primary responsibility)
2. Portfolio peak tracking and caching (lines 69-70)
3. Drawdown history management (lines 71, 288-294)
4. Statistical calculations (lines 236-301)
5. Recovery period tracking (lines 439-473)
6. Multiple inheritance coordination (lines 64-65)

**Impact:**

- Changes to caching strategy affect limit checking
- Statistical calculation changes require modifying core checker
- Testing requires complex setup due to multiple concerns
- Difficult to reuse individual capabilities

**Refactoring Recommendation:**

```python
# Separate concerns into dedicated classes
class DrawdownCalculator:
    """Handles drawdown calculations"""
    async def calculate_drawdowns(self, portfolio_state: PortfolioState) -> Dict[str, float]:
        pass

class PortfolioPeakTracker:
    """Manages portfolio peak tracking"""
    def update_peak(self, portfolio_id: str, value: float) -> float:
        pass

class DrawdownHistoryManager:
    """Manages drawdown history"""
    def record_drawdown(self, timestamp: datetime, value: float):
        pass

    def get_recovery_metrics(self) -> Dict[str, Any]:
        pass

class DrawdownChecker(LimitChecker):
    """Focused on limit checking only"""
    def __init__(self,
                 calculator: DrawdownCalculator,
                 peak_tracker: PortfolioPeakTracker,
                 history_manager: DrawdownHistoryManager):
        self.calculator = calculator
        self.peak_tracker = peak_tracker
        self.history_manager = history_manager
```

### ISSUE-3114: Open/Closed Principle Violation - Hard-coded Configuration

**File:** `/checkers/drawdown.py`
**Class:** `DrawdownConfig` (lines 31-51)
**Principle:** Open/Closed (OCP)
**Severity:** MEDIUM

**Violation:**
Configuration is hard-coded with default values directly in the dataclass, making it closed for extension without modification.

**Impact:**

- Cannot add new configuration options without modifying the class
- No strategy pattern for different drawdown calculation methods
- Difficult to test with different configurations

**Refactoring Recommendation:**

```python
from abc import ABC, abstractmethod

class DrawdownStrategy(ABC):
    """Abstract strategy for drawdown calculations"""
    @abstractmethod
    def calculate_limit(self, current_value: float, peak_value: float) -> float:
        pass

class PercentageDrawdownStrategy(DrawdownStrategy):
    def __init__(self, max_percentage: float):
        self.max_percentage = max_percentage

    def calculate_limit(self, current_value: float, peak_value: float) -> float:
        return (peak_value - current_value) / peak_value

class DrawdownConfigProvider(ABC):
    """Abstract configuration provider"""
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass
```

### ISSUE-3115: Liskov Substitution Principle Violation - Inconsistent Method Signatures

**File:** Multiple checker files
**Classes:** `DrawdownChecker`, `PositionSizeChecker`, `SimpleThresholdChecker`
**Principle:** Liskov Substitution (LSP)
**Severity:** HIGH

**Violation:**
Different checker implementations have inconsistent method signatures and return types:

- `DrawdownChecker.check()` returns `LimitCheckResult` with different metadata structure
- `PositionSizeChecker.check_limit()` has different parameter expectations
- `SimpleThresholdChecker` uses different context structure

**Impact:**

- Cannot reliably substitute one checker for another
- Client code must know specific implementation details
- Breaks polymorphism

**Refactoring Recommendation:**

```python
from typing import Protocol

class ILimitChecker(Protocol):
    """Consistent interface for all checkers"""

    async def check_limit(self,
                         limit: LimitDefinition,
                         context: CheckContext) -> LimitCheckResult:
        """Standard check method signature"""
        ...

    def supports_limit_type(self, limit_type: LimitType) -> bool:
        """Check if this checker supports the limit type"""
        ...
```

### ISSUE-3116: Interface Segregation Principle Violation - Fat Interface

**File:** `/checkers/drawdown.py`
**Class:** `DrawdownChecker`
**Principle:** Interface Segregation (ISP)
**Severity:** HIGH

**Violation:**
The class implements multiple interfaces through inheritance, forcing clients to depend on methods they don't use:

- Methods from `LimitChecker`: `check_limit`, `calculate_current_value`, `supports_limit_type`
- Methods from `ErrorHandlingMixin`: error handling methods
- Custom methods: `check`, `get_statistics`, multiple private calculation methods

**Impact:**

- Clients must handle exceptions from error handling even if not needed
- Testing requires mocking unused interface methods
- Increased coupling

**Refactoring Recommendation:**

```python
class ILimitValidator:
    """Core validation interface"""
    async def validate(self, request: ValidationRequest) -> ValidationResult:
        pass

class IStatisticsProvider:
    """Optional statistics interface"""
    def get_statistics(self) -> Dict[str, Any]:
        pass

class IErrorReporter:
    """Optional error reporting interface"""
    def report_error(self, error: Exception) -> None:
        pass

# Use composition instead of inheritance
class DrawdownChecker:
    def __init__(self,
                 error_reporter: Optional[IErrorReporter] = None,
                 stats_provider: Optional[IStatisticsProvider] = None):
        self.error_reporter = error_reporter
        self.stats_provider = stats_provider
```

### ISSUE-3117: Dependency Inversion Principle Violation - Concrete Dependencies

**File:** `/checkers/drawdown.py`
**Lines:** 14-27
**Principle:** Dependency Inversion (DIP)
**Severity:** HIGH

**Violation:**
Direct imports of concrete implementations instead of abstractions:

```python
from main.risk_management.types import RiskCheckResult, RiskMetric, RiskLevel
from main.utils.core import ErrorHandlingMixin
from main.utils.monitoring import record_metric
```

**Impact:**

- Tight coupling to specific implementations
- Cannot mock dependencies for testing
- Changes to concrete classes affect all checkers

**Refactoring Recommendation:**

```python
# Define abstractions
from abc import ABC, abstractmethod

class IMetricsRecorder(ABC):
    @abstractmethod
    def record(self, metric: str, value: float, tags: Dict[str, str]) -> None:
        pass

class IRiskResultFactory(ABC):
    @abstractmethod
    def create_result(self, **kwargs) -> Any:
        pass

# Inject dependencies
class DrawdownChecker:
    def __init__(self,
                 metrics_recorder: IMetricsRecorder,
                 result_factory: IRiskResultFactory):
        self.metrics_recorder = metrics_recorder
        self.result_factory = result_factory
```

### ISSUE-3118: Multiple Inheritance Anti-Pattern

**File:** `/checkers/drawdown.py`
**Line:** 53
**Principle:** SRP, DIP
**Severity:** HIGH

**Violation:**

```python
class DrawdownChecker(LimitChecker, ErrorHandlingMixin):
```

Multiple inheritance creates diamond problem risks and violates single responsibility.

**Impact:**

- Method resolution order (MRO) complexity
- Difficult to test in isolation
- Coupling to multiple base classes

**Refactoring Recommendation:**

```python
# Use composition over inheritance
class DrawdownChecker:
    def __init__(self,
                 base_checker: LimitChecker,
                 error_handler: ErrorHandler):
        self.base_checker = base_checker
        self.error_handler = error_handler

    async def check(self, request: CheckRequest) -> CheckResult:
        try:
            return await self._perform_check(request)
        except Exception as e:
            return self.error_handler.handle(e)
```

### ISSUE-3119: Event Manager God Object

**File:** `/events.py`
**Class:** `EventManager` (lines 179-373)
**Principle:** Single Responsibility (SRP)
**Severity:** HIGH

**Violation:**
The `EventManager` class has too many responsibilities:

1. Event subscription management (lines 224-252)
2. Event emission and distribution (lines 254-273)
3. Buffer management (lines 275-284)
4. Async task management (lines 281-283)
5. Statistics tracking (lines 356-366)
6. Circuit breaker integration (lines 213-217)
7. Event processing (lines 285-313)

**Impact:**

- Single class controls entire event system
- Changes to any aspect require modifying this class
- Difficult to test individual features

**Refactoring Recommendation:**

```python
class EventSubscriptionManager:
    """Manages event subscriptions"""
    def subscribe(self, event_type: str, handler: Callable):
        pass

class EventDistributor:
    """Distributes events to subscribers"""
    async def distribute(self, event: Event, subscribers: List[Callable]):
        pass

class EventProcessor:
    """Processes events"""
    async def process(self, event: Event):
        pass

class EventManager:
    """Coordinates event handling components"""
    def __init__(self,
                 subscription_mgr: EventSubscriptionManager,
                 distributor: EventDistributor,
                 processor: EventProcessor):
        self.subscription_mgr = subscription_mgr
        self.distributor = distributor
        self.processor = processor
```

### ISSUE-3120: Violation of DRY - Repeated Drawdown Check Logic

**File:** `/checkers/drawdown.py`
**Methods:** `_check_daily_drawdown`, `_check_weekly_drawdown`, `_check_total_drawdown` (lines 317-394)
**Principle:** DRY (Don't Repeat Yourself)
**Severity:** MEDIUM

**Violation:**
Three methods with nearly identical logic, only differing in configuration values.

**Impact:**

- Maintenance burden - changes must be made in multiple places
- Risk of inconsistency
- Code bloat

**Refactoring Recommendation:**

```python
def _check_drawdown_limit(self,
                          current_dd: float,
                          potential_loss: float,
                          portfolio_value: float,
                          max_drawdown: float,
                          check_name: str,
                          metric: RiskMetric) -> RiskCheckResult:
    """Generic drawdown limit check"""
    potential_dd = current_dd + (potential_loss / portfolio_value)
    passed = potential_dd <= max_drawdown
    utilization = (potential_dd / max_drawdown * 100 if max_drawdown > 0 else 0)

    warning = None
    if utilization > self.drawdown_config.warning_threshold * 100:
        warning = f"{check_name} drawdown at {utilization:.0f}% of limit"

    return RiskCheckResult(
        passed=passed,
        check_name=check_name,
        metric=metric,
        current_value=potential_dd,
        limit_value=max_drawdown,
        utilization=utilization,
        message=f"{check_name}: {potential_dd:.1%} vs limit {max_drawdown:.1%}",
        metadata={'warning': warning}
    )
```

### ISSUE-3121: Leaky Abstraction - Position Size Checker

**File:** `/checkers/position_size.py`
**Method:** `check_limit` (lines 22-89)
**Principle:** Abstraction
**Severity:** MEDIUM

**Violation:**
Implementation details leak through the abstraction:

- Direct datetime manipulation for violation IDs
- Hardcoded severity determination
- Portfolio value extraction logic embedded

**Impact:**

- Clients depend on implementation details
- Cannot change internal logic without affecting clients
- Testing requires knowledge of internals

**Refactoring Recommendation:**

```python
class PositionSizeChecker:
    def __init__(self,
                 id_generator: IViolationIdGenerator,
                 severity_calculator: ISeverityCalculator):
        self.id_generator = id_generator
        self.severity_calculator = severity_calculator

    def check_limit(self, limit: LimitDefinition,
                   current_value: float,
                   context: Dict[str, Any]) -> LimitCheckResult:
        # Use injected services
        violation_id = self.id_generator.generate(limit.limit_id)
        severity = self.severity_calculator.calculate(current_value, limit)
```

### ISSUE-3122: Missing Factory Pattern for Checker Creation

**File:** `/checkers/__init__.py`
**Principle:** Factory Pattern, DIP
**Severity:** MEDIUM

**Violation:**
Direct exports of concrete classes without factory abstraction.

**Impact:**

- Clients directly instantiate concrete classes
- No central point for checker creation logic
- Difficult to add new checker types

**Refactoring Recommendation:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Type

class CheckerFactory:
    """Factory for creating limit checkers"""

    _checkers: Dict[LimitType, Type[LimitChecker]] = {}

    @classmethod
    def register(cls, limit_type: LimitType, checker_class: Type[LimitChecker]):
        cls._checkers[limit_type] = checker_class

    @classmethod
    def create(cls, limit_type: LimitType, config: Dict[str, Any]) -> LimitChecker:
        checker_class = cls._checkers.get(limit_type)
        if not checker_class:
            raise ValueError(f"No checker registered for {limit_type}")
        return checker_class(**config)

# Register checkers
CheckerFactory.register(LimitType.DRAWDOWN, DrawdownChecker)
CheckerFactory.register(LimitType.POSITION_SIZE, PositionSizeChecker)
```

### ISSUE-3123: Event Coupling Anti-Pattern

**File:** `/events.py`
**Lines:** 43-93
**Principle:** Coupling/Cohesion
**Severity:** HIGH

**Violation:**
Event classes directly coupled to specific implementations:

- `ViolationEvent` knows about `LimitViolation`
- `CheckEvent` knows about `LimitCheckResult`
- Events contain business logic in `__post_init__`

**Impact:**

- Cannot reuse events in different contexts
- Changes to domain objects affect event system
- Testing requires full domain object setup

**Refactoring Recommendation:**

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class GenericEvent(Generic[T]):
    """Generic event container"""
    event_type: str
    payload: T
    metadata: Dict[str, Any]

    @classmethod
    def create(cls, event_type: str, payload: T, **metadata):
        return cls(event_type=event_type, payload=payload, metadata=metadata)

# Use generic events
violation_event = GenericEvent.create(
    "violation_detected",
    violation_data,
    severity=severity,
    timestamp=datetime.now()
)
```

### ISSUE-3124: Missing Async Context Manager Pattern

**File:** `/events.py`
**Class:** `EventManager`
**Principle:** Resource Management
**Severity:** MEDIUM

**Violation:**
No async context manager implementation for proper resource cleanup.

**Impact:**

- Manual start/stop management prone to errors
- Resources may leak if stop() not called
- No automatic cleanup on exceptions

**Refactoring Recommendation:**

```python
class EventManager:
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        if exc_type:
            logger.error(f"EventManager exiting with error: {exc_val}")
        return False

# Usage
async with EventManager() as event_mgr:
    # Use event manager
    await event_mgr.emit(event)
```

### ISSUE-3125: Violation of Composition Over Inheritance

**File:** `/checkers/simple_threshold.py`
**Class:** `SimpleThresholdChecker`
**Principle:** Composition over Inheritance
**Severity:** MEDIUM

**Violation:**
Inherits from `LimitChecker` when composition would be more flexible.

**Impact:**

- Locked into inheritance hierarchy
- Cannot dynamically change behavior
- Difficult to test in isolation

**Refactoring Recommendation:**

```python
class SimpleThresholdChecker:
    """Uses composition instead of inheritance"""

    def __init__(self,
                 validator: IValidator,
                 comparator: IComparator,
                 message_formatter: IMessageFormatter):
        self.validator = validator
        self.comparator = comparator
        self.message_formatter = message_formatter

    async def check(self, request: CheckRequest) -> CheckResult:
        # Delegate to composed objects
        is_valid = self.validator.validate(request)
        comparison = self.comparator.compare(request.value, request.threshold)
        message = self.message_formatter.format(comparison)
        return CheckResult(is_valid, message)
```

### ISSUE-3126: Hard-coded Magic Numbers

**File:** `/checkers/drawdown.py`
**Multiple locations
**Principle:** Configuration Management
**Severity:** LOW

**Violation:**
Magic numbers throughout the code:

- Line 106: `1e-10` for floating point comparison
- Line 313: `0.05` (5%) for potential move calculation
- Line 401: `0.05` (5%) drawdown threshold

**Impact:**

- Difficult to understand business rules
- Cannot configure without code changes
- Risk of inconsistency

**Refactoring Recommendation:**

```python
class DrawdownConstants:
    """Centralized constants"""
    FLOAT_EPSILON = 1e-10
    DEFAULT_ADVERSE_MOVE = 0.05
    SIGNIFICANT_DRAWDOWN_THRESHOLD = 0.05
    DEFAULT_PORTFOLIO_VALUE = 100000
    HISTORY_RETENTION_DAYS = 90
```

### ISSUE-3127: Missing Strategy Pattern for Event Processing

**File:** `/events.py`
**Method:** `_process_event` (lines 285-313)
**Principle:** Open/Closed, Strategy Pattern
**Severity:** MEDIUM

**Violation:**
Event processing logic hard-coded in single method with conditionals.

**Impact:**

- Cannot add new processing strategies without modification
- Difficult to test different processing paths
- No way to configure processing behavior

**Refactoring Recommendation:**

```python
from abc import ABC, abstractmethod

class EventProcessingStrategy(ABC):
    @abstractmethod
    async def process(self, event: Event) -> None:
        pass

class AsyncEventProcessingStrategy(EventProcessingStrategy):
    async def process(self, event: Event) -> None:
        # Async processing logic
        pass

class SyncEventProcessingStrategy(EventProcessingStrategy):
    async def process(self, event: Event) -> None:
        # Sync processing logic
        pass

class EventManager:
    def __init__(self, processing_strategy: EventProcessingStrategy):
        self.processing_strategy = processing_strategy

    async def _process_event(self, event: Event):
        await self.processing_strategy.process(event)
```

### ISSUE-3128: Temporal Coupling in DrawdownChecker

**File:** `/checkers/drawdown.py`
**Method:** `check` (lines 114-234)
**Principle:** Temporal Coupling
**Severity:** MEDIUM

**Violation:**
Methods must be called in specific order:

1. Calculate drawdowns
2. Check daily
3. Check weekly
4. Check total
5. Check recovery

**Impact:**

- Fragile code that breaks if order changes
- Hidden dependencies between steps
- Difficult to parallelize checks

**Refactoring Recommendation:**

```python
class DrawdownCheckPipeline:
    """Explicit pipeline for drawdown checks"""

    def __init__(self):
        self.steps = [
            DrawdownCalculationStep(),
            DailyCheckStep(),
            WeeklyCheckStep(),
            TotalCheckStep(),
            RecoveryCheckStep()
        ]

    async def execute(self, context: CheckContext) -> LimitCheckResult:
        results = []
        for step in self.steps:
            result = await step.execute(context)
            results.append(result)
            if not result.continue_pipeline:
                break
        return self.aggregate_results(results)
```

### ISSUE-3129: Missing Observer Pattern Implementation

**File:** `/events.py`
**Class:** `EventManager`
**Principle:** Observer Pattern
**Severity:** LOW

**Violation:**
Manual subscription management instead of proper Observer pattern.

**Impact:**

- Boilerplate code for subscription management
- No standard interface for observers
- Difficult to manage observer lifecycle

**Refactoring Recommendation:**

```python
from abc import ABC, abstractmethod

class IEventObserver(ABC):
    @abstractmethod
    async def on_event(self, event: Event) -> None:
        pass

class EventSubject:
    def __init__(self):
        self._observers: List[IEventObserver] = []

    def attach(self, observer: IEventObserver) -> None:
        self._observers.append(observer)

    def detach(self, observer: IEventObserver) -> None:
        self._observers.remove(observer)

    async def notify(self, event: Event) -> None:
        for observer in self._observers:
            await observer.on_event(event)
```

## Architectural Patterns Analysis

### Current Design Patterns Usage

1. **Registry Pattern**: Partially implemented for checker registration
2. **Strategy Pattern**: Missing, would benefit multiple areas
3. **Factory Pattern**: Not implemented, needed for checker creation
4. **Observer Pattern**: Partial implementation in EventManager
5. **Command Pattern**: Missing for event actions

### Coupling and Cohesion Metrics

- **High Coupling**: DrawdownChecker coupled to 8+ different modules
- **Low Cohesion**: EventManager handles 7+ unrelated responsibilities
- **Circular Dependency Risk**: Events depend on domain objects which may depend on events

### Module Boundaries

- **Violated**: Checkers directly access utils and monitoring modules
- **Missing**: No clear interface layer between checkers and core system
- **Leaky**: Implementation details exposed through public interfaces

## Overall Architectural Integrity Assessment

### Strengths

1. Consistent checker pattern across implementations
2. Async support throughout
3. Comprehensive event system (though over-engineered)

### Critical Weaknesses

1. **Severe SRP violations** in core components
2. **Missing dependency injection** framework
3. **No clear architectural boundaries**
4. **Over-reliance on inheritance** instead of composition
5. **Lack of abstraction layers**

### System Evolution Impact

The current architecture will make the system difficult to:

1. **Test**: Too many dependencies and responsibilities per class
2. **Extend**: Hard-coded logic and missing extension points
3. **Maintain**: Violations of DRY and high coupling
4. **Scale**: God objects will become bottlenecks

## Recommended Refactoring Priority

### Phase 1: Critical (Immediate)

1. **ISSUE-3113**: Refactor DrawdownChecker to separate concerns
2. **ISSUE-3119**: Break down EventManager god object
3. **ISSUE-3117**: Introduce dependency injection

### Phase 2: High (Next Sprint)

1. **ISSUE-3115**: Standardize checker interfaces
2. **ISSUE-3122**: Implement checker factory pattern
3. **ISSUE-3123**: Decouple events from domain objects

### Phase 3: Medium (Next Quarter)

1. **ISSUE-3120**: Eliminate code duplication
2. **ISSUE-3127**: Add strategy pattern for processing
3. **ISSUE-3124**: Implement context managers

### Phase 4: Low (Technical Debt Backlog)

1. **ISSUE-3126**: Extract magic numbers
2. **ISSUE-3129**: Formalize observer pattern
3. General code cleanup and documentation

## Long-term Architecture Recommendations

### 1. Introduce Dependency Injection Container

```python
# Example using dependency-injector
from dependency_injector import containers, providers

class CheckerContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    metrics_recorder = providers.Singleton(
        MetricsRecorder,
        config=config.metrics
    )

    drawdown_calculator = providers.Singleton(
        DrawdownCalculator
    )

    drawdown_checker = providers.Factory(
        DrawdownChecker,
        calculator=drawdown_calculator,
        metrics=metrics_recorder
    )
```

### 2. Implement Clear Layer Architecture

```
Presentation Layer (API)
    ↓
Application Layer (Use Cases)
    ↓
Domain Layer (Business Logic)
    ↓
Infrastructure Layer (External Systems)
```

### 3. Establish Module Boundaries

- Create explicit interfaces between modules
- Use dependency injection for cross-module communication
- Implement facade pattern for complex subsystems

### 4. Adopt Hexagonal Architecture

- Core domain logic independent of external systems
- Ports and adapters for external integration
- Testable business logic in isolation

## Conclusion

The risk_management module shows significant architectural debt with severe SOLID principle violations. The most critical issues are the god objects (DrawdownChecker and EventManager) and the lack of proper abstraction layers. Immediate refactoring is recommended to prevent further technical debt accumulation and ensure system maintainability.

The current implementation, while functional, will become increasingly difficult to maintain and extend. The recommended refactoring should be prioritized based on business impact and risk, with the most critical issues (god objects and dependency management) addressed first.
