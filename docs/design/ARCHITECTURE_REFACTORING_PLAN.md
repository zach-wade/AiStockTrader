# Architecture Refactoring Plan: Extract Business Logic from Infrastructure Layer

## Executive Summary

The architecture test `test_no_business_logic_in_infrastructure` is failing with 106 violations across infrastructure modules. This document provides a comprehensive plan to extract all business logic from the infrastructure layer to the domain layer, ensuring clean architecture principles and achieving 0 architecture violations.

## Architectural Impact Assessment

**Rating: HIGH**

- **Justification**: The violations span critical system components including security validation, monitoring, resilience, and observability. These violations represent significant architectural debt where business rules are mixed with infrastructure concerns, violating the Dependency Inversion Principle and making the system harder to maintain and test.

## Pattern Compliance Checklist

- ❌ Single Responsibility Principle - Infrastructure classes handle both technical and business concerns
- ❌ Dependency Inversion Principle - Infrastructure contains business rules instead of depending on abstractions
- ❌ Clean Architecture - Business logic exists in the outermost layer
- ❌ Domain-Driven Design - Business rules are scattered outside the domain layer

## Violations Analysis by Module

### 1. Infrastructure/Security Module (30 violations)

#### Current Issues

- **validation.py**: Contains trading-specific validation rules (symbols, prices, quantities)
- **hardening.py**: Has complex request verification logic with business rules
- **secrets.py**: Complex conditionals for secret retrieval logic
- **input_sanitizer.py**: Business rules for filename validation

#### Refactoring Plan

**Step 1: Create Domain Services**

```python
# src/domain/services/security_policy_service.py
class SecurityPolicyService:
    """Domain service for security policy decisions."""

    def determine_validation_rules(self, context: str) -> ValidationRules:
        """Determine validation rules based on business context."""
        pass

    def evaluate_request_risk(self, request_metadata: Dict) -> RiskLevel:
        """Evaluate request risk based on business rules."""
        pass

    def determine_sanitization_level(self, data_type: str) -> SanitizationLevel:
        """Determine sanitization requirements based on data type."""
        pass
```

```python
# src/domain/services/secrets_policy_service.py
class SecretsManagementPolicy:
    """Domain service for secrets management policies."""

    def determine_secret_access_level(self, secret_type: str, context: Dict) -> AccessLevel:
        """Determine access requirements for secrets."""
        pass

    def evaluate_secret_rotation_need(self, secret_metadata: Dict) -> bool:
        """Evaluate if secret needs rotation based on business rules."""
        pass
```

**Step 2: Extract Business Logic**

- Move all validation methods from `SecurityValidator` to `TradingValidationService` (already exists)
- Move request verification logic from `hardening.py` to `SecurityPolicyService`
- Move secret retrieval logic from `secrets.py` to `SecretsManagementPolicy`

### 2. Infrastructure/Observability Module (4 violations)

#### Current Issues

- **business_intelligence.py**: Contains portfolio performance calculations, strategy analysis
- **exporters.py**: Complex conversion logic with business rules

#### Refactoring Plan

**Step 1: Move to Existing Domain Services**

- The module already uses `PortfolioAnalyticsService` and `StrategyAnalyticsService`
- Move remaining calculation logic to these services
- Infrastructure should only handle data collection and forwarding

**Step 2: Create Observability Adapter**

```python
# src/infrastructure/observability/adapters.py
class ObservabilityAdapter:
    """Adapter for observability without business logic."""

    def __init__(self, analytics_service: PortfolioAnalyticsService):
        self.analytics_service = analytics_service

    def collect_metrics(self, raw_data: Dict) -> None:
        """Collect and forward metrics to domain service."""
        # No business logic here, just data forwarding
        processed = self.analytics_service.analyze(raw_data)
        self._forward_to_collector(processed)
```

### 3. Infrastructure/Resilience Module (10 violations)

#### Current Issues

- **health.py**: Business logic for determining overall system health
- **retry.py**: Complex delay calculation with business rules
- **circuit_breaker.py**: Business logic for failure thresholds
- **error_handling.py**: Complex error categorization logic

#### Refactoring Plan

**Step 1: Create Domain Services**

```python
# src/domain/services/system_health_policy.py
class SystemHealthPolicy:
    """Domain service for system health policies."""

    def determine_health_status(self, metrics: Dict) -> HealthStatus:
        """Determine system health based on business rules."""
        pass

    def calculate_health_score(self, components: List[ComponentHealth]) -> float:
        """Calculate overall health score."""
        pass
```

```python
# src/domain/services/resilience_policy_service.py
class ResiliencePolicyService:
    """Domain service for resilience policies."""

    def calculate_retry_delay(self, attempt: int, error_type: str) -> float:
        """Calculate retry delay based on business rules."""
        pass

    def determine_circuit_breaker_threshold(self, service: str) -> CircuitBreakerPolicy:
        """Determine circuit breaker thresholds."""
        pass

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize errors based on business rules."""
        pass
```

### 4. Infrastructure/Monitoring Module (18 violations)

#### Current Issues

- **metrics.py**: Trading metric calculations
- **health.py**: Trading status determination
- **telemetry.py**: Trading attribute additions
- **integration.py**: Resilience recommendations generation
- **performance.py**: Performance analysis and recommendations

#### Refactoring Plan

**Step 1: Create Domain Services**

```python
# src/domain/services/monitoring_policy_service.py
class MonitoringPolicyService:
    """Domain service for monitoring policies."""

    def determine_metric_thresholds(self, metric_type: str) -> MetricThresholds:
        """Determine thresholds for metrics."""
        pass

    def evaluate_performance(self, metrics: Dict) -> PerformanceAssessment:
        """Evaluate system performance."""
        pass

    def generate_recommendations(self, analysis: Dict) -> List[Recommendation]:
        """Generate performance recommendations."""
        pass
```

```python
# src/domain/services/trading_status_service.py
class TradingStatusService:
    """Domain service for trading status determination."""

    def determine_trading_status(self, market_data: Dict) -> TradingStatus:
        """Determine current trading status."""
        pass

    def evaluate_trading_health(self, metrics: Dict) -> TradingHealth:
        """Evaluate trading system health."""
        pass
```

## Implementation Strategy

### Phase 1: Create Domain Services (Week 1)

1. Create all new domain service files
2. Define interfaces and method signatures
3. Write unit tests for domain services

### Phase 2: Extract Business Logic (Week 2)

1. Move validation logic to domain services
2. Update infrastructure to use domain services
3. Ensure all tests pass

### Phase 3: Refactor Infrastructure (Week 3)

1. Remove all business logic from infrastructure
2. Convert infrastructure components to pure adapters
3. Update dependency injection

### Phase 4: Validation and Testing (Week 4)

1. Run architecture tests
2. Ensure 0 violations
3. Performance testing
4. Integration testing

## Specific Code Refactoring Examples

### Example 1: Extracting Validation Logic

**Before (Infrastructure Layer):**

```python
# src/infrastructure/security/validation.py
class SecurityValidator:
    @classmethod
    def validate_trading_symbol(cls, symbol: str) -> bool:
        if not symbol or len(symbol) > 5:
            return False
        if not symbol.isalnum():
            return False
        # Business rule: certain symbols are restricted
        if symbol in ['TEST', 'DEMO']:
            return False
        return True
```

**After (Domain Layer):**

```python
# src/domain/services/trading_validation_service.py
class TradingValidationService:
    RESTRICTED_SYMBOLS = ['TEST', 'DEMO']

    @classmethod
    def validate_trading_symbol(cls, symbol: str) -> bool:
        if not symbol or len(symbol) > 5:
            return False
        if not symbol.isalnum():
            return False
        if symbol in cls.RESTRICTED_SYMBOLS:
            return False
        return True

# src/infrastructure/security/validation.py
class SecurityValidator:
    @classmethod
    def validate_trading_symbol(cls, symbol: str) -> bool:
        # Delegate to domain service
        return TradingValidationService.validate_trading_symbol(symbol)
```

### Example 2: Extracting Health Check Logic

**Before (Infrastructure Layer):**

```python
# src/infrastructure/resilience/health.py
def get_overall_status(checks: Dict) -> str:
    if any(check['status'] == 'critical' for check in checks.values()):
        return 'critical'
    if any(check['status'] == 'degraded' for check in checks.values()):
        return 'degraded'
    return 'healthy'
```

**After (Domain Layer):**

```python
# src/domain/services/system_health_policy.py
class SystemHealthPolicy:
    @staticmethod
    def determine_overall_status(component_statuses: Dict[str, str]) -> str:
        if any(status == 'critical' for status in component_statuses.values()):
            return 'critical'
        if any(status == 'degraded' for status in component_statuses.values()):
            return 'degraded'
        return 'healthy'

# src/infrastructure/resilience/health.py
def get_overall_status(checks: Dict, health_policy: SystemHealthPolicy) -> str:
    statuses = {name: check['status'] for name, check in checks.items()}
    return health_policy.determine_overall_status(statuses)
```

## Long-term Implications

### Positive Impacts

1. **Improved Testability**: Business logic in domain layer can be tested independently
2. **Better Maintainability**: Clear separation of concerns
3. **Enhanced Flexibility**: Business rules can change without affecting infrastructure
4. **Easier Onboarding**: Clear architecture makes codebase easier to understand
5. **Reduced Technical Debt**: Clean architecture prevents future violations

### Decisions That Enhance Future Flexibility

1. Use of domain services for all business logic
2. Infrastructure as pure adapters
3. Dependency injection for loose coupling
4. Clear interfaces between layers

## Success Metrics

1. **Architecture Test**: 0 violations in `test_no_business_logic_in_infrastructure`
2. **Code Coverage**: Maintain or improve current coverage
3. **Performance**: No degradation in system performance
4. **Maintainability Index**: Improved code maintainability scores

## Risk Mitigation

1. **Gradual Migration**: Implement changes incrementally
2. **Comprehensive Testing**: Write tests before refactoring
3. **Feature Flags**: Use feature flags for gradual rollout
4. **Rollback Plan**: Maintain ability to revert changes
5. **Performance Monitoring**: Monitor for any performance impacts

## Conclusion

This refactoring plan addresses all 106 architecture violations by systematically extracting business logic from the infrastructure layer to the domain layer. The approach maintains functionality while improving architectural integrity, testability, and maintainability. Following this plan will result in a clean architecture with 0 violations and better separation of concerns.
