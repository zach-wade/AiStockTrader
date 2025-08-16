# Wave Plan
## Wave 0 — MTP Blockers
| # | File | Severity | Summary |
|---|---|---|---|
| 1 |  | Critical | Severity: CRITICAL** |
| 2 |  | Critical | 1. SRP Violations - God Functions (SEVERITY: CRITICAL) |
| 3 |  | Critical | 6. Missing Production Configurations (SEVERITY: CRITICAL) |
| 4 |  | Critical | Severity: CRITICAL** - This creates a God Object anti-pattern |
| 5 |  | Critical | Severity: CRITICAL** - Forces fat dependencies throughout the system |
| 6 |  | High | Severity: HIGH** |
| 7 |  | High | 2. Inline Class Definition (SEVERITY: HIGH) |
| 8 |  | High | 3. Service Locator Anti-Pattern (SEVERITY: HIGH) |
| 9 |  | High | 5. Mixed Presentation and Business Logic (SEVERITY: HIGH) |
| 10 |  | High | 7. DIP Violations - Direct Dependencies (SEVERITY: HIGH) |
| 11 |  | High | Severity: HIGH** - Conflates data modeling with business logic |
| 12 |  | High | Severity: HIGH** - Violates separation of concerns |
| 13 |  | High | Severity: HIGH** - Requires code changes for configuration extensions |
| 14 |  | High | Critical Issues (Severity: HIGH) |
| 15 |  | Medium | Severity: MEDIUM** |
| 16 |  | Medium | 4. Factory Pattern Inconsistency (SEVERITY: MEDIUM) |
| 17 |  | Medium | 8. Dangerous Use of globals() (SEVERITY: MEDIUM) |
| 18 |  | Medium | Severity: MEDIUM** - Adds maintenance burden to core model |
| 19 |  | Low | # Architectural Review: Trading Configuration Models |
| 20 |  | Low | Module:** `/ai_trader/src/main/config/validation_models/trading.py` |
| 21 |  | Low | Review Date:** 2025-08-13 |
| 22 |  | Low | Lines of Code:** 206 |
| 23 |  | Low | Review Focus:** SOLID Principles & Architectural Integrity |
| 24 |  | Low | Architectural Impact Assessment |
| 25 |  | Low | Rating: MEDIUM** |
| 26 |  | Low | Justification:** The module exhibits multiple SOLID violations, particularly around Single Responsibility and Interface Segregation principles. While the configuration models provide validation, they  |
| 27 |  | Low | Pattern Compliance Checklist |
| 28 |  | Low | SOLID Principles |
| 29 |  | Low | ❌ **Single Responsibility Principle (SRP)** - Multiple violations found |
| 30 |  | Low | ✅ **Open/Closed Principle (OCP)** - Generally compliant through composition |

## Wave 1 — Stability & Observability (next 50)
| # | File | Severity | Summary |
|---|---|---|---|
| 31 |  | Low | ✅ **Liskov Substitution Principle (LSP)** - No violations detected |
| 32 |  | Low | ❌ **Interface Segregation Principle (ISP)** - Fat interfaces with mixed concerns |
| 33 |  | Low | ❌ **Dependency Inversion Principle (DIP)** - Direct dependency on concrete logger |
| 34 |  | Low | Architectural Patterns |
| 35 |  | Low | ❌ **Consistency with established patterns** - Mixing data models with business logic |
| 36 |  | Low | ❌ **Proper dependency management** - Side effects in validators |
| 37 |  | Low | ❌ **Appropriate abstraction levels** - Business rules embedded in data models |
| 38 |  | Low | Violations Found |
| 39 |  | Low | 1. Single Responsibility Principle Violations |
| 40 |  | Low | **CRITICAL: Mixed Responsibilities in Configuration Models** |
| 41 |  | Low | Location:** Lines 75-206 (All config classes) |
| 42 |  | Low | Severity:** HIGH |
| 43 |  | Low | Problem:** Configuration models are responsible for: |
| 44 |  | Low | Data structure definition |
| 45 |  | Low | Validation logic |
| 46 |  | Low | Business rule enforcement |
| 47 |  | Low | Logging/warning generation |
| 48 |  | Low | Cross-field consistency checks |
| 49 |  | Low | Example:** `PositionSizingConfig` (lines 75-86) |
| 50 |  | Low | Data structure |
| 51 |  | Low | Business rule validation |
| 52 |  | Low | **MEDIUM: Side Effects in Validators** |
| 53 |  | Low | Location:** Lines 98-100, 123-126, 138-141, 155-157, 171-173 |
| 54 |  | Low | Severity:** MEDIUM |
| 55 |  | Low | Problem:** Validators produce side effects (logging warnings) which violates SRP and makes testing difficult. |
| 56 |  | Low | ```python |
| 57 |  | Low | 2. Interface Segregation Principle Violations |
| 58 |  | Low | **MEDIUM: Fat Interfaces with Optional Dependencies** |
| 59 |  | Low | Location:** Lines 104-111 (`TradingConfig`) |
| 60 |  | Low | Problem:** Large configuration classes force clients to depend on configurations they don't use. |
| 61 |  | Low | 3. Dependency Inversion Principle Violations |
| 62 |  | Low | **LOW: Direct Logger Dependency** |
| 63 |  | Low | Location:** Line 12 |
| 64 |  | Low | Severity:** LOW |
| 65 |  | Low | Problem:** Direct instantiation of logger creates tight coupling to logging implementation. |
| 66 |  | Low | 4. Abstraction Level Inconsistencies |
| 67 |  | Low | **MEDIUM: Business Logic in Data Models** |
| 68 |  | Low | Location:** Lines 82-86, 151-157, 167-173, 192-198 |
| 69 |  | Low | Problem:** Data models contain business logic that should be in domain services. |
| 70 |  | Low | Example:** Stop loss percentage validation (lines 151-157) |
| 71 |  | Low | 5. Coupling Issues |
| 72 |  | Low | **HIGH: Duplicate Position Sizing Configurations** |
| 73 |  | Low | Location:** Lines 75-86 and 114-127 |
| 74 |  | Low | Problem:** Two separate position sizing configurations (`PositionSizingConfig` and `RiskPositionSizingConfig`) with overlapping responsibilities create confusion and potential inconsistencies. |
| 75 |  | Low | Trading module |
| 76 |  | Low | Risk module   |
| 77 |  | Low | 6. Missing Abstractions |
| 78 |  | Low | **MEDIUM: No Validator Strategy Pattern** |
| 79 |  | Low | Location:** Throughout file |
| 80 |  | Low | Problem:** Validation logic is scattered across field validators and model validators without a coherent strategy. |

## Wave 2+ — Long Tail
Remaining issues grouped by top-level module:
- **unknown**: 25224 issues
- **1.**: 1 issues
- ****: 7 issues
- **market_data_repository.py**: 1 issues
- **var_calculator.py**: 1 issues
- **blackscholes_calculator.py**: 1 issues
- **stress_test_calculator.py**: 1 issues
- **company_repository.py**: 1 issues
- **sql_security.py**: 1 issues
- **Path**: 4 issues
- **paths**: 1 issues
- **str**: 3 issues
