 Memory Update: AI Trading System Project Context

  You are working on an AI Trading System - a sophisticated algorithmic trading platform designed to handle real money transactions.
  This is an enterprise-grade financial system with zero tolerance for errors.

  🎯 CRITICAL PROJECT DIRECTIVE

  Foundation MUST be airtight with 0 issues before ANY new code. This system will handle real money - there is no room for error. You
  must validate ALL existing code follows coding principles and ensure the development foundation is the BEST possible foundation to
  build upon.

  📍 Current Status

- Foundation Grade: B+ (Architecturally sound but has stability issues)
- Production Ready: ❌ NO - Critical fixes required
- Timeline: 4-6 weeks to production readiness

  🔴 Critical Quality Gates (ALL Must Pass)

- Test Pass Rate: Currently ~50% (816 failures) → Target: 100%
- MyPy Type Errors: Currently 264 errors → Target: 0
- Test Coverage: Currently 12% → Target: 80%+
- SOLID Compliance: Portfolio entity violates SRP → Must fix

  🟢 Strong Foundation Areas

- Architecture: Excellent DDD compliance with clean layer separation
- Security: 95+ score with MFA, rate limiting, HTTPS enforcement
- Performance: 94,255 orders/second (94x requirement)
- Financial Precision: All calculations use Decimal types correctly

  📋 TODO Priority Order (MUST Complete in Sequence)

  1. Fix ALL existing tests (816 failures → 0 failures)
  2. Fix ALL MyPy errors (264 errors → 0 errors)
  3. Refactor Portfolio entity (fix SOLID violations)
  4. Achieve 80% test coverage
  5. Only then consider new features

  🛠 Key Commands

# Test all

  PYTHONPATH=/Users/zachwade/StockMonitoring python -m pytest tests/ --tb=short

# MyPy check

  python -m mypy src --ignore-missing-imports --show-error-codes

# Coverage

  python -m pytest --cov=src --cov-report=term-missing

  🏗 Architecture Pattern

- Domain: Pure business logic (Order, Position, Portfolio, Money, Price, Quantity)
- Application: Use cases and coordination
- Infrastructure: External systems (brokers, database, security)

  👥 Subagent Strategy

  Use specialized subagents in parallel for efficient work:

- error-detective-analyzer: Analyze failures and type errors
- architecture-integrity-reviewer: Validate DDD/SOLID compliance
- code-quality-auditor: Find violations and anti-patterns
- code-implementation-expert: Systematic fixes

  ⚠️ Critical Requirements

- Zero Tolerance: This handles real money - every line must be perfect
- Type Safety: 100% MyPy compliance required
- Financial Precision: All money calculations use Decimal types
- Test Coverage: Minimum 80% coverage for critical paths
- Security First: Enterprise-grade protection maintained

  🏁 Success Definition

  Foundation is ready when ALL quality gates pass:

- 100% tests passing
- 0 MyPy errors
- 80%+ coverage
- SOLID compliance validated
- Security and performance maintained

  Remember: Fix existing foundation issues BEFORE adding any new features. The architectural foundation is excellent - focus on
  stability and type safety.
