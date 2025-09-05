"""
Consolidated Policy Engine

This service consolidates all policy enforcement logic from scattered policy
services into a single, cohesive engine following the Single Responsibility
Principle for policy concerns.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class PolicyLevel(Enum):
    """Policy enforcement levels."""

    STRICT = "strict"  # No violations allowed
    NORMAL = "normal"  # Standard enforcement
    RELAXED = "relaxed"  # Warnings only
    DISABLED = "disabled"  # Policy not enforced


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""

    passed: bool
    policy_name: str
    level: PolicyLevel
    message: str
    violations: list[str]
    metadata: dict[str, Any]

    @classmethod
    def success(cls, policy_name: str, level: PolicyLevel = PolicyLevel.NORMAL) -> "PolicyResult":
        """Create a successful policy result."""
        return cls(
            passed=True,
            policy_name=policy_name,
            level=level,
            message="Policy check passed",
            violations=[],
            metadata={},
        )

    @classmethod
    def failure(
        cls,
        policy_name: str,
        violations: list[str],
        level: PolicyLevel = PolicyLevel.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> "PolicyResult":
        """Create a failed policy result."""
        return cls(
            passed=False,
            policy_name=policy_name,
            level=level,
            message=f"Policy violated: {len(violations)} violation(s)",
            violations=violations,
            metadata=metadata or {},
        )


class PolicyEngine:
    """
    Unified policy engine for all system policies.

    This engine consolidates functionality from:
    - monitoring_policy_service.py
    - resilience_policy_service.py
    - secrets_policy_service.py
    - security_policy_service.py
    - system_health_policy.py
    - threshold_policy_service.py

    Providing a single point of policy enforcement across the system.
    """

    def __init__(self) -> None:
        """Initialize the policy engine with default configurations."""
        self.policies = {
            "security": PolicyLevel.STRICT,
            "monitoring": PolicyLevel.NORMAL,
            "resilience": PolicyLevel.NORMAL,
            "threshold": PolicyLevel.NORMAL,
            "secrets": PolicyLevel.STRICT,
            "system_health": PolicyLevel.NORMAL,
        }

        # Policy configurations
        self.config = {
            # Security policies
            "max_login_attempts": 5,
            "session_timeout_minutes": 30,
            "password_min_length": 12,
            "require_mfa": True,
            # Monitoring policies
            "max_log_size_mb": 100,
            "log_retention_days": 90,
            "alert_threshold_error_rate": 0.05,
            "metrics_collection_interval": 60,
            # Resilience policies
            "max_retry_attempts": 3,
            "circuit_breaker_threshold": 5,
            "timeout_seconds": 30,
            "fallback_enabled": True,
            # Threshold policies
            "max_order_size": 10000,
            "max_portfolio_positions": 50,
            "max_leverage": 3.0,
            "min_cash_balance": 1000,
            "max_daily_trades": 100,
            "max_position_concentration": 0.25,
            # System health policies
            "max_cpu_usage": 0.8,
            "max_memory_usage": 0.9,
            "max_disk_usage": 0.85,
            "min_free_connections": 10,
            "max_response_time_ms": 1000,
        }

    # ========== Security Policies ==========

    def evaluate_security_policy(self, context: dict[str, Any]) -> PolicyResult:
        """Evaluate security policies."""
        violations = []

        if self.policies["security"] == PolicyLevel.DISABLED:
            return PolicyResult.success("security", PolicyLevel.DISABLED)

        # Check authentication
        if context.get("authenticated") is False:
            violations.append("User not authenticated")

        # Check MFA if required
        if self.config["require_mfa"] and not context.get("mfa_verified"):
            violations.append("MFA verification required")

        # Check session timeout
        if "session_start" in context:
            session_duration = (datetime.now(UTC) - context["session_start"]).seconds / 60
            if session_duration > self.config["session_timeout_minutes"]:
                violations.append(f"Session expired (duration: {session_duration:.1f} minutes)")

        # Check login attempts
        if context.get("login_attempts", 0) > self.config["max_login_attempts"]:
            violations.append(f"Too many login attempts ({context.get('login_attempts')})")

        # Check password strength
        if "password" in context:
            password = context["password"]
            if len(password) < self.config["password_min_length"]:
                violations.append(f"Password too short (min: {self.config['password_min_length']})")

        # Check for secure connection
        if not context.get("https", True):
            if self.policies["security"] == PolicyLevel.STRICT:
                violations.append("Insecure connection (HTTPS required)")

        if violations:
            return PolicyResult.failure("security", violations, self.policies["security"])
        return PolicyResult.success("security", self.policies["security"])

    # ========== Monitoring Policies ==========

    def evaluate_monitoring_policy(self, metrics: dict[str, Any]) -> PolicyResult:
        """Evaluate monitoring and observability policies."""
        violations = []

        if self.policies["monitoring"] == PolicyLevel.DISABLED:
            return PolicyResult.success("monitoring", PolicyLevel.DISABLED)

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.config["alert_threshold_error_rate"]:
            violations.append(f"High error rate: {error_rate:.2%}")

        # Check log size
        log_size_mb = metrics.get("log_size_mb", 0)
        if log_size_mb > self.config["max_log_size_mb"]:
            violations.append(f"Log size exceeded: {log_size_mb}MB")

        # Check metrics collection
        last_collection = metrics.get("last_metrics_collection")
        if last_collection:
            time_since = (datetime.now(UTC) - last_collection).seconds
            if time_since > self.config["metrics_collection_interval"] * 2:
                violations.append(f"Metrics collection delayed: {time_since}s")

        # Check alert fatigue
        alerts_per_hour = metrics.get("alerts_per_hour", 0)
        if alerts_per_hour > 10:
            violations.append(f"Alert fatigue: {alerts_per_hour} alerts/hour")

        if violations:
            return PolicyResult.failure("monitoring", violations, self.policies["monitoring"])
        return PolicyResult.success("monitoring", self.policies["monitoring"])

    # ========== Resilience Policies ==========

    def evaluate_resilience_policy(self, operation: dict[str, Any]) -> PolicyResult:
        """Evaluate system resilience policies."""
        violations = []

        if self.policies["resilience"] == PolicyLevel.DISABLED:
            return PolicyResult.success("resilience", PolicyLevel.DISABLED)

        # Check retry attempts
        retry_count = operation.get("retry_count", 0)
        if retry_count > self.config["max_retry_attempts"]:
            violations.append(f"Too many retries: {retry_count}")

        # Check circuit breaker
        failure_count = operation.get("consecutive_failures", 0)
        if failure_count >= self.config["circuit_breaker_threshold"]:
            violations.append(f"Circuit breaker threshold reached: {failure_count} failures")

        # Check timeout
        duration = operation.get("duration_seconds", 0)
        if duration > self.config["timeout_seconds"]:
            violations.append(f"Operation timeout: {duration}s")

        # Check fallback availability
        if not self.config["fallback_enabled"] and operation.get("requires_fallback"):
            violations.append("Fallback required but not enabled")

        # Check dependency health
        unhealthy_deps = operation.get("unhealthy_dependencies", [])
        if unhealthy_deps:
            violations.append(f"Unhealthy dependencies: {', '.join(unhealthy_deps)}")

        if violations:
            return PolicyResult.failure("resilience", violations, self.policies["resilience"])
        return PolicyResult.success("resilience", self.policies["resilience"])

    # ========== Threshold Policies ==========

    def evaluate_threshold_policy(self, values: dict[str, Any]) -> PolicyResult:
        """Evaluate business threshold policies."""
        violations = []

        if self.policies["threshold"] == PolicyLevel.DISABLED:
            return PolicyResult.success("threshold", PolicyLevel.DISABLED)

        # Check order size
        order_size = values.get("order_size", 0)
        if order_size > self.config["max_order_size"]:
            violations.append(
                f"Order size {order_size} exceeds maximum {self.config['max_order_size']}"
            )

        # Check portfolio positions
        position_count = values.get("position_count", 0)
        if position_count > self.config["max_portfolio_positions"]:
            violations.append(f"Too many positions: {position_count}")

        # Check leverage
        leverage = values.get("leverage", 0)
        if leverage > self.config["max_leverage"]:
            violations.append(
                f"Leverage {leverage:.2f}x exceeds maximum {self.config['max_leverage']}x"
            )

        # Check cash balance
        cash_balance = values.get("cash_balance", float("inf"))
        if cash_balance < self.config["min_cash_balance"]:
            violations.append(f"Insufficient cash: ${cash_balance:.2f}")

        # Check daily trades
        daily_trades = values.get("daily_trades", 0)
        if daily_trades > self.config["max_daily_trades"]:
            violations.append(f"Too many daily trades: {daily_trades}")

        # Check position concentration
        max_concentration = values.get("max_position_concentration", 0)
        if max_concentration > self.config["max_position_concentration"]:
            violations.append(f"Position concentration {max_concentration:.1%} too high")

        if violations:
            return PolicyResult.failure("threshold", violations, self.policies["threshold"])
        return PolicyResult.success("threshold", self.policies["threshold"])

    # ========== Secrets Management Policies ==========

    def evaluate_secrets_policy(self, secret_info: dict[str, Any]) -> PolicyResult:
        """Evaluate secrets management policies."""
        violations = []

        if self.policies["secrets"] == PolicyLevel.DISABLED:
            return PolicyResult.success("secrets", PolicyLevel.DISABLED)

        # Check for hardcoded secrets
        if secret_info.get("hardcoded_secrets"):
            violations.append("Hardcoded secrets detected")

        # Check secret rotation
        last_rotation = secret_info.get("last_rotation")
        if last_rotation:
            days_since = (datetime.now(UTC) - last_rotation).days
            if days_since > 90:
                violations.append(f"Secrets not rotated for {days_since} days")

        # Check encryption
        if not secret_info.get("encrypted", True):
            violations.append("Secrets not encrypted")

        # Check access logs
        if not secret_info.get("access_logged", True):
            violations.append("Secret access not logged")

        # Check for exposed secrets
        if secret_info.get("exposed_in_logs"):
            violations.append("Secrets exposed in logs")

        if violations:
            return PolicyResult.failure("secrets", violations, self.policies["secrets"])
        return PolicyResult.success("secrets", self.policies["secrets"])

    # ========== System Health Policies ==========

    def evaluate_system_health_policy(self, health: dict[str, Any]) -> PolicyResult:
        """Evaluate system health policies."""
        violations = []

        if self.policies["system_health"] == PolicyLevel.DISABLED:
            return PolicyResult.success("system_health", PolicyLevel.DISABLED)

        # Check CPU usage
        cpu_usage = health.get("cpu_usage", 0)
        if cpu_usage > self.config["max_cpu_usage"]:
            violations.append(f"High CPU usage: {cpu_usage:.1%}")

        # Check memory usage
        memory_usage = health.get("memory_usage", 0)
        if memory_usage > self.config["max_memory_usage"]:
            violations.append(f"High memory usage: {memory_usage:.1%}")

        # Check disk usage
        disk_usage = health.get("disk_usage", 0)
        if disk_usage > self.config["max_disk_usage"]:
            violations.append(f"High disk usage: {disk_usage:.1%}")

        # Check database connections
        free_connections = health.get("free_connections", float("inf"))
        if free_connections < self.config["min_free_connections"]:
            violations.append(f"Low free connections: {free_connections}")

        # Check response time
        response_time_ms = health.get("response_time_ms", 0)
        if response_time_ms > self.config["max_response_time_ms"]:
            violations.append(f"Slow response time: {response_time_ms}ms")

        if violations:
            return PolicyResult.failure("system_health", violations, self.policies["system_health"])
        return PolicyResult.success("system_health", self.policies["system_health"])

    # ========== Composite Policy Evaluation ==========

    def evaluate_all_policies(self, context: dict[str, Any]) -> list[PolicyResult]:
        """Evaluate all applicable policies based on context."""
        results = []

        # Evaluate each policy category if data is available
        if "security" in context:
            results.append(self.evaluate_security_policy(context["security"]))

        if "metrics" in context:
            results.append(self.evaluate_monitoring_policy(context["metrics"]))

        if "operation" in context:
            results.append(self.evaluate_resilience_policy(context["operation"]))

        if "values" in context:
            results.append(self.evaluate_threshold_policy(context["values"]))

        if "secrets" in context:
            results.append(self.evaluate_secrets_policy(context["secrets"]))

        if "health" in context:
            results.append(self.evaluate_system_health_policy(context["health"]))

        return results

    def is_compliant(self, results: list[PolicyResult]) -> bool:
        """Check if all policy results are compliant."""
        return all(r.passed or r.level == PolicyLevel.RELAXED for r in results)

    def get_violations_summary(self, results: list[PolicyResult]) -> dict[str, list[str]]:
        """Get a summary of all policy violations."""
        summary = {}
        for result in results:
            if not result.passed and result.level != PolicyLevel.RELAXED:
                summary[result.policy_name] = result.violations
        return summary

    def set_policy_level(self, policy_name: str, level: PolicyLevel) -> None:
        """Set the enforcement level for a specific policy."""
        if policy_name in self.policies:
            self.policies[policy_name] = level

    def update_config(self, config_updates: dict[str, Any]) -> None:
        """Update policy configuration values."""
        self.config.update(config_updates)
