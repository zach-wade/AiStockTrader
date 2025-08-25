"""
Threshold policy domain service.

This module contains all threshold evaluation and breach detection business logic
that was previously in the infrastructure layer. It defines policies for metric
thresholds and determines when breaches occur based on business rules.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ThresholdComparison(Enum):
    """Threshold comparison operators."""

    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL_TO = "equal_to"
    NOT_EQUAL_TO = "not_equal_to"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


class ThresholdSeverity(Enum):
    """Threshold breach severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ThresholdPolicy:
    """
    Threshold policy definition.

    Encapsulates business rules for metric thresholds.
    """

    metric_name: str
    comparison: ThresholdComparison
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    emergency_threshold: float | None = None
    consecutive_breaches_required: int = 1
    evaluation_window_seconds: int | None = None
    description: str | None = None
    enabled: bool = True


@dataclass
class ThresholdBreachEvent:
    """
    Represents a threshold breach event.
    """

    metric_name: str
    current_value: float
    threshold_value: float
    severity: ThresholdSeverity
    comparison: ThresholdComparison
    timestamp: float
    consecutive_breaches: int
    message: str


class ThresholdPolicyService:
    """
    Domain service for threshold policy evaluation and breach detection.

    This service encapsulates all business logic related to metric thresholds,
    breach detection, and alert triggering based on domain rules.
    """

    # Default threshold policies (business rules)
    DEFAULT_POLICIES = {
        "cpu_usage": ThresholdPolicy(
            metric_name="cpu_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            critical_threshold=85.0,
            emergency_threshold=95.0,
            consecutive_breaches_required=3,
            evaluation_window_seconds=300,
            description="CPU usage percentage thresholds",
        ),
        "memory_usage": ThresholdPolicy(
            metric_name="memory_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=75.0,
            critical_threshold=90.0,
            emergency_threshold=95.0,
            consecutive_breaches_required=2,
            evaluation_window_seconds=180,
            description="Memory usage percentage thresholds",
        ),
        "error_rate": ThresholdPolicy(
            metric_name="error_rate",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=0.01,  # 1% error rate
            critical_threshold=0.05,  # 5% error rate
            emergency_threshold=0.10,  # 10% error rate
            consecutive_breaches_required=2,
            evaluation_window_seconds=60,
            description="Error rate thresholds",
        ),
        "response_time_ms": ThresholdPolicy(
            metric_name="response_time_ms",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=1000.0,  # 1 second
            critical_threshold=3000.0,  # 3 seconds
            emergency_threshold=5000.0,  # 5 seconds
            consecutive_breaches_required=5,
            evaluation_window_seconds=120,
            description="API response time thresholds in milliseconds",
        ),
        "order_failure_rate": ThresholdPolicy(
            metric_name="order_failure_rate",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=0.02,  # 2% failure rate
            critical_threshold=0.05,  # 5% failure rate
            emergency_threshold=0.10,  # 10% failure rate
            consecutive_breaches_required=1,
            evaluation_window_seconds=60,
            description="Trading order failure rate thresholds",
        ),
        "position_risk_score": ThresholdPolicy(
            metric_name="position_risk_score",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=0.6,
            critical_threshold=0.8,
            emergency_threshold=0.95,
            consecutive_breaches_required=1,
            evaluation_window_seconds=30,
            description="Portfolio position risk score thresholds",
        ),
    }

    def __init__(self, policies: dict[str, ThresholdPolicy] | None = None) -> None:
        """
        Initialize threshold policy service.

        Args:
            policies: Optional dictionary of custom threshold policies
        """
        self.policies = policies or self.DEFAULT_POLICIES.copy()
        self.breach_states: dict[str, dict[str, Any]] = {}
        self._initialize_breach_states()

    def _initialize_breach_states(self) -> None:
        """Initialize breach tracking states for all policies."""
        for metric_name in self.policies:
            self.breach_states[metric_name] = {
                "consecutive_breaches": 0,
                "last_breach_time": None,
                "last_breach_value": None,
                "last_breach_severity": None,
                "in_breach": False,
                "breach_start_time": None,
            }

    def evaluate_threshold(
        self, metric_name: str, current_value: float, current_time: float | None = None
    ) -> ThresholdBreachEvent | None:
        """
        Evaluate a metric value against its threshold policy.

        Args:
            metric_name: Name of the metric to evaluate
            current_value: Current metric value
            current_time: Optional timestamp (defaults to current time)

        Returns:
            ThresholdBreachEvent if breach detected, None otherwise
        """
        if metric_name not in self.policies:
            return None

        policy = self.policies[metric_name]
        if not policy.enabled:
            return None

        current_time = current_time or time.time()
        state = self.breach_states[metric_name]

        # Check for breach
        breach_info = self._check_threshold_breach(current_value, policy)

        if breach_info:
            severity, threshold_value = breach_info

            # Check if within evaluation window
            if (
                state["last_breach_time"]
                and policy.evaluation_window_seconds
                and (current_time - state["last_breach_time"]) > policy.evaluation_window_seconds
            ):
                # Reset consecutive breaches if outside window
                state["consecutive_breaches"] = 0

            # Increment consecutive breaches
            state["consecutive_breaches"] += 1
            state["last_breach_time"] = current_time
            state["last_breach_value"] = current_value
            state["last_breach_severity"] = severity

            # Check if we should trigger an alert
            if state["consecutive_breaches"] >= policy.consecutive_breaches_required:
                if not state["in_breach"]:
                    state["in_breach"] = True
                    state["breach_start_time"] = current_time

                # Create breach event
                return ThresholdBreachEvent(
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=threshold_value,
                    severity=severity,
                    comparison=policy.comparison,
                    timestamp=current_time,
                    consecutive_breaches=state["consecutive_breaches"],
                    message=self._format_breach_message(
                        metric_name, current_value, threshold_value, severity, policy.comparison
                    ),
                )
        elif state["in_breach"]:
            # Metric has recovered
            state["in_breach"] = False
            state["breach_start_time"] = None
            state["consecutive_breaches"] = 0

        return None

    def _check_threshold_breach(
        self, value: float, policy: ThresholdPolicy
    ) -> tuple[ThresholdSeverity, float] | None:
        """
        Check if a value breaches the threshold based on comparison operator.

        Args:
            value: Value to check
            policy: Threshold policy to evaluate against

        Returns:
            Tuple of (severity, threshold_value) if breached, None otherwise
        """
        # Check thresholds in order of severity (highest to lowest)
        thresholds = [
            (policy.emergency_threshold, ThresholdSeverity.EMERGENCY),
            (policy.critical_threshold, ThresholdSeverity.CRITICAL),
            (policy.warning_threshold, ThresholdSeverity.WARNING),
        ]

        for threshold_value, severity in thresholds:
            if threshold_value is not None:
                if self._compare_values(value, threshold_value, policy.comparison):
                    return (severity, threshold_value)

        return None

    def _compare_values(
        self, value: float, threshold: float, comparison: ThresholdComparison
    ) -> bool:
        """
        Compare a value against a threshold using the specified operator.

        Args:
            value: Value to compare
            threshold: Threshold value
            comparison: Comparison operator

        Returns:
            True if comparison condition is met, False otherwise
        """
        if comparison == ThresholdComparison.GREATER_THAN:
            return value > threshold
        elif comparison == ThresholdComparison.LESS_THAN:
            return value < threshold
        elif comparison == ThresholdComparison.EQUAL_TO:
            return value == threshold
        elif comparison == ThresholdComparison.NOT_EQUAL_TO:
            return value != threshold
        elif comparison == ThresholdComparison.GREATER_THAN_OR_EQUAL:
            return value >= threshold
        elif comparison == ThresholdComparison.LESS_THAN_OR_EQUAL:
            return value <= threshold
        else:
            # This is unreachable but satisfies type checker
            raise ValueError(f"Unknown comparison type: {comparison}")

    def _format_breach_message(
        self,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        severity: ThresholdSeverity,
        comparison: ThresholdComparison,
    ) -> str:
        """
        Format a human-readable breach message.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            threshold_value: Threshold that was breached
            severity: Severity of the breach
            comparison: Comparison operator used

        Returns:
            Formatted breach message
        """
        comparison_text = {
            ThresholdComparison.GREATER_THAN: "exceeded",
            ThresholdComparison.LESS_THAN: "fell below",
            ThresholdComparison.EQUAL_TO: "equals",
            ThresholdComparison.NOT_EQUAL_TO: "does not equal",
            ThresholdComparison.GREATER_THAN_OR_EQUAL: "met or exceeded",
            ThresholdComparison.LESS_THAN_OR_EQUAL: "met or fell below",
        }

        action = comparison_text.get(comparison, "breached")

        return (
            f"{severity.value.upper()}: Metric '{metric_name}' {action} threshold. "
            f"Current value: {current_value:.2f}, Threshold: {threshold_value:.2f}"
        )

    def add_policy(self, policy: ThresholdPolicy) -> None:
        """
        Add or update a threshold policy.

        Args:
            policy: Threshold policy to add
        """
        self.policies[policy.metric_name] = policy
        if policy.metric_name not in self.breach_states:
            self.breach_states[policy.metric_name] = {
                "consecutive_breaches": 0,
                "last_breach_time": None,
                "last_breach_value": None,
                "last_breach_severity": None,
                "in_breach": False,
                "breach_start_time": None,
            }

    def remove_policy(self, metric_name: str) -> None:
        """
        Remove a threshold policy.

        Args:
            metric_name: Name of the metric policy to remove
        """
        if metric_name in self.policies:
            del self.policies[metric_name]
        if metric_name in self.breach_states:
            del self.breach_states[metric_name]

    def get_policy(self, metric_name: str) -> ThresholdPolicy | None:
        """
        Get a threshold policy by metric name.

        Args:
            metric_name: Name of the metric

        Returns:
            ThresholdPolicy if found, None otherwise
        """
        return self.policies.get(metric_name)

    def get_breach_state(self, metric_name: str) -> dict[str, Any] | None:
        """
        Get the current breach state for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Breach state dictionary if found, None otherwise
        """
        return self.breach_states.get(metric_name)

    def reset_breach_state(self, metric_name: str) -> None:
        """
        Reset the breach state for a metric.

        Args:
            metric_name: Name of the metric
        """
        if metric_name in self.breach_states:
            self.breach_states[metric_name] = {
                "consecutive_breaches": 0,
                "last_breach_time": None,
                "last_breach_value": None,
                "last_breach_severity": None,
                "in_breach": False,
                "breach_start_time": None,
            }

    def get_active_breaches(self) -> list[dict[str, Any]]:
        """
        Get all currently active breaches.

        Returns:
            List of active breach information
        """
        active_breaches = []
        for metric_name, state in self.breach_states.items():
            if state["in_breach"]:
                active_breaches.append(
                    {
                        "metric_name": metric_name,
                        "severity": (
                            state["last_breach_severity"].value
                            if state["last_breach_severity"]
                            else None
                        ),
                        "current_value": state["last_breach_value"],
                        "consecutive_breaches": state["consecutive_breaches"],
                        "breach_start_time": state["breach_start_time"],
                        "last_breach_time": state["last_breach_time"],
                    }
                )
        return active_breaches

    def evaluate_all_thresholds(
        self, metrics: dict[str, float], current_time: float | None = None
    ) -> list[ThresholdBreachEvent]:
        """
        Evaluate multiple metrics against their threshold policies.

        Args:
            metrics: Dictionary of metric names and values
            current_time: Optional timestamp

        Returns:
            List of threshold breach events
        """
        breaches = []
        for metric_name, value in metrics.items():
            breach_event = self.evaluate_threshold(metric_name, value, current_time)
            if breach_event:
                breaches.append(breach_event)
        return breaches
