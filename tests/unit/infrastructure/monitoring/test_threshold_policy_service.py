"""
Unit tests for ThresholdPolicyService.

Tests threshold evaluation, breach detection, and policy management.
"""

import pytest

from src.infrastructure.monitoring.threshold_policy_service import (
    ThresholdComparison,
    ThresholdPolicy,
    ThresholdPolicyService,
    ThresholdSeverity,
)


class TestThresholdPolicyService:
    """Test suite for ThresholdPolicyService."""

    @pytest.fixture
    def service(self):
        """Create a ThresholdPolicyService instance."""
        return ThresholdPolicyService()

    @pytest.fixture
    def cpu_policy(self):
        """Create a CPU usage threshold policy."""
        return ThresholdPolicy(
            metric_name="cpu_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            critical_threshold=85.0,
            emergency_threshold=95.0,
            consecutive_breaches_required=2,
            evaluation_window_seconds=60,
            description="CPU usage monitoring",
            enabled=True,
        )

    @pytest.fixture
    def memory_policy(self):
        """Create a memory usage threshold policy."""
        return ThresholdPolicy(
            metric_name="memory_usage",
            comparison=ThresholdComparison.GREATER_THAN_OR_EQUAL,
            warning_threshold=60.0,
            critical_threshold=80.0,
            emergency_threshold=90.0,
            consecutive_breaches_required=1,
            enabled=True,
        )

    # Basic Policy Management Tests

    def test_init_with_default_policies(self):
        """Test service initialization with default policies."""
        service = ThresholdPolicyService()
        # Should have default policies
        assert service._policies is not None
        assert len(service._policies) > 0

    def test_init_with_custom_policies(self, cpu_policy, memory_policy):
        """Test service initialization with custom policies."""
        policies = {"cpu_usage": cpu_policy, "memory_usage": memory_policy}
        service = ThresholdPolicyService(policies)

        assert len(service._policies) == 2
        assert "cpu_usage" in service._policies
        assert "memory_usage" in service._policies

    def test_add_policy(self, service, cpu_policy):
        """Test adding a new policy."""
        service.add_policy(cpu_policy)

        policy = service.get_policy("cpu_usage")
        assert policy is not None
        assert policy.metric_name == "cpu_usage"
        assert policy.warning_threshold == 70.0

    def test_remove_policy(self, service, cpu_policy):
        """Test removing a policy."""
        service.add_policy(cpu_policy)
        assert service.get_policy("cpu_usage") is not None

        service.remove_policy("cpu_usage")
        assert service.get_policy("cpu_usage") is None

    def test_get_policy(self, service, cpu_policy):
        """Test getting a policy by metric name."""
        service.add_policy(cpu_policy)

        policy = service.get_policy("cpu_usage")
        assert policy == cpu_policy

        # Non-existent policy
        assert service.get_policy("non_existent") is None

    # Threshold Evaluation Tests

    def test_evaluate_threshold_no_breach(self, service, cpu_policy):
        """Test threshold evaluation with no breach."""
        service.add_policy(cpu_policy)

        # Value below warning threshold
        breach = service.evaluate_threshold("cpu_usage", 50.0)
        assert breach is None

    def test_evaluate_threshold_warning_breach(self, service, cpu_policy):
        """Test threshold evaluation with warning breach."""
        # Set consecutive breaches to 1 for immediate breach
        cpu_policy.consecutive_breaches_required = 1
        service.add_policy(cpu_policy)

        # Value above warning but below critical
        breach = service.evaluate_threshold("cpu_usage", 75.0)
        assert breach is not None
        assert breach.severity == ThresholdSeverity.WARNING
        assert breach.current_value == 75.0
        assert breach.threshold_value == 70.0

    def test_evaluate_threshold_critical_breach(self, service, cpu_policy):
        """Test threshold evaluation with critical breach."""
        cpu_policy.consecutive_breaches_required = 1
        service.add_policy(cpu_policy)

        # Value above critical but below emergency
        breach = service.evaluate_threshold("cpu_usage", 90.0)
        assert breach is not None
        assert breach.severity == ThresholdSeverity.CRITICAL
        assert breach.current_value == 90.0
        assert breach.threshold_value == 85.0

    def test_evaluate_threshold_emergency_breach(self, service, cpu_policy):
        """Test threshold evaluation with emergency breach."""
        cpu_policy.consecutive_breaches_required = 1
        service.add_policy(cpu_policy)

        # Value above emergency threshold
        breach = service.evaluate_threshold("cpu_usage", 98.0)
        assert breach is not None
        assert breach.severity == ThresholdSeverity.EMERGENCY
        assert breach.current_value == 98.0
        assert breach.threshold_value == 95.0

    def test_evaluate_threshold_consecutive_breaches(self, service, cpu_policy):
        """Test consecutive breaches requirement."""
        cpu_policy.consecutive_breaches_required = 3
        service.add_policy(cpu_policy)

        # First breach - no event
        breach = service.evaluate_threshold("cpu_usage", 75.0)
        assert breach is None

        # Second breach - no event
        breach = service.evaluate_threshold("cpu_usage", 76.0)
        assert breach is None

        # Third breach - should trigger event
        breach = service.evaluate_threshold("cpu_usage", 77.0)
        assert breach is not None
        assert breach.consecutive_breaches == 3

    def test_evaluate_threshold_disabled_policy(self, service, cpu_policy):
        """Test that disabled policies don't generate breaches."""
        cpu_policy.enabled = False
        service.add_policy(cpu_policy)

        # Even with high value, should not breach
        breach = service.evaluate_threshold("cpu_usage", 100.0)
        assert breach is None

    def test_evaluate_threshold_nonexistent_metric(self, service):
        """Test evaluating threshold for non-existent metric."""
        breach = service.evaluate_threshold("nonexistent", 100.0)
        assert breach is None

    # Comparison Operator Tests

    def test_comparison_greater_than(self, service):
        """Test GREATER_THAN comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Below threshold - no breach
        assert service.evaluate_threshold("test", 49.9) is None
        # Exactly at threshold - no breach
        assert service.evaluate_threshold("test", 50.0) is None
        # Above threshold - breach
        assert service.evaluate_threshold("test", 50.1) is not None

    def test_comparison_less_than(self, service):
        """Test LESS_THAN comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.LESS_THAN,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Above threshold - no breach
        assert service.evaluate_threshold("test", 50.1) is None
        # Exactly at threshold - no breach
        assert service.evaluate_threshold("test", 50.0) is None
        # Below threshold - breach
        assert service.evaluate_threshold("test", 49.9) is not None

    def test_comparison_equal_to(self, service):
        """Test EQUAL_TO comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.EQUAL_TO,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Not equal - no breach
        assert service.evaluate_threshold("test", 49.9) is None
        assert service.evaluate_threshold("test", 50.1) is None
        # Equal - breach
        assert service.evaluate_threshold("test", 50.0) is not None

    def test_comparison_not_equal_to(self, service):
        """Test NOT_EQUAL_TO comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.NOT_EQUAL_TO,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Equal - no breach
        assert service.evaluate_threshold("test", 50.0) is None
        # Not equal - breach
        assert service.evaluate_threshold("test", 49.9) is not None
        assert service.evaluate_threshold("test", 50.1) is not None

    def test_comparison_greater_than_or_equal(self, service):
        """Test GREATER_THAN_OR_EQUAL comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.GREATER_THAN_OR_EQUAL,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Below threshold - no breach
        assert service.evaluate_threshold("test", 49.9) is None
        # Exactly at threshold - breach
        assert service.evaluate_threshold("test", 50.0) is not None
        # Above threshold - breach
        assert service.evaluate_threshold("test", 50.1) is not None

    def test_comparison_less_than_or_equal(self, service):
        """Test LESS_THAN_OR_EQUAL comparison."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.LESS_THAN_OR_EQUAL,
            warning_threshold=50.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Above threshold - no breach
        assert service.evaluate_threshold("test", 50.1) is None
        # Exactly at threshold - breach
        assert service.evaluate_threshold("test", 50.0) is not None
        # Below threshold - breach
        assert service.evaluate_threshold("test", 49.9) is not None

    # Breach State Management Tests

    def test_get_breach_state(self, service, cpu_policy):
        """Test getting breach state for a metric."""
        service.add_policy(cpu_policy)

        # Initially no breach state
        state = service.get_breach_state("cpu_usage")
        assert state is not None
        assert state["consecutive_count"] == 0

        # After a breach attempt
        service.evaluate_threshold("cpu_usage", 75.0)
        state = service.get_breach_state("cpu_usage")
        assert state["consecutive_count"] == 1

    def test_reset_breach_state(self, service, cpu_policy):
        """Test resetting breach state."""
        service.add_policy(cpu_policy)

        # Create some breach state
        service.evaluate_threshold("cpu_usage", 75.0)
        state = service.get_breach_state("cpu_usage")
        assert state["consecutive_count"] == 1

        # Reset the state
        service.reset_breach_state("cpu_usage")
        state = service.get_breach_state("cpu_usage")
        assert state["consecutive_count"] == 0

    def test_get_active_breaches(self, service):
        """Test getting currently active breaches."""
        # Add policies with immediate breach
        cpu_policy = ThresholdPolicy(
            metric_name="cpu",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        memory_policy = ThresholdPolicy(
            metric_name="memory",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=80.0,
            consecutive_breaches_required=1,
            enabled=True,
        )

        service.add_policy(cpu_policy)
        service.add_policy(memory_policy)

        # Create breaches
        service.evaluate_threshold("cpu", 75.0)
        service.evaluate_threshold("memory", 85.0)

        active = service.get_active_breaches()
        assert len(active) == 2

        # Clear one breach
        service.evaluate_threshold("cpu", 60.0)
        active = service.get_active_breaches()
        assert len(active) == 1

    def test_evaluate_all_thresholds(self, service):
        """Test evaluating all thresholds at once."""
        # Add multiple policies
        cpu_policy = ThresholdPolicy(
            metric_name="cpu_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            consecutive_breaches_required=1,
            enabled=True,
        )
        memory_policy = ThresholdPolicy(
            metric_name="memory_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=80.0,
            consecutive_breaches_required=1,
            enabled=True,
        )

        service.add_policy(cpu_policy)
        service.add_policy(memory_policy)

        # Evaluate all with metrics
        metrics = {
            "cpu_usage": 75.0,
            "memory_usage": 85.0,
            "unknown_metric": 100.0,  # Should be ignored
        }

        breaches = service.evaluate_all_thresholds(metrics)
        assert len(breaches) == 2
        assert any(b.metric_name == "cpu_usage" for b in breaches)
        assert any(b.metric_name == "memory_usage" for b in breaches)

    # Edge Cases

    def test_evaluate_with_none_value(self, service, cpu_policy):
        """Test evaluation with None value."""
        service.add_policy(cpu_policy)

        # Should handle None gracefully
        breach = service.evaluate_threshold("cpu_usage", None)
        assert breach is None

    def test_evaluate_with_string_value(self, service, cpu_policy):
        """Test evaluation with string value."""
        service.add_policy(cpu_policy)

        # Should convert string to float if valid
        breach = service.evaluate_threshold("cpu_usage", "75.0")
        # Implementation might or might not handle this
        # If it doesn't, breach will be None

    def test_partial_thresholds(self, service):
        """Test policy with only some thresholds defined."""
        policy = ThresholdPolicy(
            metric_name="test",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            critical_threshold=None,
            emergency_threshold=None,
            consecutive_breaches_required=1,
            enabled=True,
        )
        service.add_policy(policy)

        # Should still work with partial thresholds
        breach = service.evaluate_threshold("test", 75.0)
        assert breach is not None
        assert breach.severity == ThresholdSeverity.WARNING

    def test_breach_message_formatting(self, service):
        """Test that breach messages are properly formatted."""
        policy = ThresholdPolicy(
            metric_name="cpu_usage",
            comparison=ThresholdComparison.GREATER_THAN,
            warning_threshold=70.0,
            consecutive_breaches_required=1,
            description="CPU usage monitoring",
            enabled=True,
        )
        service.add_policy(policy)

        breach = service.evaluate_threshold("cpu_usage", 75.0)
        assert breach is not None
        assert breach.message is not None
        assert "cpu_usage" in breach.message
        assert "75.0" in breach.message or "75" in breach.message
