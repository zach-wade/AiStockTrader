# File: risk_management/pre_trade/unified_limit_checker/utils.py

"""
Utility functions for the Unified Limit Checker.

Provides convenience functions and helpers for common limit checking operations.
"""

# Standard library imports
import json

# Local imports
from main.interfaces.events import IEventBus
from main.utils.core import get_logger

from .config import LimitConfig, get_default_config
from .events import create_event_manager_with_defaults
from .models import AggregatedCheckResult, LimitCheckResult, LimitDefinition, LimitViolation
from .registry import create_default_registry
from .types import ComparisonOperator, LimitAction, LimitScope, LimitType, ViolationSeverity

logger = get_logger(__name__)


class LimitTemplates:
    """Common limit templates for quick setup."""

    @staticmethod
    def create_position_size_limit(
        limit_id: str,
        max_percentage: float,
        scope: LimitScope = LimitScope.POSITION,
        action: LimitAction = LimitAction.BLOCK_TRADE,
    ) -> LimitDefinition:
        """Create a position size limit."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Position Size Limit ({max_percentage}%)",
            description="Maximum position size as percentage of portfolio",
            limit_type=LimitType.POSITION_SIZE,
            scope=scope,
            threshold_value=max_percentage,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=action,
            soft_threshold=max_percentage * 0.8,
            soft_violation_action=LimitAction.ALERT,
        )

    @staticmethod
    def create_drawdown_limit(
        limit_id: str,
        max_drawdown: float,
        lookback_days: int = 30,
        action: LimitAction = LimitAction.PAUSE_STRATEGY,
    ) -> LimitDefinition:
        """Create a drawdown limit."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Max Drawdown Limit ({max_drawdown}%)",
            description=f"Maximum drawdown over {lookback_days} days",
            limit_type=LimitType.DRAWDOWN,
            scope=LimitScope.GLOBAL,
            threshold_value=max_drawdown,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=action,
            soft_threshold=max_drawdown * 0.7,
            soft_violation_action=LimitAction.ALERT,
            metadata={"lookback_days": lookback_days},
        )

    @staticmethod
    def create_concentration_limit(
        limit_id: str,
        max_concentration: float,
        scope: LimitScope = LimitScope.SECTOR,
        group_by: str = "sector",
    ) -> LimitDefinition:
        """Create a concentration limit."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Concentration Limit ({max_concentration}%)",
            description=f"Maximum concentration by {group_by}",
            limit_type=LimitType.CONCENTRATION,
            scope=scope,
            threshold_value=max_concentration,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.REDUCE_POSITION,
            soft_threshold=max_concentration * 0.9,
            metadata={"group_by": group_by},
        )

    @staticmethod
    def create_leverage_limit(
        limit_id: str, max_leverage: float, include_derivatives: bool = True
    ) -> LimitDefinition:
        """Create a leverage limit."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Leverage Limit ({max_leverage}x)",
            description="Maximum portfolio leverage",
            limit_type=LimitType.LEVERAGE,
            scope=LimitScope.GLOBAL,
            threshold_value=max_leverage,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.BLOCK_TRADE,
            metadata={"include_derivatives": include_derivatives},
        )

    @staticmethod
    def create_volatility_limit(
        limit_id: str, max_volatility: float, lookback_days: int = 20
    ) -> LimitDefinition:
        """Create a volatility limit."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Volatility Limit ({max_volatility}%)",
            description="Maximum annualized volatility",
            limit_type=LimitType.VOLATILITY,
            scope=LimitScope.GLOBAL,
            threshold_value=max_volatility,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.ALERT,
            metadata={"lookback_days": lookback_days},
        )


def create_basic_portfolio_limits() -> list[LimitDefinition]:
    """Create a basic set of portfolio limits."""
    return [
        LimitTemplates.create_position_size_limit("pos_size_10", 10.0),
        LimitTemplates.create_drawdown_limit("max_dd_20", 20.0),
        LimitTemplates.create_concentration_limit("sector_conc_30", 30.0),
        LimitTemplates.create_leverage_limit("max_lev_2", 2.0),
        LimitTemplates.create_volatility_limit("max_vol_50", 50.0),
    ]


def create_comprehensive_portfolio_limits() -> list[LimitDefinition]:
    """Create a comprehensive set of portfolio limits."""
    limits = []

    # Position limits
    limits.extend(
        [
            LimitTemplates.create_position_size_limit("pos_size_5", 5.0),
            LimitTemplates.create_position_size_limit(
                "pos_size_10_warn", 10.0, action=LimitAction.ALERT
            ),
            LimitTemplates.create_position_size_limit("pos_size_15_block", 15.0),
        ]
    )

    # Drawdown limits
    limits.extend(
        [
            LimitTemplates.create_drawdown_limit("dd_daily_5", 5.0, lookback_days=1),
            LimitTemplates.create_drawdown_limit("dd_weekly_10", 10.0, lookback_days=7),
            LimitTemplates.create_drawdown_limit("dd_monthly_15", 15.0, lookback_days=30),
            LimitTemplates.create_drawdown_limit("dd_yearly_25", 25.0, lookback_days=365),
        ]
    )

    # Concentration limits
    limits.extend(
        [
            LimitTemplates.create_concentration_limit("sector_conc_25", 25.0),
            LimitTemplates.create_concentration_limit(
                "currency_conc_40", 40.0, scope=LimitScope.CURRENCY, group_by="currency"
            ),
            LimitTemplates.create_concentration_limit(
                "country_conc_30", 30.0, scope=LimitScope.GEOGRAPHIC, group_by="country"
            ),
        ]
    )

    # Risk limits
    limits.extend(
        [
            LimitTemplates.create_leverage_limit("gross_lev_3", 3.0),
            LimitTemplates.create_leverage_limit("net_lev_2", 2.0),
            LimitTemplates.create_volatility_limit("vol_30d_40", 40.0, lookback_days=30),
            LimitTemplates.create_volatility_limit("vol_60d_35", 35.0, lookback_days=60),
        ]
    )

    # Trading limits
    limits.append(
        LimitDefinition(
            limit_id="trade_velocity_100",
            name="Trading Velocity Limit",
            description="Maximum trades per hour",
            limit_type=LimitType.TRADING_VELOCITY,
            scope=LimitScope.GLOBAL,
            threshold_value=100,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.ALERT,
            soft_threshold=80,
        )
    )

    return limits


def validate_limit_definition(limit: LimitDefinition) -> list[str]:
    """
    Validate a limit definition.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if not limit.limit_id:
        errors.append("limit_id is required")

    if not limit.name:
        errors.append("name is required")

    if limit.threshold_value < 0:
        errors.append("threshold_value must be non-negative")

    # Check operator-specific validations
    if limit.operator in [ComparisonOperator.BETWEEN, ComparisonOperator.OUTSIDE]:
        if limit.upper_threshold is None:
            errors.append(f"{limit.operator.value} operator requires upper_threshold")
        elif limit.threshold_value >= limit.upper_threshold:
            errors.append("threshold_value must be less than upper_threshold")

    # Check soft threshold
    if limit.soft_threshold is not None:
        if limit.operator in [ComparisonOperator.LESS_THAN, ComparisonOperator.LESS_EQUAL]:
            if limit.soft_threshold >= limit.threshold_value:
                errors.append("soft_threshold must be less than threshold_value for LESS operators")
        elif limit.operator in [ComparisonOperator.GREATER_THAN, ComparisonOperator.GREATER_EQUAL]:
            if limit.soft_threshold <= limit.threshold_value:
                errors.append(
                    "soft_threshold must be greater than threshold_value for GREATER operators"
                )

    return errors


def format_limit_summary(limit: LimitDefinition) -> str:
    """Format a limit definition as a human-readable summary."""
    parts = [
        f"Limit: {limit.name}",
        f"Type: {limit.limit_type.value}",
        f"Scope: {limit.scope.value}",
        f"Threshold: {limit.threshold_value} ({limit.operator.value})",
    ]

    if limit.soft_threshold:
        parts.append(f"Soft threshold: {limit.soft_threshold}")

    if limit.upper_threshold:
        parts.append(f"Upper threshold: {limit.upper_threshold}")

    parts.append(f"Action: {limit.violation_action.value}")
    parts.append(f"Enabled: {limit.enabled}")

    return " | ".join(parts)


def format_violation_summary(violation: LimitViolation) -> str:
    """Format a violation as a human-readable summary."""
    parts = [
        f"Violation: {violation.limit_name}",
        f"Severity: {violation.severity.value}",
        f"Current: {violation.current_value:.2f}",
        f"Threshold: {violation.threshold_value:.2f}",
        (
            f"Breach: {violation.breach_magnitude:.1%}"
            if violation.breach_magnitude
            else "Breach: N/A"
        ),
    ]

    if violation.resolved:
        parts.append("Status: RESOLVED")
        if violation.duration_seconds:
            parts.append(f"Duration: {violation.duration_seconds:.1f}s")
    else:
        parts.append("Status: ACTIVE")

    return " | ".join(parts)


def format_check_result_summary(result: LimitCheckResult) -> str:
    """Format a check result as a human-readable summary."""
    status = "PASSED" if result.passed else "FAILED"
    parts = [
        f"[{status}]",
        f"Limit: {result.limit_id}",
        f"Value: {result.current_value:.2f}",
        f"Threshold: {result.threshold_value:.2f}",
    ]

    if result.violation:
        parts.append(f"Severity: {result.violation.severity.value}")

    if result.check_duration_ms:
        parts.append(f"Duration: {result.check_duration_ms:.1f}ms")

    return " | ".join(parts)


def aggregate_check_results(results: list[LimitCheckResult]) -> AggregatedCheckResult:
    """Aggregate multiple check results."""
    aggregated = AggregatedCheckResult()

    for result in results:
        aggregated.add_result(result)

    return aggregated


def filter_violations_by_severity(
    violations: list[LimitViolation], min_severity: ViolationSeverity
) -> list[LimitViolation]:
    """Filter violations by minimum severity."""
    severity_order = {
        ViolationSeverity.INFO: 0,
        ViolationSeverity.WARNING: 1,
        ViolationSeverity.SOFT_BREACH: 2,
        ViolationSeverity.HARD_BREACH: 3,
        ViolationSeverity.CRITICAL: 4,
    }

    min_level = severity_order[min_severity]

    return [v for v in violations if severity_order.get(v.severity, 0) >= min_level]


def get_active_violations(violations: list[LimitViolation]) -> list[LimitViolation]:
    """Get only active (unresolved) violations."""
    return [v for v in violations if not v.resolved]


def get_violations_by_limit_type(
    violations: list[LimitViolation], limit_types: dict[str, LimitType]
) -> dict[LimitType, list[LimitViolation]]:
    """Group violations by limit type."""
    grouped: dict[LimitType, list[LimitViolation]] = {}

    for violation in violations:
        limit_type = limit_types.get(violation.limit_id)
        if limit_type:
            if limit_type not in grouped:
                grouped[limit_type] = []
            grouped[limit_type].append(violation)

    return grouped


def calculate_violation_duration_stats(violations: list[LimitViolation]) -> dict[str, float]:
    """Calculate duration statistics for resolved violations."""
    durations = [
        v.duration_seconds for v in violations if v.resolved and v.duration_seconds is not None
    ]

    if not durations:
        return {"count": 0, "total": 0.0, "average": 0.0, "min": 0.0, "max": 0.0}

    return {
        "count": len(durations),
        "total": sum(durations),
        "average": sum(durations) / len(durations),
        "min": min(durations),
        "max": max(durations),
    }


# Convenience functions for creating configured instances


def create_limit_checker(
    config: LimitConfig | None = None, event_bus: IEventBus | None = None
) -> "UnifiedLimitChecker":
    """
    Create a UnifiedLimitChecker with configuration.

    This function imports UnifiedLimitChecker locally to avoid circular imports.
    """
    from .unified_limit_checker import UnifiedLimitChecker

    if config is None:
        config = get_default_config()

    event_manager = create_event_manager_with_defaults(event_bus)
    registry = create_default_registry(config, event_manager)

    return UnifiedLimitChecker(config, registry, event_manager)


def create_limit_checker_with_defaults() -> "UnifiedLimitChecker":
    """Create a UnifiedLimitChecker with default configuration and limits."""
    checker = create_limit_checker()

    # Add basic portfolio limits
    limits = create_basic_portfolio_limits()
    for limit in limits:
        checker.add_limit(limit)

    return checker


async def setup_basic_portfolio_limits(checker: "UnifiedLimitChecker") -> list[LimitDefinition]:
    """
    Setup basic portfolio limits on a checker.

    Returns:
        List of added limits
    """
    limits = create_basic_portfolio_limits()

    for limit in limits:
        await checker.add_limit(limit)

    logger.info(f"Added {len(limits)} basic portfolio limits")
    return limits


async def setup_comprehensive_portfolio_limits(
    checker: "UnifiedLimitChecker",
) -> list[LimitDefinition]:
    """
    Setup comprehensive portfolio limits on a checker.

    Returns:
        List of added limits
    """
    limits = create_comprehensive_portfolio_limits()

    for limit in limits:
        await checker.add_limit(limit)

    logger.info(f"Added {len(limits)} comprehensive portfolio limits")
    return limits


def export_limits_to_json(limits: list[LimitDefinition]) -> str:
    """Export limits to JSON format."""
    data = []
    for limit in limits:
        limit_dict = {
            "limit_id": limit.limit_id,
            "name": limit.name,
            "description": limit.description,
            "limit_type": limit.limit_type.value,
            "scope": limit.scope.value,
            "threshold_value": limit.threshold_value,
            "operator": limit.operator.value,
            "soft_threshold": limit.soft_threshold,
            "upper_threshold": limit.upper_threshold,
            "violation_action": limit.violation_action.value,
            "soft_violation_action": limit.soft_violation_action.value,
            "scope_filter": limit.scope_filter,
            "enabled": limit.enabled,
            "priority": limit.priority,
            "tags": limit.tags,
            "metadata": limit.metadata,
        }
        data.append(limit_dict)

    return json.dumps(data, indent=2)


def import_limits_from_json(json_str: str) -> list[LimitDefinition]:
    """Import limits from JSON format."""
    data = json.loads(json_str)
    limits = []

    for item in data:
        limit = LimitDefinition(
            limit_id=item["limit_id"],
            name=item["name"],
            description=item["description"],
            limit_type=LimitType(item["limit_type"]),
            scope=LimitScope(item["scope"]),
            threshold_value=item["threshold_value"],
            operator=ComparisonOperator(item["operator"]),
            soft_threshold=item.get("soft_threshold"),
            upper_threshold=item.get("upper_threshold"),
            violation_action=LimitAction(item["violation_action"]),
            soft_violation_action=LimitAction(item["soft_violation_action"]),
            scope_filter=item.get("scope_filter", {}),
            enabled=item.get("enabled", True),
            priority=item.get("priority", 0),
            tags=item.get("tags", []),
            metadata=item.get("metadata", {}),
        )
        limits.append(limit)

    return limits
