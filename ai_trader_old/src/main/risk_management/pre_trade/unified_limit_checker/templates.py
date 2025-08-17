"""
Unified Limit Checker Templates Module

This module provides predefined limit templates for common use cases.
"""

from .models import LimitDefinition
from .types import ComparisonOperator, LimitAction, LimitScope, LimitType


class LimitTemplates:
    """Predefined limit templates for common use cases."""

    @staticmethod
    def create_position_size_limit(
        limit_id: str, max_position_pct: float, scope_filter: dict | None = None
    ) -> LimitDefinition:
        """Create position size limit as percentage of portfolio."""
        return LimitDefinition(
            limit_id=limit_id,
            name=f"Position Size Limit ({max_position_pct}%)",
            description=f"Maximum position size of {max_position_pct}% of portfolio value",
            limit_type=LimitType.POSITION_SIZE,
            scope=LimitScope.POSITION,
            threshold_value=max_position_pct,
            soft_threshold=max_position_pct * 0.8,  # 80% of limit for warning
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.BLOCK_TRADE,
            soft_violation_action=LimitAction.ALERT,
            scope_filter=dict(scope_filter or {}, relative=True),
        )

    @staticmethod
    def create_sector_concentration_limit(sector: str, max_pct: float) -> LimitDefinition:
        """Create sector concentration limit."""
        return LimitDefinition(
            limit_id=f"sector_concentration_{sector.lower()}",
            name=f"{sector} Sector Concentration Limit",
            description=f"Maximum exposure to {sector} sector: {max_pct}%",
            limit_type=LimitType.SECTOR_EXPOSURE,
            scope=LimitScope.SECTOR,
            threshold_value=max_pct,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.ALERT,
            scope_filter={"sector": sector},
        )

    @staticmethod
    def create_drawdown_limit(max_drawdown_pct: float) -> LimitDefinition:
        """Create maximum drawdown limit."""
        return LimitDefinition(
            limit_id="max_drawdown",
            name="Maximum Drawdown Limit",
            description=f"Maximum allowed portfolio drawdown: {max_drawdown_pct}%",
            limit_type=LimitType.DRAWDOWN,
            scope=LimitScope.GLOBAL,
            threshold_value=max_drawdown_pct / 100,  # Convert to decimal
            soft_threshold=max_drawdown_pct * 0.8 / 100,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.PAUSE_STRATEGY,
            soft_violation_action=LimitAction.ALERT,
        )

    @staticmethod
    def create_var_utilization_limit(max_var_pct: float) -> LimitDefinition:
        """Create VaR utilization limit."""
        return LimitDefinition(
            limit_id="var_utilization",
            name="VaR Utilization Limit",
            description=f"Maximum VaR utilization: {max_var_pct}%",
            limit_type=LimitType.VAR_UTILIZATION,
            scope=LimitScope.GLOBAL,
            threshold_value=max_var_pct / 100,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.REDUCE_POSITION,
            scope_filter={"metric_key": "var_95"},
        )

    @staticmethod
    def create_leverage_limit(max_leverage: float) -> LimitDefinition:
        """Create leverage limit."""
        return LimitDefinition(
            limit_id="leverage_limit",
            name="Leverage Limit",
            description=f"Maximum portfolio leverage: {max_leverage}x",
            limit_type=LimitType.LEVERAGE,
            scope=LimitScope.GLOBAL,
            threshold_value=max_leverage,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.REDUCE_POSITION,
            scope_filter={"metric_key": "leverage"},
        )

    @staticmethod
    def create_volatility_limit(max_volatility_pct: float) -> LimitDefinition:
        """Create volatility limit."""
        return LimitDefinition(
            limit_id="volatility_limit",
            name="Volatility Limit",
            description=f"Maximum portfolio volatility: {max_volatility_pct}%",
            limit_type=LimitType.VOLATILITY,
            scope=LimitScope.GLOBAL,
            threshold_value=max_volatility_pct / 100,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.ALERT,
            scope_filter={"metric_key": "volatility"},
        )

    @staticmethod
    def create_correlation_limit(max_correlation: float) -> LimitDefinition:
        """Create correlation limit."""
        return LimitDefinition(
            limit_id="correlation_limit",
            name="Correlation Limit",
            description=f"Maximum position correlation: {max_correlation}",
            limit_type=LimitType.CORRELATION,
            scope=LimitScope.POSITION,
            threshold_value=max_correlation,
            operator=ComparisonOperator.LESS_EQUAL,
            violation_action=LimitAction.ALERT,
            scope_filter={"metric_key": "correlation"},
        )


def create_basic_portfolio_limits(
    max_position_pct: float = 10.0, max_drawdown_pct: float = 20.0, max_leverage: float = 2.0
) -> list[LimitDefinition]:
    """Create basic portfolio limit set."""
    return [
        LimitTemplates.create_position_size_limit("max_position_size", max_position_pct),
        LimitTemplates.create_drawdown_limit(max_drawdown_pct),
        LimitTemplates.create_leverage_limit(max_leverage),
        LimitTemplates.create_var_utilization_limit(80.0),  # 80% VaR utilization
    ]


def create_comprehensive_portfolio_limits(
    max_position_pct: float = 5.0,
    max_drawdown_pct: float = 15.0,
    max_leverage: float = 1.5,
    max_volatility_pct: float = 25.0,
    max_correlation: float = 0.8,
) -> list[LimitDefinition]:
    """Create comprehensive portfolio limit set."""
    return [
        LimitTemplates.create_position_size_limit("max_position_size", max_position_pct),
        LimitTemplates.create_drawdown_limit(max_drawdown_pct),
        LimitTemplates.create_leverage_limit(max_leverage),
        LimitTemplates.create_var_utilization_limit(70.0),  # 70% VaR utilization
        LimitTemplates.create_volatility_limit(max_volatility_pct),
        LimitTemplates.create_correlation_limit(max_correlation),
    ]
