# File: risk_management/position_sizing/var_position_sizer.py

"""
VaR-Based Position Sizing System

This module provides advanced position sizing based on Value at Risk (VaR) calculations,
supporting multiple confidence levels, volatility adjustments, and correlation-aware
risk budgeting for optimal portfolio risk management.

Key Features:
- 95% and 99% confidence level VaR calculations
- Volatility-adjusted position sizing
- Correlation-aware risk budgeting
- Dynamic position sizing based on current portfolio state
- Multiple VaR methodologies (Historical, Parametric, Monte Carlo)
- Risk contribution analysis
- Portfolio optimization integration
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any

# Third-party imports
import numpy as np
from scipy import stats

# Local imports
# SECURITY FIX: Import secure random for G2.4 vulnerability fix
from main.utils.core import secure_numpy_uniform


class VaRMethod(Enum):
    """VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


# Local imports
# Import risk management components
from main.feature_pipeline.calculators.risk import RiskConfig, RiskMetricsFacade
from main.models.common import OrderSide, Position

# Import position management
from main.trading_engine.core.position_manager import PositionManager as UnifiedPositionManager

# Import cache components
from main.utils.cache import CacheType

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methodologies."""

    FIXED_PERCENTAGE = "fixed_percentage"
    VAR_TARGET = "var_target"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    CORRELATION_ADJUSTED = "correlation_adjusted"


class RiskBudgetingMethod(Enum):
    """Risk budgeting methodologies."""

    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    RISK_WEIGHTED = "risk_weighted"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CORRELATION_WEIGHTED = "correlation_weighted"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""

    # VaR settings
    var_confidence_levels: list[float] = field(default_factory=lambda: [0.95, 0.99])
    var_method: VaRMethod = VaRMethod.HISTORICAL
    var_time_horizon: int = 1  # days

    # Risk budgeting
    portfolio_var_target: float = 0.02  # 2% of portfolio
    max_position_var: float = 0.005  # 0.5% of portfolio per position
    risk_budget_method: RiskBudgetingMethod = RiskBudgetingMethod.EQUAL_RISK_CONTRIBUTION

    # Position constraints
    min_position_size: Decimal = Decimal("100")  # Minimum position value
    max_position_size: Decimal = Decimal("50000")  # Maximum position value
    max_portfolio_weight: float = 0.15  # 15% max per position
    min_portfolio_weight: float = 0.01  # 1% min per position

    # Volatility adjustments
    volatility_lookback: int = 60  # days
    volatility_adjustment: bool = True
    volatility_floor: float = 0.05  # 5% minimum volatility
    volatility_cap: float = 1.0  # 100% maximum volatility

    # Correlation settings
    correlation_lookback: int = 120  # days
    correlation_threshold: float = 0.7  # High correlation threshold
    correlation_adjustment: bool = True

    # Rebalancing
    rebalance_threshold: float = 0.05  # 5% drift threshold
    max_turnover: float = 0.20  # 20% max turnover per rebalance


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation with detailed analysis."""

    symbol: str
    side: OrderSide

    # Recommended sizing
    recommended_quantity: Decimal
    recommended_value: Decimal
    portfolio_weight: float

    # Risk metrics
    position_var_95: Decimal
    position_var_99: Decimal
    incremental_var: Decimal
    marginal_var: Decimal

    # Risk contribution
    var_contribution_pct: float
    risk_budget_utilization: float

    # Volatility analysis
    asset_volatility: float
    adjusted_volatility: float
    volatility_rank: float

    # Correlation analysis
    avg_correlation: float
    max_correlation: float
    correlation_adjustment_factor: float

    # Constraints
    constraint_violations: list[str] = field(default_factory=list)
    sizing_warnings: list[str] = field(default_factory=list)

    # Metadata
    sizing_method: PositionSizingMethod = PositionSizingMethod.VAR_TARGET
    confidence_score: float = 0.0
    calculation_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Alternative sizing scenarios
    conservative_size: Decimal | None = None
    aggressive_size: Decimal | None = None


@dataclass
class PortfolioRiskBudget:
    """Portfolio-level risk budget allocation."""

    total_var_budget: Decimal
    allocated_var: Decimal
    remaining_var_budget: Decimal

    # Risk allocation by position
    position_risk_budgets: dict[str, Decimal] = field(default_factory=dict)
    position_var_contributions: dict[str, Decimal] = field(default_factory=dict)

    # Risk concentration
    concentration_score: float = 0.0
    diversification_ratio: float = 0.0

    # Budget utilization
    budget_utilization_pct: float = 0.0
    over_budget_positions: list[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class VaRPositionSizer:
    """
    Advanced position sizing system based on VaR calculations and risk budgeting.
    """

    def __init__(
        self, position_manager: UnifiedPositionManager, config: PositionSizingConfig | None = None
    ):
        """
        Initialize VaR-based position sizer.

        Args:
            position_manager: Position manager for portfolio state
            config: Position sizing configuration
        """
        self.position_manager = position_manager
        self.config = config or PositionSizingConfig()

        # Initialize risk metrics calculator
        risk_config = RiskConfig(
            var_confidence_levels=self.config.var_confidence_levels,
            var_time_horizons=[self.config.var_time_horizon],
        )
        self.risk_metrics = RiskMetricsFacade(risk_config)

        # Market data cache for calculations
        self.cache = get_global_cache()
        self.correlation_matrix: np.ndarray | None = None

        # Risk budget tracking
        self.current_risk_budget: PortfolioRiskBudget | None = None
        self.last_rebalance: datetime | None = None

        logger.info("VaRPositionSizer initialized with advanced risk budgeting")

    async def calculate_position_size(
        self,
        symbol: str,
        side: OrderSide,
        target_price: float | Decimal,
        strategy_signal_strength: float = 1.0,
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size using VaR-based methodology.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            target_price: Target execution price
            strategy_signal_strength: Signal strength (0-1)

        Returns:
            Position sizing recommendation with detailed analysis
        """
        target_price = Decimal(str(target_price))

        try:
            # 1. Get current portfolio state
            portfolio_value = await self.position_manager.get_portfolio_value()
            current_positions = self.position_manager.get_all_positions()

            # 2. Update risk budget
            await self._update_risk_budget(portfolio_value, current_positions)

            # 3. Get asset returns and volatility
            asset_returns = await self._get_asset_returns(symbol)
            asset_volatility = self._calculate_asset_volatility(asset_returns)

            # 4. Calculate correlation adjustments
            correlation_data = await self._calculate_correlation_adjustments(
                symbol, current_positions
            )

            # 5. Calculate base position size using primary method
            base_size = await self._calculate_base_position_size(
                symbol, target_price, portfolio_value, asset_volatility
            )

            # 6. Apply adjustments
            adjusted_size = self._apply_sizing_adjustments(
                base_size, asset_volatility, correlation_data, strategy_signal_strength
            )

            # 7. Validate constraints
            final_size, violations, warnings = self._validate_position_constraints(
                adjusted_size, target_price, portfolio_value, symbol
            )

            # 8. Calculate risk metrics for final size
            risk_metrics = await self._calculate_position_risk_metrics(
                symbol, final_size, target_price, portfolio_value
            )

            # 9. Generate alternative scenarios
            conservative_size, aggressive_size = self._generate_alternative_sizes(
                final_size, risk_metrics, portfolio_value
            )

            # 10. Create recommendation
            recommendation = PositionSizeRecommendation(
                symbol=symbol,
                side=side,
                recommended_quantity=final_size,
                recommended_value=final_size * target_price,
                portfolio_weight=(
                    float((final_size * target_price) / portfolio_value)
                    if portfolio_value > 0
                    else 0
                ),
                position_var_95=risk_metrics.get("var_95", Decimal("0")),
                position_var_99=risk_metrics.get("var_99", Decimal("0")),
                incremental_var=risk_metrics.get("incremental_var", Decimal("0")),
                marginal_var=risk_metrics.get("marginal_var", Decimal("0")),
                var_contribution_pct=risk_metrics.get("var_contribution_pct", 0.0),
                risk_budget_utilization=risk_metrics.get("budget_utilization", 0.0),
                asset_volatility=asset_volatility,
                adjusted_volatility=correlation_data.get("adjusted_volatility", asset_volatility),
                volatility_rank=self._calculate_volatility_rank(symbol, asset_volatility),
                avg_correlation=correlation_data.get("avg_correlation", 0.0),
                max_correlation=correlation_data.get("max_correlation", 0.0),
                correlation_adjustment_factor=correlation_data.get("adjustment_factor", 1.0),
                constraint_violations=violations,
                sizing_warnings=warnings,
                sizing_method=PositionSizingMethod.VAR_TARGET,
                confidence_score=self._calculate_sizing_confidence(
                    final_size, risk_metrics, violations, warnings
                ),
                conservative_size=conservative_size,
                aggressive_size=aggressive_size,
            )

            logger.info(
                f"Position size calculated for {symbol}: {final_size} shares "
                f"(${final_size * target_price:,.2f}, {recommendation.portfolio_weight:.2%} of portfolio)"
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}", exc_info=True)

            # Return minimal safe size on error
            return PositionSizeRecommendation(
                symbol=symbol,
                side=side,
                recommended_quantity=Decimal("1"),
                recommended_value=target_price,
                portfolio_weight=0.0,
                position_var_95=Decimal("0"),
                position_var_99=Decimal("0"),
                incremental_var=Decimal("0"),
                marginal_var=Decimal("0"),
                var_contribution_pct=0.0,
                risk_budget_utilization=0.0,
                asset_volatility=0.0,
                adjusted_volatility=0.0,
                volatility_rank=0.0,
                avg_correlation=0.0,
                max_correlation=0.0,
                correlation_adjustment_factor=1.0,
                constraint_violations=[f"Calculation error: {e!s}"],
                confidence_score=0.0,
            )

    async def _update_risk_budget(
        self, portfolio_value: Decimal, current_positions: dict[str, Position]
    ):
        """Update portfolio risk budget allocation."""

        # Calculate total VaR budget
        total_var_budget = portfolio_value * Decimal(str(self.config.portfolio_var_target))

        # Calculate current VaR contributions
        position_vars = {}
        total_allocated_var = Decimal("0")

        if current_positions:
            # Placeholder for actual VaR calculation
            # In practice, this would calculate each position's VaR contribution
            for symbol, position in current_positions.items():
                if not position.is_flat:
                    # Simplified VaR estimation
                    position_var = abs(position.market_value) * Decimal("0.02")  # 2% estimate
                    position_vars[symbol] = position_var
                    total_allocated_var += position_var

        # Update risk budget
        self.current_risk_budget = PortfolioRiskBudget(
            total_var_budget=total_var_budget,
            allocated_var=total_allocated_var,
            remaining_var_budget=total_var_budget - total_allocated_var,
            position_var_contributions=position_vars,
            budget_utilization_pct=(
                float(total_allocated_var / total_var_budget * 100) if total_var_budget > 0 else 0
            ),
            concentration_score=self._calculate_concentration_score(position_vars),
            diversification_ratio=self._calculate_diversification_ratio(current_positions),
        )

    async def _get_asset_returns(self, symbol: str) -> np.ndarray:
        """Get historical returns for asset (placeholder implementation)."""

        # In practice, this would fetch real historical data
        # For now, generate synthetic returns for demonstration
        cache_key = f"returns:{symbol}:{self.config.volatility_lookback}"
        cached_returns = await self.cache.get(CacheType.FEATURES, cache_key)

        if cached_returns is None:
            np.random.seed(hash(symbol) % 1000)  # Consistent randomness per symbol
            returns = secure_numpy_normal(0.001, 0.02, self.config.volatility_lookback)
            # Cache for 1 hour
            await self.cache.set(CacheType.FEATURES, cache_key, returns.tolist(), 3600)
            return returns

        return np.array(cached_returns)

    def _calculate_asset_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility for asset."""

        if len(returns) < 10:
            return self.config.volatility_floor

        daily_vol = np.std(returns, ddof=1)
        annualized_vol = daily_vol * np.sqrt(252)

        # Apply volatility constraints
        vol = max(self.config.volatility_floor, min(self.config.volatility_cap, annualized_vol))

        return vol

    async def _calculate_correlation_adjustments(
        self, symbol: str, current_positions: dict[str, Position]
    ) -> dict[str, Any]:
        """Calculate correlation-based adjustments."""

        if not current_positions or not self.config.correlation_adjustment:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "adjustment_factor": 1.0,
                "adjusted_volatility": await self._get_cached_volatility(symbol),
            }

        # Get returns for current positions
        position_symbols = [pos.symbol for pos in current_positions.values() if not pos.is_flat]

        if not position_symbols:
            return {
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "adjustment_factor": 1.0,
                "adjusted_volatility": await self._get_cached_volatility(symbol),
            }

        # Calculate correlations (simplified)
        correlations = []
        for pos_symbol in position_symbols:
            if pos_symbol != symbol:
                # Simplified correlation calculation
                # SECURITY FIX: G2.4 - Replace insecure np.secure_uniform() with cryptographically secure alternative
                corr = secure_numpy_uniform(-0.5, 0.8)  # Placeholder
                correlations.append(corr)

        if correlations:
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)

            # Adjustment factor based on correlation
            adjustment_factor = 1.0
            if max_correlation > self.config.correlation_threshold:
                adjustment_factor = (
                    1.0 - (max_correlation - self.config.correlation_threshold) * 0.5
                )

            # Adjusted volatility
            base_vol = await self._get_cached_volatility(symbol)
            adjusted_vol = base_vol * (1 + avg_correlation * 0.2)  # Correlation adjustment

            return {
                "avg_correlation": avg_correlation,
                "max_correlation": max_correlation,
                "adjustment_factor": adjustment_factor,
                "adjusted_volatility": adjusted_vol,
            }

        return {
            "avg_correlation": 0.0,
            "max_correlation": 0.0,
            "adjustment_factor": 1.0,
            "adjusted_volatility": await self._get_cached_volatility(symbol),
        }

    async def _calculate_base_position_size(
        self, symbol: str, target_price: Decimal, portfolio_value: Decimal, asset_volatility: float
    ) -> Decimal:
        """Calculate base position size using VaR targeting."""

        # Target VaR for this position
        target_position_var = min(
            self.config.max_position_var * float(portfolio_value),
            float(self.current_risk_budget.remaining_var_budget) * 0.5,  # Don't use all remaining
        )

        if target_position_var <= 0:
            return Decimal("0")

        # VaR-based sizing
        # VaR = Position_Value * Z_score * Volatility * sqrt(time_horizon)
        z_score = stats.norm.ppf(
            self.config.var_confidence_levels[0]
        )  # Use primary confidence level
        time_factor = np.sqrt(self.config.var_time_horizon)

        # Solve for position value: Position_Value = VaR / (Z_score * Volatility * time_factor)
        var_factor = abs(z_score) * asset_volatility * time_factor

        if var_factor > 0:
            target_position_value = target_position_var / var_factor
            base_quantity = Decimal(str(target_position_value)) / target_price
        else:
            base_quantity = Decimal("0")

        return max(Decimal("0"), base_quantity)

    def _apply_sizing_adjustments(
        self,
        base_size: Decimal,
        asset_volatility: float,
        correlation_data: dict[str, Any],
        signal_strength: float,
    ) -> Decimal:
        """Apply various adjustments to base position size."""

        adjusted_size = base_size

        # 1. Volatility adjustment
        if self.config.volatility_adjustment:
            # Reduce size for high volatility assets
            vol_adjustment = min(1.0, self.config.volatility_floor / asset_volatility)
            adjusted_size *= Decimal(str(vol_adjustment))

        # 2. Correlation adjustment
        correlation_adjustment = correlation_data.get("adjustment_factor", 1.0)
        adjusted_size *= Decimal(str(correlation_adjustment))

        # 3. Signal strength adjustment
        signal_adjustment = max(0.1, min(1.0, signal_strength))  # Constrain to 10%-100%
        adjusted_size *= Decimal(str(signal_adjustment))

        return adjusted_size

    def _validate_position_constraints(
        self, size: Decimal, price: Decimal, portfolio_value: Decimal, symbol: str
    ) -> tuple[Decimal, list[str], list[str]]:
        """Validate and adjust position size against constraints."""

        violations = []
        warnings = []
        final_size = size
        position_value = size * price

        # 1. Minimum size constraint
        if position_value < self.config.min_position_size:
            if self.config.min_position_size / price <= size * Decimal("2"):
                final_size = self.config.min_position_size / price
                warnings.append(
                    f"Position size increased to minimum: ${self.config.min_position_size}"
                )
            else:
                final_size = Decimal("0")
                violations.append("Position below minimum viable size")

        # 2. Maximum size constraint
        if position_value > self.config.max_position_size:
            final_size = self.config.max_position_size / price
            violations.append(f"Position size capped at maximum: ${self.config.max_position_size}")

        # 3. Portfolio weight constraints
        portfolio_weight = float(position_value / portfolio_value) if portfolio_value > 0 else 0

        if portfolio_weight > self.config.max_portfolio_weight:
            max_value = portfolio_value * Decimal(str(self.config.max_portfolio_weight))
            final_size = max_value / price
            violations.append(f"Position weight capped at {self.config.max_portfolio_weight:.1%}")

        if portfolio_weight < self.config.min_portfolio_weight and final_size > 0:
            min_value = portfolio_value * Decimal(str(self.config.min_portfolio_weight))
            if min_value <= self.config.max_position_size:
                final_size = min_value / price
                warnings.append(
                    f"Position size increased to minimum weight: {self.config.min_portfolio_weight:.1%}"
                )

        # 4. Risk budget constraint
        if self.current_risk_budget:
            remaining_budget = self.current_risk_budget.remaining_var_budget
            estimated_var = position_value * Decimal("0.02")  # Simplified estimate

            if estimated_var > remaining_budget:
                if remaining_budget > 0:
                    risk_adjusted_value = remaining_budget / Decimal("0.02")
                    final_size = risk_adjusted_value / price
                    violations.append("Position size reduced due to risk budget constraints")
                else:
                    final_size = Decimal("0")
                    violations.append("No remaining risk budget for new positions")

        return final_size, violations, warnings

    async def _calculate_position_risk_metrics(
        self, symbol: str, quantity: Decimal, price: Decimal, portfolio_value: Decimal
    ) -> dict[str, Any]:
        """Calculate comprehensive risk metrics for position."""

        position_value = quantity * price

        # Simplified risk calculations (would use actual VaR models in practice)
        asset_volatility = await self._get_cached_volatility(symbol)

        # VaR calculations
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        var_95 = position_value * Decimal(str(abs(z_95) * asset_volatility))
        var_99 = position_value * Decimal(str(abs(z_99) * asset_volatility))

        # Risk contribution
        var_contribution_pct = float(var_95 / portfolio_value * 100) if portfolio_value > 0 else 0

        # Budget utilization
        if self.current_risk_budget and self.current_risk_budget.total_var_budget > 0:
            budget_utilization = float(var_95 / self.current_risk_budget.total_var_budget * 100)
        else:
            budget_utilization = 0.0

        return {
            "var_95": var_95,
            "var_99": var_99,
            "incremental_var": var_95,  # Simplified
            "marginal_var": var_95,  # Simplified
            "var_contribution_pct": var_contribution_pct,
            "budget_utilization": budget_utilization,
        }

    def _generate_alternative_sizes(
        self, base_size: Decimal, risk_metrics: dict[str, Any], portfolio_value: Decimal
    ) -> tuple[Decimal | None, Decimal | None]:
        """Generate conservative and aggressive sizing alternatives."""

        conservative_size = base_size * Decimal("0.7")  # 70% of base
        aggressive_size = base_size * Decimal("1.3")  # 130% of base

        # Validate alternatives against constraints
        max_portfolio_weight = self.config.max_portfolio_weight
        max_weight_value = portfolio_value * Decimal(str(max_portfolio_weight))

        # Don't suggest aggressive if it would exceed portfolio weight limit
        if aggressive_size and risk_metrics.get(
            "var_95", Decimal("0")
        ) > max_weight_value * Decimal("0.02"):
            aggressive_size = None

        return conservative_size, aggressive_size

    async def _calculate_volatility_rank(self, symbol: str, volatility: float) -> float:
        """Calculate volatility percentile rank among all assets."""

        # Simplified ranking (would use actual universe in practice)
        # Get current volatility for comparison
        current_vol = await self._get_cached_volatility(symbol)

        # For simplicity, return a fixed percentile based on volatility level
        if current_vol < 0.01:
            return 0.2  # Low volatility
        elif current_vol < 0.03:
            return 0.5  # Medium volatility
        else:
            return 0.8  # High volatility

        rank = sum(1 for v in all_volatilities if v < volatility) / len(all_volatilities)
        return rank

    def _calculate_sizing_confidence(
        self,
        size: Decimal,
        risk_metrics: dict[str, Any],
        violations: list[str],
        warnings: list[str],
    ) -> float:
        """Calculate confidence score for sizing recommendation."""

        confidence = 100.0

        # Reduce confidence for violations and warnings
        confidence -= len(violations) * 25
        confidence -= len(warnings) * 10

        # Reduce confidence if size is at constraints
        if size <= Decimal("1"):
            confidence -= 30

        # Reduce confidence for high risk budget utilization
        budget_util = risk_metrics.get("budget_utilization", 0)
        if budget_util > 80:
            confidence -= 20
        elif budget_util > 60:
            confidence -= 10

        return max(0.0, min(100.0, confidence))

    def _calculate_concentration_score(self, position_vars: dict[str, Decimal]) -> float:
        """Calculate portfolio concentration score."""

        if not position_vars:
            return 0.0

        # Herfindahl-Hirschman Index
        total_var = sum(position_vars.values())
        if total_var == 0:
            return 0.0

        weights = [float(var / total_var) for var in position_vars.values()]
        hhi = sum(w**2 for w in weights)

        # Normalize to 0-100 scale
        return hhi * 100

    def _calculate_diversification_ratio(self, positions: dict[str, Position]) -> float:
        """Calculate portfolio diversification ratio."""

        if len(positions) <= 1:
            return 0.0

        # Simplified diversification ratio
        # In practice, would use actual correlation matrix
        effective_positions = len([p for p in positions.values() if not p.is_flat])
        max_positions = 20  # Assumed maximum for scaling

        return min(1.0, effective_positions / max_positions)

    # Public API methods

    def get_current_risk_budget(self) -> PortfolioRiskBudget | None:
        """Get current portfolio risk budget."""
        return self.current_risk_budget

    async def rebalance_portfolio_risk(self) -> dict[str, PositionSizeRecommendation]:
        """Recommend position rebalancing based on risk budget."""

        recommendations = {}
        current_positions = self.position_manager.get_all_positions()

        if not current_positions:
            return recommendations

        # Check if rebalancing is needed
        if not self._needs_rebalancing():
            return recommendations

        # Generate rebalancing recommendations for each position
        for symbol, position in current_positions.items():
            if not position.is_flat:
                try:
                    rebalance_rec = await self.calculate_position_size(
                        symbol=symbol,
                        side=OrderSide.BUY if position.side.value == "long" else OrderSide.SELL,
                        target_price=position.current_price,
                    )

                    # Check if significant change is recommended
                    current_size = position.abs_quantity
                    recommended_size = rebalance_rec.recommended_quantity

                    size_change_pct = (
                        float(abs(recommended_size - current_size) / current_size)
                        if current_size > 0
                        else 0
                    )

                    if size_change_pct > self.config.rebalance_threshold:
                        recommendations[symbol] = rebalance_rec

                except Exception as e:
                    logger.error(f"Error calculating rebalancing for {symbol}: {e}")

        self.last_rebalance = datetime.now(UTC)

        return recommendations

    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing."""

        if not self.last_rebalance:
            return True

        # Check time since last rebalance
        time_since_rebalance = datetime.now(UTC) - self.last_rebalance
        if time_since_rebalance > timedelta(days=7):  # Weekly rebalancing
            return True

        # Check risk budget utilization
        if self.current_risk_budget and self.current_risk_budget.budget_utilization_pct > 90:
            return True

        return False

    async def _get_cached_volatility(self, symbol: str) -> float:
        """Get cached volatility for a symbol."""
        cache_key = f"volatility:{symbol}"
        cached_vol = await self.cache.get(CacheType.METRICS, cache_key)

        if cached_vol is None:
            # Calculate volatility from returns
            returns = await self._get_historical_returns(symbol)
            volatility = self._calculate_asset_volatility(returns)

            # Cache for 30 minutes
            await self.cache.set(CacheType.METRICS, cache_key, volatility, 1800)
            return volatility

        return cached_vol

    async def _get_cache_info(self) -> dict[str, Any]:
        """Get cache information for statistics."""
        try:
            # Get cache size info if available
            if hasattr(self.cache, "get_stats"):
                cache_stats = await self.cache.get_stats()
                return cache_stats
            else:
                return {"status": "cache_available"}
        except Exception:
            return {"status": "cache_unavailable"}

    async def get_sizing_statistics(self) -> dict[str, Any]:
        """Get position sizing statistics."""

        budget = self.current_risk_budget

        return {
            "config": {
                "var_confidence_levels": self.config.var_confidence_levels,
                "portfolio_var_target": self.config.portfolio_var_target,
                "max_position_var": self.config.max_position_var,
                "max_portfolio_weight": self.config.max_portfolio_weight,
            },
            "current_budget": {
                "total_var_budget": float(budget.total_var_budget) if budget else 0,
                "allocated_var": float(budget.allocated_var) if budget else 0,
                "remaining_budget": float(budget.remaining_var_budget) if budget else 0,
                "utilization_pct": budget.budget_utilization_pct if budget else 0,
                "concentration_score": budget.concentration_score if budget else 0,
                "diversification_ratio": budget.diversification_ratio if budget else 0,
            },
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
            "cache_info": await self._get_cache_info(),
        }
