# File: risk_management/real_time/position_liquidator.py

"""
Automated Position Liquidation System for Emergency Risk Management

This module provides comprehensive emergency position liquidation capabilities
with multiple liquidation strategies, market impact minimization, and
portfolio rebalancing for the AI Trader V3 system.

Key Features:
- Emergency liquidation triggers for critical risk breaches
- Gradual position reduction strategies (TWAP, VWAP, Iceberg)
- Market impact minimization during liquidation
- Portfolio rebalancing after emergency events
- Real-time liquidation progress monitoring
- Risk-adjusted liquidation prioritization
- Partial liquidation and scaling strategies
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LiquidationStrategy(Enum):
    """Liquidation strategy types."""

    IMMEDIATE = "immediate"  # Market orders, fastest execution
    GRADUAL_TWAP = "gradual_twap"  # Time-weighted average price
    GRADUAL_VWAP = "gradual_vwap"  # Volume-weighted average price
    ICEBERG = "iceberg"  # Large orders broken into smaller chunks
    ADAPTIVE = "adaptive"  # Dynamically adjust based on market conditions


class LiquidationUrgency(Enum):
    """Liquidation urgency levels."""

    LOW = "low"  # Normal position reduction
    MEDIUM = "medium"  # Elevated risk, faster liquidation
    HIGH = "high"  # Critical risk, aggressive liquidation
    EMERGENCY = "emergency"  # Immediate liquidation required


class LiquidationStatus(Enum):
    """Liquidation status types."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class LiquidationTarget:
    """Individual position liquidation target."""

    symbol: str
    current_quantity: Decimal
    target_quantity: Decimal  # Final target (can be 0 for full liquidation)
    liquidation_quantity: Decimal  # Amount to liquidate
    urgency: LiquidationUrgency
    strategy: LiquidationStrategy

    # Market data
    current_price: Decimal
    bid_ask_spread: Decimal
    avg_daily_volume: int

    # Risk metrics
    portfolio_weight: float
    risk_contribution: float
    correlation_risk: float
    liquidity_score: float

    # Execution parameters
    max_participation_rate: float = 0.20  # Max % of volume
    time_horizon_minutes: int = 30
    price_tolerance_bps: int = 50  # 50 basis points

    # Status tracking
    status: LiquidationStatus = LiquidationStatus.PENDING
    executed_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    avg_execution_price: Decimal = Decimal("0")
    market_impact_bps: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        self.remaining_quantity = self.liquidation_quantity - self.executed_quantity


@dataclass
class LiquidationPlan:
    """Complete liquidation plan for portfolio."""

    plan_id: str
    timestamp: datetime
    urgency: LiquidationUrgency
    total_portfolio_value: Decimal

    # Targets
    targets: list[LiquidationTarget]
    total_liquidation_value: Decimal
    liquidation_percentage: float

    # Execution parameters
    estimated_duration_minutes: int
    max_market_impact_bps: float
    rebalancing_required: bool

    # Progress tracking
    status: LiquidationStatus = LiquidationStatus.PENDING
    start_time: datetime | None = None
    completion_time: datetime | None = None
    executed_value: Decimal = Decimal("0")
    remaining_value: Decimal = Decimal("0")

    # Results
    actual_duration_minutes: float = 0.0
    actual_market_impact_bps: float = 0.0
    execution_quality_score: float = 0.0


@dataclass
class MarketImpactModel:
    """Market impact estimation model."""

    # Linear impact model parameters
    permanent_impact_coef: float = 0.1  # Permanent price impact coefficient
    temporary_impact_coef: float = 0.05  # Temporary price impact coefficient

    # Volume participation limits
    max_participation_rate: float = 0.25  # Max 25% of daily volume
    aggressive_participation_rate: float = 0.40  # For emergency liquidation

    # Spread impact
    spread_impact_multiplier: float = 0.5  # Impact from bid-ask spread

    def estimate_impact(
        self,
        liquidation_value: Decimal,
        avg_daily_volume: int,
        avg_price: Decimal,
        bid_ask_spread: Decimal,
        participation_rate: float,
    ) -> dict[str, float]:
        """Estimate market impact for liquidation."""

        # Calculate liquidation as % of daily volume
        daily_volume_value = avg_daily_volume * float(avg_price)
        volume_participation = float(liquidation_value) / daily_volume_value

        # Linear impact model
        permanent_impact = self.permanent_impact_coef * volume_participation
        temporary_impact = self.temporary_impact_coef * volume_participation

        # Spread impact
        spread_impact = float(bid_ask_spread / avg_price) * self.spread_impact_multiplier

        # Participation rate adjustment
        participation_adjustment = max(1.0, participation_rate / self.max_participation_rate)

        total_impact = (
            permanent_impact + temporary_impact + spread_impact
        ) * participation_adjustment

        return {
            "permanent_impact_bps": permanent_impact * 10000,
            "temporary_impact_bps": temporary_impact * 10000,
            "spread_impact_bps": spread_impact * 10000,
            "total_impact_bps": total_impact * 10000,
            "participation_rate": participation_rate,
            "volume_participation": volume_participation,
        }


class PositionLiquidator:
    """
    Comprehensive automated position liquidation system.

    Provides multiple liquidation strategies with market impact minimization
    and real-time progress monitoring for emergency risk management.
    """

    def __init__(self, position_manager, order_manager, config: dict[str, Any] | None = None):
        """
        Initialize position liquidator.

        Args:
            position_manager: Unified position manager
            order_manager: Order manager for execution
            config: Liquidation configuration
        """
        self.position_manager = position_manager
        self.order_manager = order_manager
        self.config = config or {}

        # Market impact model
        self.impact_model = MarketImpactModel()

        # Active liquidations
        self.active_liquidations: dict[str, LiquidationPlan] = {}
        self.liquidation_history: list[LiquidationPlan] = []

        # Execution tracking
        self.execution_callbacks: list[Callable] = []
        self.progress_callbacks: list[Callable] = []

        # Configuration parameters
        self.max_concurrent_liquidations = config.get("max_concurrent_liquidations", 5)
        self.default_time_horizon_minutes = config.get("default_time_horizon_minutes", 30)
        self.emergency_time_horizon_minutes = config.get("emergency_time_horizon_minutes", 5)
        self.rebalancing_threshold = config.get("rebalancing_threshold", 0.10)  # 10%

        # Risk thresholds
        self.max_portfolio_liquidation_pct = config.get(
            "max_portfolio_liquidation_pct", 0.50
        )  # 50%
        self.emergency_liquidation_threshold = config.get(
            "emergency_liquidation_threshold", 0.15
        )  # 15%

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._execution_tasks: dict[str, asyncio.Task] = {}

        logger.info("Position liquidator initialized")

    async def create_liquidation_plan(
        self,
        positions: dict[str, Any],
        liquidation_percentage: float,
        urgency: LiquidationUrgency = LiquidationUrgency.MEDIUM,
        target_symbols: list[str] | None = None,
    ) -> LiquidationPlan:
        """
        Create comprehensive liquidation plan.

        Args:
            positions: Current positions dictionary
            liquidation_percentage: Percentage of portfolio to liquidate (0-1)
            urgency: Liquidation urgency level
            target_symbols: Specific symbols to liquidate (None for all)

        Returns:
            Complete liquidation plan
        """

        plan_id = f"liquidation_{int(datetime.now().timestamp() * 1000)}"
        timestamp = datetime.now(UTC)

        # Get portfolio value
        portfolio_value = await self.position_manager.get_portfolio_value()
        total_liquidation_value = portfolio_value * Decimal(str(liquidation_percentage))

        logger.info(
            f"Creating liquidation plan {plan_id}: {liquidation_percentage:.1%} of portfolio (${total_liquidation_value:,.2f})"
        )

        # Select positions to liquidate
        liquidation_targets = await self._select_liquidation_targets(
            positions, liquidation_percentage, urgency, target_symbols
        )

        # Determine liquidation strategy based on urgency
        default_strategy = self._get_default_strategy(urgency)
        time_horizon = self._get_time_horizon(urgency)

        # Create targets with execution parameters
        targets = []
        for symbol, target_data in liquidation_targets.items():
            position = positions[symbol]

            # Get market data for target
            market_data = await self._get_market_data(symbol)

            # Calculate risk metrics
            risk_metrics = await self._calculate_position_risk_metrics(
                symbol, position, portfolio_value
            )

            target = LiquidationTarget(
                symbol=symbol,
                current_quantity=position.quantity,
                target_quantity=target_data["target_quantity"],
                liquidation_quantity=target_data["liquidation_quantity"],
                urgency=urgency,
                strategy=target_data.get("strategy", default_strategy),
                current_price=market_data["price"],
                bid_ask_spread=market_data["spread"],
                avg_daily_volume=market_data["avg_volume"],
                portfolio_weight=risk_metrics["portfolio_weight"],
                risk_contribution=risk_metrics["risk_contribution"],
                correlation_risk=risk_metrics["correlation_risk"],
                liquidity_score=risk_metrics["liquidity_score"],
                time_horizon_minutes=time_horizon,
            )

            targets.append(target)

        # Estimate total execution time and market impact
        estimated_duration = await self._estimate_execution_duration(targets)
        max_impact = await self._estimate_max_market_impact(targets)

        # Determine if rebalancing is needed
        rebalancing_required = liquidation_percentage > self.rebalancing_threshold

        # Create liquidation plan
        plan = LiquidationPlan(
            plan_id=plan_id,
            timestamp=timestamp,
            urgency=urgency,
            total_portfolio_value=portfolio_value,
            targets=targets,
            total_liquidation_value=total_liquidation_value,
            liquidation_percentage=liquidation_percentage,
            estimated_duration_minutes=estimated_duration,
            max_market_impact_bps=max_impact,
            rebalancing_required=rebalancing_required,
        )

        plan.remaining_value = total_liquidation_value

        logger.info(
            f"Liquidation plan created: {len(targets)} positions, estimated duration {estimated_duration} minutes"
        )

        return plan

    async def execute_liquidation_plan(self, plan: LiquidationPlan) -> bool:
        """
        Execute liquidation plan with real-time monitoring.

        Args:
            plan: Liquidation plan to execute

        Returns:
            True if execution started successfully
        """

        if plan.plan_id in self.active_liquidations:
            logger.warning(f"Liquidation plan {plan.plan_id} already active")
            return False

        logger.critical(
            f"🔴 EXECUTING LIQUIDATION PLAN {plan.plan_id}: {plan.liquidation_percentage:.1%} of portfolio"
        )

        # Add to active liquidations
        self.active_liquidations[plan.plan_id] = plan
        plan.status = LiquidationStatus.IN_PROGRESS
        plan.start_time = datetime.now(UTC)

        # Start execution task
        self._execution_tasks[plan.plan_id] = asyncio.create_task(
            self._execute_liquidation_plan_async(plan)
        )

        # Notify callbacks
        for callback in self.execution_callbacks:
            try:
                callback(plan, "started")
            except Exception as e:
                logger.error(f"Error in execution callback: {e}")

        return True

    async def emergency_liquidate_all(
        self, positions: dict[str, Any], reason: str = "Emergency liquidation"
    ) -> LiquidationPlan:
        """
        Emergency liquidation of all positions.

        Args:
            positions: All current positions
            reason: Reason for emergency liquidation

        Returns:
            Emergency liquidation plan
        """

        logger.critical(f"🔴 EMERGENCY LIQUIDATION INITIATED: {reason}")

        # Create emergency plan for 100% liquidation
        plan = await self.create_liquidation_plan(
            positions=positions,
            liquidation_percentage=1.0,  # 100% liquidation
            urgency=LiquidationUrgency.EMERGENCY,
        )

        # Override strategy for all targets to immediate execution
        for target in plan.targets:
            target.strategy = LiquidationStrategy.IMMEDIATE
            target.time_horizon_minutes = 1  # Execute ASAP
            target.max_participation_rate = 0.50  # Aggressive participation

        # Execute immediately
        await self.execute_liquidation_plan(plan)

        return plan

    async def partial_liquidate_position(
        self,
        symbol: str,
        reduction_percentage: float,
        urgency: LiquidationUrgency = LiquidationUrgency.MEDIUM,
    ) -> bool:
        """
        Partially liquidate a specific position.

        Args:
            symbol: Symbol to partially liquidate
            reduction_percentage: Percentage of position to liquidate (0-1)
            urgency: Liquidation urgency

        Returns:
            True if liquidation started successfully
        """

        position = self.position_manager.get_position(symbol)
        if not position:
            logger.error(f"Cannot liquidate {symbol}: position not found")
            return False

        # Create single-position liquidation plan
        positions = {symbol: position}

        # Calculate target quantities
        current_qty = position.quantity
        liquidation_qty = current_qty * Decimal(str(reduction_percentage))
        target_qty = current_qty - liquidation_qty

        # Create mini liquidation plan
        plan = await self.create_liquidation_plan(
            positions=positions,
            liquidation_percentage=reduction_percentage,
            urgency=urgency,
            target_symbols=[symbol],
        )

        # Execute the plan
        return await self.execute_liquidation_plan(plan)

    async def _execute_liquidation_plan_async(self, plan: LiquidationPlan):
        """Asynchronous execution of liquidation plan."""

        try:
            start_time = datetime.now()

            # Sort targets by priority (urgency, risk contribution, liquidity)
            sorted_targets = self._prioritize_liquidation_targets(plan.targets)

            # Execute liquidations based on strategy
            if plan.urgency == LiquidationUrgency.EMERGENCY:
                # Parallel execution for emergency
                await self._execute_parallel_liquidation(sorted_targets)
            else:
                # Sequential execution for controlled liquidation
                await self._execute_sequential_liquidation(sorted_targets)

            # Calculate final results
            plan.completion_time = datetime.now(UTC)
            plan.actual_duration_minutes = (
                plan.completion_time - plan.start_time
            ).total_seconds() / 60

            # Update plan status
            completed_targets = [t for t in plan.targets if t.status == LiquidationStatus.COMPLETED]
            if len(completed_targets) == len(plan.targets):
                plan.status = LiquidationStatus.COMPLETED
            elif len(completed_targets) > 0:
                plan.status = LiquidationStatus.PARTIAL
            else:
                plan.status = LiquidationStatus.FAILED

            # Calculate execution quality
            plan.execution_quality_score = self._calculate_execution_quality(plan)

            # Post-liquidation rebalancing if needed
            if plan.rebalancing_required and plan.status in [
                LiquidationStatus.COMPLETED,
                LiquidationStatus.PARTIAL,
            ]:
                await self._perform_post_liquidation_rebalancing(plan)

            logger.info(
                f"Liquidation plan {plan.plan_id} completed with status {plan.status.value}"
            )

        except Exception as e:
            logger.error(f"Error executing liquidation plan {plan.plan_id}: {e}", exc_info=True)
            plan.status = LiquidationStatus.FAILED

        finally:
            # Clean up
            if plan.plan_id in self.active_liquidations:
                del self.active_liquidations[plan.plan_id]

            self.liquidation_history.append(plan)

            # Keep only recent history
            if len(self.liquidation_history) > 100:
                self.liquidation_history = self.liquidation_history[-100:]

            # Notify completion
            for callback in self.execution_callbacks:
                try:
                    callback(plan, "completed")
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")

    async def _execute_parallel_liquidation(self, targets: list[LiquidationTarget]):
        """Execute liquidation targets in parallel for emergency situations."""

        logger.info(f"Executing {len(targets)} liquidations in parallel (emergency mode)")

        # Create execution tasks for all targets
        tasks = []
        for target in targets:
            task = asyncio.create_task(self._execute_single_target(target))
            tasks.append(task)

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_sequential_liquidation(self, targets: list[LiquidationTarget]):
        """Execute liquidation targets sequentially for controlled liquidation."""

        logger.info(f"Executing {len(targets)} liquidations sequentially")

        for target in targets:
            try:
                await self._execute_single_target(target)

                # Brief pause between executions to assess market impact
                if target.strategy != LiquidationStrategy.IMMEDIATE:
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error executing target {target.symbol}: {e}")
                target.status = LiquidationStatus.FAILED

    async def _execute_single_target(self, target: LiquidationTarget):
        """Execute liquidation for a single target."""

        logger.info(
            f"Executing liquidation: {target.symbol} {target.liquidation_quantity} shares ({target.strategy.value})"
        )

        target.status = LiquidationStatus.IN_PROGRESS

        try:
            if target.strategy == LiquidationStrategy.IMMEDIATE:
                await self._execute_immediate_liquidation(target)
            elif target.strategy == LiquidationStrategy.GRADUAL_TWAP:
                await self._execute_twap_liquidation(target)
            elif target.strategy == LiquidationStrategy.GRADUAL_VWAP:
                await self._execute_vwap_liquidation(target)
            elif target.strategy == LiquidationStrategy.ICEBERG:
                await self._execute_iceberg_liquidation(target)
            elif target.strategy == LiquidationStrategy.ADAPTIVE:
                await self._execute_adaptive_liquidation(target)

            # Mark as completed if fully executed
            if target.executed_quantity >= target.liquidation_quantity * Decimal(
                "0.95"
            ):  # 95% threshold
                target.status = LiquidationStatus.COMPLETED
            else:
                target.status = LiquidationStatus.PARTIAL

        except Exception as e:
            logger.error(f"Error executing {target.symbol} liquidation: {e}")
            target.status = LiquidationStatus.FAILED

    async def _execute_immediate_liquidation(self, target: LiquidationTarget):
        """Execute immediate market order liquidation."""

        # Create market sell order
        order_side = "sell" if target.liquidation_quantity > 0 else "buy"

        order = await self.order_manager.submit_market_order(
            symbol=target.symbol, side=order_side, quantity=abs(float(target.liquidation_quantity))
        )

        if order:
            # Track execution (simplified)
            target.executed_quantity = target.liquidation_quantity
            target.avg_execution_price = target.current_price  # Simplified
            logger.info(f"Immediate liquidation executed: {target.symbol}")

    async def _execute_twap_liquidation(self, target: LiquidationTarget):
        """Execute time-weighted average price liquidation."""

        # Split into time slices
        num_slices = min(10, target.time_horizon_minutes)
        slice_quantity = target.liquidation_quantity / num_slices
        slice_interval = (target.time_horizon_minutes * 60) / num_slices

        logger.info(
            f"TWAP liquidation: {target.symbol} in {num_slices} slices over {target.time_horizon_minutes} minutes"
        )

        for i in range(num_slices):
            try:
                # Submit slice order
                order_side = "sell" if slice_quantity > 0 else "buy"

                order = await self.order_manager.submit_limit_order(
                    symbol=target.symbol,
                    side=order_side,
                    quantity=abs(float(slice_quantity)),
                    price=float(target.current_price),  # Could be more sophisticated
                )

                if order:
                    target.executed_quantity += slice_quantity

                # Wait for next slice (except last one)
                if i < num_slices - 1:
                    await asyncio.sleep(slice_interval)

            except Exception as e:
                logger.error(f"Error in TWAP slice {i} for {target.symbol}: {e}")
                break

    async def _execute_vwap_liquidation(self, target: LiquidationTarget):
        """Execute volume-weighted average price liquidation."""

        # This would implement VWAP strategy based on historical volume patterns
        # Simplified implementation here
        await self._execute_twap_liquidation(target)  # Fallback to TWAP

    async def _execute_iceberg_liquidation(self, target: LiquidationTarget):
        """Execute iceberg order liquidation."""

        # Break large order into smaller visible chunks
        chunk_size = min(
            target.liquidation_quantity / 5,  # 5 chunks
            Decimal(str(target.avg_daily_volume * 0.01)),  # 1% of daily volume
        )

        remaining = target.liquidation_quantity

        while remaining > 0:
            current_chunk = min(chunk_size, remaining)

            order_side = "sell" if current_chunk > 0 else "buy"

            order = await self.order_manager.submit_limit_order(
                symbol=target.symbol,
                side=order_side,
                quantity=abs(float(current_chunk)),
                price=float(target.current_price),
            )

            if order:
                target.executed_quantity += current_chunk
                remaining -= current_chunk
            else:
                break

            # Brief pause between chunks
            await asyncio.sleep(5)

    async def _execute_adaptive_liquidation(self, target: LiquidationTarget):
        """Execute adaptive liquidation based on real-time market conditions."""

        # Assess current market conditions and choose best strategy
        market_volatility = await self._assess_market_volatility(target.symbol)

        if market_volatility > 0.05:  # High volatility
            target.strategy = LiquidationStrategy.ICEBERG
            await self._execute_iceberg_liquidation(target)
        else:
            target.strategy = LiquidationStrategy.GRADUAL_TWAP
            await self._execute_twap_liquidation(target)

    # Helper methods

    async def _select_liquidation_targets(
        self,
        positions: dict[str, Any],
        liquidation_percentage: float,
        urgency: LiquidationUrgency,
        target_symbols: list[str] | None,
    ) -> dict[str, dict]:
        """Select and prioritize positions for liquidation."""

        targets = {}

        if target_symbols:
            # Liquidate specific symbols
            for symbol in target_symbols:
                if symbol in positions:
                    position = positions[symbol]
                    targets[symbol] = {
                        "target_quantity": Decimal("0"),  # Full liquidation
                        "liquidation_quantity": position.quantity,
                    }
        else:
            # Select positions based on risk and liquidation criteria
            portfolio_value = await self.position_manager.get_portfolio_value()
            target_liquidation_value = portfolio_value * Decimal(str(liquidation_percentage))

            # Sort positions by liquidation priority
            position_scores = []
            for symbol, position in positions.items():
                score = await self._calculate_liquidation_priority_score(position, portfolio_value)
                position_scores.append((symbol, position, score))

            # Sort by score (highest first)
            position_scores.sort(key=lambda x: x[2], reverse=True)

            # Select positions until target liquidation value is reached
            accumulated_value = Decimal("0")
            for symbol, position, score in position_scores:
                position_value = abs(position.market_value)

                if accumulated_value + position_value <= target_liquidation_value:
                    # Full liquidation
                    targets[symbol] = {
                        "target_quantity": Decimal("0"),
                        "liquidation_quantity": position.quantity,
                    }
                    accumulated_value += position_value
                else:
                    # Partial liquidation
                    remaining_value = target_liquidation_value - accumulated_value
                    if remaining_value > 0:
                        partial_ratio = remaining_value / position_value
                        liquidation_qty = position.quantity * partial_ratio
                        targets[symbol] = {
                            "target_quantity": position.quantity - liquidation_qty,
                            "liquidation_quantity": liquidation_qty,
                        }
                    break

        return targets

    async def _calculate_liquidation_priority_score(
        self, position, portfolio_value: Decimal
    ) -> float:
        """Calculate liquidation priority score for a position."""

        # Factors: risk contribution, liquidity, size, correlation
        weight = abs(position.market_value) / portfolio_value

        # Higher score = higher priority for liquidation
        score = weight * 100  # Base score from position size

        # Adjust for risk factors (would need actual risk calculations)
        # This is simplified
        if hasattr(position, "beta") and position.beta > 1.5:
            score += 20  # High beta positions

        return score

    async def _get_market_data(self, symbol: str) -> dict[str, Any]:
        """Get market data for a symbol."""

        # This would fetch real market data
        # Placeholder implementation
        return {"price": Decimal("100.00"), "spread": Decimal("0.10"), "avg_volume": 1000000}

    async def _calculate_position_risk_metrics(
        self, symbol: str, position, portfolio_value: Decimal
    ) -> dict[str, float]:
        """Calculate risk metrics for a position."""

        return {
            "portfolio_weight": float(abs(position.market_value) / portfolio_value),
            "risk_contribution": 0.05,  # Placeholder
            "correlation_risk": 0.03,  # Placeholder
            "liquidity_score": 0.8,  # Placeholder
        }

    def _get_default_strategy(self, urgency: LiquidationUrgency) -> LiquidationStrategy:
        """Get default liquidation strategy based on urgency."""

        strategy_map = {
            LiquidationUrgency.LOW: LiquidationStrategy.GRADUAL_TWAP,
            LiquidationUrgency.MEDIUM: LiquidationStrategy.ICEBERG,
            LiquidationUrgency.HIGH: LiquidationStrategy.GRADUAL_VWAP,
            LiquidationUrgency.EMERGENCY: LiquidationStrategy.IMMEDIATE,
        }

        return strategy_map.get(urgency, LiquidationStrategy.GRADUAL_TWAP)

    def _get_time_horizon(self, urgency: LiquidationUrgency) -> int:
        """Get time horizon in minutes based on urgency."""

        horizon_map = {
            LiquidationUrgency.LOW: 60,  # 1 hour
            LiquidationUrgency.MEDIUM: 30,  # 30 minutes
            LiquidationUrgency.HIGH: 15,  # 15 minutes
            LiquidationUrgency.EMERGENCY: 1,  # 1 minute
        }

        return horizon_map.get(urgency, self.default_time_horizon_minutes)

    async def _estimate_execution_duration(self, targets: list[LiquidationTarget]) -> int:
        """Estimate total execution duration for all targets."""

        max_duration = max([target.time_horizon_minutes for target in targets])
        return max_duration

    async def _estimate_max_market_impact(self, targets: list[LiquidationTarget]) -> float:
        """Estimate maximum market impact across all targets."""

        max_impact = 0.0
        for target in targets:
            impact = self.impact_model.estimate_impact(
                liquidation_value=target.liquidation_quantity * target.current_price,
                avg_daily_volume=target.avg_daily_volume,
                avg_price=target.current_price,
                bid_ask_spread=target.bid_ask_spread,
                participation_rate=target.max_participation_rate,
            )
            max_impact = max(max_impact, impact["total_impact_bps"])

        return max_impact

    def _prioritize_liquidation_targets(
        self, targets: list[LiquidationTarget]
    ) -> list[LiquidationTarget]:
        """Prioritize liquidation targets by urgency and risk."""

        def priority_key(target):
            urgency_priority = {
                LiquidationUrgency.EMERGENCY: 4,
                LiquidationUrgency.HIGH: 3,
                LiquidationUrgency.MEDIUM: 2,
                LiquidationUrgency.LOW: 1,
            }

            return (
                urgency_priority.get(target.urgency, 0),
                target.risk_contribution,
                -target.liquidity_score,  # Higher liquidity = easier to liquidate
            )

        return sorted(targets, key=priority_key, reverse=True)

    def _calculate_execution_quality(self, plan: LiquidationPlan) -> float:
        """Calculate execution quality score (0-100)."""

        if not plan.targets:
            return 0.0

        total_score = 0.0
        for target in plan.targets:
            # Factors: completion rate, market impact, execution speed
            completion_rate = (
                float(target.executed_quantity / target.liquidation_quantity)
                if target.liquidation_quantity > 0
                else 0
            )

            # Market impact penalty
            impact_penalty = min(50, target.market_impact_bps / 10)  # 10 bps = 5 point penalty

            # Speed bonus/penalty
            time_ratio = (
                plan.actual_duration_minutes / plan.estimated_duration_minutes
                if plan.estimated_duration_minutes > 0
                else 1
            )
            speed_score = max(0, 100 - abs(time_ratio - 1) * 100)

            target_score = (completion_rate * 100 - impact_penalty + speed_score * 0.2) / 1.2
            total_score += max(0, min(100, target_score))

        return total_score / len(plan.targets)

    async def _perform_post_liquidation_rebalancing(self, plan: LiquidationPlan):
        """Perform portfolio rebalancing after significant liquidation."""

        logger.info(f"Performing post-liquidation rebalancing for plan {plan.plan_id}")

        # This would implement portfolio rebalancing logic
        # Placeholder for now
        pass

    async def _assess_market_volatility(self, symbol: str) -> float:
        """Assess current market volatility for a symbol."""

        # This would calculate real-time volatility
        # Placeholder
        return 0.02  # 2% volatility

    # Public API methods

    def get_active_liquidations(self) -> dict[str, LiquidationPlan]:
        """Get all active liquidation plans."""
        return self.active_liquidations.copy()

    def get_liquidation_history(self) -> list[LiquidationPlan]:
        """Get liquidation history."""
        return self.liquidation_history.copy()

    def add_execution_callback(self, callback: Callable):
        """Add callback for liquidation execution events."""
        self.execution_callbacks.append(callback)

    def add_progress_callback(self, callback: Callable):
        """Add callback for liquidation progress updates."""
        self.progress_callbacks.append(callback)

    async def cancel_liquidation(self, plan_id: str) -> bool:
        """Cancel an active liquidation plan."""

        if plan_id not in self.active_liquidations:
            return False

        plan = self.active_liquidations[plan_id]
        plan.status = LiquidationStatus.CANCELLED

        # Cancel execution task
        if plan_id in self._execution_tasks:
            self._execution_tasks[plan_id].cancel()

        logger.warning(f"Liquidation plan {plan_id} cancelled")
        return True

    async def get_liquidation_status(self, plan_id: str) -> dict[str, Any] | None:
        """Get detailed status of a liquidation plan."""

        plan = self.active_liquidations.get(plan_id)
        if not plan:
            # Check history
            for historical_plan in self.liquidation_history:
                if historical_plan.plan_id == plan_id:
                    plan = historical_plan
                    break

        if not plan:
            return None

        return {
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "progress_percentage": (
                float(plan.executed_value / plan.total_liquidation_value * 100)
                if plan.total_liquidation_value > 0
                else 0
            ),
            "start_time": plan.start_time.isoformat() if plan.start_time else None,
            "estimated_completion": (
                (plan.start_time + timedelta(minutes=plan.estimated_duration_minutes)).isoformat()
                if plan.start_time
                else None
            ),
            "targets_completed": len(
                [t for t in plan.targets if t.status == LiquidationStatus.COMPLETED]
            ),
            "total_targets": len(plan.targets),
            "execution_quality_score": plan.execution_quality_score,
        }
