"""
Comprehensive risk management system.

This module provides a unified risk management framework that combines
position validation, risk metrics, and real-time monitoring to ensure
safe trading operations.
"""

# Standard library imports
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from typing import Any

# Local imports
from main.models.common import AccountInfo, Order, OrderSide, OrderStatus, Position, PositionSide
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.trading_engine.core.position_risk_validator import (
    PositionRiskValidator,
    RiskLimits,
    RiskMetrics,
)
from main.trading_engine.core.position_validator import (
    PositionLimits,
    PositionValidator,
    ValidationContext,
)
from main.utils.core import (
    AITraderException,
    ValidationResult,
    create_event_tracker,
    create_task_safely,
)
from main.utils.database import DatabasePool
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class RiskManagerException(AITraderException):
    """Base exception for risk manager errors."""

    pass


class RiskLimitExceeded(RiskManagerException):
    """Raised when risk limits are exceeded."""

    pass


@dataclass
class RiskConfig:
    """Complete risk configuration."""

    position_limits: PositionLimits
    risk_limits: RiskLimits
    emergency_stop_loss: Decimal  # Emergency stop loss percentage
    daily_loss_limit: Decimal  # Maximum daily loss allowed
    circuit_breaker_threshold: Decimal  # Threshold to halt trading
    risk_check_interval: int = 60  # Seconds between risk checks


@dataclass
class RiskState:
    """Current risk state of the system."""

    timestamp: datetime
    portfolio_metrics: RiskMetrics
    daily_pnl: Decimal
    open_risk: Decimal  # Total open risk across positions
    margin_usage: Decimal  # Margin usage percentage
    violations: list[str] = field(default_factory=list)
    is_emergency_stop: bool = False
    trading_halted: bool = False


@dataclass
class RiskAlert:
    """Risk alert/warning."""

    timestamp: datetime
    severity: str  # info, warning, critical
    category: str  # position, portfolio, market, system
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Comprehensive risk management system.

    Coordinates position validation, risk metrics, real-time monitoring,
    and emergency controls to maintain safe trading operations.
    """

    def __init__(
        self,
        risk_config: RiskConfig,
        broker: BrokerInterface,
        database: DatabasePool,
        market_data_provider: Any,
        metrics_collector: MetricsCollector | None = None,
    ):
        """
        Initialize risk manager.

        Args:
            risk_config: Complete risk configuration
            broker: Broker interface
            database: Database manager
            market_data_provider: Market data provider
            metrics_collector: Optional metrics collector
        """
        self.config = risk_config
        self.broker = broker
        self.db = database
        self.market_data = market_data_provider
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("risk_manager")

        # Initialize validators
        self.position_validator = PositionValidator(risk_config.position_limits, metrics_collector)

        self.risk_validator = PositionRiskValidator(
            risk_config.risk_limits, market_data_provider, metrics_collector
        )

        # Risk state tracking
        self._current_state: RiskState | None = None
        self._risk_alerts: list[RiskAlert] = []
        self._daily_pnl: Decimal = Decimal("0")
        self._emergency_stop_triggered = False
        self._trading_halted = False

        # Alert callbacks
        self._alert_callbacks: dict[str, list[callable]] = defaultdict(list)

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start risk manager."""
        logger.info("Starting risk manager")
        self._running = True

        # Initialize daily P&L from database
        self._daily_pnl = await self._calculate_daily_pnl()

        # Start monitoring
        self._monitoring_task = create_task_safely(self._monitor_risk())

        # Perform initial risk check
        await self.check_portfolio_risk()

    async def stop(self) -> None:
        """Stop risk manager."""
        logger.info("Stopping risk manager")
        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            await asyncio.gather(self._monitoring_task, return_exceptions=True)

    async def validate_order(self, order: Order, account_info: AccountInfo) -> ValidationResult:
        """
        Validate order against all risk constraints.

        Args:
            order: Order to validate
            account_info: Current account information

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Check if trading is halted
        if self._trading_halted:
            errors.append("Trading is currently halted due to risk limits")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Get current positions
        positions = await self.broker.get_positions()
        pending_orders = await self._get_pending_orders()

        # Create validation context
        context = ValidationContext(
            account_info=account_info,
            existing_positions=positions,
            pending_orders=pending_orders,
            position_limits=self.config.position_limits,
        )

        # Position validation
        position_result = await self.position_validator.validate_order(order, context)
        errors.extend(position_result.errors)
        warnings.extend(position_result.warnings)

        # Risk validation
        if not position_result.errors:  # Only check risk if position validation passed
            position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT

            risk_result = await self.risk_validator.validate_position_risk(
                order.symbol, position_side, order.quantity, context
            )
            errors.extend(risk_result.errors)
            warnings.extend(risk_result.warnings)

        # Check daily loss limit
        if self._daily_pnl < -self.config.daily_loss_limit:
            errors.append(f"Daily loss limit exceeded: {self._daily_pnl:.2f}")

        # Check circuit breaker
        if self._daily_pnl < -self.config.circuit_breaker_threshold:
            errors.append("Circuit breaker triggered - trading halted")
            await self._trigger_circuit_breaker()

        # Record validation
        is_valid = len(errors) == 0
        self._track_validation(order, is_valid, errors, warnings)

        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    async def check_portfolio_risk(self) -> RiskState:
        """
        Perform comprehensive portfolio risk check.

        Returns:
            Current risk state
        """
        account_info = await self.broker.get_account_info()
        positions = await self.broker.get_positions()

        # Create validation context
        context = ValidationContext(
            account_info=account_info,
            existing_positions=positions,
            pending_orders=[],
            position_limits=self.config.position_limits,
        )

        # Validate portfolio
        portfolio_result = await self.risk_validator.validate_portfolio_risk(context)

        # Calculate portfolio metrics
        portfolio_metrics = await self.risk_validator._calculate_portfolio_metrics(positions)

        # Calculate open risk
        open_risk = await self._calculate_open_risk(positions)

        # Calculate margin usage
        margin_usage = self._calculate_margin_usage(account_info, positions)

        # Update daily P&L
        self._daily_pnl = await self._calculate_daily_pnl()

        # Check for emergency conditions
        violations = portfolio_result.errors + portfolio_result.warnings
        is_emergency = self._check_emergency_conditions(
            portfolio_metrics, self._daily_pnl, margin_usage
        )

        # Create risk state
        self._current_state = RiskState(
            timestamp=datetime.utcnow(),
            portfolio_metrics=portfolio_metrics,
            daily_pnl=self._daily_pnl,
            open_risk=open_risk,
            margin_usage=margin_usage,
            violations=violations,
            is_emergency_stop=is_emergency,
            trading_halted=self._trading_halted,
        )

        # Generate alerts
        await self._generate_risk_alerts(self._current_state)

        # Track metrics
        if self.metrics:
            self.metrics.gauge("risk_manager.daily_pnl", float(self._daily_pnl))
            self.metrics.gauge("risk_manager.open_risk", float(open_risk))
            self.metrics.gauge("risk_manager.margin_usage", float(margin_usage))
            self.metrics.gauge("risk_manager.violations", len(violations))

        return self._current_state

    async def handle_emergency_stop(self) -> None:
        """Execute emergency stop - close all positions."""
        logger.warning("EMERGENCY STOP TRIGGERED - Closing all positions")

        self._emergency_stop_triggered = True
        self._trading_halted = True

        # Create critical alert
        alert = RiskAlert(
            timestamp=datetime.utcnow(),
            severity="critical",
            category="system",
            message="Emergency stop triggered - closing all positions",
            details={"daily_pnl": float(self._daily_pnl), "trigger": "emergency_stop"},
        )
        await self._process_alert(alert)

        # Close all positions
        try:
            order_ids = await self.broker.close_all_positions()
            logger.info(f"Emergency stop: submitted {len(order_ids)} closing orders")

            # Track event
            self.event_tracker.track(
                "emergency_stop",
                {"positions_closed": len(order_ids), "daily_pnl": float(self._daily_pnl)},
            )

        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

    def register_alert_callback(self, severity: str, callback: callable) -> None:
        """Register callback for risk alerts."""
        self._alert_callbacks[severity].append(callback)

    async def get_risk_state(self) -> RiskState | None:
        """Get current risk state."""
        return self._current_state

    async def get_recent_alerts(
        self, hours: int = 24, severity: str | None = None
    ) -> list[RiskAlert]:
        """Get recent risk alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        alerts = [a for a in self._risk_alerts if a.timestamp >= cutoff]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    async def override_halt(self, reason: str) -> None:
        """Override trading halt (admin function)."""
        logger.warning(f"Trading halt override: {reason}")

        self._trading_halted = False

        alert = RiskAlert(
            timestamp=datetime.utcnow(),
            severity="warning",
            category="system",
            message=f"Trading halt overridden: {reason}",
            details={"override_reason": reason},
        )

        await self._process_alert(alert)

    async def _monitor_risk(self) -> None:
        """Background risk monitoring task."""
        while self._running:
            try:
                # Perform risk check
                await self.check_portfolio_risk()

                # Wait for next check
                await asyncio.sleep(self.config.risk_check_interval)

            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(self.config.risk_check_interval)

    async def _calculate_daily_pnl(self) -> Decimal:
        """Calculate P&L for current day."""
        try:
            # Get today's trades
            today = datetime.utcnow().date()
            fills = await self.db.get_fills_by_date(today)

            # Calculate realized P&L
            realized_pnl = Decimal("0")
            for fill in fills:
                if fill.side == OrderSide.SELL:
                    # For sells, calculate P&L
                    cost_basis = await self.db.get_position_cost_basis(fill.symbol, fill.timestamp)
                    if cost_basis:
                        pnl = (fill.price - cost_basis) * fill.quantity
                        realized_pnl += pnl

            # Get unrealized P&L
            positions = await self.broker.get_positions()
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)

            return realized_pnl + unrealized_pnl

        except Exception as e:
            logger.error(f"Error calculating daily P&L: {e}")
            return Decimal("0")

    async def _calculate_open_risk(self, positions: list[Position]) -> Decimal:
        """Calculate total open risk across positions."""
        total_risk = Decimal("0")

        for position in positions:
            # Simple risk calculation: position value * volatility estimate
            position_risk = position.market_value * Decimal("0.02")  # 2% daily vol estimate
            total_risk += position_risk

        return total_risk

    def _calculate_margin_usage(
        self, account_info: AccountInfo, positions: list[Position]
    ) -> Decimal:
        """Calculate margin usage percentage."""
        if account_info.buying_power <= 0:
            return Decimal("100")

        total_position_value = sum(p.market_value for p in positions)
        equity = account_info.portfolio_value

        if equity <= 0:
            return Decimal("0")

        # Simple margin calculation
        margin_used = total_position_value - account_info.cash
        margin_available = account_info.buying_power

        if margin_available > 0:
            return (margin_used / margin_available) * 100
        else:
            return Decimal("100")

    def _check_emergency_conditions(
        self, metrics: RiskMetrics, daily_pnl: Decimal, margin_usage: Decimal
    ) -> bool:
        """Check if emergency conditions are met."""
        # Check emergency stop loss
        if daily_pnl < -self.config.emergency_stop_loss:
            logger.critical(f"Emergency stop loss triggered: {daily_pnl}")
            return True

        # Check extreme VaR
        if metrics.var_99 > self.config.risk_limits.max_var_99 * Decimal("1.5"):
            logger.critical(f"Extreme VaR detected: {metrics.var_99}")
            return True

        # Check margin call
        if margin_usage > Decimal("90"):
            logger.critical(f"Margin call risk: {margin_usage}%")
            return True

        return False

    async def _trigger_circuit_breaker(self) -> None:
        """Trigger circuit breaker to halt trading."""
        if not self._trading_halted:
            self._trading_halted = True

            alert = RiskAlert(
                timestamp=datetime.utcnow(),
                severity="critical",
                category="system",
                message="Circuit breaker triggered - trading halted",
                details={
                    "daily_pnl": float(self._daily_pnl),
                    "threshold": float(self.config.circuit_breaker_threshold),
                },
            )

            await self._process_alert(alert)

    async def _generate_risk_alerts(self, state: RiskState) -> None:
        """Generate alerts based on risk state."""
        # Check for critical violations
        if state.violations:
            for violation in state.violations:
                if "exceeds limit" in violation:
                    alert = RiskAlert(
                        timestamp=datetime.utcnow(),
                        severity="warning",
                        category="portfolio",
                        message=violation,
                    )
                    await self._process_alert(alert)

        # Check margin usage
        if state.margin_usage > Decimal("80"):
            alert = RiskAlert(
                timestamp=datetime.utcnow(),
                severity="warning",
                category="margin",
                message=f"High margin usage: {state.margin_usage:.1f}%",
                details={"margin_usage": float(state.margin_usage)},
            )
            await self._process_alert(alert)

        # Check daily P&L
        if state.daily_pnl < -self.config.daily_loss_limit * Decimal("0.8"):
            alert = RiskAlert(
                timestamp=datetime.utcnow(),
                severity="warning",
                category="pnl",
                message=f"Approaching daily loss limit: {state.daily_pnl:.2f}",
                details={"daily_pnl": float(state.daily_pnl)},
            )
            await self._process_alert(alert)

    async def _process_alert(self, alert: RiskAlert) -> None:
        """Process and distribute risk alert."""
        # Store alert
        self._risk_alerts.append(alert)

        # Log alert
        log_method = getattr(logger, alert.severity, logger.info)
        log_method(f"Risk Alert: {alert.message}")

        # Track event
        self.event_tracker.track(
            "risk_alert",
            {"severity": alert.severity, "category": alert.category, "message": alert.message},
        )

        # Trigger callbacks
        callbacks = self._alert_callbacks.get(alert.severity, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _get_pending_orders(self) -> list[Order]:
        """Get pending orders from broker."""
        try:
            return await self.broker.get_orders(status=OrderStatus.NEW)
        except Exception as e:
            logger.error(f"Error getting pending orders: {e}")
            return []

    def _track_validation(
        self, order: Order, is_valid: bool, errors: list[str], warnings: list[str]
    ) -> None:
        """Track order validation metrics."""
        self.event_tracker.track(
            "order_validation",
            {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "is_valid": is_valid,
                "error_count": len(errors),
                "warning_count": len(warnings),
            },
        )

        if self.metrics:
            self.metrics.increment(
                "risk_manager.order_validations",
                tags={"result": "valid" if is_valid else "invalid", "symbol": order.symbol},
            )
