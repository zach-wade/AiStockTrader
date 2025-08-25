"""
Business Intelligence for AI Trading System

Trading-specific monitoring and analysis:
- Portfolio performance tracking
- Risk metrics monitoring
- Trading strategy effectiveness
- Market exposure tracking
- Compliance monitoring
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from ...domain.services.portfolio_analytics_service import (
    PortfolioAnalyticsService,
    PortfolioValue,
    PositionInfo,
)
from ...domain.services.portfolio_analytics_service import TradeRecord as PortfolioTradeRecord
from ...domain.services.strategy_analytics_service import (
    StrategyAnalyticsService,
    StrategyTradeRecord,
)
from .collector import ObservabilityEvent
from .events import BusinessEvent

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point."""

    timestamp: float
    portfolio_id: str
    metric_name: str
    value: float
    benchmark_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetric:
    """Risk metric data point."""

    timestamp: float
    portfolio_id: str
    risk_type: str  # var, drawdown, beta, sharpe, etc.
    value: float
    limit: float | None = None
    severity: str = "normal"  # normal, warning, critical
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingAnalysis:
    """Comprehensive trading analysis results."""

    analysis_period: tuple[float, float]  # start_time, end_time
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades_by_symbol: dict[str, int]
    trades_by_strategy: dict[str, int]
    performance_by_timeframe: dict[str, dict[str, Any]]
    risk_metrics: dict[str, float]


@dataclass
class ComplianceViolation:
    """Compliance violation record."""

    timestamp: float
    violation_type: str
    severity: str  # low, medium, high, critical
    portfolio_id: str | None = None
    order_id: str | None = None
    symbol: str | None = None
    description: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: float | None = None


class TradingIntelligence:
    """
    Business intelligence system for trading operations.

    Provides:
    - Portfolio performance analysis
    - Risk monitoring and alerting
    - Trading strategy effectiveness measurement
    - Market exposure analysis
    - Compliance monitoring
    - Performance benchmarking
    """

    def __init__(
        self,
        max_history_days: int = 90,
        performance_calculation_interval: float = 300.0,  # 5 minutes
        risk_check_interval: float = 60.0,  # 1 minute
    ):
        self.max_history_days = max_history_days
        self.performance_calculation_interval = performance_calculation_interval
        self.risk_check_interval = risk_check_interval

        # Initialize domain services for business logic
        self.portfolio_analytics = PortfolioAnalyticsService()
        self.strategy_analytics = StrategyAnalyticsService()

        # Data storage
        self.performance_history: dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=1000))
        self.risk_history: dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=1000))
        self.trade_history: deque[Any] = deque(maxlen=10000)
        self.compliance_violations: deque[Any] = deque(maxlen=1000)

        # Current state tracking
        self.portfolio_positions: dict[str, dict[str, Any]] = defaultdict(dict)
        self.portfolio_values: dict[str, float] = {}
        self.portfolio_pnl: dict[str, float] = defaultdict(float)
        self.active_orders: dict[str, dict[str, Any]] = {}

        # Strategy tracking
        self.strategy_performance: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "avg_trade_duration": 0.0,
                "symbols_traded": set(),
                "last_trade_time": None,
            }
        )

        # Risk limits and thresholds
        self.risk_limits = self._initialize_risk_limits()

        # Background tasks
        self._analysis_task: asyncio.Task[None] | None = None
        self._stop_analysis = False

        logger.info("Trading intelligence system initialized")

    def _initialize_risk_limits(self) -> dict[str, dict[str, float]]:
        """Initialize default risk limits."""
        return {
            "portfolio": {
                "max_drawdown_percent": 10.0,
                "max_daily_loss_percent": 5.0,
                "max_position_size_percent": 20.0,
                "max_sector_exposure_percent": 30.0,
                "min_cash_percent": 5.0,
            },
            "position": {
                "max_position_value": 100000.0,
                "max_leverage": 2.0,
                "stop_loss_percent": 5.0,
            },
            "trading": {
                "max_orders_per_minute": 10,
                "max_daily_trades": 100,
                "min_order_value": 100.0,
            },
        }

    def process_trading_event(self, event: ObservabilityEvent) -> None:
        """Route trading event to appropriate handler."""
        try:
            # Simple routing based on event type
            event_type = event.event_type

            # Route to handler
            handlers = {
                "order": self._process_order_event,
                "portfolio": self._process_portfolio_event,
                "risk": self._process_risk_event,
                "market_data": self._process_market_data_event,
            }

            handler = handlers.get(event_type)
            if handler:
                handler(event)

        except Exception as e:
            logger.error(f"Failed to process trading event: {e}")

    def _process_order_event(self, event: ObservabilityEvent) -> None:
        """Simple order event recording."""
        order_id = event.order_id
        if not order_id:
            return

        # Simple operation dispatch
        handlers = {
            "submitted": self._handle_order_submitted,
            "filled": self._handle_order_filled,
            "partially_filled": self._handle_order_filled,
            "cancelled": self._handle_order_cancelled,
            "rejected": self._handle_order_cancelled,
        }

        handler = handlers.get(event.operation)
        if handler:
            handler(event)

    def _handle_order_submitted(self, event: ObservabilityEvent) -> None:
        """Record order submission."""
        order_id = event.order_id
        if not order_id:
            return

        self.active_orders[order_id] = {
            "timestamp": event.timestamp,
            "symbol": event.symbol,
            "portfolio_id": event.portfolio_id,
            "strategy": event.strategy,
            "status": "submitted",
            **event.context,
        }

    def _handle_order_filled(self, event: ObservabilityEvent) -> None:
        """Record order fill."""
        order_id = event.order_id
        if not order_id or order_id not in self.active_orders:
            return

        # Create and store trade record
        trade_record = self._create_trade_record(event)
        self.trade_history.append(trade_record)

        # Update related data
        if event.strategy:
            self._update_strategy_performance(event.strategy, trade_record)
        if event.portfolio_id:
            self._update_portfolio_positions(event.portfolio_id, trade_record)

        # Clean up if fully filled
        if event.operation == "filled":
            del self.active_orders[order_id]

    def _handle_order_cancelled(self, event: ObservabilityEvent) -> None:
        """Record order cancellation."""
        order_id = event.order_id
        if order_id and order_id in self.active_orders:
            del self.active_orders[order_id]

    def _create_trade_record(self, event: ObservabilityEvent) -> dict[str, Any]:
        """Create trade record from event."""
        order_id = event.order_id if event.order_id else ""
        order_info = self.active_orders.get(order_id, {})
        return {
            "timestamp": event.timestamp,
            "order_id": event.order_id,
            "symbol": event.symbol,
            "portfolio_id": event.portfolio_id,
            "strategy": event.strategy,
            "operation": event.operation,
            "status": event.status,
            "duration_ms": event.duration_ms,
            "context": event.context,
            "submit_time": order_info.get("timestamp"),
        }

    def _process_portfolio_event(self, event: ObservabilityEvent) -> None:
        """Simple portfolio event recording."""
        portfolio_id = event.portfolio_id
        if not portfolio_id:
            return

        # Simple operation dispatch
        handlers = {
            "value_update": self._handle_portfolio_value_update,
            "pnl_update": self._handle_portfolio_pnl_update,
        }

        handler = handlers.get(event.operation)
        if handler:
            handler(event)

    def _handle_portfolio_value_update(self, event: ObservabilityEvent) -> None:
        """Record portfolio value update."""
        value = self._extract_value_from_event(event)
        if value is None:
            return

        portfolio_id = event.portfolio_id
        if not portfolio_id:
            return

        self.portfolio_values[portfolio_id] = float(value)

        # Record metric
        metric = PerformanceMetric(
            timestamp=event.timestamp,
            portfolio_id=portfolio_id,
            metric_name="portfolio_value",
            value=float(value),
            metadata=event.context or {},
        )
        self.performance_history[portfolio_id].append(metric)

    def _handle_portfolio_pnl_update(self, event: ObservabilityEvent) -> None:
        """Record portfolio P&L update."""
        portfolio_id = event.portfolio_id
        if portfolio_id and isinstance(event, BusinessEvent):
            pnl = self._extract_pnl(event)
            if pnl is not None:
                self.portfolio_pnl[portfolio_id] = pnl

    def _extract_value_from_event(self, event: ObservabilityEvent) -> float | None:
        """Extract value from event context or metrics."""
        context = event.context or {}
        metrics = event.metrics or {}
        value = context.get("value") or metrics.get("value")
        return float(value) if value is not None else None

    def _extract_pnl(self, event: BusinessEvent) -> float | None:
        """Extract PnL from event."""
        context = event.context or {}
        metrics = event.metrics or {}
        pnl = context.get("pnl") or metrics.get("pnl")

        if pnl is not None:
            return float(pnl)
        return None

    def _process_risk_event(self, event: ObservabilityEvent) -> None:
        """Process risk-related events."""
        portfolio_id = event.portfolio_id or "default"

        if event.operation == "risk_calculation":
            # Extract risk metrics from event
            for metric_name, value in event.metrics.items():
                if isinstance(value, (int, float)):
                    risk_metric = RiskMetric(
                        timestamp=event.timestamp,
                        portfolio_id=portfolio_id,
                        risk_type=metric_name,
                        value=float(value),
                        metadata=event.context,
                    )

                    # Check against limits
                    self._check_risk_limits(risk_metric)

                    self.risk_history[portfolio_id].append(risk_metric)

    def _process_market_data_event(self, event: ObservabilityEvent) -> None:
        """Process market data events."""
        # Market data events can be used for performance attribution
        # and market exposure analysis
        pass

    def _update_strategy_performance(self, strategy: str, trade_record: dict[str, Any]) -> None:
        """Update strategy performance tracking data.

        Simple data accumulation only - business logic is in domain services.
        """
        perf = self.strategy_performance[strategy]
        perf["trades"] += 1
        perf["last_trade_time"] = trade_record["timestamp"]

        # Track symbol
        symbol = trade_record.get("symbol")
        if symbol:
            perf["symbols_traded"].add(symbol)

        # Track P&L
        context = trade_record.get("context", {})
        pnl = context.get("pnl") if context else None
        if pnl is not None:
            perf["total_pnl"] += float(pnl)
            # Simple win/loss tracking
            if pnl > 0:
                perf["wins"] += 1
            elif pnl < 0:
                perf["losses"] += 1

    def _update_portfolio_positions(self, portfolio_id: str, trade_record: dict[str, Any]) -> None:
        """Update portfolio position tracking."""
        symbol = trade_record.get("symbol")
        if not symbol:
            return

        if symbol not in self.portfolio_positions[portfolio_id]:
            self.portfolio_positions[portfolio_id][symbol] = {
                "quantity": 0,
                "avg_cost": 0.0,
                "market_value": 0.0,
                "unrealized_pnl": 0.0,
                "last_update": trade_record["timestamp"],
            }

        # Update position (simplified - would need more complex logic for actual implementation)
        position = self.portfolio_positions[portfolio_id][symbol]
        position["last_update"] = trade_record["timestamp"]

    def _check_risk_limits(self, risk_metric: RiskMetric) -> None:
        """Check risk metric against limits."""
        limits = self.risk_limits.get("portfolio", {})
        limit_key = f"max_{risk_metric.risk_type}_percent"

        if limit_key in limits:
            limit_value = limits[limit_key]

            if risk_metric.value > limit_value:
                # Create compliance violation
                violation = ComplianceViolation(
                    timestamp=risk_metric.timestamp,
                    violation_type=f"risk_limit_breach_{risk_metric.risk_type}",
                    severity="high" if risk_metric.value > limit_value * 1.5 else "medium",
                    portfolio_id=risk_metric.portfolio_id,
                    description=f"{risk_metric.risk_type} exceeded limit: {risk_metric.value:.2f}% > {limit_value}%",
                    details={
                        "risk_type": risk_metric.risk_type,
                        "current_value": risk_metric.value,
                        "limit_value": limit_value,
                        "breach_percentage": (risk_metric.value - limit_value) / limit_value * 100,
                    },
                )

                self.compliance_violations.append(violation)
                logger.warning(f"Risk limit breached: {violation.description}")

    def calculate_portfolio_performance(
        self, portfolio_id: str, period_days: int = 30
    ) -> dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics.

        Delegates business logic to domain service.
        """
        end_time = time.time()
        start_time = end_time - (period_days * 24 * 3600)

        # Get performance history for period
        performance_data = [
            metric
            for metric in self.performance_history.get(portfolio_id, [])
            if start_time <= metric.timestamp <= end_time
        ]

        if not performance_data:
            return {"error": "No performance data available for period"}

        # Convert to domain value objects
        portfolio_values = [
            PortfolioValue(
                timestamp=m.timestamp,
                portfolio_id=m.portfolio_id,
                value=m.value,
                metadata=m.metadata,
            )
            for m in performance_data
            if m.metric_name == "portfolio_value"
        ]

        if len(portfolio_values) < 2:
            return {"error": "Insufficient data points for calculation"}

        # Get trades for period
        trades = [t for t in self.trade_history if t.get("portfolio_id") == portfolio_id]
        period_trades = [t for t in trades if start_time <= t["timestamp"] <= end_time]

        # Convert trades to domain objects
        domain_trades = [
            PortfolioTradeRecord(
                timestamp=t["timestamp"],
                order_id=t.get("order_id", ""),
                symbol=t.get("symbol", ""),
                portfolio_id=t.get("portfolio_id", ""),
                strategy=t.get("strategy"),
                operation=t.get("operation", ""),
                status=t.get("status", ""),
                duration_ms=t.get("duration_ms"),
                context=t.get("context"),
                submit_time=t.get("submit_time"),
            )
            for t in period_trades
        ]

        # Delegate to domain service for business logic
        try:
            metrics = self.portfolio_analytics.calculate_performance_metrics(
                portfolio_values=portfolio_values,
                trades=domain_trades,
                current_positions_count=len(self.portfolio_positions.get(portfolio_id, {})),
                period_days=period_days,
            )

            # Convert domain result to dictionary
            return {
                "portfolio_id": metrics.portfolio_id,
                "period_days": metrics.period_days,
                "start_value": metrics.start_value,
                "end_value": metrics.end_value,
                "total_return_percent": metrics.total_return_percent,
                "annualized_volatility_percent": metrics.annualized_volatility_percent,
                "max_drawdown_percent": metrics.max_drawdown_percent,
                "sharpe_ratio": metrics.sharpe_ratio,
                "total_trades": metrics.total_trades,
                "current_positions": metrics.current_positions,
                "data_points": metrics.data_points,
                "analysis_timestamp": metrics.analysis_timestamp,
            }
        except ValueError as e:
            return {"error": str(e)}

    def analyze_strategy_performance(self, strategy: str | None = None) -> dict[str, Any]:
        """Analyze trading strategy performance.

        Delegates business logic to domain service.
        """
        if strategy:
            strategies = {strategy: self.strategy_performance[strategy]}
        else:
            strategies = dict(self.strategy_performance)

        analysis = {}

        for strat_name, strat_data in strategies.items():
            if strat_data["trades"] == 0:
                continue

            # Get strategy trades from history
            strategy_trades = [t for t in self.trade_history if t.get("strategy") == strat_name]

            # Convert to domain objects
            domain_trades = [
                StrategyTradeRecord(
                    timestamp=t["timestamp"],
                    order_id=t.get("order_id", ""),
                    symbol=t.get("symbol", ""),
                    strategy=strat_name,
                    pnl=t.get("context", {}).get("pnl") if t.get("context") else None,
                    duration_seconds=(
                        (t.get("duration_ms", 0) / 1000) if t.get("duration_ms") else None
                    ),
                    side=t.get("context", {}).get("side") if t.get("context") else None,
                    quantity=t.get("context", {}).get("quantity") if t.get("context") else None,
                    price=t.get("context", {}).get("price") if t.get("context") else None,
                )
                for t in strategy_trades
            ]

            # Delegate to domain service
            metrics = self.strategy_analytics.calculate_strategy_performance(
                strategy_name=strat_name, trades=domain_trades
            )

            # Convert domain result to dictionary
            analysis[strat_name] = {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "win_rate_percent": metrics.win_rate_percent,
                "total_pnl": metrics.total_pnl,
                "avg_pnl_per_trade": metrics.avg_pnl_per_trade,
                "avg_trade_duration_seconds": metrics.avg_trade_duration_seconds,
                "symbols_traded": metrics.symbols_traded,
                "last_trade_time": metrics.last_trade_time,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "profit_factor": metrics.profit_factor,
                "max_consecutive_wins": metrics.max_consecutive_wins,
                "max_consecutive_losses": metrics.max_consecutive_losses,
                "expectancy": metrics.expectancy,
                "max_drawdown": metrics.max_drawdown,
                "recovery_factor": metrics.recovery_factor,
                "recent_trades": len(
                    [
                        t
                        for t in strategy_trades
                        if t["timestamp"] > time.time() - 24 * 3600  # Last 24 hours
                    ]
                ),
            }

        return analysis

    def get_risk_summary(self, portfolio_id: str | None = None) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        if portfolio_id:
            portfolios = [portfolio_id]
        else:
            portfolios = list(self.risk_history.keys())

        risk_summary: dict[str, Any] = {"timestamp": time.time(), "portfolios": {}}

        for pid in portfolios:
            risk_data = list(self.risk_history.get(pid, []))
            if not risk_data:
                continue

            # Get latest risk metrics
            latest_metrics = {}
            for metric in reversed(risk_data):  # Start from most recent
                if metric.risk_type not in latest_metrics:
                    latest_metrics[metric.risk_type] = metric

            # Calculate risk trends (last 24 hours)
            recent_time = time.time() - 24 * 3600
            recent_metrics = [m for m in risk_data if m.timestamp > recent_time]

            portfolio_risk = {
                "current_metrics": {
                    risk_type: {
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "severity": metric.severity,
                    }
                    for risk_type, metric in latest_metrics.items()
                },
                "recent_violations": len(
                    [
                        v
                        for v in self.compliance_violations
                        if v.portfolio_id == pid and v.timestamp > recent_time
                    ]
                ),
                "metric_count_24h": len(recent_metrics),
            }

            risk_summary["portfolios"][pid] = portfolio_risk

        # Overall compliance status
        recent_violations = [
            v
            for v in self.compliance_violations
            if v.timestamp > time.time() - 24 * 3600 and not v.resolved
        ]

        risk_summary["compliance"] = {
            "open_violations": len([v for v in self.compliance_violations if not v.resolved]),
            "recent_violations_24h": len(recent_violations),
            "critical_violations": len([v for v in recent_violations if v.severity == "critical"]),
            "high_violations": len([v for v in recent_violations if v.severity == "high"]),
        }

        return risk_summary

    def get_market_exposure(self, portfolio_id: str | None = None) -> dict[str, Any]:
        """Analyze market exposure by sector, geography, etc.

        Delegates business logic to domain service.
        """
        if portfolio_id:
            portfolios = {portfolio_id: self.portfolio_positions[portfolio_id]}
        else:
            portfolios = self.portfolio_positions

        exposure_analysis: dict[str, Any] = {"timestamp": time.time(), "portfolios": {}}

        for pid, positions in portfolios.items():
            if not positions:
                continue

            # Convert to domain objects
            domain_positions = {}
            for symbol, pos in positions.items():
                domain_positions[symbol] = PositionInfo(
                    symbol=symbol,
                    quantity=pos.get("quantity", 0),
                    avg_cost=pos.get("avg_cost", 0),
                    market_value=pos.get("market_value", 0),
                    unrealized_pnl=pos.get("unrealized_pnl", 0),
                    last_update=pos.get("last_update", time.time()),
                )

            # Calculate weights using domain service
            position_weights = self.portfolio_analytics.calculate_position_weights(domain_positions)

            # Calculate total value
            total_value = sum(pos.get("market_value", 0) for pos in positions.values())

            portfolio_exposure = {
                "total_market_value": total_value,
                "position_count": len(positions),
                "positions": {},
            }

            # Add position details with calculated weights
            for symbol, weight_data in position_weights.items():
                portfolio_exposure["positions"][symbol] = {
                    "market_value": weight_data["market_value"],
                    "weight_percent": weight_data["weight_percent"],
                    "quantity": weight_data["quantity"],
                    "avg_cost": weight_data["avg_cost"],
                    "unrealized_pnl": weight_data["unrealized_pnl"],
                    "last_update": positions[symbol].get("last_update"),
                }

            exposure_analysis["portfolios"][pid] = portfolio_exposure

        return exposure_analysis

    def generate_comprehensive_report(self, portfolio_id: str | None = None) -> dict[str, Any]:
        """Generate comprehensive business intelligence report."""
        return {
            "report_timestamp": time.time(),
            "performance_analysis": self.calculate_portfolio_performance(
                portfolio_id or "all", period_days=30
            ),
            "strategy_analysis": self.analyze_strategy_performance(),
            "risk_summary": self.get_risk_summary(portfolio_id),
            "market_exposure": self.get_market_exposure(portfolio_id),
            "trading_activity": {
                "total_trades_30d": len(
                    [t for t in self.trade_history if t["timestamp"] > time.time() - 30 * 24 * 3600]
                ),
                "active_orders": len(self.active_orders),
                "strategies_active": len(
                    [
                        s
                        for s, data in self.strategy_performance.items()
                        if data.get("last_trade_time", 0) > time.time() - 24 * 3600
                    ]
                ),
            },
            "compliance_status": {
                "violations_24h": len(
                    [v for v in self.compliance_violations if v.timestamp > time.time() - 24 * 3600]
                ),
                "critical_issues": len(
                    [
                        v
                        for v in self.compliance_violations
                        if v.severity == "critical" and not v.resolved
                    ]
                ),
            },
        }

    async def start_background_analysis(self) -> None:
        """Start background analysis tasks."""
        if self._analysis_task and not self._analysis_task.done():
            logger.warning("Background analysis already running")
            return

        self._stop_analysis = False
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("Started background business intelligence analysis")

    async def stop_background_analysis(self) -> None:
        """Stop background analysis tasks."""
        self._stop_analysis = True

        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background business intelligence analysis")

    async def _analysis_loop(self) -> None:
        """Background analysis loop."""
        while not self._stop_analysis:
            try:
                # Perform periodic analysis
                self._cleanup_old_data()
                self._update_real_time_metrics()

                await asyncio.sleep(self.performance_calculation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in business intelligence analysis loop: {e}")
                await asyncio.sleep(10)

    def _cleanup_old_data(self) -> None:
        """Clean up old data beyond retention period."""
        cutoff_time = time.time() - (self.max_history_days * 24 * 3600)

        # Clean up trade history
        while self.trade_history and self.trade_history[0]["timestamp"] < cutoff_time:
            self.trade_history.popleft()

        # Clean up compliance violations
        while self.compliance_violations and self.compliance_violations[0].timestamp < cutoff_time:
            self.compliance_violations.popleft()

    def _update_real_time_metrics(self) -> None:
        """Record current portfolio values for tracking.

        Simple value recording - calculations are in domain services.
        """
        current_time = time.time()

        # Record current portfolio values
        for portfolio_id, positions in self.portfolio_positions.items():
            # Sum market values
            total = 0.0
            for pos in positions.values():
                val = pos.get("market_value", 0)
                total += val

            # Record if non-zero
            if total > 0:
                metric = PerformanceMetric(
                    timestamp=current_time,
                    portfolio_id=portfolio_id,
                    metric_name="real_time_value",
                    value=total,
                )
                self.performance_history[portfolio_id].append(metric)


# Global trading intelligence instance
_trading_intelligence: TradingIntelligence | None = None


def initialize_trading_intelligence(**kwargs: Any) -> TradingIntelligence:
    """Initialize global trading intelligence."""
    global _trading_intelligence
    _trading_intelligence = TradingIntelligence(**kwargs)
    return _trading_intelligence


def get_trading_intelligence() -> TradingIntelligence:
    """Get global trading intelligence instance."""
    if not _trading_intelligence:
        raise RuntimeError(
            "Trading intelligence not initialized. Call initialize_trading_intelligence() first."
        )
    return _trading_intelligence
