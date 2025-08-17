# File: risk_management/dashboards/live_risk_dashboard.py

"""
Live Risk Dashboard and Alerting System

This module provides a comprehensive real-time risk dashboard with integrated
alerting capabilities for live trading risk monitoring. It displays real-time
risk metrics, position analysis, and circuit breaker status with automated
alert generation and distribution.

Key Features:
- Real-time risk metrics visualization
- Live portfolio monitoring dashboard
- Automated risk alerting system
- Circuit breaker status monitoring
- VaR utilization tracking
- Position concentration analysis
- Risk limit breach notifications
- Emergency alert system
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from enum import Enum
import logging
import smtplib
from typing import Any

# Local imports
# Import helpers
from main.feature_pipeline.calculators.helpers import safe_divide

# Import risk management components
from main.risk_management.real_time.live_risk_monitor import LiveRiskMonitor
from main.risk_management.types import RiskAlertLevel

# Import position management
from main.trading_engine.core.unified_position_manager import UnifiedPositionManager

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TERMINAL = "terminal"
    LOG = "log"


class DashboardUpdateType(Enum):
    """Types of dashboard updates."""

    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER = "circuit_breaker"
    POSITION_UPDATE = "position_update"
    RISK_DECISION = "risk_decision"
    SYSTEM_STATUS = "system_status"


@dataclass
class AlertConfiguration:
    """Configuration for risk alerts."""

    # Alert thresholds
    var_utilization_warning: float = 70.0  # % of VaR limit
    var_utilization_critical: float = 85.0  # % of VaR limit
    drawdown_warning: float = 5.0  # % drawdown
    drawdown_critical: float = 8.0  # % drawdown
    concentration_warning: float = 15.0  # % of portfolio in single position
    concentration_critical: float = 20.0  # % of portfolio in single position

    # Alert channels
    enabled_channels: list[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.LOG, AlertChannel.TERMINAL]
    )

    # Email settings
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: list[str] = field(default_factory=list)

    # Webhook settings
    webhook_urls: list[str] = field(default_factory=list)

    # Alert frequency limits
    min_alert_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    max_alerts_per_hour: int = 10

    # Alert persistence
    alert_history_days: int = 30


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""

    timestamp: datetime

    # Portfolio overview
    portfolio_value: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: float
    unrealized_pnl: Decimal
    realized_pnl: Decimal

    # Position metrics
    total_positions: int
    long_positions: int
    short_positions: int
    largest_position_pct: float
    top_5_concentration_pct: float

    # Risk metrics
    portfolio_var_95: Decimal
    portfolio_var_99: Decimal
    var_utilization_pct: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    volatility: float

    # Exposure metrics
    gross_exposure: Decimal
    net_exposure: Decimal
    leverage: float

    # Circuit breaker status
    circuit_breaker_status: str
    trading_allowed: bool
    active_breakers: list[str] = field(default_factory=list)

    # Alert summary
    active_alerts: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0

    # System performance
    risk_decisions_today: int = 0
    blocked_trades_today: int = 0
    avg_decision_time_ms: float = 0.0


@dataclass
class RiskAlertMessage:
    """Formatted risk alert message."""

    alert_id: str
    alert_type: str
    severity: RiskAlertLevel
    title: str
    message: str
    timestamp: datetime

    # Alert context
    symbol: str | None = None
    current_value: float | None = None
    threshold_value: float | None = None

    # Formatted content
    formatted_message: str = ""
    html_message: str = ""

    # Delivery tracking
    delivered_channels: list[AlertChannel] = field(default_factory=list)
    delivery_attempts: int = 0

    def format_messages(self):
        """Format alert messages for different channels."""

        # Plain text format
        self.formatted_message = f"""
🚨 RISK ALERT - {self.severity.value.upper()} 🚨

{self.title}

{self.message}

Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Alert ID: {self.alert_id}
"""
        if self.symbol:
            self.formatted_message += f"Symbol: {self.symbol}\n"
        if self.current_value is not None and self.threshold_value is not None:
            self.formatted_message += (
                f"Current: {self.current_value:.2f} | Threshold: {self.threshold_value:.2f}\n"
            )

        # HTML format for email
        severity_colors = {
            RiskAlertLevel.LOW: "#28a745",
            RiskAlertLevel.MEDIUM: "#ffc107",
            RiskAlertLevel.HIGH: "#fd7e14",
            RiskAlertLevel.CRITICAL: "#dc3545",
        }

        color = severity_colors.get(self.severity, "#6c757d")

        self.html_message = f"""
<html>
<head>
    <style>
        .alert-container {{ padding: 20px; border-left: 5px solid {color}; background-color: #f8f9fa; }}
        .alert-title {{ color: {color}; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
        .alert-message {{ font-size: 14px; margin-bottom: 15px; }}
        .alert-details {{ font-size: 12px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="alert-container">
        <div class="alert-title">🚨 RISK ALERT - {self.severity.value.upper()}</div>
        <div class="alert-title">{self.title}</div>
        <div class="alert-message">{self.message}</div>
        <div class="alert-details">
            <strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
            <strong>Alert ID:</strong> {self.alert_id}<br>
"""

        if self.symbol:
            self.html_message += f"            <strong>Symbol:</strong> {self.symbol}<br>\n"
        if self.current_value is not None and self.threshold_value is not None:
            self.html_message += (
                f"            <strong>Current Value:</strong> {self.current_value:.2f}<br>\n"
            )
            self.html_message += (
                f"            <strong>Threshold:</strong> {self.threshold_value:.2f}<br>\n"
            )

        self.html_message += """        </div>
    </div>
</body>
</html>"""


class LiveRiskDashboard:
    """
    Comprehensive live risk dashboard with real-time monitoring and alerting.
    """

    def __init__(
        self,
        risk_monitor: LiveRiskMonitor,
        position_manager: UnifiedPositionManager,
        alert_config: AlertConfiguration | None = None,
    ):
        """
        Initialize live risk dashboard.

        Args:
            risk_monitor: Live risk monitor instance
            position_manager: Position manager for portfolio data
            alert_config: Alert configuration
        """
        self.risk_monitor = risk_monitor
        self.position_manager = position_manager
        self.alert_config = alert_config or AlertConfiguration()

        # Dashboard state
        self.current_metrics: DashboardMetrics | None = None
        self.dashboard_clients: list[Callable] = []  # WebSocket clients or callback functions

        # Alert management
        self.active_alerts: dict[str, RiskAlertMessage] = {}
        self.alert_history: list[RiskAlertMessage] = []
        self.last_alert_times: dict[str, datetime] = {}
        self.alerts_sent_this_hour: int = 0
        self.hour_reset_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

        # Background tasks
        self._dashboard_task: asyncio.Task | None = None
        self._alert_task: asyncio.Task | None = None

        # Subscribe to risk monitor events
        self.risk_monitor.add_risk_decision_handler(self._handle_risk_decision)
        self.risk_monitor.add_portfolio_update_handler(self._handle_portfolio_update)
        self.risk_monitor.add_emergency_handler(self._handle_emergency_alert)

        logger.info("LiveRiskDashboard initialized with comprehensive monitoring")

    async def start_dashboard(self):
        """Start the live risk dashboard."""

        if self._dashboard_task is None or self._dashboard_task.done():
            self._dashboard_task = asyncio.create_task(self._dashboard_update_loop())

        if self._alert_task is None or self._alert_task.done():
            self._alert_task = asyncio.create_task(self._alert_processing_loop())

        logger.info("Live risk dashboard started")

    async def stop_dashboard(self):
        """Stop the live risk dashboard."""

        if self._dashboard_task:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass

        if self._alert_task:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass

        logger.info("Live risk dashboard stopped")

    async def _dashboard_update_loop(self):
        """Main dashboard update loop."""

        while True:
            try:
                await self._update_dashboard_metrics()
                await self._broadcast_dashboard_update()
                await asyncio.sleep(2)  # Update every 2 seconds

            except asyncio.CancelledError:
                logger.info("Dashboard update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Longer delay on error

    async def _alert_processing_loop(self):
        """Alert processing and delivery loop."""

        while True:
            try:
                # Reset hourly alert counter
                current_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
                if current_hour > self.hour_reset_time:
                    self.alerts_sent_this_hour = 0
                    self.hour_reset_time = current_hour

                # Process pending alerts
                await self._process_pending_alerts()

                # Clean up old alerts
                self._cleanup_old_alerts()

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                logger.info("Alert processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Longer delay on error

    async def _update_dashboard_metrics(self):
        """Update comprehensive dashboard metrics."""

        try:
            # Get current portfolio snapshot
            snapshot = self.risk_monitor.get_current_snapshot()
            if not snapshot:
                return

            # Get position metrics
            position_metrics = self.position_manager.get_position_metrics()

            # Get risk statistics
            risk_stats = self.risk_monitor.get_risk_statistics()

            # Calculate additional metrics
            leverage = float(
                safe_divide(snapshot.gross_exposure, snapshot.portfolio_value, default_value=0.0)
            )

            # Count active alerts by severity
            critical_alerts = len(
                [a for a in self.active_alerts.values() if a.severity == RiskAlertLevel.CRITICAL]
            )
            warning_alerts = len(
                [
                    a
                    for a in self.active_alerts.values()
                    if a.severity in [RiskAlertLevel.HIGH, RiskAlertLevel.MEDIUM]
                ]
            )

            # Create dashboard metrics
            self.current_metrics = DashboardMetrics(
                timestamp=datetime.now(UTC),
                portfolio_value=snapshot.portfolio_value,
                daily_pnl=snapshot.daily_pnl,
                daily_pnl_pct=snapshot.daily_pnl_pct,
                unrealized_pnl=snapshot.unrealized_pnl,
                realized_pnl=snapshot.realized_pnl_today,
                total_positions=snapshot.total_positions,
                long_positions=position_metrics.get("long_positions", 0),
                short_positions=position_metrics.get("short_positions", 0),
                largest_position_pct=snapshot.largest_position_pct,
                top_5_concentration_pct=snapshot.top_5_concentration_pct,
                portfolio_var_95=snapshot.var_95_1day,
                portfolio_var_99=snapshot.var_99_1day,
                var_utilization_pct=snapshot.var_utilization_pct,
                current_drawdown_pct=snapshot.current_drawdown,
                max_drawdown_pct=snapshot.max_drawdown,
                volatility=snapshot.volatility,
                gross_exposure=snapshot.gross_exposure,
                net_exposure=snapshot.net_exposure,
                leverage=leverage,
                circuit_breaker_status=snapshot.circuit_breaker_status,
                trading_allowed=snapshot.trading_allowed,
                active_breakers=snapshot.breached_limits,
                active_alerts=len(self.active_alerts),
                critical_alerts=critical_alerts,
                warning_alerts=warning_alerts,
                risk_decisions_today=risk_stats.get("total_decisions", 0),
                blocked_trades_today=risk_stats.get("blocked_trades", 0),
                avg_decision_time_ms=risk_stats.get("avg_decision_time_ms", 0),
            )

            # Check for new alerts based on metrics
            await self._check_metric_alerts()

        except Exception as e:
            logger.error(f"Error updating dashboard metrics: {e}", exc_info=True)

    async def _check_metric_alerts(self):
        """Check current metrics for alert conditions."""

        if not self.current_metrics:
            return

        metrics = self.current_metrics

        # VaR utilization alerts
        if metrics.var_utilization_pct >= self.alert_config.var_utilization_critical:
            await self._create_alert(
                alert_type="var_utilization_critical",
                severity=RiskAlertLevel.CRITICAL,
                title="Critical VaR Utilization",
                message=f"VaR utilization is critically high at {metrics.var_utilization_pct:.1f}%",
                current_value=metrics.var_utilization_pct,
                threshold_value=self.alert_config.var_utilization_critical,
            )
        elif metrics.var_utilization_pct >= self.alert_config.var_utilization_warning:
            await self._create_alert(
                alert_type="var_utilization_warning",
                severity=RiskAlertLevel.HIGH,
                title="High VaR Utilization",
                message=f"VaR utilization is approaching limits at {metrics.var_utilization_pct:.1f}%",
                current_value=metrics.var_utilization_pct,
                threshold_value=self.alert_config.var_utilization_warning,
            )

        # Drawdown alerts
        if metrics.current_drawdown_pct >= self.alert_config.drawdown_critical:
            await self._create_alert(
                alert_type="drawdown_critical",
                severity=RiskAlertLevel.CRITICAL,
                title="Critical Portfolio Drawdown",
                message=f"Portfolio drawdown is critically high at {metrics.current_drawdown_pct:.1f}%",
                current_value=metrics.current_drawdown_pct,
                threshold_value=self.alert_config.drawdown_critical,
            )
        elif metrics.current_drawdown_pct >= self.alert_config.drawdown_warning:
            await self._create_alert(
                alert_type="drawdown_warning",
                severity=RiskAlertLevel.HIGH,
                title="High Portfolio Drawdown",
                message=f"Portfolio drawdown is elevated at {metrics.current_drawdown_pct:.1f}%",
                current_value=metrics.current_drawdown_pct,
                threshold_value=self.alert_config.drawdown_warning,
            )

        # Concentration alerts
        if metrics.largest_position_pct >= self.alert_config.concentration_critical:
            await self._create_alert(
                alert_type="concentration_critical",
                severity=RiskAlertLevel.CRITICAL,
                title="Critical Position Concentration",
                message=f"Largest position represents {metrics.largest_position_pct:.1f}% of portfolio",
                current_value=metrics.largest_position_pct,
                threshold_value=self.alert_config.concentration_critical,
            )
        elif metrics.largest_position_pct >= self.alert_config.concentration_warning:
            await self._create_alert(
                alert_type="concentration_warning",
                severity=RiskAlertLevel.MEDIUM,
                title="High Position Concentration",
                message=f"Largest position represents {metrics.largest_position_pct:.1f}% of portfolio",
                current_value=metrics.largest_position_pct,
                threshold_value=self.alert_config.concentration_warning,
            )

        # Trading halt alert
        if not metrics.trading_allowed and "trading_halt" not in self.active_alerts:
            await self._create_alert(
                alert_type="trading_halt",
                severity=RiskAlertLevel.CRITICAL,
                title="Trading Halted",
                message=f"Trading has been halted. Active breakers: {', '.join(metrics.active_breakers)}",
            )

    async def _create_alert(
        self,
        alert_type: str,
        severity: RiskAlertLevel,
        title: str,
        message: str,
        symbol: str | None = None,
        current_value: float | None = None,
        threshold_value: float | None = None,
    ):
        """Create and queue a new risk alert."""

        # Check alert frequency limits
        now = datetime.now(UTC)
        last_alert_time = self.last_alert_times.get(alert_type)

        if last_alert_time and (now - last_alert_time) < self.alert_config.min_alert_interval:
            return  # Skip due to frequency limit

        if self.alerts_sent_this_hour >= self.alert_config.max_alerts_per_hour:
            return  # Skip due to hourly limit

        # Create alert
        alert_id = f"{alert_type}_{int(now.timestamp())}"

        alert = RiskAlertMessage(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=now,
            symbol=symbol,
            current_value=current_value,
            threshold_value=threshold_value,
        )

        alert.format_messages()

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_type] = now

        logger.warning(f"Risk alert created: {alert_type} - {title}")

    async def _process_pending_alerts(self):
        """Process and deliver pending alerts."""

        for alert in list(self.active_alerts.values()):
            if not alert.delivered_channels:
                await self._deliver_alert(alert)

    async def _deliver_alert(self, alert: RiskAlertMessage):
        """Deliver alert through configured channels."""

        for channel in self.alert_config.enabled_channels:
            try:
                if channel == AlertChannel.LOG:
                    await self._deliver_log_alert(alert)
                elif channel == AlertChannel.TERMINAL:
                    await self._deliver_terminal_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    await self._deliver_email_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._deliver_webhook_alert(alert)

                alert.delivered_channels.append(channel)

            except Exception as e:
                logger.error(f"Failed to deliver alert via {channel.value}: {e}")
                alert.delivery_attempts += 1

        # Update alerts sent counter
        if alert.delivered_channels:
            self.alerts_sent_this_hour += 1

    async def _deliver_log_alert(self, alert: RiskAlertMessage):
        """Deliver alert to log."""

        if alert.severity == RiskAlertLevel.CRITICAL:
            logger.critical(alert.formatted_message)
        elif alert.severity == RiskAlertLevel.HIGH:
            logger.error(alert.formatted_message)
        elif alert.severity == RiskAlertLevel.MEDIUM:
            logger.warning(alert.formatted_message)
        else:
            logger.info(alert.formatted_message)

    async def _deliver_terminal_alert(self, alert: RiskAlertMessage):
        """Deliver alert to terminal/console."""

        # ANSI color codes for terminal
        colors = {
            RiskAlertLevel.CRITICAL: "\033[91m",  # Red
            RiskAlertLevel.HIGH: "\033[93m",  # Yellow
            RiskAlertLevel.MEDIUM: "\033[94m",  # Blue
            RiskAlertLevel.LOW: "\033[92m",  # Green
        }
        reset_color = "\033[0m"

        color = colors.get(alert.severity, "")
        print(f"{color}{alert.formatted_message}{reset_color}")

    async def _deliver_email_alert(self, alert: RiskAlertMessage):
        """Deliver alert via email."""

        if not self.alert_config.email_recipients or not self.alert_config.email_username:
            return

        try:
            msg = MimeMultipart("alternative")
            msg["Subject"] = f"Risk Alert: {alert.title}"
            msg["From"] = self.alert_config.email_username
            msg["To"] = ", ".join(self.alert_config.email_recipients)

            # Add plain text and HTML parts
            text_part = MimeText(alert.formatted_message, "plain")
            html_part = MimeText(alert.html_message, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(
                self.alert_config.email_smtp_server, self.alert_config.email_smtp_port
            ) as server:
                server.starttls()
                server.login(self.alert_config.email_username, self.alert_config.email_password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise

    async def _deliver_webhook_alert(self, alert: RiskAlertMessage):
        """Deliver alert via webhook."""

        if not self.alert_config.webhook_urls:
            return

        # Third-party imports
        import aiohttp

        alert_data = {
            "alert_id": alert.alert_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "symbol": alert.symbol,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
        }

        for webhook_url in self.alert_config.webhook_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(webhook_url, json=alert_data, timeout=10) as response:
                        if response.status == 200:
                            logger.info(f"Webhook alert sent to {webhook_url}: {alert.alert_id}")
                        else:
                            logger.warning(f"Webhook alert failed: {response.status}")

            except Exception as e:
                logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
                raise

    def _cleanup_old_alerts(self):
        """Clean up old alerts from memory."""

        cutoff_time = datetime.now(UTC) - timedelta(days=self.alert_config.alert_history_days)

        # Clean up alert history
        self.alert_history = [
            alert for alert in self.alert_history if alert.timestamp > cutoff_time
        ]

        # Clean up active alerts that are resolved
        resolved_alerts = []
        for alert_id, alert in self.active_alerts.items():
            # Consider alert resolved if it's old and delivered
            if alert.timestamp < cutoff_time and alert.delivered_channels:
                resolved_alerts.append(alert_id)

        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]

    async def _broadcast_dashboard_update(self):
        """Broadcast dashboard update to all clients."""

        if not self.current_metrics:
            return

        update_data = {
            "type": DashboardUpdateType.PORTFOLIO_SNAPSHOT.value,
            "timestamp": self.current_metrics.timestamp.isoformat(),
            "data": asdict(self.current_metrics),
        }

        # Send to all registered clients
        for client_callback in self.dashboard_clients:
            try:
                if asyncio.iscoroutinefunction(client_callback):
                    await client_callback(update_data)
                else:
                    client_callback(update_data)
            except Exception as e:
                logger.error(f"Error broadcasting to dashboard client: {e}")

    # Event handlers

    def _handle_risk_decision(self, decision):
        """Handle risk decision events."""

        # Create alert for blocked trades
        if not decision.allowed:
            asyncio.create_task(
                self._create_alert(
                    alert_type="trade_blocked",
                    severity=RiskAlertLevel.MEDIUM,
                    title="Trade Blocked by Risk System",
                    message=f"Trade blocked: {decision.symbol} {decision.side.value} {decision.quantity}",
                    symbol=decision.symbol,
                )
            )

    def _handle_portfolio_update(self, snapshot):
        """Handle portfolio update events."""

        # Dashboard will be updated in the main loop
        pass

    async def _handle_emergency_alert(self, event_type: str, event_data: dict):
        """Handle emergency events."""

        await self._create_alert(
            alert_type="emergency",
            severity=RiskAlertLevel.CRITICAL,
            title=f"EMERGENCY: {event_type}",
            message=f"Emergency event triggered: {event_type}. Immediate attention required.",
        )

    # Public API methods

    def register_dashboard_client(self, callback: Callable):
        """Register a dashboard client for real-time updates."""
        self.dashboard_clients.append(callback)

    def unregister_dashboard_client(self, callback: Callable):
        """Unregister a dashboard client."""
        if callback in self.dashboard_clients:
            self.dashboard_clients.remove(callback)

    def get_current_dashboard_data(self) -> dict[str, Any] | None:
        """Get current dashboard data."""

        if not self.current_metrics:
            return None

        return {
            "metrics": asdict(self.current_metrics),
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "alert_summary": {
                "total_active": len(self.active_alerts),
                "critical": len(
                    [
                        a
                        for a in self.active_alerts.values()
                        if a.severity == RiskAlertLevel.CRITICAL
                    ]
                ),
                "high": len(
                    [a for a in self.active_alerts.values() if a.severity == RiskAlertLevel.HIGH]
                ),
                "medium": len(
                    [a for a in self.active_alerts.values() if a.severity == RiskAlertLevel.MEDIUM]
                ),
            },
            "system_status": {
                "dashboard_running": self._dashboard_task is not None
                and not self._dashboard_task.done(),
                "alerts_running": self._alert_task is not None and not self._alert_task.done(),
                "last_update": self.current_metrics.timestamp.isoformat(),
            },
        }

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert."""

        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert acknowledged: {alert_id}")
            return True

        return False

    def get_alert_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get alert history for specified hours."""

        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        recent_alerts = [
            asdict(alert) for alert in self.alert_history if alert.timestamp > cutoff_time
        ]

        return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)

    def get_dashboard_statistics(self) -> dict[str, Any]:
        """Get dashboard and alerting statistics."""

        return {
            "dashboard": {
                "clients_connected": len(self.dashboard_clients),
                "last_update": (
                    self.current_metrics.timestamp.isoformat() if self.current_metrics else None
                ),
                "update_frequency_seconds": 2,
            },
            "alerts": {
                "active_alerts": len(self.active_alerts),
                "total_history": len(self.alert_history),
                "alerts_sent_this_hour": self.alerts_sent_this_hour,
                "enabled_channels": [c.value for c in self.alert_config.enabled_channels],
                "alert_frequency_limit_minutes": self.alert_config.min_alert_interval.total_seconds()
                / 60,
                "hourly_limit": self.alert_config.max_alerts_per_hour,
            },
        }
