"""
Alert Setup Example

This example demonstrates how to set up monitoring and alerting
for the AI Trader system using the alerting service.
"""

# Standard library imports
import asyncio
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Third-party imports
import structlog

# Local imports
from main.config.config_manager import get_config
from main.utils.alerting.alerting_service import AlertChannel, AlertingService, AlertPriority

logger = structlog.get_logger(__name__)


class TradingMonitor:
    """
    Example trading monitor that sends alerts for various conditions.
    """

    def __init__(self, config):
        self.config = config
        self.alerting_service = AlertingService(config)

        # Track thresholds
        self.thresholds = {
            "max_drawdown": 0.05,  # 5% drawdown
            "position_size": 0.15,  # 15% position size
            "daily_loss": 0.02,  # 2% daily loss
            "error_rate": 0.1,  # 10% error rate
        }

    async def monitor_portfolio_drawdown(self, current_value: float, peak_value: float):
        """Monitor portfolio drawdown and alert if threshold exceeded."""
        drawdown = (peak_value - current_value) / peak_value

        if drawdown > self.thresholds["max_drawdown"]:
            await self.alerting_service.send_alert(
                title="Portfolio Drawdown Alert",
                message=f"Portfolio drawdown of {drawdown:.1%} exceeds threshold of {self.thresholds['max_drawdown']:.1%}",
                priority=AlertPriority.HIGH,
                context={
                    "current_value": current_value,
                    "peak_value": peak_value,
                    "drawdown": drawdown,
                    "threshold": self.thresholds["max_drawdown"],
                },
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            )

            logger.warning(
                "Portfolio drawdown alert sent",
                drawdown=drawdown,
                threshold=self.thresholds["max_drawdown"],
            )

    async def monitor_position_size(
        self, symbol: str, position_value: float, portfolio_value: float
    ):
        """Monitor individual position sizes."""
        position_pct = position_value / portfolio_value

        if position_pct > self.thresholds["position_size"]:
            await self.alerting_service.send_alert(
                title="Position Size Alert",
                message=f"{symbol} position is {position_pct:.1%} of portfolio, exceeds {self.thresholds['position_size']:.1%} threshold",
                priority=AlertPriority.MEDIUM,
                context={
                    "symbol": symbol,
                    "position_value": position_value,
                    "portfolio_value": portfolio_value,
                    "position_pct": position_pct,
                    "threshold": self.thresholds["position_size"],
                },
                channels=[AlertChannel.SLACK],
            )

    async def monitor_trading_errors(
        self, error_count: int, total_operations: int, error_details: dict[str, Any]
    ):
        """Monitor trading system errors."""
        error_rate = error_count / total_operations if total_operations > 0 else 0

        if error_rate > self.thresholds["error_rate"]:
            await self.alerting_service.send_alert(
                title="High Error Rate Alert",
                message=f"Trading system error rate is {error_rate:.1%}, exceeds {self.thresholds['error_rate']:.1%} threshold",
                priority=AlertPriority.CRITICAL,
                context={
                    "error_count": error_count,
                    "total_operations": total_operations,
                    "error_rate": error_rate,
                    "threshold": self.thresholds["error_rate"],
                    "recent_errors": error_details,
                },
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.PAGERDUTY],
            )

    async def monitor_daily_performance(
        self, daily_pnl: float, portfolio_value: float, trades_executed: int
    ):
        """Monitor daily trading performance."""
        daily_return = daily_pnl / portfolio_value

        # Alert on significant daily loss
        if daily_return < -self.thresholds["daily_loss"]:
            await self.alerting_service.send_alert(
                title="Daily Loss Alert",
                message=f"Daily loss of {-daily_return:.1%} exceeds threshold",
                priority=AlertPriority.HIGH,
                context={
                    "daily_pnl": daily_pnl,
                    "daily_return": daily_return,
                    "portfolio_value": portfolio_value,
                    "trades_executed": trades_executed,
                },
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            )

        # Info alert for daily summary
        await self.alerting_service.send_alert(
            title="Daily Trading Summary",
            message=f"Daily P&L: ${daily_pnl:,.2f} ({daily_return:.2%}), Trades: {trades_executed}",
            priority=AlertPriority.LOW,
            context={
                "daily_pnl": daily_pnl,
                "daily_return": daily_return,
                "trades_executed": trades_executed,
                "timestamp": datetime.now().isoformat(),
            },
            channels=[AlertChannel.SLACK],
        )

    async def monitor_system_health(
        self, cpu_usage: float, memory_usage: float, api_latency: float, database_connections: int
    ):
        """Monitor system health metrics."""
        issues = []

        if cpu_usage > 80:
            issues.append(f"High CPU usage: {cpu_usage:.1f}%")

        if memory_usage > 85:
            issues.append(f"High memory usage: {memory_usage:.1f}%")

        if api_latency > 1000:  # 1 second
            issues.append(f"High API latency: {api_latency:.0f}ms")

        if database_connections > 90:
            issues.append(f"High database connections: {database_connections}")

        if issues:
            await self.alerting_service.send_alert(
                title="System Health Alert",
                message=f"System health issues detected: {', '.join(issues)}",
                priority=AlertPriority.HIGH,
                context={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "api_latency": api_latency,
                    "database_connections": database_connections,
                    "issues": issues,
                },
                channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            )


async def main():
    """Example usage of monitoring and alerting."""

    # Load configuration
    config = get_config(config_name="prod", environment="dev")

    # Create monitor
    monitor = TradingMonitor(config)

    print("=== Trading Monitor Alert Examples ===\n")

    # Example 1: Portfolio drawdown alert
    print("1. Testing portfolio drawdown alert...")
    await monitor.monitor_portfolio_drawdown(current_value=95000, peak_value=100000)
    print("   Alert sent for 5% drawdown\n")

    # Example 2: Position size alert
    print("2. Testing position size alert...")
    await monitor.monitor_position_size(symbol="AAPL", position_value=16000, portfolio_value=100000)
    print("   Alert sent for oversized position\n")

    # Example 3: Daily performance summary
    print("3. Testing daily performance alert...")
    await monitor.monitor_daily_performance(
        daily_pnl=1250.50, portfolio_value=100000, trades_executed=15
    )
    print("   Daily summary sent\n")

    # Example 4: System health monitoring
    print("4. Testing system health alert...")
    await monitor.monitor_system_health(
        cpu_usage=85.5, memory_usage=72.3, api_latency=250, database_connections=45
    )
    print("   System health alert sent\n")

    # Example 5: Error rate monitoring
    print("5. Testing error rate alert...")
    error_details = {
        "recent_errors": [
            {"time": "2024-01-22T10:30:00", "error": "API timeout", "symbol": "MSFT"},
            {"time": "2024-01-22T10:31:00", "error": "Order rejected", "symbol": "GOOGL"},
            {"time": "2024-01-22T10:32:00", "error": "Data validation failed", "symbol": "AMZN"},
        ]
    }
    await monitor.monitor_trading_errors(
        error_count=12, total_operations=100, error_details=error_details
    )
    print("   High error rate alert sent\n")

    # Show alerting configuration
    print("\n=== Alert Configuration ===")
    print(f"Configured channels: {monitor.alerting_service.enabled_channels}")
    print("Alert thresholds:")
    for key, value in monitor.thresholds.items():
        print(f"  {key}: {value:.1%}" if key != "error_rate" else f"  {key}: {value:.0%}")

    print("\n=== Alert Priority Levels ===")
    print("- CRITICAL: System failures, major issues")
    print("- HIGH: Significant losses, breached thresholds")
    print("- MEDIUM: Warning conditions, unusual activity")
    print("- LOW: Informational, daily summaries")

    print("\n✅ All example alerts have been configured!")
    print("\nNote: In production, configure actual notification endpoints in .env:")
    print("  SLACK_WEBHOOK_URL=https://hooks.slack.com/...")
    print("  SMTP_HOST=smtp.gmail.com")
    print("  PAGERDUTY_ROUTING_KEY=your-key-here")


if __name__ == "__main__":
    asyncio.run(main())
