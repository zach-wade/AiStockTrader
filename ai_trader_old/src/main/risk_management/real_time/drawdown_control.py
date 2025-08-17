"""
Portfolio drawdown limits
Created: 2025-06-16
"""

"""
Portfolio-level drawdown control and protection.

Monitors portfolio drawdown and automatically reduces exposure
or halts trading when drawdown limits are exceeded.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

# Third-party imports
import numpy as np

# Local imports
from main.trading_engine.core.order_manager import OrderManager
from main.trading_engine.core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class DrawdownLevel(Enum):
    """Drawdown severity levels."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    MAXIMUM = "maximum"


@dataclass
class DrawdownState:
    """Current drawdown state."""

    current_drawdown: float
    max_drawdown: float
    peak_value: float
    trough_value: float
    peak_date: datetime
    current_date: datetime
    duration_days: int
    level: DrawdownLevel
    recovery_days: int
    underwater_curve: list[float]


class DrawdownController:
    """Controls portfolio exposure based on drawdown levels."""

    def __init__(
        self,
        config: "Config",
        portfolio_manager: PortfolioManager,
        risk_manager: "RiskManager",
        order_manager: OrderManager,
    ):
        """Initialize drawdown controller."""
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager

        # Drawdown thresholds
        self.warning_threshold = config.get("drawdown.warning_threshold", 0.05)  # 5%
        self.critical_threshold = config.get("drawdown.critical_threshold", 0.10)  # 10%
        self.max_threshold = config.get("drawdown.max_threshold", 0.15)  # 15%

        # Response actions
        self.reduce_new_positions = config.get("drawdown.reduce_new_positions", True)
        self.scale_down_positions = config.get("drawdown.scale_down_positions", True)
        self.halt_trading = config.get("drawdown.halt_trading", True)

        # Position scaling factors by level
        self.position_scale = {
            DrawdownLevel.NORMAL: 1.0,
            DrawdownLevel.WARNING: config.get("drawdown.warning_scale", 0.7),
            DrawdownLevel.CRITICAL: config.get("drawdown.critical_scale", 0.3),
            DrawdownLevel.MAXIMUM: 0.0,  # No new positions
        }

        # Recovery parameters
        self.recovery_threshold = config.get("drawdown.recovery_threshold", 0.5)  # 50% recovery
        self.gradual_recovery = config.get("drawdown.gradual_recovery", True)

        # State tracking
        self.portfolio_values = []
        self.peak_value = 0
        self.peak_date = None
        self.current_state = None
        self.action_history = []

        # Initialize state
        self._update_state()

    def _update_state(self):
        """Update current drawdown state."""
        current_value = self.portfolio_manager.get_total_value()
        current_date = datetime.now()

        # Track portfolio value
        self.portfolio_values.append({"date": current_date, "value": current_value})

        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.peak_date = current_date

        # Calculate drawdown
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            current_drawdown = 0

        # Calculate max drawdown
        values = [pv["value"] for pv in self.portfolio_values]
        if values:
            peaks = np.maximum.accumulate(values)
            drawdowns = (peaks - values) / peaks
            max_drawdown = np.max(drawdowns)

            # Find trough
            max_dd_idx = np.argmax(drawdowns)
            trough_value = values[max_dd_idx]

            # Underwater curve (last 252 days)
            underwater_curve = list(drawdowns[-252:])
        else:
            max_drawdown = 0
            trough_value = current_value
            underwater_curve = []

        # Determine level
        if current_drawdown >= self.max_threshold:
            level = DrawdownLevel.MAXIMUM
        elif current_drawdown >= self.critical_threshold:
            level = DrawdownLevel.CRITICAL
        elif current_drawdown >= self.warning_threshold:
            level = DrawdownLevel.WARNING
        else:
            level = DrawdownLevel.NORMAL

        # Calculate duration and recovery
        duration_days = (current_date - self.peak_date).days if self.peak_date else 0

        # Recovery days (if recovering)
        recovery_days = 0
        if len(values) > 1 and current_value > trough_value:
            for i in range(len(values) - 1, 0, -1):
                if values[i] <= trough_value:
                    recovery_days = len(values) - i - 1
                    break

        self.current_state = DrawdownState(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            peak_value=self.peak_value,
            trough_value=trough_value,
            peak_date=self.peak_date,
            current_date=current_date,
            duration_days=duration_days,
            level=level,
            recovery_days=recovery_days,
            underwater_curve=underwater_curve,
        )

    def check_drawdown(self) -> DrawdownState:
        """
        Check current drawdown and take appropriate action.

        Returns:
            Current drawdown state
        """
        # Update state
        self._update_state()

        # Log if level changed
        if hasattr(self, "_last_level") and self._last_level != self.current_state.level:
            logger.warning(
                f"Drawdown level changed: {self._last_level.value} -> "
                f"{self.current_state.level.value} "
                f"({self.current_state.current_drawdown:.1%})"
            )

            # Record action
            self.action_history.append(
                {
                    "timestamp": datetime.now(),
                    "level": self.current_state.level,
                    "drawdown": self.current_state.current_drawdown,
                    "action": self._get_action_for_level(self.current_state.level),
                }
            )

        self._last_level = self.current_state.level

        # Take action based on level
        if self.current_state.level != DrawdownLevel.NORMAL:
            self._handle_drawdown_breach()

        return self.current_state

    def _handle_drawdown_breach(self):
        """Handle drawdown threshold breach."""
        level = self.current_state.level

        logger.warning(
            f"Drawdown breach: {self.current_state.current_drawdown:.1%} " f"(Level: {level.value})"
        )

        # Reduce new positions
        if self.reduce_new_positions and level != DrawdownLevel.NORMAL:
            scale_factor = self.position_scale[level]
            self.risk_manager.set_position_scale_factor(scale_factor)
            logger.info(f"Position scale factor set to {scale_factor:.1%}")

        # Scale down existing positions
        if self.scale_down_positions and level in [DrawdownLevel.CRITICAL, DrawdownLevel.MAXIMUM]:
            self._scale_down_positions(level)

        # Halt trading if maximum drawdown
        if self.halt_trading and level == DrawdownLevel.MAXIMUM:
            self._halt_all_trading()

    def _scale_down_positions(self, level: DrawdownLevel):
        """Scale down existing positions based on drawdown level."""
        scale_factor = self.position_scale[level]

        if scale_factor >= 1.0:
            return  # No scaling needed

        positions = self.portfolio_manager.get_all_positions()

        for symbol, position in positions.items():
            if position.quantity > 0:  # Only scale down long positions for now
                # Calculate target quantity
                target_quantity = int(position.quantity * scale_factor)
                reduce_quantity = position.quantity - target_quantity

                if reduce_quantity > 0:
                    logger.info(f"Scaling down {symbol}: {position.quantity} -> {target_quantity}")

                    # Submit reduce order
                    order = {
                        "symbol": symbol,
                        "quantity": reduce_quantity,
                        "side": "sell",
                        "type": "market",
                        "metadata": {
                            "reason": f"drawdown_control_{level.value}",
                            "original_quantity": position.quantity,
                            "scale_factor": scale_factor,
                        },
                    }

                    self.order_manager.submit_order(order)

    def _halt_all_trading(self):
        """Halt all trading activities."""
        logger.critical("HALTING ALL TRADING - Maximum drawdown reached")

        # Cancel all pending orders
        pending_orders = self.order_manager.get_pending_orders()
        for order in pending_orders:
            self.order_manager.cancel_order(order.order_id)

        # Set risk manager to reject all new trades
        self.risk_manager.set_trading_enabled(False)

        # Close all positions if configured
        if self.config.get("drawdown.close_all_on_max", False):
            self._close_all_positions()

    def _close_all_positions(self):
        """Close all open positions."""
        positions = self.portfolio_manager.get_all_positions()

        for symbol, position in positions.items():
            if position.quantity != 0:
                logger.info(f"Closing position: {symbol} ({position.quantity} shares)")

                order = {
                    "symbol": symbol,
                    "quantity": abs(position.quantity),
                    "side": "sell" if position.quantity > 0 else "buy",
                    "type": "market",
                    "metadata": {
                        "reason": "emergency_drawdown_close",
                        "drawdown": self.current_state.current_drawdown,
                    },
                }

                self.order_manager.submit_order(order)

    def check_recovery(self) -> bool:
        """
        Check if portfolio is recovering from drawdown.

        Returns:
            True if recovery conditions are met
        """
        if self.current_state.level == DrawdownLevel.NORMAL:
            return True

        # Calculate recovery percentage
        recovery_from_trough = (
            self.portfolio_manager.get_total_value() - self.current_state.trough_value
        ) / (self.current_state.peak_value - self.current_state.trough_value)

        if recovery_from_trough >= self.recovery_threshold:
            logger.info(f"Recovery threshold met: {recovery_from_trough:.1%}")

            if self.gradual_recovery:
                # Gradually increase position sizing
                new_scale = min(1.0, self.position_scale[self.current_state.level] + 0.1)
                self.risk_manager.set_position_scale_factor(new_scale)
                logger.info(f"Increasing position scale to {new_scale:.1%}")
            else:
                # Full recovery
                self.risk_manager.set_position_scale_factor(1.0)
                self.risk_manager.set_trading_enabled(True)
                logger.info("Full trading resumed")

            return True

        return False

    def get_position_scale_factor(self) -> float:
        """Get current position scale factor based on drawdown."""
        if self.current_state:
            return self.position_scale[self.current_state.level]
        return 1.0

    def get_drawdown_report(self) -> dict:
        """Get comprehensive drawdown report."""
        if not self.current_state:
            return {}

        return {
            "current_drawdown": f"{self.current_state.current_drawdown:.2%}",
            "max_drawdown": f"{self.current_state.max_drawdown:.2%}",
            "level": self.current_state.level.value,
            "peak_value": f"${self.current_state.peak_value:,.2f}",
            "peak_date": self.current_state.peak_date.strftime("%Y-%m-%d"),
            "duration_days": self.current_state.duration_days,
            "recovery_days": self.current_state.recovery_days,
            "position_scale": self.get_position_scale_factor(),
            "thresholds": {
                "warning": f"{self.warning_threshold:.1%}",
                "critical": f"{self.critical_threshold:.1%}",
                "maximum": f"{self.max_threshold:.1%}",
            },
            "actions_taken": len(self.action_history),
            "last_action": self.action_history[-1] if self.action_history else None,
        }

    def _get_action_for_level(self, level: DrawdownLevel) -> str:
        """Get action description for drawdown level."""
        actions = {
            DrawdownLevel.NORMAL: "Normal trading",
            DrawdownLevel.WARNING: f"Reduce positions to {self.position_scale[level]:.0%}",
            DrawdownLevel.CRITICAL: f"Scale down to {self.position_scale[level]:.0%}",
            DrawdownLevel.MAXIMUM: "Halt all trading",
        }
        return actions.get(level, "Unknown")

    def plot_underwater_curve(self) -> dict:
        """Get data for plotting underwater curve."""
        if not self.portfolio_values:
            return {}

        dates = [pv["date"] for pv in self.portfolio_values]
        values = [pv["value"] for pv in self.portfolio_values]

        # Calculate running peak and drawdown
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks * 100  # Percentage

        return {
            "dates": dates,
            "drawdowns": drawdowns,
            "current_level": self.current_state.level.value if self.current_state else "normal",
            "thresholds": {
                "warning": self.warning_threshold * 100,
                "critical": self.critical_threshold * 100,
                "maximum": self.max_threshold * 100,
            },
        }

    def get_risk_metrics(self) -> dict:
        """Calculate risk metrics related to drawdown."""
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return {}

        values = np.array([pv["value"] for pv in self.portfolio_values])
        returns = np.diff(values) / values[:-1]

        # Calmar ratio (annualized return / max drawdown)
        if len(returns) > 252 and self.current_state.max_drawdown > 0:
            annual_return = (values[-1] / values[-252] - 1) if len(values) > 252 else 0
            calmar_ratio = annual_return / self.current_state.max_drawdown
        else:
            calmar_ratio = 0

        # Recovery factor (total return / max drawdown)
        total_return = (values[-1] / values[0] - 1) if values[0] > 0 else 0
        recovery_factor = (
            total_return / self.current_state.max_drawdown
            if self.current_state.max_drawdown > 0
            else 0
        )

        # Average drawdown
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks
        avg_drawdown = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0

        return {
            "calmar_ratio": calmar_ratio,
            "recovery_factor": recovery_factor,
            "avg_drawdown": avg_drawdown,
            "drawdown_frequency": np.sum(drawdowns > 0.05)
            / len(drawdowns),  # % time in 5%+ drawdown
            "max_duration": self.current_state.duration_days,
            "current_underwater": self.current_state.current_drawdown > 0,
        }
