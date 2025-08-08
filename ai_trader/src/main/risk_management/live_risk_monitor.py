"""
Live Risk Monitoring System

Provides real-time risk monitoring and alerting for trading activities.
Monitors positions, exposure, and risk metrics in real-time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum

from main.risk_management.risk_calculator import RiskCalculator
from main.risk_management.portfolio_risk import PortfolioRiskManager
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.models.common import Position, Order, OrderStatus
from main.utils.monitoring import record_metric

logger = logging.getLogger(__name__)


class RiskAlertLevel(Enum):
    """Risk alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class RiskMetricType(Enum):
    """Types of risk metrics to monitor."""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    SECTOR_EXPOSURE = "sector_exposure"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    DAILY_LOSS = "daily_loss"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    MARGIN_USAGE = "margin_usage"


@dataclass
class RiskAlert:
    """Risk alert information."""
    timestamp: datetime
    level: RiskAlertLevel
    metric_type: RiskMetricType
    symbol: Optional[str]
    message: str
    current_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    metric_type: RiskMetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class RiskSnapshot:
    """Point-in-time risk metrics snapshot."""
    timestamp: datetime
    total_exposure: float
    portfolio_var: float
    max_position_size: float
    concentration_ratio: float
    leverage_ratio: float
    daily_pnl: float
    margin_usage: float
    position_count: int
    alerts_count: Dict[RiskAlertLevel, int]
    

class LiveRiskMonitor:
    """
    Real-time risk monitoring system.
    
    Features:
    - Real-time position and exposure monitoring
    - Dynamic risk limit checking
    - Alert generation and management
    - Risk metric calculation and tracking
    - Integration with portfolio risk management
    """
    
    def __init__(self, 
                 broker: BrokerInterface,
                 risk_calculator: RiskCalculator,
                 portfolio_risk_manager: PortfolioRiskManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk monitor.
        
        Args:
            broker: Broker interface for position/order data
            risk_calculator: Risk calculation engine
            portfolio_risk_manager: Portfolio-level risk manager
            config: Configuration parameters
        """
        self.broker = broker
        self.risk_calculator = risk_calculator
        self.portfolio_risk_manager = portfolio_risk_manager
        self.config = config or {}
        
        # Monitoring parameters
        self.update_interval = self.config.get('update_interval', 5)  # seconds
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # 5 minutes
        self.snapshot_interval = self.config.get('snapshot_interval', 60)  # 1 minute
        
        # Risk limits
        self.risk_limits = self._initialize_risk_limits()
        
        # State tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.risk_snapshots: List[RiskSnapshot] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Cache for performance
        self._position_cache: Dict[str, Position] = {}
        self._last_position_update = datetime.min
        
        logger.info("Live risk monitor initialized")
    
    def _initialize_risk_limits(self) -> Dict[RiskMetricType, RiskLimit]:
        """Initialize default risk limits from config."""
        limits = {}
        
        # Position size limits (% of portfolio)
        limits[RiskMetricType.POSITION_SIZE] = RiskLimit(
            metric_type=RiskMetricType.POSITION_SIZE,
            warning_threshold=self.config.get('position_size_warning', 0.15),
            critical_threshold=self.config.get('position_size_critical', 0.20),
            emergency_threshold=self.config.get('position_size_emergency', 0.25)
        )
        
        # Portfolio VaR limits (% of portfolio)
        limits[RiskMetricType.PORTFOLIO_VAR] = RiskLimit(
            metric_type=RiskMetricType.PORTFOLIO_VAR,
            warning_threshold=self.config.get('var_warning', 0.02),
            critical_threshold=self.config.get('var_critical', 0.03),
            emergency_threshold=self.config.get('var_emergency', 0.05)
        )
        
        # Sector exposure limits
        limits[RiskMetricType.SECTOR_EXPOSURE] = RiskLimit(
            metric_type=RiskMetricType.SECTOR_EXPOSURE,
            warning_threshold=self.config.get('sector_warning', 0.30),
            critical_threshold=self.config.get('sector_critical', 0.40)
        )
        
        # Concentration limits (HHI)
        limits[RiskMetricType.CONCENTRATION] = RiskLimit(
            metric_type=RiskMetricType.CONCENTRATION,
            warning_threshold=self.config.get('concentration_warning', 0.15),
            critical_threshold=self.config.get('concentration_critical', 0.25)
        )
        
        # Leverage limits
        limits[RiskMetricType.LEVERAGE] = RiskLimit(
            metric_type=RiskMetricType.LEVERAGE,
            warning_threshold=self.config.get('leverage_warning', 1.5),
            critical_threshold=self.config.get('leverage_critical', 2.0),
            emergency_threshold=self.config.get('leverage_emergency', 2.5)
        )
        
        # Daily loss limits (% of portfolio)
        limits[RiskMetricType.DAILY_LOSS] = RiskLimit(
            metric_type=RiskMetricType.DAILY_LOSS,
            warning_threshold=self.config.get('daily_loss_warning', 0.02),
            critical_threshold=self.config.get('daily_loss_critical', 0.03),
            emergency_threshold=self.config.get('daily_loss_emergency', 0.05)
        )
        
        # Margin usage limits
        limits[RiskMetricType.MARGIN_USAGE] = RiskLimit(
            metric_type=RiskMetricType.MARGIN_USAGE,
            warning_threshold=self.config.get('margin_warning', 0.70),
            critical_threshold=self.config.get('margin_critical', 0.85),
            emergency_threshold=self.config.get('margin_emergency', 0.95)
        )
        
        return limits
    
    async def start_monitoring(self):
        """Start the risk monitoring loop."""
        if self._running:
            logger.warning("Risk monitor already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Risk monitoring started")
    
    async def stop_monitoring(self):
        """Stop the risk monitoring loop."""
        self._running = False
        if self._monitoring_task:
            await self._monitoring_task
        logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        last_snapshot_time = datetime.now(timezone.utc)
        
        while self._running:
            try:
                # Update positions
                await self._update_positions()
                
                # Check all risk metrics
                await self._check_risk_metrics()
                
                # Create snapshot if interval elapsed
                current_time = datetime.now(timezone.utc)
                if (current_time - last_snapshot_time).total_seconds() >= self.snapshot_interval:
                    await self._create_risk_snapshot()
                    last_snapshot_time = current_time
                
                # Clean up old alerts
                self._cleanup_alerts()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)
    
    async def _update_positions(self):
        """Update position cache from broker."""
        try:
            positions = await self.broker.get_positions()
            
            # Update cache
            self._position_cache = {pos.symbol: pos for pos in positions}
            self._last_position_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    async def _check_risk_metrics(self):
        """Check all risk metrics against limits."""
        if not self._position_cache:
            return
        
        # Check position-level risks
        await self._check_position_risks()
        
        # Check portfolio-level risks
        await self._check_portfolio_risks()
        
        # Check market risks
        await self._check_market_risks()
    
    async def _check_position_risks(self):
        """Check position-level risk metrics."""
        portfolio_value = await self._get_portfolio_value()
        
        for symbol, position in self._position_cache.items():
            # Check position size
            position_value = abs(position.quantity * position.current_price)
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            await self._check_limit(
                RiskMetricType.POSITION_SIZE,
                position_pct,
                symbol=symbol,
                metadata={'position_value': position_value}
            )
            
            # Check position-specific volatility
            volatility = await self.risk_calculator.calculate_volatility(
                symbol, lookback_days=20
            )
            
            if volatility > self.config.get('high_volatility_threshold', 0.5):
                await self._create_alert(
                    level=RiskAlertLevel.WARNING,
                    metric_type=RiskMetricType.VOLATILITY,
                    symbol=symbol,
                    message=f"High volatility detected for {symbol}",
                    current_value=volatility,
                    threshold=0.5
                )
    
    async def _check_portfolio_risks(self):
        """Check portfolio-level risk metrics."""
        portfolio_value = await self._get_portfolio_value()
        
        # Calculate portfolio VaR
        positions_df = self._positions_to_dataframe()
        if not positions_df.empty:
            portfolio_var = await self.portfolio_risk_manager.calculate_portfolio_var(
                positions_df, confidence_level=0.95
            )
            
            var_pct = portfolio_var / portfolio_value if portfolio_value > 0 else 0
            
            await self._check_limit(
                RiskMetricType.PORTFOLIO_VAR,
                var_pct,
                metadata={'var_amount': portfolio_var}
            )
        
        # Check concentration
        concentration = self._calculate_concentration()
        await self._check_limit(
            RiskMetricType.CONCENTRATION,
            concentration
        )
        
        # Check leverage
        leverage = await self._calculate_leverage()
        await self._check_limit(
            RiskMetricType.LEVERAGE,
            leverage
        )
        
        # Check daily P&L
        daily_pnl = await self._calculate_daily_pnl()
        daily_pnl_pct = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        
        if daily_pnl_pct < 0:  # Only check losses
            await self._check_limit(
                RiskMetricType.DAILY_LOSS,
                abs(daily_pnl_pct),
                metadata={'pnl_amount': daily_pnl}
            )
        
        # Check margin usage
        margin_usage = await self._calculate_margin_usage()
        await self._check_limit(
            RiskMetricType.MARGIN_USAGE,
            margin_usage
        )
    
    async def _check_market_risks(self):
        """Check market-wide risk factors."""
        # Check correlations
        correlations = await self._calculate_portfolio_correlations()
        
        # Alert on high correlations
        for (symbol1, symbol2), corr in correlations.items():
            if abs(corr) > self.config.get('high_correlation_threshold', 0.8):
                await self._create_alert(
                    level=RiskAlertLevel.WARNING,
                    metric_type=RiskMetricType.CORRELATION,
                    symbol=f"{symbol1}-{symbol2}",
                    message=f"High correlation between {symbol1} and {symbol2}",
                    current_value=corr,
                    threshold=0.8
                )
    
    async def _check_limit(self, 
                          metric_type: RiskMetricType, 
                          current_value: float,
                          symbol: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """Check a metric against its limits and create alerts if needed."""
        limit = self.risk_limits.get(metric_type)
        if not limit or not limit.enabled:
            return
        
        alert_level = None
        threshold = None
        
        # Determine alert level
        if limit.emergency_threshold and current_value >= limit.emergency_threshold:
            alert_level = RiskAlertLevel.EMERGENCY
            threshold = limit.emergency_threshold
        elif current_value >= limit.critical_threshold:
            alert_level = RiskAlertLevel.CRITICAL
            threshold = limit.critical_threshold
        elif current_value >= limit.warning_threshold:
            alert_level = RiskAlertLevel.WARNING
            threshold = limit.warning_threshold
        
        # Create alert if threshold breached
        if alert_level:
            message = f"{metric_type.value} exceeded {alert_level.value} threshold"
            if symbol:
                message = f"{message} for {symbol}"
            
            await self._create_alert(
                level=alert_level,
                metric_type=metric_type,
                symbol=symbol,
                message=message,
                current_value=current_value,
                threshold=threshold,
                metadata=metadata
            )
    
    async def _create_alert(self,
                           level: RiskAlertLevel,
                           metric_type: RiskMetricType,
                           symbol: Optional[str],
                           message: str,
                           current_value: float,
                           threshold: float,
                           metadata: Optional[Dict[str, Any]] = None):
        """Create a new risk alert if not in cooldown."""
        # Check cooldown
        alert_key = f"{metric_type.value}_{symbol or 'portfolio'}_{level.value}"
        last_alert_time = self.last_alert_times.get(alert_key)
        
        current_time = datetime.now(timezone.utc)
        if last_alert_time:
            time_since_last = (current_time - last_alert_time).total_seconds()
            if time_since_last < self.alert_cooldown:
                return  # Skip alert due to cooldown
        
        # Create alert
        alert = RiskAlert(
            timestamp=current_time,
            level=level,
            metric_type=metric_type,
            symbol=symbol,
            message=message,
            current_value=current_value,
            threshold=threshold,
            metadata=metadata or {}
        )
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Update cooldown tracker
        self.last_alert_times[alert_key] = current_time
        
        # Log alert
        logger.warning(f"Risk Alert: {alert.message} (Value: {current_value:.4f}, Threshold: {threshold:.4f})")
        
        # Record metric
        record_metric(
            'risk_monitor.alert_created',
            1,
            tags={
                'level': level.value,
                'metric_type': metric_type.value,
                'symbol': symbol or 'portfolio'
            }
        )
        
        # Execute alert actions based on level
        await self._execute_alert_actions(alert)
    
    async def _execute_alert_actions(self, alert: RiskAlert):
        """Execute actions based on alert level."""
        if alert.level == RiskAlertLevel.EMERGENCY:
            # Emergency actions - could trigger position reduction
            logger.critical(f"EMERGENCY ALERT: {alert.message}")
            # In production, could send notifications, reduce positions, etc.
            
        elif alert.level == RiskAlertLevel.CRITICAL:
            # Critical actions - prevent new positions
            logger.error(f"CRITICAL ALERT: {alert.message}")
            
        elif alert.level == RiskAlertLevel.WARNING:
            # Warning actions - log and monitor
            logger.warning(f"WARNING ALERT: {alert.message}")
    
    async def _create_risk_snapshot(self):
        """Create a point-in-time risk snapshot."""
        portfolio_value = await self._get_portfolio_value()
        
        # Calculate metrics
        positions_df = self._positions_to_dataframe()
        
        portfolio_var = 0.0
        if not positions_df.empty:
            portfolio_var = await self.portfolio_risk_manager.calculate_portfolio_var(
                positions_df, confidence_level=0.95
            )
        
        # Count alerts by level
        alert_counts = defaultdict(int)
        for alert in self.active_alerts:
            alert_counts[alert.level] += 1
        
        # Create snapshot
        snapshot = RiskSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_exposure=sum(abs(p.quantity * p.current_price) 
                             for p in self._position_cache.values()),
            portfolio_var=portfolio_var,
            max_position_size=max((abs(p.quantity * p.current_price) 
                                 for p in self._position_cache.values()), default=0),
            concentration_ratio=self._calculate_concentration(),
            leverage_ratio=await self._calculate_leverage(),
            daily_pnl=await self._calculate_daily_pnl(),
            margin_usage=await self._calculate_margin_usage(),
            position_count=len(self._position_cache),
            alerts_count=dict(alert_counts)
        )
        
        self.risk_snapshots.append(snapshot)
        
        # Keep only recent snapshots (e.g., last 24 hours)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.risk_snapshots = [s for s in self.risk_snapshots 
                              if s.timestamp > cutoff_time]
        
        # Record metrics
        record_metric('risk_monitor.total_exposure', snapshot.total_exposure)
        record_metric('risk_monitor.portfolio_var', snapshot.portfolio_var)
        record_metric('risk_monitor.position_count', snapshot.position_count)
    
    def _cleanup_alerts(self):
        """Remove resolved alerts from active list."""
        # Keep alerts for a certain duration
        alert_duration = timedelta(minutes=self.config.get('alert_duration', 30))
        current_time = datetime.now(timezone.utc)
        
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - alert.timestamp) < alert_duration
        ]
    
    async def _get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        account_info = await self.broker.get_account_info()
        return account_info.get('portfolio_value', 0.0)
    
    def _positions_to_dataframe(self) -> pd.DataFrame:
        """Convert positions to DataFrame for risk calculations."""
        if not self._position_cache:
            return pd.DataFrame()
        
        data = []
        for symbol, position in self._position_cache.items():
            data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'current_price': position.current_price,
                'market_value': position.quantity * position.current_price,
                'unrealized_pnl': position.unrealized_pnl
            })
        
        return pd.DataFrame(data)
    
    def _calculate_concentration(self) -> float:
        """Calculate portfolio concentration (HHI)."""
        if not self._position_cache:
            return 0.0
        
        total_value = sum(abs(p.quantity * p.current_price) 
                         for p in self._position_cache.values())
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum((abs(p.quantity * p.current_price) / total_value) ** 2 
                  for p in self._position_cache.values())
        
        return hhi
    
    async def _calculate_leverage(self) -> float:
        """Calculate current leverage ratio."""
        account_info = await self.broker.get_account_info()
        
        equity = account_info.get('equity', 0)
        total_exposure = sum(abs(p.quantity * p.current_price) 
                           for p in self._position_cache.values())
        
        if equity > 0:
            return total_exposure / equity
        return 0.0
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        daily_pnl = 0.0
        
        for position in self._position_cache.values():
            # This is simplified - in production would track from market open
            daily_pnl += position.unrealized_pnl
        
        return daily_pnl
    
    async def _calculate_margin_usage(self) -> float:
        """Calculate current margin usage percentage."""
        account_info = await self.broker.get_account_info()
        
        margin_used = account_info.get('margin_used', 0)
        margin_available = account_info.get('margin_available', 0)
        
        total_margin = margin_used + margin_available
        if total_margin > 0:
            return margin_used / total_margin
        return 0.0
    
    async def _calculate_portfolio_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations for portfolio positions."""
        correlations = {}
        
        symbols = list(self._position_cache.keys())
        if len(symbols) < 2:
            return correlations
        
        # Get historical data for correlation calculation
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Calculate correlation (simplified - would use actual returns data)
                corr = await self.risk_calculator.calculate_correlation(
                    symbol1, symbol2, lookback_days=30
                )
                
                correlations[(symbol1, symbol2)] = corr
        
        return correlations
    
    def get_active_alerts(self, 
                         level: Optional[RiskAlertLevel] = None,
                         metric_type: Optional[RiskMetricType] = None) -> List[RiskAlert]:
        """Get active alerts, optionally filtered."""
        alerts = self.active_alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if metric_type:
            alerts = [a for a in alerts if a.metric_type == metric_type]
        
        return alerts
    
    def get_latest_snapshot(self) -> Optional[RiskSnapshot]:
        """Get the most recent risk snapshot."""
        if self.risk_snapshots:
            return self.risk_snapshots[-1]
        return None
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of current risk status."""
        latest_snapshot = self.get_latest_snapshot()
        
        summary = {
            'monitoring_active': self._running,
            'position_count': len(self._position_cache),
            'active_alerts': len(self.active_alerts),
            'alerts_by_level': defaultdict(int),
            'last_update': self._last_position_update
        }
        
        # Count alerts by level
        for alert in self.active_alerts:
            summary['alerts_by_level'][alert.level.value] += 1
        
        # Add snapshot data if available
        if latest_snapshot:
            summary.update({
                'total_exposure': latest_snapshot.total_exposure,
                'portfolio_var': latest_snapshot.portfolio_var,
                'leverage_ratio': latest_snapshot.leverage_ratio,
                'margin_usage': latest_snapshot.margin_usage
            })
        
        return summary
    
    def update_risk_limit(self, metric_type: RiskMetricType, limit: RiskLimit):
        """Update a risk limit configuration."""
        self.risk_limits[metric_type] = limit
        logger.info(f"Updated risk limit for {metric_type.value}")
    
    async def force_risk_check(self):
        """Force an immediate risk check outside of regular schedule."""
        logger.info("Forcing immediate risk check")
        await self._update_positions()
        await self._check_risk_metrics()
        await self._create_risk_snapshot()