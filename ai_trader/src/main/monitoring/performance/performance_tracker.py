"""
Performance Tracker - Modular Implementation

Clean, focused performance tracker that coordinates the modular components.
This replaces the monolithic unified_performance_tracker.py.
"""

import asyncio
import logging
import time
import psutil
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator, Generator

import numpy as np
import pandas as pd

from .models import (
    PerformanceMetrics, PerformanceMetricType, TimeFrame,
    TradeRecord, SystemPerformanceRecord, PerformanceAlertData
)
from .calculators import (
    ReturnCalculator, RiskCalculator, RiskAdjustedCalculator, 
    TradingMetricsCalculator
)
from .alerts.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Modular performance tracking system coordinating specialized components.
    
    This implementation uses:
    - Modular calculators for different metric types
    - Separate models for data structures
    - Alert management system
    - Clean separation of concerns
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 risk_free_rate: float = 0.02,
                 enable_alerts: bool = True):
        """Initialize performance tracker."""
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.enable_alerts = enable_alerts
        
        # Data storage
        self.trade_records: List[TradeRecord] = []
        self.system_records: List[SystemPerformanceRecord] = []
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Performance metrics cache
        self._metrics_cache: Dict[str, PerformanceMetrics] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        
        # Alert system
        self.alert_manager = AlertManager() if enable_alerts else None
        
        # System monitoring
        self.start_time = datetime.now(timezone.utc)
        self.system_stats = {
            'total_operations': 0,
            'failed_operations': 0,
            'memory_peaks': deque(maxlen=100),
            'cpu_peaks': deque(maxlen=100)
        }
        
        logger.info("Performance tracker initialized with modular architecture")
    
    def add_trade(self, trade: TradeRecord):
        """Add a trade record."""
        self.trade_records.append(trade)
        self._invalidate_cache()
    
    def add_portfolio_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add portfolio value data point."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.portfolio_values.append(value)
        self.timestamps.append(timestamp)
        self._invalidate_cache()
    
    def add_system_record(self, record: SystemPerformanceRecord):
        """Add system performance record."""
        self.system_records.append(record)
        self._update_system_stats(record)
    
    def calculate_metrics(self, timeframe: TimeFrame = TimeFrame.INCEPTION,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        # Check cache first
        cache_key = f"{timeframe.value}_{start_date}_{end_date}"
        if not self._is_cache_expired(cache_key):
            cached_metrics = self._metrics_cache.get(cache_key)
            if cached_metrics:
                return cached_metrics
        
        # Calculate metrics
        metrics = self._calculate_metrics(timeframe, start_date, end_date)
        
        # Check alerts if enabled
        if self.alert_manager:
            alerts = self.alert_manager.check_alerts(metrics)
            if alerts:
                logger.info(f"Generated {len(alerts)} performance alerts")
        
        # Cache results
        self._metrics_cache[cache_key] = metrics
        self._cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        return metrics
    
    def _calculate_metrics(self, timeframe: TimeFrame, start_date: Optional[datetime], 
                          end_date: Optional[datetime]) -> PerformanceMetrics:
        """Internal method to calculate metrics."""
        if not self.portfolio_values:
            return self._create_empty_metrics()
        
        # Filter data based on timeframe
        filtered_values, filtered_timestamps = self._filter_data(timeframe, start_date, end_date)
        filtered_trades = self._filter_trades(start_date, end_date)
        
        if not filtered_values:
            return self._create_empty_metrics()
        
        # Calculate returns
        daily_returns = ReturnCalculator.daily_returns(filtered_values)
        cumulative_returns = ReturnCalculator.cumulative_returns(daily_returns)
        
        # Calculate base metrics
        total_return = ReturnCalculator.total_return(self.initial_capital, filtered_values[-1])
        period_days = (filtered_timestamps[-1] - filtered_timestamps[0]).days if len(filtered_timestamps) > 1 else 1
        annualized_return = ReturnCalculator.annualized_return(total_return, period_days)
        
        # Risk metrics
        volatility = RiskCalculator.volatility(daily_returns)
        downside_volatility = RiskCalculator.downside_volatility(daily_returns)
        max_drawdown = RiskCalculator.max_drawdown(cumulative_returns)
        current_drawdown = RiskCalculator.current_drawdown(cumulative_returns)
        var_95 = RiskCalculator.var(daily_returns, 0.95)
        var_99 = RiskCalculator.var(daily_returns, 0.99)
        cvar_95 = RiskCalculator.cvar(daily_returns, 0.95)
        cvar_99 = RiskCalculator.cvar(daily_returns, 0.99)
        
        # Risk-adjusted metrics
        sharpe_ratio = RiskAdjustedCalculator.sharpe_ratio(daily_returns, self.risk_free_rate)
        sortino_ratio = RiskAdjustedCalculator.sortino_ratio(daily_returns, 0.0, self.risk_free_rate)
        calmar_ratio = RiskAdjustedCalculator.calmar_ratio(daily_returns, self.risk_free_rate)
        
        # Trading metrics
        win_rate = TradingMetricsCalculator.win_rate(filtered_trades)
        profit_factor = TradingMetricsCalculator.profit_factor(filtered_trades)
        avg_win, avg_loss = TradingMetricsCalculator.average_win_loss(filtered_trades)
        largest_win, largest_loss = TradingMetricsCalculator.largest_win_loss(filtered_trades)
        execution_metrics = TradingMetricsCalculator.execution_metrics(filtered_trades)
        
        # System metrics
        system_metrics = self._calculate_system_metrics()
        
        # Create metrics object
        return PerformanceMetrics(
            start_date=filtered_timestamps[0] if filtered_timestamps else datetime.now(timezone.utc),
            end_date=filtered_timestamps[-1] if filtered_timestamps else datetime.now(timezone.utc),
            period_days=period_days,
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=ReturnCalculator.cagr(self.initial_capital, filtered_values[-1], period_days/365.25),
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=len(filtered_trades),
            winning_trades=len([t for t in filtered_trades if t.pnl > 0]),
            losing_trades=len([t for t in filtered_trades if t.pnl < 0]),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            positions_count=len(set(t.symbol for t in filtered_trades)),
            portfolio_value=filtered_values[-1],
            cash_balance=0.0,  # Would need to be tracked separately
            total_pnl=sum(t.pnl for t in filtered_trades),
            realized_pnl=sum(t.pnl for t in filtered_trades if t.is_closed),
            unrealized_pnl=sum(t.pnl for t in filtered_trades if not t.is_closed),
            total_commission=execution_metrics['total_commission'],
            total_slippage=execution_metrics['total_slippage'],
            total_fees=execution_metrics['total_fees'],
            execution_cost_bps=self._calculate_execution_cost_bps(filtered_trades),
            avg_execution_time=execution_metrics['avg_execution_time'],
            system_uptime_pct=system_metrics['uptime_pct'],
            api_success_rate=system_metrics['success_rate'],
            memory_usage_mb=system_metrics['memory_usage_mb'],
            cpu_usage_pct=system_metrics['cpu_usage_pct'],
            risk_free_rate=self.risk_free_rate
        )
    
    def _filter_data(self, timeframe: TimeFrame, start_date: Optional[datetime], 
                    end_date: Optional[datetime]) -> tuple[List[float], List[datetime]]:
        """Filter portfolio data based on timeframe."""
        if not self.portfolio_values:
            return [], []
        
        # Use provided dates or calculate based on timeframe
        if start_date is None or end_date is None:
            end_date = datetime.now(timezone.utc)
            
            if timeframe == TimeFrame.DAILY:
                start_date = end_date - timedelta(days=1)
            elif timeframe == TimeFrame.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
            elif timeframe == TimeFrame.MONTHLY:
                start_date = end_date - timedelta(days=30)
            elif timeframe == TimeFrame.QUARTERLY:
                start_date = end_date - timedelta(days=90)
            elif timeframe == TimeFrame.YEARLY:
                start_date = end_date - timedelta(days=365)
            else:  # INCEPTION
                start_date = self.timestamps[0] if self.timestamps else end_date
        
        # Filter data
        filtered_values = []
        filtered_timestamps = []
        
        for value, timestamp in zip(self.portfolio_values, self.timestamps):
            if start_date <= timestamp <= end_date:
                filtered_values.append(value)
                filtered_timestamps.append(timestamp)
        
        return filtered_values, filtered_timestamps
    
    def _filter_trades(self, start_date: Optional[datetime], 
                      end_date: Optional[datetime]) -> List[TradeRecord]:
        """Filter trades based on date range."""
        if start_date is None or end_date is None:
            return self.trade_records
        
        return [trade for trade in self.trade_records 
                if start_date <= trade.entry_time <= end_date]
    
    def _calculate_execution_cost_bps(self, trades: List[TradeRecord]) -> float:
        """Calculate execution cost in basis points."""
        if not trades:
            return 0.0
        
        total_value = sum(trade.quantity * trade.entry_price for trade in trades)
        total_costs = sum(trade.commission + trade.slippage + trade.fees for trade in trades)
        
        if total_value == 0:
            return 0.0
        
        return (total_costs / total_value) * 10000  # Convert to basis points
    
    def _calculate_system_metrics(self) -> dict:
        """Calculate system performance metrics."""
        if not self.system_records:
            return {
                'uptime_pct': 1.0,
                'success_rate': 1.0,
                'memory_usage_mb': 0.0,
                'cpu_usage_pct': 0.0
            }
        
        # Calculate uptime
        uptime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        uptime_pct = min(1.0, uptime_hours / 24)  # Assume 24/7 target
        
        # Calculate success rate
        total_ops = self.system_stats['total_operations']
        failed_ops = self.system_stats['failed_operations']
        success_rate = (total_ops - failed_ops) / total_ops if total_ops > 0 else 1.0
        
        # Get current system usage
        try:
            memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
            cpu_usage = psutil.cpu_percent()
        except Exception:
            memory_usage = 0.0
            cpu_usage = 0.0
        
        return {
            'uptime_pct': uptime_pct,
            'success_rate': success_rate,
            'memory_usage_mb': memory_usage,
            'cpu_usage_pct': cpu_usage / 100.0
        }
    
    def _update_system_stats(self, record: SystemPerformanceRecord):
        """Update system statistics."""
        self.system_stats['total_operations'] += 1
        if not record.success:
            self.system_stats['failed_operations'] += 1
        
        self.system_stats['memory_peaks'].append(record.memory_usage_mb)
        self.system_stats['cpu_peaks'].append(record.cpu_usage_pct)
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object."""
        now = datetime.now(timezone.utc)
        return PerformanceMetrics(
            start_date=now,
            end_date=now,
            period_days=0,
            risk_free_rate=self.risk_free_rate
        )
    
    def _is_cache_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired."""
        expiry_time = self._cache_expiry.get(cache_key)
        if expiry_time is None:
            return True
        return datetime.now(timezone.utc) > expiry_time
    
    def _invalidate_cache(self):
        """Invalidate all cached metrics."""
        self._metrics_cache.clear()
        self._cache_expiry.clear()
    
    @contextmanager
    def performance_context(self, operation: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            duration_ms = (end_time - start_time) * 1000
            
            record = SystemPerformanceRecord(
                timestamp=datetime.now(timezone.utc),
                operation=operation,
                duration_ms=duration_ms,
                memory_usage_mb=end_memory,
                cpu_usage_pct=end_cpu,
                success=success,
                error_message=error_message
            )
            
            self.add_system_record(record)
    
    def get_alerts(self) -> List[PerformanceAlertData]:
        """Get active performance alerts."""
        if not self.alert_manager:
            return []
        return self.alert_manager.get_active_alerts()
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge a performance alert."""
        if self.alert_manager:
            self.alert_manager.acknowledge_alert(alert_id)
    
    def export_metrics(self, timeframe: TimeFrame = TimeFrame.INCEPTION) -> dict:
        """Export metrics as dictionary."""
        metrics = self.calculate_metrics(timeframe)
        return metrics.to_dict()


# Factory function for backward compatibility
def create_performance_tracker(initial_capital: float = 100000.0,
                             risk_free_rate: float = 0.02,
                             enable_alerts: bool = True) -> PerformanceTracker:
    """Create a new performance tracker instance."""
    return PerformanceTracker(initial_capital, risk_free_rate, enable_alerts)