"""
Performance metrics logging for strategy and system analysis.

This module provides specialized logging for performance metrics,
strategy analytics, and benchmark comparisons.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import logging
import numpy as np

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.monitoring.metrics.unified_metrics_integration import UnifiedMetricsAdapter

logger = get_logger(__name__)


class MetricType(Enum):
    """Performance metric types."""
    RETURN = "return"
    RISK = "risk"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    DRAWDOWN = "drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPOSURE = "exposure"
    TURNOVER = "turnover"


@dataclass
class PerformanceLogEntry:
    """Base performance log entry."""
    timestamp: datetime
    log_id: str
    category: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class MetricLogEntry(PerformanceLogEntry):
    """Metric-specific log entry."""
    metric_type: MetricType = MetricType.RETURN
    metric_value: float = 0.0
    period: str = "daily"  # 'daily', 'weekly', 'monthly', etc.
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "metric"


@dataclass
class StrategyLogEntry(PerformanceLogEntry):
    """Strategy performance log entry."""
    strategy_name: str = ""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades_count: int = 0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "strategy"


@dataclass
class BenchmarkLogEntry(PerformanceLogEntry):
    """Benchmark comparison log entry."""
    strategy_name: str = ""
    benchmark_name: str = ""
    strategy_return: float = 0.0
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    period: str = "daily"
    
    def __post_init__(self):
        """Initialize base fields."""
        self.category = "benchmark"


class PerformanceLogger(ErrorHandlingMixin):
    """
    Logger for performance metrics and analytics.
    
    Features:
    - Strategy performance tracking
    - Risk metrics logging
    - Benchmark comparisons
    - Portfolio analytics
    - Time-series performance data
    """
    
    def __init__(
        self,
        db_pool: DatabasePool,
        log_dir: str = "logs/performance",
        calculation_interval: int = 300,  # 5 minutes
        retention_days: int = 365
    ):
        """Initialize performance logger."""
        super().__init__()
        self.db_pool = db_pool
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.calculation_interval = calculation_interval
        self.retention_days = retention_days
        
        # Initialize metrics adapter if not provided
        if metrics_adapter is None:
            from main.monitoring.metrics.unified_metrics import UnifiedMetrics
            unified_metrics = UnifiedMetrics(db_pool)
            self.metrics_adapter = UnifiedMetricsAdapter(unified_metrics)
        else:
            self.metrics_adapter = metrics_adapter
        
        # Performance cache
        self._performance_cache: Dict[str, Dict[str, Any]] = {}
        self._metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # File handlers
        self._file_handlers = self._setup_file_handlers()
        
        # State
        self._is_running = False
        self._calculation_task: Optional[asyncio.Task] = None
        
        # Metrics tracking
        self._log_count = 0
        self._calculation_count = 0
        
        # Benchmark data
        self._benchmark_data: Dict[str, Dict[str, float]] = {}
    
    def _setup_file_handlers(self) -> Dict[str, logging.Handler]:
        """Setup file handlers for performance logs."""
        handlers = {}
        
        # Daily performance log
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            self.log_dir / "performance_daily.log",
            when='midnight',
            interval=1,
            backupCount=self.retention_days
        )
        daily_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['daily'] = daily_handler
        
        # Metrics log
        metrics_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "metrics.log",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        metrics_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['metrics'] = metrics_handler
        
        # Strategy performance log
        strategy_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "strategy_performance.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        strategy_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        handlers['strategy'] = strategy_handler
        
        # JSON log for analysis
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.json",
            maxBytes=100*1024*1024,
            backupCount=5
        )
        handlers['json'] = json_handler
        
        return handlers
    
    async def start(self):
        """Start the performance logger."""
        with self._handle_error("starting performance logger"):
            if self._is_running:
                logger.warning("Performance logger already running")
                return
            
            self._is_running = True
            self._calculation_task = asyncio.create_task(
                self._calculation_loop()
            )
            
            # Load benchmark data
            await self._load_benchmark_data()
            
            logger.info("Started performance logger")
    
    async def stop(self):
        """Stop the performance logger."""
        self._is_running = False
        
        if self._calculation_task:
            self._calculation_task.cancel()
            try:
                await self._calculation_task
            except asyncio.CancelledError:
                pass
        
        # Final calculations
        await self._calculate_all_metrics()
        
        logger.info("Stopped performance logger")
    
    def log_metric(
        self,
        metric_type: MetricType,
        value: float,
        period: str = "current",
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a performance metric."""
        with self._handle_error("logging metric"):
            entry = MetricLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"MET_{self._generate_log_id()}",
                message=f"{metric_type.value}: {value:.4f}",
                metric_type=metric_type,
                metric_value=value,
                period=period,
                symbol=symbol,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            self._log_entry(entry)
            
            # Update metric history
            key = f"{strategy or 'portfolio'}:{metric_type.value}"
            if key not in self._metric_history:
                self._metric_history[key] = []
            self._metric_history[key].append((datetime.utcnow(), value))
            
            # Trim old history
            cutoff = datetime.utcnow() - timedelta(days=30)
            self._metric_history[key] = [
                (ts, val) for ts, val in self._metric_history[key]
                if ts > cutoff
            ]
            
            # Record metric
            self.metrics_adapter.record_metric(
                f'performance.{metric_type.value}',
                value,
                tags={
                    'strategy': strategy or 'portfolio',
                    'symbol': symbol or 'all',
                    'period': period
                }
            )
    
    def log_daily_performance(
        self,
        date: date,
        daily_return: float,
        cumulative_return: float,
        portfolio_value: float,
        cash_balance: float,
        positions_count: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log daily performance summary."""
        with self._handle_error("logging daily performance"):
            message = (
                f"Daily Performance - "
                f"Return: {daily_return:.2%}, "
                f"Cumulative: {cumulative_return:.2%}, "
                f"Portfolio: ${portfolio_value:,.2f}"
            )
            
            entry = PerformanceLogEntry(
                timestamp=datetime.combine(date, datetime.min.time()),
                log_id=f"DAY_{self._generate_log_id()}",
                category="daily",
                message=message,
                metadata={
                    'date': date.isoformat(),
                    'daily_return': daily_return,
                    'cumulative_return': cumulative_return,
                    'portfolio_value': portfolio_value,
                    'cash_balance': cash_balance,
                    'positions_count': positions_count,
                    **(metadata or {})
                }
            )
            
            self._log_entry(entry)
            
            # Log to daily file
            record = logging.LogRecord(
                name=__name__,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            self._file_handlers['daily'].emit(record)
    
    async def log_strategy_performance(
        self,
        strategy_name: str,
        period_start: datetime,
        period_end: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log strategy performance summary."""
        with self._handle_error("logging strategy performance"):
            # Calculate performance metrics
            metrics = await self._calculate_strategy_metrics(
                strategy_name,
                period_start,
                period_end
            )
            
            if not metrics:
                return
            
            entry = StrategyLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"STR_{self._generate_log_id()}",
                message=(
                    f"Strategy {strategy_name} - "
                    f"Return: {metrics['total_return']:.2%}, "
                    f"Sharpe: {metrics['sharpe_ratio']:.2f}"
                ),
                strategy_name=strategy_name,
                total_return=metrics['total_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                trades_count=metrics['trades_count'],
                period_start=period_start,
                period_end=period_end,
                metadata=metadata or {}
            )
            
            self._log_entry(entry)
            
            # Update cache
            self._performance_cache[strategy_name] = metrics
    
    async def log_benchmark_comparison(
        self,
        strategy_name: str,
        benchmark_name: str = "SPY",
        period: str = "YTD",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log benchmark comparison."""
        with self._handle_error("logging benchmark comparison"):
            # Get strategy and benchmark returns
            strategy_data = await self._get_strategy_returns(strategy_name, period)
            benchmark_data = await self._get_benchmark_returns(benchmark_name, period)
            
            if not strategy_data or not benchmark_data:
                return
            
            # Calculate comparison metrics
            comparison = self._calculate_comparison_metrics(
                strategy_data,
                benchmark_data
            )
            
            entry = BenchmarkLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"BEN_{self._generate_log_id()}",
                message=(
                    f"{strategy_name} vs {benchmark_name} - "
                    f"Alpha: {comparison['alpha']:.2%}, "
                    f"Beta: {comparison['beta']:.2f}"
                ),
                strategy_name=strategy_name,
                benchmark_name=benchmark_name,
                strategy_return=comparison['strategy_return'],
                benchmark_return=comparison['benchmark_return'],
                alpha=comparison['alpha'],
                beta=comparison['beta'],
                correlation=comparison['correlation'],
                tracking_error=comparison['tracking_error'],
                information_ratio=comparison['information_ratio'],
                period=period,
                metadata=metadata or {}
            )
            
            self._log_entry(entry)
    
    def log_risk_metrics(
        self,
        strategy: Optional[str] = None,
        var_95: float = 0,
        var_99: float = 0,
        cvar_95: float = 0,
        downside_deviation: float = 0,
        max_drawdown: float = 0,
        current_drawdown: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log risk metrics."""
        with self._handle_error("logging risk metrics"):
            metrics_data = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'downside_deviation': downside_deviation,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                **(metadata or {})
            }
            
            # Log individual metrics
            self.log_metric(
                MetricType.DRAWDOWN,
                max_drawdown,
                strategy=strategy,
                metadata={'type': 'max'}
            )
            
            # Create summary entry
            entry = PerformanceLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"RSK_{self._generate_log_id()}",
                category="risk",
                message=(
                    f"Risk Metrics - "
                    f"VaR95: {var_95:.2%}, "
                    f"MaxDD: {max_drawdown:.2%}"
                ),
                metadata=metrics_data
            )
            
            self._log_entry(entry)
    
    def log_portfolio_analytics(
        self,
        total_value: float,
        positions_count: int,
        long_exposure: float,
        short_exposure: float,
        net_exposure: float,
        gross_exposure: float,
        concentration: float,
        turnover: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log portfolio analytics."""
        with self._handle_error("logging portfolio analytics"):
            analytics_data = {
                'total_value': total_value,
                'positions_count': positions_count,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_exposure': net_exposure,
                'gross_exposure': gross_exposure,
                'concentration': concentration,
                'turnover': turnover,
                **(metadata or {})
            }
            
            entry = PerformanceLogEntry(
                timestamp=datetime.utcnow(),
                log_id=f"PRT_{self._generate_log_id()}",
                category="portfolio",
                message=(
                    f"Portfolio - "
                    f"Value: ${total_value:,.0f}, "
                    f"Positions: {positions_count}, "
                    f"Net Exposure: {net_exposure:.1%}"
                ),
                metadata=analytics_data
            )
            
            self._log_entry(entry)
            
            # Log exposure metrics
            self.log_metric(
                MetricType.EXPOSURE,
                net_exposure,
                metadata={'type': 'net'}
            )
            self.log_metric(
                MetricType.TURNOVER,
                turnover,
                metadata={'period': 'daily'}
            )
    
    async def generate_performance_report(
        self,
        strategy: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._handle_error("generating performance report"):
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            # Get performance data
            if strategy:
                metrics = await self._calculate_strategy_metrics(
                    strategy,
                    start_date,
                    end_date
                )
            else:
                metrics = await self._calculate_portfolio_metrics(
                    start_date,
                    end_date
                )
            
            # Get metric history
            metric_series = {}
            for metric_type in MetricType:
                key = f"{strategy or 'portfolio'}:{metric_type.value}"
                if key in self._metric_history:
                    series = [
                        {'timestamp': ts.isoformat(), 'value': val}
                        for ts, val in self._metric_history[key]
                        if ts >= start_date
                    ]
                    if series:
                        metric_series[metric_type.value] = series
            
            # Generate report
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'strategy': strategy or 'portfolio',
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': period_days
                },
                'summary': metrics,
                'time_series': metric_series,
                'statistics': self._calculate_statistics(metrics)
            }
            
            # Save report
            report_file = (
                self.log_dir / 
                f"report_{strategy or 'portfolio'}_{end_date.strftime('%Y%m%d')}.json"
            )
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report
    
    async def _calculation_loop(self):
        """Periodic metric calculation loop."""
        while self._is_running:
            try:
                await asyncio.sleep(self.calculation_interval)
                await self._calculate_all_metrics()
                self._calculation_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in calculation loop: {e}")
    
    async def _calculate_all_metrics(self):
        """Calculate all performance metrics."""
        try:
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(
                datetime.utcnow() - timedelta(days=1),
                datetime.utcnow()
            )
            
            if portfolio_metrics:
                # Log key metrics
                self.log_metric(
                    MetricType.RETURN,
                    portfolio_metrics.get('daily_return', 0)
                )
                self.log_metric(
                    MetricType.SHARPE,
                    portfolio_metrics.get('sharpe_ratio', 0)
                )
                
            # Calculate strategy metrics
            strategies = await self._get_active_strategies()
            for strategy in strategies:
                strategy_metrics = await self._calculate_strategy_metrics(
                    strategy,
                    datetime.utcnow() - timedelta(days=1),
                    datetime.utcnow()
                )
                
                if strategy_metrics:
                    self.log_metric(
                        MetricType.RETURN,
                        strategy_metrics.get('daily_return', 0),
                        strategy=strategy
                    )
                    
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
    
    async def _calculate_portfolio_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        async with self.db_pool.acquire() as conn:
            # Get portfolio values
            query = """
                SELECT 
                    date,
                    portfolio_value,
                    daily_return,
                    cash_balance
                FROM portfolio_history
                WHERE date BETWEEN $1 AND $2
                ORDER BY date
            """
            
            rows = await conn.fetch(query, start_date.date(), end_date.date())
            
            if not rows:
                return {}
            
            # Extract data
            returns = [float(row['daily_return']) for row in rows if row['daily_return']]
            
            if not returns:
                return {}
            
            # Calculate metrics
            returns_array = np.array(returns)
            
            return {
                'total_return': np.prod(1 + returns_array) - 1,
                'daily_return': returns_array[-1] if len(returns_array) > 0 else 0,
                'annualized_return': (np.prod(1 + returns_array) ** (252 / len(returns_array))) - 1,
                'volatility': np.std(returns_array) * np.sqrt(252),
                'sharpe_ratio': (
                    np.mean(returns_array) * 252 / (np.std(returns_array) * np.sqrt(252))
                    if np.std(returns_array) > 0 else 0
                ),
                'max_drawdown': self._calculate_max_drawdown(returns_array),
                'portfolio_value': float(rows[-1]['portfolio_value'])
            }
    
    async def _calculate_strategy_metrics(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate strategy-specific metrics."""
        async with self.db_pool.acquire() as conn:
            # Get strategy trades
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    SUM(pnl) as total_pnl
                FROM completed_trades
                WHERE strategy_name = $1
                AND exit_time BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(query, strategy_name, start_date, end_date)
            
            if not result or result['total_trades'] == 0:
                return {}
            
            total_trades = result['total_trades']
            winning_trades = result['winning_trades'] or 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = float(result['avg_win'] or 0)
            avg_loss = float(result['avg_loss'] or 0)
            
            profit_factor = (
                (avg_win * winning_trades) / abs(avg_loss * (total_trades - winning_trades))
                if avg_loss != 0 and (total_trades - winning_trades) > 0
                else 0
            )
            
            # Get returns for risk metrics
            returns = await self._get_strategy_returns(strategy_name, "daily")
            
            sharpe_ratio = 0
            max_drawdown = 0
            
            if returns:
                returns_array = np.array([r['return'] for r in returns])
                if len(returns_array) > 1:
                    sharpe_ratio = (
                        np.mean(returns_array) * 252 / (np.std(returns_array) * np.sqrt(252))
                        if np.std(returns_array) > 0 else 0
                    )
                    max_drawdown = self._calculate_max_drawdown(returns_array)
            
            return {
                'total_return': float(result['total_pnl'] or 0),
                'trades_count': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        if len(returns) == 0:
            return 0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    def _calculate_comparison_metrics(
        self,
        strategy_data: List[Dict[str, float]],
        benchmark_data: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate comparison metrics between strategy and benchmark."""
        # Align data by date
        strategy_returns = {d['date']: d['return'] for d in strategy_data}
        benchmark_returns = {d['date']: d['return'] for d in benchmark_data}
        
        common_dates = sorted(
            set(strategy_returns.keys()) & set(benchmark_returns.keys())
        )
        
        if not common_dates:
            return {}
        
        # Extract aligned returns
        strat_ret = np.array([strategy_returns[d] for d in common_dates])
        bench_ret = np.array([benchmark_returns[d] for d in common_dates])
        
        # Calculate metrics
        strategy_total = np.prod(1 + strat_ret) - 1
        benchmark_total = np.prod(1 + bench_ret) - 1
        
        # Beta
        if np.var(bench_ret) > 0:
            beta = np.cov(strat_ret, bench_ret)[0, 1] / np.var(bench_ret)
        else:
            beta = 1.0
        
        # Alpha (simplified)
        alpha = strategy_total - benchmark_total
        
        # Correlation
        if len(strat_ret) > 1:
            correlation = np.corrcoef(strat_ret, bench_ret)[0, 1]
        else:
            correlation = 0
        
        # Tracking error
        tracking_error = np.std(strat_ret - bench_ret) * np.sqrt(252)
        
        # Information ratio
        if tracking_error > 0:
            information_ratio = (
                np.mean(strat_ret - bench_ret) * 252 / tracking_error
            )
        else:
            information_ratio = 0
        
        return {
            'strategy_return': strategy_total,
            'benchmark_return': benchmark_total,
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_statistics(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate additional statistics from metrics."""
        stats = {
            'risk_adjusted_return': (
                metrics.get('total_return', 0) / abs(metrics.get('max_drawdown', 1))
                if metrics.get('max_drawdown', 0) != 0 else 0
            ),
            'calmar_ratio': (
                metrics.get('annualized_return', 0) / abs(metrics.get('max_drawdown', 1))
                if metrics.get('max_drawdown', 0) != 0 else 0
            ),
            'recovery_factor': (
                metrics.get('total_return', 0) / abs(metrics.get('max_drawdown', 1))
                if metrics.get('max_drawdown', 0) != 0 else 0
            )
        }
        
        return stats
    
    async def _get_strategy_returns(
        self,
        strategy_name: str,
        period: str
    ) -> List[Dict[str, float]]:
        """Get strategy returns for period."""
        # Placeholder - would query from database
        return []
    
    async def _get_benchmark_returns(
        self,
        benchmark_name: str,
        period: str
    ) -> List[Dict[str, float]]:
        """Get benchmark returns for period."""
        # Placeholder - would query from database
        return []
    
    async def _get_active_strategies(self) -> List[str]:
        """Get list of active strategies."""
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT strategy_name FROM trading_strategies WHERE active = true"
            )
            return [row['strategy_name'] for row in rows]
    
    async def _load_benchmark_data(self):
        """Load benchmark data for comparisons."""
        # Placeholder - would load benchmark data
        self._benchmark_data = {
            'SPY': {'daily_return': 0.0008, 'volatility': 0.15},
            'QQQ': {'daily_return': 0.0010, 'volatility': 0.20},
            'IWM': {'daily_return': 0.0007, 'volatility': 0.18}
        }
    
    def _log_entry(self, entry: PerformanceLogEntry):
        """Log entry to files and database."""
        self._log_count += 1
        
        # Log to JSON
        with open(self.log_dir / "performance.json", "a") as f:
            json.dump(entry.to_dict(), f)
            f.write("\n")
        
        # Log to appropriate handler
        handler = self._file_handlers.get(entry.category, self._file_handlers['metrics'])
        
        record = logging.LogRecord(
            name=__name__,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=entry.message,
            args=(),
            exc_info=None
        )
        handler.emit(record)
        
        # Store in database (async task)
        asyncio.create_task(self._store_in_database(entry))
    
    async def _store_in_database(self, entry: PerformanceLogEntry):
        """Store log entry in database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO performance_logs (
                        log_id, timestamp, category, message, metadata
                    ) VALUES ($1, $2, $3, $4, $5)
                    """,
                    entry.log_id,
                    entry.timestamp,
                    entry.category,
                    entry.message,
                    json.dumps(entry.metadata)
                )
        except Exception as e:
            logger.error(f"Error storing performance log: {e}")
    
    def _generate_log_id(self) -> str:
        """Generate unique log ID."""
        return f"{datetime.utcnow().timestamp():.0f}_{self._log_count}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            'log_count': self._log_count,
            'calculation_count': self._calculation_count,
            'cached_strategies': len(self._performance_cache),
            'metric_series': len(self._metric_history),
            'is_running': self._is_running
        }