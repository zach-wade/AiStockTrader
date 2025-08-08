"""
Drawdown limit checker for pre-trade risk validation.

This module checks if executing a trade would breach drawdown limits,
preventing trades that could deepen losses beyond acceptable thresholds.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from main.risk_management.types import (
    RiskCheckResult, RiskMetric, RiskLevel
)
from main.risk_management.pre_trade.unified_limit_checker.models import (
    LimitCheckResult
)
from main.risk_management.pre_trade.unified_limit_checker.types import (
    CheckContext, PortfolioState, LimitType
)
from main.risk_management.pre_trade.unified_limit_checker.registry import LimitChecker
from main.risk_management.pre_trade.unified_limit_checker.models import LimitDefinition
from main.utils.core import ErrorHandlingMixin
from main.utils.monitoring import record_metric

logger = logging.getLogger(__name__)


@dataclass
class DrawdownConfig:
    """Configuration for drawdown limits."""
    max_daily_drawdown: float = 0.02  # 2% daily
    max_weekly_drawdown: float = 0.05  # 5% weekly
    max_monthly_drawdown: float = 0.10  # 10% monthly
    max_total_drawdown: float = 0.20  # 20% from peak
    
    # Thresholds for warnings
    warning_threshold: float = 0.8  # Warn at 80% of limit
    
    # Recovery requirements
    require_recovery_period: bool = True
    min_recovery_days: int = 3  # Days before allowing aggressive trades after drawdown
    recovery_threshold: float = 0.5  # Must recover 50% of drawdown
    
    # Position restrictions during drawdown
    reduce_position_size: bool = True
    position_reduction_factor: float = 0.5  # Reduce to 50% during drawdown
    block_new_positions: bool = False  # Block entirely if drawdown severe


class DrawdownChecker(LimitChecker, ErrorHandlingMixin):
    """
    Check drawdown limits before trade execution.
    
    Prevents trades that would breach maximum drawdown limits
    and enforces position restrictions during drawdown periods.
    """
    
    def __init__(self, checker_id: str = "drawdown_checker", limit_config=None, config: Optional[DrawdownConfig] = None):
        """Initialize drawdown checker."""
        # Initialize parent classes
        LimitChecker.__init__(self, checker_id, limit_config)
        ErrorHandlingMixin.__init__(self)
        self.drawdown_config = config or DrawdownConfig()
        
        # Cache for performance
        self._portfolio_peaks: Dict[str, float] = {}
        self._drawdown_history: List[Tuple[datetime, float]] = []
        
        # State tracking
        self._check_count = 0
        self._breach_count = 0
        
        logger.info("Drawdown checker initialized")
    
    def supports_limit_type(self, limit_type: LimitType) -> bool:
        """Check if this checker supports the given limit type."""
        return limit_type == LimitType.DRAWDOWN
    
    async def check_limit(self, 
                         limit: LimitDefinition, 
                         current_value: float,
                         context: Dict[str, Any]) -> LimitCheckResult:
        """Check a limit against current value and context."""
        # Delegate to the existing check method
        check_context = context.get('check_context')
        if not check_context:
            # Create a basic context if none provided
            from main.risk_management.pre_trade.unified_limit_checker.types import CheckContext, PortfolioState
            portfolio_state = PortfolioState(total_value=context.get('portfolio_value', 100000))
            check_context = CheckContext(portfolio_state=portfolio_state)
        
        return await self.check(
            symbol=context.get('symbol', ''),
            quantity=context.get('quantity', 0),
            price=context.get('price', 0),
            side=context.get('side', 'buy'),
            context=check_context
        )
    
    async def calculate_current_value(self,
                                    limit: LimitDefinition,
                                    context: Dict[str, Any]) -> float:
        """Calculate the current value for a limit check."""
        check_context = context.get('check_context')
        if not check_context:
            return 0.0
        
        drawdowns = await self._calculate_drawdowns(check_context.portfolio_state)
        return drawdowns.get('total', 0.0)
    
    async def check(self,
                   symbol: str,
                   quantity: float,
                   price: float,
                   side: str,
                   context: CheckContext) -> LimitCheckResult:
        """Check if trade would breach drawdown limits."""
        with self._handle_error("checking drawdown limits"):
            self._check_count += 1
            
            # Get current portfolio state
            portfolio_value = context.portfolio_state.total_value
            current_positions = context.portfolio_state.positions
            
            # Calculate potential impact
            trade_value = quantity * price
            potential_loss = self._calculate_potential_loss(
                symbol, quantity, price, side, current_positions
            )
            
            # Get current drawdowns
            drawdowns = await self._calculate_drawdowns(context.portfolio_state)
            
            # Check each drawdown limit
            checks_failed = []
            warnings = []
            utilization_scores = []
            
            # Daily drawdown check
            daily_check = self._check_daily_drawdown(
                drawdowns['daily'],
                potential_loss,
                portfolio_value
            )
            if not daily_check.passed:
                checks_failed.append("daily_drawdown")
            utilization_scores.append(daily_check.utilization)
            if daily_check.warning:
                warnings.append(daily_check.warning)
            
            # Weekly drawdown check
            weekly_check = self._check_weekly_drawdown(
                drawdowns['weekly'],
                potential_loss,
                portfolio_value
            )
            if not weekly_check.passed:
                checks_failed.append("weekly_drawdown")
            utilization_scores.append(weekly_check.utilization)
            if weekly_check.warning:
                warnings.append(weekly_check.warning)
            
            # Total drawdown check
            total_check = self._check_total_drawdown(
                drawdowns['total'],
                potential_loss,
                portfolio_value
            )
            if not total_check.passed:
                checks_failed.append("total_drawdown")
            utilization_scores.append(total_check.utilization)
            if total_check.warning:
                warnings.append(total_check.warning)
            
            # Recovery period check
            if self.drawdown_config.require_recovery_period:
                recovery_check = self._check_recovery_requirements(
                    drawdowns,
                    context.portfolio_state
                )
                if not recovery_check.passed:
                    checks_failed.append("recovery_period")
                if recovery_check.warning:
                    warnings.append(recovery_check.warning)
            
            # Overall result
            passed = len(checks_failed) == 0
            max_utilization = max(utilization_scores) if utilization_scores else 0.0
            
            if not passed:
                self._breach_count += 1
                message = f"Drawdown limits breached: {', '.join(checks_failed)}"
            else:
                message = "All drawdown checks passed"
            
            # Position size adjustment
            adjusted_quantity = quantity
            if self.drawdown_config.reduce_position_size and drawdowns['total'] > 0.05:
                adjusted_quantity = quantity * self.drawdown_config.position_reduction_factor
                warnings.append(
                    f"Position size reduced to {adjusted_quantity:.0f} due to drawdown"
                )
            
            # Record metrics
            record_metric(
                'risk.drawdown_check',
                1,
                tags={
                    'passed': str(passed),
                    'symbol': symbol,
                    'utilization': f"{max_utilization:.0f}"
                }
            )
            
            return LimitCheckResult(
                passed=passed,
                check_name="drawdown_limits",
                limit_type="drawdown",
                current_value=drawdowns['total'],
                limit_value=self.drawdown_config.max_total_drawdown,
                utilization=max_utilization,
                message=message,
                warnings=warnings,
                metadata={
                    'daily_drawdown': drawdowns['daily'],
                    'weekly_drawdown': drawdowns['weekly'],
                    'total_drawdown': drawdowns['total'],
                    'checks_failed': checks_failed,
                    'adjusted_quantity': adjusted_quantity
                }
            )
    
    async def _calculate_drawdowns(self, portfolio_state: PortfolioState) -> Dict[str, float]:
        """Calculate current drawdown levels."""
        current_value = portfolio_state.total_value
        
        # Update peak if necessary
        portfolio_id = portfolio_state.portfolio_id or "default"
        if portfolio_id not in self._portfolio_peaks:
            self._portfolio_peaks[portfolio_id] = current_value
        else:
            self._portfolio_peaks[portfolio_id] = max(
                self._portfolio_peaks[portfolio_id],
                current_value
            )
        
        peak_value = self._portfolio_peaks[portfolio_id]
        
        # Calculate total drawdown from peak
        total_drawdown = 0.0
        if peak_value > 0:
            total_drawdown = (peak_value - current_value) / peak_value
        
        # Get historical values for period calculations
        history = portfolio_state.value_history or []
        
        # Daily drawdown
        daily_drawdown = 0.0
        if history:
            yesterday = datetime.utcnow() - timedelta(days=1)
            daily_values = [v for t, v in history if t >= yesterday]
            if daily_values:
                daily_peak = max(daily_values)
                daily_drawdown = (daily_peak - current_value) / daily_peak if daily_peak > 0 else 0
        
        # Weekly drawdown
        weekly_drawdown = 0.0
        if history:
            week_ago = datetime.utcnow() - timedelta(days=7)
            weekly_values = [v for t, v in history if t >= week_ago]
            if weekly_values:
                weekly_peak = max(weekly_values)
                weekly_drawdown = (weekly_peak - current_value) / weekly_peak if weekly_peak > 0 else 0
        
        # Monthly drawdown
        monthly_drawdown = 0.0
        if history:
            month_ago = datetime.utcnow() - timedelta(days=30)
            monthly_values = [v for t, v in history if t >= month_ago]
            if monthly_values:
                monthly_peak = max(monthly_values)
                monthly_drawdown = (monthly_peak - current_value) / monthly_peak if monthly_peak > 0 else 0
        
        # Update history
        self._drawdown_history.append((datetime.utcnow(), total_drawdown))
        
        # Keep only recent history
        cutoff = datetime.utcnow() - timedelta(days=90)
        self._drawdown_history = [
            (t, dd) for t, dd in self._drawdown_history if t > cutoff
        ]
        
        return {
            'daily': daily_drawdown,
            'weekly': weekly_drawdown,
            'monthly': monthly_drawdown,
            'total': total_drawdown
        }
    
    def _calculate_potential_loss(self,
                                symbol: str,
                                quantity: float,
                                price: float,
                                side: str,
                                current_positions: Dict[str, Any]) -> float:
        """Calculate potential loss from trade."""
        # Simplified calculation - could be enhanced with VaR
        # Assume 5% adverse move as potential loss
        potential_move = 0.05
        trade_value = quantity * price
        
        return trade_value * potential_move
    
    def _check_daily_drawdown(self,
                            current_dd: float,
                            potential_loss: float,
                            portfolio_value: float) -> RiskCheckResult:
        """Check daily drawdown limit."""
        potential_dd = current_dd + (potential_loss / portfolio_value)
        
        passed = potential_dd <= self.drawdown_config.max_daily_drawdown
        utilization = (potential_dd / self.drawdown_config.max_daily_drawdown * 100 
                      if self.drawdown_config.max_daily_drawdown > 0 else 0)
        
        warning = None
        if utilization > self.drawdown_config.warning_threshold * 100:
            warning = f"Daily drawdown at {utilization:.0f}% of limit"
        
        return RiskCheckResult(
            passed=passed,
            check_name="daily_drawdown",
            metric=RiskMetric.CURRENT_DRAWDOWN,
            current_value=potential_dd,
            limit_value=self.drawdown_config.max_daily_drawdown,
            utilization=utilization,
            message=f"Daily drawdown: {potential_dd:.1%} vs limit {self.drawdown_config.max_daily_drawdown:.1%}",
            metadata={'warning': warning}
        )
    
    def _check_weekly_drawdown(self,
                             current_dd: float,
                             potential_loss: float,
                             portfolio_value: float) -> RiskCheckResult:
        """Check weekly drawdown limit."""
        potential_dd = current_dd + (potential_loss / portfolio_value)
        
        passed = potential_dd <= self.drawdown_config.max_weekly_drawdown
        utilization = (potential_dd / self.drawdown_config.max_weekly_drawdown * 100 
                      if self.drawdown_config.max_weekly_drawdown > 0 else 0)
        
        warning = None
        if utilization > self.drawdown_config.warning_threshold * 100:
            warning = f"Weekly drawdown at {utilization:.0f}% of limit"
        
        return RiskCheckResult(
            passed=passed,
            check_name="weekly_drawdown",
            metric=RiskMetric.CURRENT_DRAWDOWN,
            current_value=potential_dd,
            limit_value=self.drawdown_config.max_weekly_drawdown,
            utilization=utilization,
            message=f"Weekly drawdown: {potential_dd:.1%} vs limit {self.drawdown_config.max_weekly_drawdown:.1%}",
            metadata={'warning': warning}
        )
    
    def _check_total_drawdown(self,
                            current_dd: float,
                            potential_loss: float,
                            portfolio_value: float) -> RiskCheckResult:
        """Check total drawdown limit."""
        peak_value = portfolio_value / (1 - current_dd) if current_dd < 1 else portfolio_value
        potential_dd = current_dd + (potential_loss / peak_value)
        
        passed = potential_dd <= self.drawdown_config.max_total_drawdown
        utilization = (potential_dd / self.drawdown_config.max_total_drawdown * 100 
                      if self.drawdown_config.max_total_drawdown > 0 else 0)
        
        warning = None
        if utilization > self.drawdown_config.warning_threshold * 100:
            warning = f"Total drawdown at {utilization:.0f}% of limit"
        
        return RiskCheckResult(
            passed=passed,
            check_name="total_drawdown",
            metric=RiskMetric.MAX_DRAWDOWN,
            current_value=potential_dd,
            limit_value=self.drawdown_config.max_total_drawdown,
            utilization=utilization,
            message=f"Total drawdown: {potential_dd:.1%} vs limit {self.drawdown_config.max_total_drawdown:.1%}",
            metadata={'warning': warning}
        )
    
    def _check_recovery_requirements(self,
                                   drawdowns: Dict[str, float],
                                   portfolio_state: PortfolioState) -> RiskCheckResult:
        """Check if recovery requirements are met."""
        # Check if in significant drawdown
        if drawdowns['total'] < 0.05:  # Less than 5% drawdown
            return RiskCheckResult(
                passed=True,
                check_name="recovery_period",
                metric=RiskMetric.DRAWDOWN_DURATION,
                current_value=0,
                limit_value=self.drawdown_config.min_recovery_days,
                utilization=0,
                message="Not in significant drawdown"
            )
        
        # Check recovery progress
        recovery_days = self._calculate_recovery_days(portfolio_state)
        recovery_progress = self._calculate_recovery_progress(drawdowns, portfolio_state)
        
        passed = (recovery_days >= self.drawdown_config.min_recovery_days or
                 recovery_progress >= self.drawdown_config.recovery_threshold)
        
        warning = None
        if not passed:
            warning = (f"In drawdown recovery period: {recovery_days} days, "
                      f"{recovery_progress:.0%} recovered")
        
        return RiskCheckResult(
            passed=passed,
            check_name="recovery_period",
            metric=RiskMetric.DRAWDOWN_DURATION,
            current_value=recovery_days,
            limit_value=self.drawdown_config.min_recovery_days,
            utilization=(recovery_days / self.drawdown_config.min_recovery_days * 100 
                       if self.drawdown_config.min_recovery_days > 0 else 0),
            message=f"Recovery period: {recovery_days} days",
            metadata={
                'warning': warning,
                'recovery_progress': recovery_progress
            }
        )
    
    def _calculate_recovery_days(self, portfolio_state: PortfolioState) -> int:
        """Calculate days since drawdown started recovering."""
        if not self._drawdown_history:
            return 0
        
        # Find when drawdown started improving
        max_dd_value = 0.0
        max_dd_date = datetime.utcnow()
        
        for date, dd in reversed(self._drawdown_history):
            if dd > max_dd_value:
                max_dd_value = dd
                max_dd_date = date
        
        recovery_days = (datetime.utcnow() - max_dd_date).days
        return recovery_days
    
    def _calculate_recovery_progress(self,
                                   drawdowns: Dict[str, float],
                                   portfolio_state: PortfolioState) -> float:
        """Calculate percentage recovered from maximum drawdown."""
        if not self._drawdown_history:
            return 0.0
        
        # Find maximum drawdown in history
        max_dd = max(dd for _, dd in self._drawdown_history)
        
        if max_dd == 0:
            return 1.0  # Fully recovered
        
        # Calculate recovery progress
        current_dd = drawdowns['total']
        recovery = 1 - (current_dd / max_dd) if max_dd > 0 else 1.0
        
        return max(0.0, min(1.0, recovery))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checker statistics."""
        return {
            'check_count': self._check_count,
            'breach_count': self._breach_count,
            'breach_rate': self._breach_count / self._check_count if self._check_count > 0 else 0,
            'tracked_portfolios': len(self._portfolio_peaks),
            'current_drawdowns': {
                portfolio_id: (peak - peak * (1 - dd)) / peak if peak > 0 else 0
                for portfolio_id, peak in self._portfolio_peaks.items()
                for dd in [self._drawdown_history[-1][1] if self._drawdown_history else 0]
            }
        }