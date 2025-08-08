"""
Drawdown Circuit Breaker

Monitors portfolio drawdown and triggers protection when drawdown
exceeds safe levels or accelerates rapidly.

Created: 2025-07-15
"""

import logging
from typing import Dict, Any
from datetime import datetime
from collections import deque
import numpy as np

from ..types import BreakerType, MarketConditions, BreakerMetrics
from ..registry import BaseBreaker
from ..config import BreakerConfig

logger = logging.getLogger(__name__)


class DrawdownBreaker(BaseBreaker):
    """
    Circuit breaker for drawdown protection.
    
    Monitors:
    - Current portfolio drawdown from peak
    - Drawdown acceleration/velocity
    - Underwater periods (time below peak)
    """
    
    def __init__(self, breaker_type: BreakerType, config: BreakerConfig):
        """Initialize drawdown breaker."""
        super().__init__(breaker_type, config)
        
        # Drawdown tracking
        self.portfolio_peak = 0.0
        self.drawdown_history: deque = deque(maxlen=100)
        self.portfolio_value_history: deque = deque(maxlen=100)
        
        # Configuration
        self.max_drawdown = config.max_drawdown
        self.warning_threshold = self.max_drawdown * 0.8  # 80% of max drawdown
        self.drawdown_acceleration_threshold = self.max_drawdown / 20  # 5% of max drawdown per period
        
        # Performance tracking
        self.last_peak_time = datetime.now()
        self.underwater_periods = 0
        self.max_underwater_duration = 0
        
        logger.info(f"Drawdown breaker initialized - max drawdown: {self.max_drawdown:.2%}")
    
    async def check(self, 
                   portfolio_value: float,
                   positions: Dict[str, Any],
                   market_conditions: MarketConditions) -> bool:
        """
        Check if drawdown breaker should trip.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            market_conditions: Current market conditions
            
        Returns:
            True if breaker should trip
        """
        if not self.is_enabled():
            return False
        
        # Update portfolio peak
        if portfolio_value > self.portfolio_peak:
            self.portfolio_peak = portfolio_value
            self.last_peak_time = datetime.now()
        
        # Update history
        now = datetime.now()
        self.portfolio_value_history.append((now, portfolio_value))
        
        # Calculate current drawdown
        current_drawdown = self._calculate_drawdown(portfolio_value)
        self.drawdown_history.append((now, current_drawdown))
        
        # Check if drawdown exceeds maximum
        if current_drawdown > self.max_drawdown:
            self.logger.error(f"Drawdown limit breached: {current_drawdown:.2%} > {self.max_drawdown:.2%}")
            return True
        
        # Check drawdown acceleration
        if await self._check_drawdown_acceleration():
            return True
        
        # Check extended underwater period
        if await self._check_underwater_duration():
            return True
        
        return False
    
    async def check_warning_conditions(self, 
                                     portfolio_value: float,
                                     positions: Dict[str, Any],
                                     market_conditions: MarketConditions) -> bool:
        """
        Check if drawdown breaker should be in warning state.
        
        Returns:
            True if breaker should be in warning state
        """
        if not self.is_enabled():
            return False
        
        current_drawdown = self._calculate_drawdown(portfolio_value)
        
        # Warning if approaching drawdown limit
        if current_drawdown > self.warning_threshold:
            self.logger.warning(f"Approaching drawdown limit: {current_drawdown:.2%}")
            return True
        
        # Warning if drawdown is accelerating
        if len(self.drawdown_history) >= 5:
            recent_drawdowns = [dd for _, dd in list(self.drawdown_history)[-5:]]
            if len(recent_drawdowns) >= 2:
                drawdown_acceleration = (recent_drawdowns[-1] - recent_drawdowns[0]) / len(recent_drawdowns)
                if drawdown_acceleration > self.drawdown_acceleration_threshold / 2:
                    self.logger.warning(f"Drawdown accelerating: {drawdown_acceleration:.4%}/period")
                    return True
        
        return False
    
    def _calculate_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown percentage."""
        if self.portfolio_peak <= 0:
            return 0.0
        
        drawdown = (self.portfolio_peak - portfolio_value) / self.portfolio_peak
        return max(0.0, drawdown)  # Ensure non-negative
    
    async def _check_drawdown_acceleration(self) -> bool:
        """Check if drawdown is accelerating rapidly."""
        if len(self.drawdown_history) < 5:
            return False
        
        # Calculate drawdown acceleration over recent periods
        recent_drawdowns = [dd for _, dd in list(self.drawdown_history)[-5:]]
        drawdown_acceleration = (recent_drawdowns[-1] - recent_drawdowns[0]) / len(recent_drawdowns)
        
        if drawdown_acceleration > self.drawdown_acceleration_threshold:
            self.logger.warning(f"Drawdown accelerating: {drawdown_acceleration:.4%}/period")
            return True
        
        return False
    
    async def _check_underwater_duration(self) -> bool:
        """Check if portfolio has been underwater for too long."""
        if self.portfolio_peak <= 0:
            return False
        
        # Calculate time since last peak
        underwater_duration = (datetime.now() - self.last_peak_time).total_seconds() / 3600  # hours
        
        # Trip if underwater for more than 24 hours with significant drawdown
        current_drawdown = self._calculate_drawdown(self.portfolio_value_history[-1][1] if self.portfolio_value_history else 0)
        
        if underwater_duration > 24 and current_drawdown > 0.05:  # 5% drawdown for 24+ hours
            self.logger.warning(f"Extended underwater period: {underwater_duration:.1f} hours with {current_drawdown:.2%} drawdown")
            return True
        
        return False
    
    def get_metrics(self) -> BreakerMetrics:
        """Get current drawdown metrics."""
        metrics = BreakerMetrics()
        
        if self.portfolio_value_history:
            current_value = self.portfolio_value_history[-1][1]
            metrics.current_drawdown = self._calculate_drawdown(current_value)
            metrics.portfolio_peak = self.portfolio_peak
        
        return metrics
    
    def get_drawdown_statistics(self) -> Dict[str, float]:
        """Get detailed drawdown statistics."""
        if not self.drawdown_history:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'average_drawdown': 0.0,
                'drawdown_trend': 0.0,
                'underwater_duration_hours': 0.0,
                'portfolio_peak': self.portfolio_peak,
                'recovery_factor': 0.0
            }
        
        drawdowns = [dd for _, dd in self.drawdown_history]
        current_drawdown = drawdowns[-1]
        max_drawdown = max(drawdowns)
        avg_drawdown = np.mean(drawdowns)
        
        # Calculate trend
        if len(drawdowns) >= 5:
            recent_drawdowns = drawdowns[-5:]
            drawdown_trend = (recent_drawdowns[-1] - recent_drawdowns[0]) / len(recent_drawdowns)
        else:
            drawdown_trend = 0.0
        
        # Calculate underwater duration
        underwater_duration_hours = (datetime.now() - self.last_peak_time).total_seconds() / 3600
        
        # Calculate recovery factor (how much portfolio needs to recover)
        recovery_factor = 1 / (1 - current_drawdown) - 1 if current_drawdown < 1 else float('inf')
        
        return {
            'current_drawdown': float(current_drawdown),
            'max_drawdown': float(max_drawdown),
            'average_drawdown': float(avg_drawdown),
            'drawdown_trend': float(drawdown_trend),
            'underwater_duration_hours': float(underwater_duration_hours),
            'portfolio_peak': float(self.portfolio_peak),
            'recovery_factor': float(recovery_factor)
        }
    
    def reset_peak(self, new_peak: float = None):
        """Reset portfolio peak to current value or specified value."""
        if new_peak is not None:
            self.portfolio_peak = new_peak
        elif self.portfolio_value_history:
            self.portfolio_peak = self.portfolio_value_history[-1][1]
        
        self.last_peak_time = datetime.now()
        self.logger.info(f"Portfolio peak reset to: {self.portfolio_peak}")
    
    def get_recovery_analysis(self) -> Dict[str, Any]:
        """Get analysis of recovery patterns."""
        if len(self.drawdown_history) < 10:
            return {'insufficient_data': True}
        
        # Find recovery periods
        drawdowns = [dd for _, dd in self.drawdown_history]
        
        # Identify peaks and valleys in drawdown
        recovery_periods = []
        in_recovery = False
        recovery_start = None
        
        for i in range(1, len(drawdowns)):
            if drawdowns[i] < drawdowns[i-1]:  # Drawdown decreasing (recovery)
                if not in_recovery:
                    recovery_start = i
                    in_recovery = True
            elif drawdowns[i] > drawdowns[i-1]:  # Drawdown increasing
                if in_recovery and recovery_start is not None:
                    recovery_periods.append({
                        'start': recovery_start,
                        'end': i-1,
                        'duration': i - recovery_start,
                        'recovery_amount': drawdowns[recovery_start] - drawdowns[i-1]
                    })
                    in_recovery = False
        
        if not recovery_periods:
            return {'no_recovery_periods': True}
        
        # Calculate recovery statistics
        recovery_durations = [rp['duration'] for rp in recovery_periods]
        recovery_amounts = [rp['recovery_amount'] for rp in recovery_periods]
        
        return {
            'recovery_periods': len(recovery_periods),
            'avg_recovery_duration': np.mean(recovery_durations),
            'avg_recovery_amount': np.mean(recovery_amounts),
            'max_recovery_amount': max(recovery_amounts),
            'recovery_success_rate': len([rp for rp in recovery_periods if rp['recovery_amount'] > 0.01]) / len(recovery_periods)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get breaker information including drawdown-specific details."""
        base_info = super().get_info()
        
        drawdown_stats = self.get_drawdown_statistics()
        recovery_analysis = self.get_recovery_analysis()
        
        base_info.update({
            'max_drawdown_threshold': self.max_drawdown,
            'warning_threshold': self.warning_threshold,
            'acceleration_threshold': self.drawdown_acceleration_threshold,
            'portfolio_peak': self.portfolio_peak,
            'last_peak_time': self.last_peak_time.isoformat(),
            'current_stats': drawdown_stats,
            'recovery_analysis': recovery_analysis
        })
        
        return base_info