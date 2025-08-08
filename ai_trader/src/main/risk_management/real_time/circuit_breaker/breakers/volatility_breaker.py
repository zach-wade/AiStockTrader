"""
Volatility Circuit Breaker

Monitors market volatility and triggers protection when volatility
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


class VolatilityBreaker(BaseBreaker):
    """
    Circuit breaker for volatility-based protection.
    
    Monitors:
    - Spot volatility levels
    - Volatility acceleration/trends
    - Volatility breakouts from normal ranges
    """
    
    def __init__(self, breaker_type: BreakerType, config: BreakerConfig):
        """Initialize volatility breaker."""
        super().__init__(breaker_type, config)
        
        # Volatility tracking
        self.volatility_history: deque = deque(maxlen=100)
        self.volatility_threshold = config.volatility_threshold
        self.volatility_acceleration_threshold = self.volatility_threshold / 10  # 10% of main threshold
        
        # Configuration
        self.min_history_length = 10  # Minimum data points for acceleration calculation
        self.warning_threshold = self.volatility_threshold * 0.8  # 80% of threshold
        
        logger.info(f"Volatility breaker initialized - threshold: {self.volatility_threshold:.2%}")
    
    async def check(self, 
                   portfolio_value: float,
                   positions: Dict[str, Any],
                   market_conditions: MarketConditions) -> bool:
        """
        Check if volatility breaker should trip.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            market_conditions: Current market conditions
            
        Returns:
            True if breaker should trip
        """
        if not self.is_enabled():
            return False
        
        # Update volatility history
        self.volatility_history.append((datetime.now(), market_conditions.volatility))
        
        # Check spot volatility
        if market_conditions.volatility > self.volatility_threshold:
            self.logger.warning(f"High volatility detected: {market_conditions.volatility:.2%} > {self.volatility_threshold:.2%}")
            return True
        
        # Check volatility acceleration
        if await self._check_volatility_acceleration():
            return True
        
        # Check volatility breakout
        if await self._check_volatility_breakout(market_conditions.volatility):
            return True
        
        return False
    
    async def check_warning_conditions(self, 
                                     portfolio_value: float,
                                     positions: Dict[str, Any],
                                     market_conditions: MarketConditions) -> bool:
        """
        Check if volatility breaker should be in warning state.
        
        Returns:
            True if breaker should be in warning state
        """
        if not self.is_enabled():
            return False
        
        # Warning if approaching volatility threshold
        if market_conditions.volatility > self.warning_threshold:
            self.logger.warning(f"Approaching volatility threshold: {market_conditions.volatility:.2%}")
            return True
        
        # Warning if volatility is accelerating
        if len(self.volatility_history) >= self.min_history_length:
            recent_vols = [v for _, v in list(self.volatility_history)[-5:]]
            if len(recent_vols) >= 2:
                vol_trend = (recent_vols[-1] - recent_vols[0]) / len(recent_vols)
                if vol_trend > self.volatility_acceleration_threshold / 2:
                    self.logger.warning(f"Volatility accelerating: {vol_trend:.4%}/period")
                    return True
        
        return False
    
    async def _check_volatility_acceleration(self) -> bool:
        """Check if volatility is accelerating rapidly."""
        if len(self.volatility_history) < self.min_history_length:
            return False
        
        # Calculate volatility acceleration over recent periods
        recent_vols = [v for _, v in list(self.volatility_history)[-self.min_history_length:]]
        vol_acceleration = (recent_vols[-1] - recent_vols[0]) / len(recent_vols)
        
        if vol_acceleration > self.volatility_acceleration_threshold:
            self.logger.warning(f"Volatility accelerating: {vol_acceleration:.4%}/period")
            return True
        
        return False
    
    async def _check_volatility_breakout(self, current_volatility: float) -> bool:
        """Check if volatility is breaking out from normal range."""
        if len(self.volatility_history) < 20:  # Need sufficient history
            return False
        
        # Calculate volatility statistics
        historical_vols = [v for _, v in list(self.volatility_history)[:-1]]  # Exclude current
        mean_vol = np.mean(historical_vols)
        std_vol = np.std(historical_vols)
        
        # Check if current volatility is a significant outlier
        if std_vol > 0:
            z_score = (current_volatility - mean_vol) / std_vol
            if z_score > 3.0:  # 3 standard deviations
                self.logger.warning(f"Volatility breakout detected: {z_score:.2f} standard deviations")
                return True
        
        return False
    
    def get_metrics(self) -> BreakerMetrics:
        """Get current volatility metrics."""
        metrics = BreakerMetrics()
        
        if self.volatility_history:
            current_vol = self.volatility_history[-1][1]
            metrics.recent_volatility = current_vol
            
            # Calculate volatility statistics
            if len(self.volatility_history) >= 2:
                historical_vols = [v for _, v in self.volatility_history]
                metrics.recent_volatility = np.mean(historical_vols[-10:]) if len(historical_vols) >= 10 else np.mean(historical_vols)
        
        return metrics
    
    def get_volatility_statistics(self) -> Dict[str, float]:
        """Get detailed volatility statistics."""
        if len(self.volatility_history) < 2:
            return {
                'current_volatility': 0.0,
                'mean_volatility': 0.0,
                'volatility_std': 0.0,
                'volatility_trend': 0.0,
                'time_above_threshold': 0.0
            }
        
        vols = [v for _, v in self.volatility_history]
        current_vol = vols[-1]
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        
        # Calculate trend
        if len(vols) >= 5:
            recent_vols = vols[-5:]
            volatility_trend = (recent_vols[-1] - recent_vols[0]) / len(recent_vols)
        else:
            volatility_trend = 0.0
        
        # Calculate time above threshold
        above_threshold = sum(1 for v in vols if v > self.warning_threshold)
        time_above_threshold = above_threshold / len(vols)
        
        return {
            'current_volatility': float(current_vol),
            'mean_volatility': float(mean_vol),
            'volatility_std': float(std_vol),
            'volatility_trend': float(volatility_trend),
            'time_above_threshold': float(time_above_threshold)
        }
    
    def reset_history(self):
        """Reset volatility history."""
        self.volatility_history.clear()
        self.logger.info("Volatility history reset")
    
    def get_info(self) -> Dict[str, Any]:
        """Get breaker information including volatility-specific details."""
        base_info = super().get_info()
        
        vol_stats = self.get_volatility_statistics()
        
        base_info.update({
            'volatility_threshold': self.volatility_threshold,
            'warning_threshold': self.warning_threshold,
            'acceleration_threshold': self.volatility_acceleration_threshold,
            'history_length': len(self.volatility_history),
            'current_stats': vol_stats
        })
        
        return base_info