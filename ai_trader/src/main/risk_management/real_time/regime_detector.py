"""
Market Regime Detection Engine

This module provides market regime detection capabilities for identifying
significant changes in market behavior, volatility patterns, and risk levels.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np
import statistics

from .anomaly_types import AnomalyType, AnomalySeverity
from .anomaly_models import AnomalyEvent, MarketRegime

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market regime changes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize regime detector."""
        self.config = config or {}
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[MarketRegime] = []
        
        # Regime thresholds
        self.volatility_thresholds = {
            'low': self.config.get('vol_low', 0.15),    # <15% annual volatility
            'medium': self.config.get('vol_medium', 0.25), # 15-25% annual volatility
            'high': self.config.get('vol_high', 0.40),   # 25-40% annual volatility
            'extreme': self.config.get('vol_extreme', 1.0)  # >40% annual volatility
        }
        
        # Detection thresholds
        self.volatility_change_threshold = self.config.get('vol_change_threshold', 0.10)
        self.trend_change_threshold = self.config.get('trend_change_threshold', 0.02)
        self.min_data_points = self.config.get('min_data_points', 50)
        
        logger.info(f"Regime detector initialized with volatility thresholds: {self.volatility_thresholds}")
    
    def detect_regime_change(self,
                           price_history: List[float],
                           volume_history: List[int] = None) -> List[AnomalyEvent]:
        """Detect market regime changes."""
        
        if len(price_history) < self.min_data_points:
            return []
        
        # Calculate current regime characteristics
        current_regime = self._calculate_current_regime(price_history, volume_history or [])
        
        anomalies = []
        
        # Check for regime change
        if self.current_regime and self._is_regime_change(self.current_regime, current_regime):
            
            # Create regime change anomaly
            anomaly = AnomalyEvent(
                timestamp=datetime.now(timezone.utc),
                symbol="MARKET",
                anomaly_type=AnomalyType.REGIME_CHANGE,
                severity=self._determine_regime_change_severity(self.current_regime, current_regime),
                z_score=3.0,  # Simplified
                p_value=0.01,
                confidence_level=95.0,
                current_value=current_regime.stability_score,
                expected_value=self.current_regime.stability_score,
                deviation=abs(current_regime.stability_score - self.current_regime.stability_score),
                lookback_window=self.min_data_points,
                market_context={
                    'previous_regime': self.current_regime.name,
                    'new_regime': current_regime.name,
                    'risk_level_change': f"{self.current_regime.risk_level} -> {current_regime.risk_level}",
                    'volatility_change': current_regime.avg_volatility - self.current_regime.avg_volatility
                },
                contributing_factors=self._identify_regime_change_factors(self.current_regime, current_regime),
                risk_score=self._calculate_regime_risk_score(current_regime),
                trading_halt_recommended=current_regime.risk_level == 'extreme',
                position_reduction_recommended=current_regime.risk_level in ['high', 'extreme'],
                detection_method="regime_characteristics_analysis",
                model_confidence=current_regime.confidence,
                message=f"Market regime change: {self.current_regime.name} -> {current_regime.name}"
            )
            
            anomalies.append(anomaly)
            
            # Archive previous regime
            if self.current_regime:
                self.regime_history.append(self.current_regime)
        
        # Update current regime
        self.current_regime = current_regime
        
        return anomalies
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get current market regime."""
        return self.current_regime
    
    def get_regime_history(self, limit: int = None) -> List[MarketRegime]:
        """Get regime history with optional limit."""
        history = self.regime_history
        if limit:
            history = history[-limit:]
        return history
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics."""
        if not self.regime_history:
            return {}
        
        # Calculate regime durations
        durations = []
        for i in range(1, len(self.regime_history)):
            duration = (self.regime_history[i].start_time - self.regime_history[i-1].start_time).total_seconds()
            durations.append(duration)
        
        # Risk level distribution
        risk_levels = [regime.risk_level for regime in self.regime_history]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
        
        return {
            'total_regimes': len(self.regime_history),
            'current_regime': self.current_regime.name if self.current_regime else None,
            'avg_regime_duration_minutes': statistics.mean(durations) / 60 if durations else 0,
            'risk_level_distribution': risk_distribution,
            'current_risk_level': self.current_regime.risk_level if self.current_regime else None,
            'current_volatility': self.current_regime.avg_volatility if self.current_regime else None
        }
    
    def _calculate_current_regime(self, price_history: List[float], volume_history: List[int]) -> MarketRegime:
        """Calculate current market regime characteristics."""
        
        # Calculate returns and volatility
        returns = np.diff(np.log(price_history[-30:]))  # Last 30 periods
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Determine volatility regime
        vol_regime = 'low'
        for regime, threshold in self.volatility_thresholds.items():
            if volatility < threshold:
                vol_regime = regime
                break
        else:
            vol_regime = 'extreme'
        
        # Determine trend direction
        short_ma = np.mean(price_history[-10:])  # 10-period MA
        long_ma = np.mean(price_history[-30:])   # 30-period MA
        
        if short_ma > long_ma * (1 + self.trend_change_threshold):
            trend = 'bullish'
        elif short_ma < long_ma * (1 - self.trend_change_threshold):
            trend = 'bearish'
        else:
            trend = 'sideways'
        
        # Create regime name
        regime_name = f"{vol_regime.title()} Volatility {trend.title()}"
        
        # Calculate stability score
        stability = max(0, 100 - volatility * 200)  # Higher volatility = lower stability
        
        # Risk level mapping
        risk_mapping = {
            'low': 'low' if trend != 'bearish' else 'medium',
            'medium': 'medium' if trend != 'bearish' else 'high',
            'high': 'high',
            'extreme': 'extreme'
        }
        
        risk_level = risk_mapping.get(vol_regime, 'high')
        
        # Calculate correlation (simplified - would use actual correlation matrix)
        avg_correlation = 0.5  # Placeholder
        
        return MarketRegime(
            regime_id=f"{int(datetime.now().timestamp())}",
            name=regime_name,
            start_time=datetime.now(timezone.utc),
            avg_volatility=volatility,
            avg_correlation=avg_correlation,
            trend_direction=trend,
            risk_level=risk_level,
            stability_score=stability,
            confidence=85.0  # Model confidence
        )
    
    def _is_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime) -> bool:
        """Determine if a regime change has occurred."""
        
        # Check for significant changes
        volatility_change = abs(new_regime.avg_volatility - old_regime.avg_volatility)
        trend_change = new_regime.trend_direction != old_regime.trend_direction
        risk_change = new_regime.risk_level != old_regime.risk_level
        
        # Regime change criteria
        return (volatility_change > self.volatility_change_threshold or  # Significant volatility change
                trend_change or              # Trend direction change
                risk_change)                 # Risk level change
    
    def _determine_regime_change_severity(self, old_regime: MarketRegime, new_regime: MarketRegime) -> AnomalySeverity:
        """Determine severity of regime change."""
        
        volatility_increase = new_regime.avg_volatility - old_regime.avg_volatility
        
        if new_regime.risk_level == 'extreme':
            return AnomalySeverity.CRITICAL
        elif volatility_increase > 0.20:  # 20% volatility increase
            return AnomalySeverity.HIGH
        elif new_regime.risk_level in ['high'] or volatility_increase > 0.10:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _identify_regime_change_factors(self, old_regime: MarketRegime, new_regime: MarketRegime) -> List[str]:
        """Identify factors causing regime change."""
        
        factors = []
        
        volatility_change = new_regime.avg_volatility - old_regime.avg_volatility
        if volatility_change > 0.15:
            factors.append("Significant volatility increase")
        elif volatility_change < -0.15:
            factors.append("Significant volatility decrease")
        
        if old_regime.trend_direction != new_regime.trend_direction:
            factors.append(f"Trend change: {old_regime.trend_direction} -> {new_regime.trend_direction}")
        
        if old_regime.risk_level != new_regime.risk_level:
            factors.append(f"Risk level change: {old_regime.risk_level} -> {new_regime.risk_level}")
        
        # Stability change
        stability_change = new_regime.stability_score - old_regime.stability_score
        if stability_change < -20:
            factors.append("Significant stability decrease")
        elif stability_change > 20:
            factors.append("Significant stability increase")
        
        return factors
    
    def _calculate_regime_risk_score(self, regime: MarketRegime) -> float:
        """Calculate risk score for a regime."""
        
        risk_scores = {
            'low': 20,
            'medium': 40,
            'high': 70,
            'extreme': 95
        }
        
        base_score = risk_scores.get(regime.risk_level, 50)
        
        # Adjust for volatility and stability
        vol_adjustment = min(30, regime.avg_volatility * 100)
        stability_adjustment = (100 - regime.stability_score) * 0.3
        
        return min(100, base_score + vol_adjustment + stability_adjustment)