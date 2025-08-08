"""
Correlation Anomaly Detection Engine

This module provides correlation analysis capabilities for detecting correlation breakdown
and systemic risk events in portfolio and market data.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any
import numpy as np
from collections import deque

from .anomaly_types import AnomalyType, AnomalySeverity
from .anomaly_models import AnomalyEvent, CorrelationMatrix

logger = logging.getLogger(__name__)


class CorrelationAnomalyDetector:
    """Detect correlation breakdown and systemic risk."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize correlation detector."""
        self.config = config or {}
        self.baseline_correlations: Dict[str, np.ndarray] = {}
        self.correlation_history: deque = deque(maxlen=self.config.get('history_size', 100))
        
        # Detection thresholds
        self.breakdown_threshold = self.config.get('breakdown_threshold', 0.3)
        self.severe_threshold = self.config.get('severe_threshold', 0.6)
        self.high_threshold = self.config.get('high_threshold', 0.5)
        
        logger.info(f"Correlation detector initialized with breakdown threshold: {self.breakdown_threshold}")
        
    def update_baseline_correlation(self, symbols: List[str], correlation_matrix: np.ndarray):
        """Update baseline correlation matrix."""
        
        key = "_".join(sorted(symbols))
        self.baseline_correlations[key] = correlation_matrix.copy()
        logger.debug(f"Updated baseline correlation for {len(symbols)} symbols")
        
    def detect_correlation_breakdown(self,
                                   symbols: List[str],
                                   current_correlations: np.ndarray) -> List[AnomalyEvent]:
        """Detect correlation breakdown events."""
        
        key = "_".join(sorted(symbols))
        
        if key not in self.baseline_correlations:
            # Store current as baseline if no baseline exists
            self.update_baseline_correlation(symbols, current_correlations)
            return []
        
        baseline = self.baseline_correlations[key]
        
        # Calculate correlation changes
        correlation_change = np.abs(current_correlations - baseline)
        
        # Mask diagonal (self-correlations)
        mask = np.eye(len(symbols), dtype=bool)
        correlation_change[mask] = 0
        
        # Calculate breakdown metrics
        max_change = np.max(correlation_change)
        avg_change = np.mean(correlation_change[~mask])
        
        # Store correlation matrix
        correlation_data = CorrelationMatrix(
            timestamp=datetime.now(timezone.utc),
            symbols=symbols,
            correlation_matrix=current_correlations,
            baseline_correlation=baseline,
            correlation_change=correlation_change,
            max_correlation_change=max_change,
            avg_correlation_change=avg_change,
            breakdown_score=max_change * 100,
            systemic_risk_score=self._calculate_systemic_risk(current_correlations),
            diversification_score=self._calculate_diversification_score(current_correlations)
        )
        
        self.correlation_history.append(correlation_data)
        
        anomalies = []
        
        # Check for significant correlation breakdown
        if max_change > self.breakdown_threshold:
            severity = self._get_breakdown_severity(max_change)
            
            anomaly = AnomalyEvent(
                timestamp=datetime.now(timezone.utc),
                symbol="PORTFOLIO",
                anomaly_type=AnomalyType.CORRELATION_BREAKDOWN,
                severity=severity,
                z_score=max_change / 0.1,  # Normalized to 0.1 as standard unit
                p_value=0.01,  # Simplified
                confidence_level=95.0,
                current_value=max_change,
                expected_value=0.0,
                deviation=max_change,
                lookback_window=len(self.correlation_history),
                market_context={
                    'avg_correlation_change': avg_change,
                    'systemic_risk_score': correlation_data.systemic_risk_score,
                    'diversification_score': correlation_data.diversification_score
                },
                contributing_factors=self._identify_correlation_factors(correlation_data),
                risk_score=min(100, max_change * 200),
                trading_halt_recommended=max_change > 0.5,
                position_reduction_recommended=max_change > 0.4,
                detection_method="correlation_matrix_comparison",
                model_confidence=85.0,
                message=f"Correlation breakdown detected: max change {max_change:.1%}, avg change {avg_change:.1%}"
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def get_correlation_history(self, limit: int = None) -> List[CorrelationMatrix]:
        """Get correlation history with optional limit."""
        history = list(self.correlation_history)
        if limit:
            history = history[-limit:]
        return history
    
    def get_current_systemic_risk(self, symbols: List[str]) -> float:
        """Get current systemic risk score for given symbols."""
        if not self.correlation_history:
            return 0.0
        
        # Find most recent correlation data for these symbols
        key = "_".join(sorted(symbols))
        for correlation_data in reversed(self.correlation_history):
            if "_".join(sorted(correlation_data.symbols)) == key:
                return correlation_data.systemic_risk_score
        
        return 0.0
    
    def get_diversification_score(self, symbols: List[str]) -> float:
        """Get current diversification score for given symbols."""
        if not self.correlation_history:
            return 0.0
        
        # Find most recent correlation data for these symbols
        key = "_".join(sorted(symbols))
        for correlation_data in reversed(self.correlation_history):
            if "_".join(sorted(correlation_data.symbols)) == key:
                return correlation_data.diversification_score
        
        return 0.0
    
    def _get_breakdown_severity(self, max_change: float) -> AnomalySeverity:
        """Determine severity based on correlation change magnitude."""
        if max_change > self.severe_threshold:
            return AnomalySeverity.CRITICAL
        elif max_change > self.high_threshold:
            return AnomalySeverity.HIGH
        elif max_change > self.breakdown_threshold:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def _calculate_systemic_risk(self, correlation_matrix: np.ndarray) -> float:
        """Calculate systemic risk score from correlation matrix."""
        
        # High average correlation indicates systemic risk
        mask = np.eye(len(correlation_matrix), dtype=bool)
        off_diagonal = correlation_matrix[~mask]
        avg_correlation = np.mean(np.abs(off_diagonal))
        
        # Score 0-100, where higher correlation = higher systemic risk
        return min(100, avg_correlation * 150)
    
    def _calculate_diversification_score(self, correlation_matrix: np.ndarray) -> float:
        """Calculate diversification score from correlation matrix."""
        
        # Lower average correlation = better diversification
        mask = np.eye(len(correlation_matrix), dtype=bool)
        off_diagonal = correlation_matrix[~mask]
        avg_correlation = np.mean(np.abs(off_diagonal))
        
        # Score 0-100, where lower correlation = higher diversification
        return max(0, 100 - avg_correlation * 150)
    
    def _identify_correlation_factors(self, correlation_data: CorrelationMatrix) -> List[str]:
        """Identify factors contributing to correlation breakdown."""
        
        factors = []
        
        if correlation_data.max_correlation_change > 0.5:
            factors.append("Severe correlation breakdown")
        
        if correlation_data.systemic_risk_score > 80:
            factors.append("High systemic risk")
        
        if correlation_data.diversification_score < 20:
            factors.append("Poor diversification")
        
        if correlation_data.avg_correlation_change > 0.3:
            factors.append("Widespread correlation changes")
        
        return factors