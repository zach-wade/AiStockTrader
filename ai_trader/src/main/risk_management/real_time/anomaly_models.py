"""
Anomaly detection data models and configurations.

This module provides data models for anomaly events, market regimes,
and detection configurations used throughout the anomaly detection system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

from .anomaly_types import AnomalyType, AnomalySeverity


class RegimeType(Enum):
    """Market regime types."""
    BULL_QUIET = "bull_quiet"         # Uptrend with low volatility
    BULL_VOLATILE = "bull_volatile"   # Uptrend with high volatility
    BEAR_QUIET = "bear_quiet"         # Downtrend with low volatility
    BEAR_VOLATILE = "bear_volatile"   # Downtrend with high volatility
    SIDEWAYS = "sideways"             # Range-bound market
    CRISIS = "crisis"                 # Market crisis/crash
    RECOVERY = "recovery"             # Post-crisis recovery
    UNKNOWN = "unknown"               # Unclassified regime


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event."""
    event_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    symbol: Optional[str]
    timestamp: datetime
    message: str
    
    # Detection details
    detected_value: float
    expected_value: float
    threshold: float
    deviation: float  # Number of standard deviations
    
    # Model information
    detection_method: str
    model_confidence: float  # 0-1 confidence score
    false_positive_probability: float
    
    # Context
    market_regime: Optional['MarketRegime'] = None
    correlated_symbols: List[str] = field(default_factory=list)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    suggested_actions: List[str] = field(default_factory=list)
    auto_actions_taken: List[str] = field(default_factory=list)
    requires_manual_review: bool = False
    
    @property
    def deviation_percentage(self) -> float:
        """Calculate percentage deviation from expected."""
        if self.expected_value != 0:
            return abs((self.detected_value - self.expected_value) / self.expected_value) * 100
        return 0.0
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score (0-100) based on severity and confidence."""
        severity_weights = {
            AnomalySeverity.LOW: 25,
            AnomalySeverity.MEDIUM: 50,
            AnomalySeverity.HIGH: 75,
            AnomalySeverity.CRITICAL: 100
        }
        
        base_score = severity_weights.get(self.severity, 50)
        confidence_factor = self.model_confidence
        
        return base_score * confidence_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'detected_value': self.detected_value,
            'expected_value': self.expected_value,
            'threshold': self.threshold,
            'deviation': self.deviation,
            'deviation_percentage': self.deviation_percentage,
            'detection_method': self.detection_method,
            'model_confidence': self.model_confidence,
            'false_positive_probability': self.false_positive_probability,
            'risk_score': self.risk_score,
            'market_regime': self.market_regime.name if self.market_regime else None,
            'correlated_symbols': self.correlated_symbols,
            'suggested_actions': self.suggested_actions,
            'auto_actions_taken': self.auto_actions_taken,
            'requires_manual_review': self.requires_manual_review
        }


@dataclass
class MarketRegime:
    """Represents current market regime/state."""
    regime_type: RegimeType
    start_time: datetime
    confidence: float  # 0-1 confidence in regime classification
    
    # Regime characteristics
    trend_strength: float  # -1 to 1, negative for downtrend
    volatility_level: float  # 0-1, normalized volatility
    correlation_level: float  # Average pairwise correlation
    liquidity_score: float  # 0-1, market liquidity measure
    
    # Supporting metrics
    average_return: float
    return_volatility: float
    volume_ratio: float  # Current vs average volume
    breadth_indicator: float  # Advance/decline ratio
    
    # Transition probabilities
    transition_probabilities: Dict[RegimeType, float] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get regime duration in hours."""
        return (datetime.utcnow() - self.start_time).total_seconds() / 3600
    
    @property
    def is_high_risk(self) -> bool:
        """Check if regime is high risk."""
        return self.regime_type in [
            RegimeType.BEAR_VOLATILE,
            RegimeType.CRISIS
        ]
    
    @property
    def stability_score(self) -> float:
        """Calculate regime stability (0-1)."""
        # Lower score means less stable
        volatility_factor = 1 - self.volatility_level
        confidence_factor = self.confidence
        duration_factor = min(self.duration / 24, 1)  # More stable if lasting > 24h
        
        return (volatility_factor + confidence_factor + duration_factor) / 3


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection system."""
    # Detection thresholds
    price_spike_threshold: float = 3.0  # Standard deviations
    volume_surge_threshold: float = 4.0
    volatility_spike_threshold: float = 3.5
    correlation_breakdown_threshold: float = 0.3  # Correlation drop
    
    # Time windows
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'price': 100,
        'volume': 50,
        'volatility': 20,
        'correlation': 60
    })
    
    # Model parameters
    min_data_points: int = 30
    outlier_fraction: float = 0.05  # Expected fraction of outliers
    
    # Severity mapping
    severity_thresholds: Dict[AnomalySeverity, float] = field(default_factory=lambda: {
        AnomalySeverity.LOW: 2.0,
        AnomalySeverity.MEDIUM: 3.0,
        AnomalySeverity.HIGH: 4.0,
        AnomalySeverity.CRITICAL: 5.0
    })
    
    # Feature toggles
    enable_ml_detection: bool = True
    enable_statistical_detection: bool = True
    enable_pattern_detection: bool = True
    enable_correlation_monitoring: bool = True
    enable_regime_detection: bool = True
    
    # Response configuration
    auto_response_enabled: bool = False
    notification_cooldown: int = 300  # Seconds between notifications
    max_false_positives: int = 5  # Before disabling detector
    
    def get_threshold_for_type(self, anomaly_type: AnomalyType) -> float:
        """Get detection threshold for anomaly type."""
        thresholds = {
            AnomalyType.PRICE_SPIKE: self.price_spike_threshold,
            AnomalyType.PRICE_CRASH: self.price_spike_threshold,
            AnomalyType.VOLUME_SURGE: self.volume_surge_threshold,
            AnomalyType.VOLATILITY_SPIKE: self.volatility_spike_threshold,
            AnomalyType.CORRELATION_BREAKDOWN: self.correlation_breakdown_threshold
        }
        return thresholds.get(anomaly_type, 3.0)


@dataclass
class AnomalyStatistics:
    """Statistics for anomaly detection performance."""
    total_anomalies_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # By type
    anomalies_by_type: Dict[AnomalyType, int] = field(default_factory=dict)
    
    # By severity
    anomalies_by_severity: Dict[AnomalySeverity, int] = field(default_factory=dict)
    
    # Timing
    average_detection_latency: float = 0.0  # Milliseconds
    max_detection_latency: float = 0.0
    
    # Model performance
    model_accuracy: Dict[str, float] = field(default_factory=dict)
    
    @property
    def precision(self) -> float:
        """Calculate precision (true positives / all positives)."""
        total_positives = self.true_positives + self.false_positives
        if total_positives > 0:
            return self.true_positives / total_positives
        return 0.0
    
    @property
    def recall(self) -> float:
        """Calculate recall (true positives / all actual anomalies)."""
        total_actual = self.true_positives + self.false_negatives
        if total_actual > 0:
            return self.true_positives / total_actual
        return 0.0
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return 0.0


@dataclass
class AnomalyContext:
    """Context information for anomaly analysis."""
    # Market context
    market_hours: bool
    pre_market: bool
    after_hours: bool
    
    # News/events
    recent_news: List[str] = field(default_factory=list)
    scheduled_events: List[str] = field(default_factory=list)
    earnings_nearby: bool = False
    
    # Technical context
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend_direction: str = "neutral"  # "up", "down", "neutral"
    
    # Historical context
    similar_anomalies: List[str] = field(default_factory=list)  # Event IDs
    last_occurrence: Optional[datetime] = None
    typical_resolution_time: Optional[float] = None  # Hours
    
    # Risk context
    portfolio_exposure: float = 0.0
    correlated_positions: List[str] = field(default_factory=list)
    potential_impact: float = 0.0  # Estimated P&L impact