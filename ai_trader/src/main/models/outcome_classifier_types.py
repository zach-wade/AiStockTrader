"""
Data types for outcome classifier system.

This module defines the data structures used throughout the outcome
classification system for labeling training data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from decimal import Decimal


class OutcomeLabel(Enum):
    """Outcome labels for position results."""
    STRONG_WIN = "strong_win"      # > 5% gain
    WIN = "win"                    # 2-5% gain
    NEUTRAL = "neutral"            # -2% to 2%
    LOSS = "loss"                  # -5% to -2%
    STRONG_LOSS = "strong_loss"    # < -5% loss
    UNKNOWN = "unknown"            # Not enough data
    
    @classmethod
    def from_return(cls, return_pct: float, thresholds: Optional[Dict[str, float]] = None) -> 'OutcomeLabel':
        """
        Convert return percentage to outcome label.
        
        Args:
            return_pct: Return percentage (e.g., 0.05 for 5%)
            thresholds: Optional custom thresholds
            
        Returns:
            Outcome label
        """
        if thresholds is None:
            thresholds = {
                'strong_win': 0.05,
                'win': 0.02,
                'loss': -0.02,
                'strong_loss': -0.05
            }
            
        if return_pct >= thresholds['strong_win']:
            return cls.STRONG_WIN
        elif return_pct >= thresholds['win']:
            return cls.WIN
        elif return_pct <= thresholds['strong_loss']:
            return cls.STRONG_LOSS
        elif return_pct <= thresholds['loss']:
            return cls.LOSS
        else:
            return cls.NEUTRAL


class TimeHorizon(Enum):
    """Time horizons for outcome evaluation."""
    INTRADAY = "intraday"      # Same day
    SHORT = "short"            # 1-5 days
    MEDIUM = "medium"          # 5-20 days
    LONG = "long"              # 20+ days
    
    @classmethod
    def from_days(cls, days: int) -> 'TimeHorizon':
        """Get time horizon from number of days."""
        if days < 1:
            return cls.INTRADAY
        elif days <= 5:
            return cls.SHORT
        elif days <= 20:
            return cls.MEDIUM
        else:
            return cls.LONG


@dataclass
class OutcomeMetrics:
    """Metrics calculated for position outcomes."""
    entry_price: Decimal
    exit_price: Optional[Decimal]
    max_price: Decimal
    min_price: Decimal
    holding_period_days: float
    return_pct: float
    max_drawdown_pct: float
    max_gain_pct: float
    volatility: float
    sharpe_ratio: Optional[float] = None
    volume_profile: Optional[Dict[str, float]] = None
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio."""
        if self.max_drawdown_pct > 0:
            return abs(self.max_gain_pct / self.max_drawdown_pct)
        return None


@dataclass
class LabelingConfig:
    """Configuration for outcome labeling."""
    # Return thresholds
    return_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'strong_win': 0.05,
        'win': 0.02,
        'loss': -0.02,
        'strong_loss': -0.05
    })
    
    # Time horizons to evaluate (in days)
    evaluation_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Minimum data requirements
    min_data_points: int = 20
    min_volume: int = 10000
    
    # Risk adjustments
    use_risk_adjusted_returns: bool = True
    benchmark_symbol: str = "SPY"
    
    # Exit criteria
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.03


@dataclass
class SignalOutcome:
    """Outcome data for a trading signal."""
    signal_id: str
    symbol: str
    signal_timestamp: datetime
    signal_strength: float
    signal_metadata: Dict[str, Any]
    
    # Outcomes at different horizons
    outcomes: Dict[TimeHorizon, OutcomeMetrics]
    labels: Dict[TimeHorizon, OutcomeLabel]
    
    # Additional context
    market_conditions: Optional[Dict[str, Any]] = None
    execution_quality: Optional[Dict[str, Any]] = None
    
    def get_best_horizon(self) -> Tuple[TimeHorizon, OutcomeLabel]:
        """Get the time horizon with best outcome."""
        best_return = float('-inf')
        best_horizon = TimeHorizon.SHORT
        best_label = OutcomeLabel.UNKNOWN
        
        for horizon, metrics in self.outcomes.items():
            if metrics.return_pct > best_return:
                best_return = metrics.return_pct
                best_horizon = horizon
                best_label = self.labels[horizon]
                
        return best_horizon, best_label


@dataclass
class TrainingExample:
    """Training example for ML models."""
    features: Dict[str, float]
    label: OutcomeLabel
    metadata: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'features': self.features,
            'label': self.label.value,
            'metadata': self.metadata,
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """Create from dictionary."""
        return cls(
            features=data['features'],
            label=OutcomeLabel(data['label']),
            metadata=data['metadata'],
            weight=data.get('weight', 1.0)
        )


@dataclass
class OutcomeStatistics:
    """Statistics for outcome classification results."""
    total_signals: int
    labeled_signals: int
    unlabeled_signals: int
    
    # Label distribution
    label_counts: Dict[OutcomeLabel, int]
    horizon_counts: Dict[TimeHorizon, int]
    
    # Performance metrics
    avg_return_by_label: Dict[OutcomeLabel, float]
    avg_holding_period_by_label: Dict[OutcomeLabel, float]
    win_rate: float
    
    # Data quality
    signals_with_insufficient_data: int
    signals_with_execution_issues: int
    
    def get_label_distribution(self) -> Dict[OutcomeLabel, float]:
        """Get percentage distribution of labels."""
        total = sum(self.label_counts.values())
        if total == 0:
            return {}
            
        return {
            label: count / total
            for label, count in self.label_counts.items()
        }
    
    def get_class_balance_score(self) -> float:
        """
        Calculate class balance score (0-1).
        1 = perfectly balanced, 0 = completely imbalanced.
        """
        distribution = self.get_label_distribution()
        if not distribution:
            return 0.0
            
        # Calculate entropy as measure of balance
        import math
        entropy = -sum(p * math.log(p) for p in distribution.values() if p > 0)
        max_entropy = math.log(len(distribution))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0