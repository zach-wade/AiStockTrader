"""
Feature configuration and mappings.

This module contains configuration initialization, alert mappings,
and other configuration-related functionality for the feature pipeline.
"""

from typing import Dict, List, Optional, Any
from dataclasses import field

from main.utils.core import get_logger
from main.events.types import AlertType
from main.events.handlers.feature_pipeline_helpers.feature_types import (
    FeatureGroup,
    FeatureGroupConfig
)

logger = get_logger(__name__)


def initialize_group_configs() -> Dict[FeatureGroup, FeatureGroupConfig]:
    """Initialize feature group configurations."""
    return {
        # Market data features
        FeatureGroup.PRICE: FeatureGroupConfig(
            name="Price Features",
            description="Basic price-based features",
            required_data=["prices", "quotes"],
            computation_params={
                "lookback_periods": [5, 10, 20, 50],
                "return_periods": [1, 5, 20]
            }
        ),
        
        FeatureGroup.VOLUME: FeatureGroupConfig(
            name="Volume Features",
            description="Volume and liquidity features",
            required_data=["trades", "quotes"],
            computation_params={
                "volume_windows": [5, 10, 30],
                "vwap_periods": [5, 15, 60]
            }
        ),
        
        FeatureGroup.VOLATILITY: FeatureGroupConfig(
            name="Volatility Features",
            description="Volatility and variance features",
            required_data=["prices"],
            dependencies=[FeatureGroup.PRICE],
            computation_params={
                "volatility_windows": [10, 20, 50],
                "ewm_spans": [10, 20]
            }
        ),
        
        FeatureGroup.MOMENTUM: FeatureGroupConfig(
            name="Momentum Features",
            description="Momentum and trend strength indicators",
            required_data=["prices"],
            dependencies=[FeatureGroup.PRICE],
            computation_params={
                "rsi_periods": [9, 14, 21],
                "macd_params": [(12, 26, 9)]
            }
        ),
        
        FeatureGroup.TREND: FeatureGroupConfig(
            name="Trend Features",
            description="Trend identification and strength",
            required_data=["prices"],
            dependencies=[FeatureGroup.PRICE, FeatureGroup.MOMENTUM],
            computation_params={
                "ma_periods": [10, 20, 50, 200],
                "channel_periods": [20, 50]
            }
        ),
        
        # Technical indicators
        FeatureGroup.TECHNICAL_BASIC: FeatureGroupConfig(
            name="Basic Technical Indicators",
            description="Common technical analysis indicators",
            required_data=["prices", "volume"],
            dependencies=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            computation_params={
                "bb_periods": [20],
                "bb_stds": [2],
                "stoch_periods": [(14, 3)]
            }
        ),
        
        FeatureGroup.TECHNICAL_ADVANCED: FeatureGroupConfig(
            name="Advanced Technical Indicators",
            description="Complex technical indicators",
            required_data=["prices", "volume"],
            dependencies=[
                FeatureGroup.TECHNICAL_BASIC,
                FeatureGroup.VOLATILITY,
                FeatureGroup.MOMENTUM
            ],
            computation_params={
                "ichimoku_params": [(9, 26, 52)],
                "pivot_types": ["standard", "fibonacci"]
            },
            priority_boost=1
        ),
        
        # Market microstructure
        FeatureGroup.MICROSTRUCTURE: FeatureGroupConfig(
            name="Market Microstructure Features",
            description="Order book and market microstructure features",
            required_data=["order_book", "trades"],
            computation_params={
                "depth_levels": [5, 10],
                "imbalance_windows": [1, 5, 15]
            },
            priority_boost=2
        ),
        
        FeatureGroup.ORDER_FLOW: FeatureGroupConfig(
            name="Order Flow Features",
            description="Order flow analysis features",
            required_data=["trades", "order_book"],
            dependencies=[FeatureGroup.MICROSTRUCTURE],
            computation_params={
                "flow_windows": [5, 15, 60],
                "aggressor_side": True
            },
            priority_boost=2
        ),
        
        # Sentiment features
        FeatureGroup.NEWS_SENTIMENT: FeatureGroupConfig(
            name="News Sentiment Features",
            description="News-based sentiment analysis",
            required_data=["news"],
            computation_params={
                "lookback_hours": [1, 6, 24],
                "sentiment_model": "finbert"
            }
        ),
        
        FeatureGroup.SOCIAL_SENTIMENT: FeatureGroupConfig(
            name="Social Sentiment Features",
            description="Social media sentiment analysis",
            required_data=["social_media"],
            computation_params={
                "platforms": ["twitter", "reddit"],
                "lookback_hours": [1, 4, 12]
            }
        ),
        
        # Risk features
        FeatureGroup.RISK_METRICS: FeatureGroupConfig(
            name="Risk Metrics",
            description="Risk and drawdown metrics",
            required_data=["prices", "returns"],
            dependencies=[FeatureGroup.PRICE, FeatureGroup.VOLATILITY],
            computation_params={
                "var_confidence": [0.95, 0.99],
                "cvar_confidence": [0.95]
            }
        ),
        
        FeatureGroup.CORRELATION: FeatureGroupConfig(
            name="Correlation Features",
            description="Cross-asset correlation features",
            required_data=["prices", "market_indices"],
            dependencies=[FeatureGroup.PRICE],
            computation_params={
                "correlation_windows": [20, 60, 252],
                "reference_indices": ["SPY", "QQQ", "IWM"]
            }
        ),
        
        # Event-driven features
        FeatureGroup.EARNINGS: FeatureGroupConfig(
            name="Earnings Features",
            description="Earnings-related features",
            required_data=["earnings_calendar", "historical_earnings"],
            computation_params={
                "lookback_quarters": 4,
                "forward_estimates": True
            }
        ),
        
        FeatureGroup.CORPORATE_ACTIONS: FeatureGroupConfig(
            name="Corporate Action Features",
            description="Features related to corporate actions",
            required_data=["corporate_actions"],
            computation_params={
                "action_types": ["dividend", "split", "merger"]
            }
        ),
        
        # ML features
        FeatureGroup.ML_SIGNALS: FeatureGroupConfig(
            name="ML Model Signals",
            description="Pre-computed ML model outputs",
            required_data=["model_predictions"],
            dependencies=[
                FeatureGroup.PRICE,
                FeatureGroup.VOLUME,
                FeatureGroup.TECHNICAL_BASIC
            ],
            computation_params={
                "models": ["momentum_classifier", "mean_reversion_regressor"]
            },
            priority_boost=3
        ),
        
        FeatureGroup.EMBEDDINGS: FeatureGroupConfig(
            name="Feature Embeddings",
            description="Deep learning embeddings",
            required_data=["raw_features"],
            dependencies=[FeatureGroup.ML_SIGNALS],
            computation_params={
                "embedding_dims": 64,
                "model_version": "v2.0"
            },
            priority_boost=3
        )
    }


def initialize_alert_mappings() -> Dict[AlertType, List[FeatureGroup]]:
    """Initialize mappings from alert types to feature groups."""
    return {
        # Scanner alerts
        AlertType.HIGH_VOLUME: [
            FeatureGroup.VOLUME,
            FeatureGroup.PRICE,
            FeatureGroup.VOLATILITY,
            FeatureGroup.ORDER_FLOW
        ],
        
        AlertType.PRICE_BREAKOUT: [
            FeatureGroup.PRICE,
            FeatureGroup.TREND,
            FeatureGroup.MOMENTUM,
            FeatureGroup.TECHNICAL_BASIC
        ],
        
        AlertType.VOLATILITY_SPIKE: [
            FeatureGroup.VOLATILITY,
            FeatureGroup.PRICE,
            FeatureGroup.RISK_METRICS,
            FeatureGroup.MICROSTRUCTURE
        ],
        
        AlertType.MOMENTUM_SHIFT: [
            FeatureGroup.MOMENTUM,
            FeatureGroup.TREND,
            FeatureGroup.VOLUME,
            FeatureGroup.TECHNICAL_ADVANCED
        ],
        
        AlertType.UNUSUAL_OPTIONS: [
            FeatureGroup.MICROSTRUCTURE,
            FeatureGroup.ORDER_FLOW,
            FeatureGroup.VOLATILITY,
            FeatureGroup.RISK_METRICS
        ],
        
        # News and sentiment alerts
        AlertType.NEWS_SENTIMENT: [
            FeatureGroup.NEWS_SENTIMENT,
            FeatureGroup.PRICE,
            FeatureGroup.VOLUME,
            FeatureGroup.VOLATILITY
        ],
        
        AlertType.SOCIAL_SENTIMENT: [
            FeatureGroup.SOCIAL_SENTIMENT,
            FeatureGroup.NEWS_SENTIMENT,
            FeatureGroup.PRICE,
            FeatureGroup.VOLUME
        ],
        
        # Technical alerts
        AlertType.TECHNICAL_SIGNAL: [
            FeatureGroup.TECHNICAL_BASIC,
            FeatureGroup.TECHNICAL_ADVANCED,
            FeatureGroup.TREND,
            FeatureGroup.MOMENTUM
        ],
        
        AlertType.SUPPORT_RESISTANCE: [
            FeatureGroup.PRICE,
            FeatureGroup.TECHNICAL_BASIC,
            FeatureGroup.VOLUME,
            FeatureGroup.MICROSTRUCTURE
        ],
        
        # Risk alerts
        AlertType.RISK_ALERT: [
            FeatureGroup.RISK_METRICS,
            FeatureGroup.VOLATILITY,
            FeatureGroup.CORRELATION,
            FeatureGroup.PRICE
        ],
        
        # ML alerts
        AlertType.ML_SIGNAL: [
            FeatureGroup.ML_SIGNALS,
            FeatureGroup.EMBEDDINGS,
            FeatureGroup.TECHNICAL_ADVANCED,
            FeatureGroup.MICROSTRUCTURE
        ],
        
        # Default mapping
        AlertType.UNKNOWN: [
            FeatureGroup.PRICE,
            FeatureGroup.VOLUME
        ]
    }


def get_conditional_group_rules() -> Dict[str, Any]:
    """
    Get rules for conditionally adding feature groups based on alert data.
    
    Returns:
        Dictionary of rules for conditional feature group addition
    """
    return {
        "high_score_threshold": 0.8,
        "high_score_additions": [
            FeatureGroup.ML_SIGNALS,
            FeatureGroup.TECHNICAL_ADVANCED
        ],
        
        "volume_spike_multiplier": 3.0,
        "volume_spike_additions": [
            FeatureGroup.ORDER_FLOW,
            FeatureGroup.MICROSTRUCTURE
        ],
        
        "news_keyword_additions": {
            "earnings": [FeatureGroup.EARNINGS],
            "merger": [FeatureGroup.CORPORATE_ACTIONS],
            "dividend": [FeatureGroup.CORPORATE_ACTIONS]
        },
        
        "time_based_additions": {
            "pre_market": [FeatureGroup.NEWS_SENTIMENT],
            "post_market": [FeatureGroup.SOCIAL_SENTIMENT],
            "earnings_season": [FeatureGroup.EARNINGS]
        }
    }


def get_priority_calculation_rules() -> Dict[str, Any]:
    """
    Get rules for calculating request priority.
    
    Returns:
        Dictionary of priority calculation rules
    """
    return {
        "base_priority_map": {
            AlertType.ML_SIGNAL: 8,
            AlertType.UNUSUAL_OPTIONS: 7,
            AlertType.VOLATILITY_SPIKE: 6,
            AlertType.MOMENTUM_SHIFT: 5,
            AlertType.PRICE_BREAKOUT: 5,
            AlertType.HIGH_VOLUME: 4,
            AlertType.TECHNICAL_SIGNAL: 3,
            AlertType.NEWS_SENTIMENT: 3,
            AlertType.SOCIAL_SENTIMENT: 2,
            AlertType.SUPPORT_RESISTANCE: 2,
            AlertType.RISK_ALERT: 1,
            AlertType.UNKNOWN: 0
        },
        
        "score_multiplier": 3,  # alert.score * multiplier added to base
        "max_priority": 10,
        "min_priority": 0,
        
        "market_phase_boost": {
            "pre_market": 2,
            "market_open": 1,
            "market_close": 1,
            "post_market": 0
        },
        
        "volatility_boost": {
            "low": 0,
            "normal": 0,
            "high": 1,
            "extreme": 2
        }
    }