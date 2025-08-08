"""
Anomaly Detection Types and Enums

This module defines the core types and enums used throughout the anomaly detection system.
Provides clean type definitions for anomaly classification and severity levels.
"""

from enum import Enum


class AnomalyType(Enum):
    """Types of market anomalies."""
    PRICE_SPIKE = "price_spike"
    PRICE_CRASH = "price_crash"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    REGIME_CHANGE = "regime_change"
    FLASH_CRASH = "flash_crash"
    CIRCUIT_BREAKER_RISK = "circuit_breaker_risk"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MARKET_MANIPULATION = "market_manipulation"


class AnomalySeverity(Enum):
    """Anomaly severity levels."""
    LOW = "low"          # 2-3 sigma event
    MEDIUM = "medium"    # 3-4 sigma event
    HIGH = "high"        # 4-5 sigma event
    CRITICAL = "critical"  # >5 sigma event