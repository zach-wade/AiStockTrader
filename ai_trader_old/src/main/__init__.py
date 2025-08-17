"""
AI Trader - Algorithmic Trading System

A comprehensive trading system with data pipeline, backtesting, risk management,
and live trading capabilities.
"""

__version__ = "3.0.0"
__author__ = "AI Trader Team"

# Minimal imports to avoid circular dependencies
# Only import what's absolutely necessary at the module level

# Core components will be imported on-demand
# This prevents the circular import chain that was causing issues

__all__ = [
    # Core orchestration
    "UnifiedOrchestrator",
    "TradingSystem",
    "LiveRiskMonitor",
    # Scanners
    "ScanAlert",
    "AlertType",
    "Layer2CatalystOrchestrator",
    # Validation
    "ValidationPipeline",
    "ValidationStage",
    "validate_on_ingest",
    "validate_post_etl",
    "validate_pre_feature",
    # Configuration
    "get_config",
    # Utils
    "UniverseLoader",
    "UniverseHelpers",
    # Version info
    "__version__",
]
