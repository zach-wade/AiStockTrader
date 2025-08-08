"""
AI Trader Scanner System

The scanner system implements the "Field Agents" in our Intelligence Agency architecture.
Scanners monitor the market for interesting signals and catalysts, producing standardized
ScanAlert objects that are consumed by the feature pipeline.

Architecture:
- layers/: Multi-layer filtering system (Universe → Liquidity → Strategy Affinity → Catalysts)
- catalysts/: Specialized scanners for different types of market events
- utils/: Common utilities and data models

Four-Layer Filtering Funnel:
- Layer 0: Static Universe (~8,000 symbols) - Quarterly
- Layer 1: Liquidity Filter (~1,500 symbols) - Nightly
- Layer 2: Catalyst Scanner (~100-200 symbols) - Nightly/Pre-market
- Layer 3: Pre-Market Confirmation (20-50 symbols) - 8:30-9:25 AM
"""

from main.events.types import ScanAlert, AlertType

__all__ = [
    'ScanAlert',
    'AlertType'
]