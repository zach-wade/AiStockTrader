"""
Trading execution algorithms module.

This module provides various execution algorithms for optimal order execution.
"""

from .base_algorithm import (
    BaseAlgorithm,
    ChildOrder,
    ExecutionParameters,
    ExecutionState,
    ExecutionStatus,
    ExecutionSummary,
    OrderSide,
    SlicingStrategy,
)
from .iceberg import IcebergAlgorithm
from .twap import TWAPAlgorithm
from .vwap import VolumeProfile, VWAPAlgorithm

__all__ = [
    # Base classes and types
    "BaseAlgorithm",
    "ExecutionStatus",
    "OrderSide",
    "SlicingStrategy",
    "ExecutionParameters",
    "ChildOrder",
    "ExecutionState",
    "ExecutionSummary",
    # Algorithm implementations
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "VolumeProfile",
    "IcebergAlgorithm",
]
