"""
Trading execution algorithms module.

This module provides various execution algorithms for optimal order execution.
"""

from .base_algorithm import (
    BaseAlgorithm,
    ExecutionStatus,
    OrderSide,
    SlicingStrategy,
    ExecutionParameters,
    ChildOrder,
    ExecutionState,
    ExecutionSummary
)
from .twap import TWAPAlgorithm
from .vwap import VWAPAlgorithm, VolumeProfile
from .iceberg import IcebergAlgorithm

__all__ = [
    # Base classes and types
    'BaseAlgorithm',
    'ExecutionStatus',
    'OrderSide',
    'SlicingStrategy',
    'ExecutionParameters',
    'ChildOrder',
    'ExecutionState',
    'ExecutionSummary',
    
    # Algorithm implementations
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    'VolumeProfile',
    'IcebergAlgorithm'
]