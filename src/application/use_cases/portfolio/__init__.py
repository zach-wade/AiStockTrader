"""
Portfolio Use Cases Package

Contains all portfolio-related use cases that extract orchestration logic
from the Portfolio entity to follow Single Responsibility Principle.

These use cases handle portfolio management operations while maintaining
the exact same functionality as the original Portfolio entity methods.
"""

from .calculate_metrics_use_case import (
    CalculateMetricsRequest,
    CalculateMetricsResponse,
    CalculatePortfolioMetricsUseCase,
    PortfolioMetrics,
)
from .close_position_use_case import (
    ClosePositionRequest,
    ClosePositionResponse,
    ClosePositionUseCase,
)
from .open_position_use_case import OpenPositionRequest, OpenPositionResponse, OpenPositionUseCase
from .update_position_use_case import (
    UpdatePositionPricesRequest,
    UpdatePositionPricesUseCase,
    UpdatePositionRequest,
    UpdatePositionResponse,
    UpdatePositionUseCase,
)
from .validate_portfolio_use_case import (
    PositionValidationInfo,
    ValidatePortfolioRequest,
    ValidatePortfolioResponse,
    ValidatePortfolioUseCase,
)

__all__ = [
    # Open Position
    "OpenPositionUseCase",
    "OpenPositionRequest",
    "OpenPositionResponse",
    # Close Position
    "ClosePositionUseCase",
    "ClosePositionRequest",
    "ClosePositionResponse",
    # Update Position
    "UpdatePositionUseCase",
    "UpdatePositionRequest",
    "UpdatePositionResponse",
    "UpdatePositionPricesUseCase",
    "UpdatePositionPricesRequest",
    # Calculate Metrics
    "CalculatePortfolioMetricsUseCase",
    "CalculateMetricsRequest",
    "CalculateMetricsResponse",
    "PortfolioMetrics",
    # Validate Portfolio
    "ValidatePortfolioUseCase",
    "ValidatePortfolioRequest",
    "ValidatePortfolioResponse",
    "PositionValidationInfo",
]
