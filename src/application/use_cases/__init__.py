"""
Application Use Cases Layer

This layer contains the business logic orchestration for the trading system.
Use cases coordinate between domain services and repositories to implement
business workflows while maintaining clean architecture boundaries.
"""

# Trading use cases
# Market data use cases
from .market_data import GetHistoricalDataUseCase, GetLatestPriceUseCase, GetMarketDataUseCase

# Market simulation use cases
from .market_simulation import (
    CheckOrderTriggerUseCase,
    ProcessPendingOrdersUseCase,
    UpdateMarketPriceUseCase,
)

# Order execution use cases
from .order_execution import (
    CalculateCommissionUseCase,
    ProcessOrderFillUseCase,
    SimulateOrderExecutionUseCase,
)

# Portfolio management use cases
from .portfolio import (
    CalculatePortfolioMetricsUseCase,
    ClosePositionUseCase,
    OpenPositionUseCase,
    UpdatePositionUseCase,
    ValidatePortfolioUseCase,
)

# Risk management use cases
from .risk import CalculateRiskUseCase, GetRiskMetricsUseCase, ValidateOrderRiskUseCase
from .trading import (
    CancelOrderUseCase,
    GetOrderStatusUseCase,
    ModifyOrderUseCase,
    PlaceOrderUseCase,
)

__all__ = [
    # Trading use cases
    "PlaceOrderUseCase",
    "CancelOrderUseCase",
    "ModifyOrderUseCase",
    "GetOrderStatusUseCase",
    # Order execution use cases
    "ProcessOrderFillUseCase",
    "SimulateOrderExecutionUseCase",
    "CalculateCommissionUseCase",
    # Market simulation use cases
    "UpdateMarketPriceUseCase",
    "ProcessPendingOrdersUseCase",
    "CheckOrderTriggerUseCase",
    # Portfolio use cases
    "OpenPositionUseCase",
    "ClosePositionUseCase",
    "UpdatePositionUseCase",
    "CalculatePortfolioMetricsUseCase",
    "ValidatePortfolioUseCase",
    # Risk management use cases
    "CalculateRiskUseCase",
    "ValidateOrderRiskUseCase",
    "GetRiskMetricsUseCase",
    # Market data use cases
    "GetMarketDataUseCase",
    "GetLatestPriceUseCase",
    "GetHistoricalDataUseCase",
]
