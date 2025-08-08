# trading_engine/core/__init__.py
"""
Trading Engine Core Components

Core execution, order management, and portfolio management.
"""

from .execution_engine import ExecutionEngine
from .order_manager import OrderManager
from .portfolio_manager import PortfolioManager
from ..risk.risk_manager import RiskManager

# New modular position management components
from .position_manager import PositionManager, create_position_manager
from .position_tracker import PositionTracker
from .fill_processor import FillProcessor, FillResult
from .position_validator import PositionValidator, ValidationResult
from .position_risk_validator import PositionRiskValidator, PositionRiskAssessment
from .broker_reconciler import BrokerReconciler, ReconciliationResult
from .position_events import (
    PositionEvent, PositionEventType, PositionOpenedEvent, PositionClosedEvent,
    PositionIncreasedEvent, PositionDecreasedEvent, PositionReversedEvent,
    FillProcessedEvent, PriceUpdateEvent, BrokerSyncEvent, RiskLimitBreachEvent,
    create_position_event
)

__all__ = [
    'ExecutionEngine',
    'OrderManager',
    'PortfolioManager',
    'RiskManager',
    'TradingEngine',
    # New modular position management
    'PositionManager',
    'create_position_manager',
    'PositionTracker',
    'FillProcessor',
    'FillResult',
    'PositionValidator',
    'ValidationResult',
    'PositionRiskValidator',
    'PositionRiskAssessment',
    'BrokerReconciler',
    'ReconciliationResult',
    'PositionEvent',
    'PositionEventType',
    'PositionOpenedEvent',
    'PositionClosedEvent',
    'PositionIncreasedEvent',
    'PositionDecreasedEvent',
    'PositionReversedEvent',
    'FillProcessedEvent',
    'PriceUpdateEvent',
    'BrokerSyncEvent',
    'RiskLimitBreachEvent',
    'create_position_event',
]

class TradingEngine:
    """Main trading engine that coordinates all components"""
    
    def __init__(self, broker, config=None):
        self.execution_engine = ExecutionEngine(broker)
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager(config)
        
    def start(self):
        """Start the trading engine"""
        self.execution_engine.start()
        self.order_manager.start()
        
    def stop(self):
        """Stop the trading engine"""
        self.execution_engine.stop()
        self.order_manager.stop()