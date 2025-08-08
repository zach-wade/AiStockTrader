"""
AI Trader Backtesting Framework

Comprehensive backtesting system providing:
- Event-driven backtesting engine
- Performance analytics
- Risk analysis
- Strategy validation
- Cost modeling
"""

# Import interfaces first to avoid circular dependencies
from main.interfaces.backtesting import (
    BacktestConfig, BacktestMode, BacktestResult,
    IBacktestEngine, IBacktestEngineFactory
)

# Import factory for DI pattern
from .factories import BacktestEngineFactory, get_backtest_factory

# Core engine modules (commented to avoid circular import)
# from main.backtesting.engine.backtest_engine import BacktestEngine
from main.backtesting.engine.cost_model import (
    CostModel,
    FixedCommission,
    PercentageCommission,
    TieredCommission,
    FixedSlippage,
    SpreadSlippage,
    LinearSlippage,
    SquareRootSlippage,
    AdaptiveSlippage,
    create_default_cost_model,
    get_broker_cost_model
)
from main.backtesting.engine.market_simulator import MarketSimulator
from main.backtesting.engine.portfolio import Portfolio

# Analysis modules
from main.backtesting.analysis.performance_metrics import PerformanceAnalyzer as PerformanceMetrics
from main.backtesting.analysis.risk_analysis import RiskAnalysis
from main.backtesting.analysis.correlation_matrix import CorrelationMatrix
from main.backtesting.analysis.symbol_selector import SymbolSelector
from main.backtesting.analysis.validation_suite import StrategyValidationSuite

__all__ = [
    # Interfaces
    'BacktestConfig',
    'BacktestMode', 
    'BacktestResult',
    'IBacktestEngine',
    'IBacktestEngineFactory',
    
    # Factory
    'BacktestEngineFactory',
    'get_backtest_factory',
    
    # Engine
    # 'BacktestEngine',  # Commented to avoid circular import
    'MarketSimulator',
    'Portfolio',
    
    # Cost models
    'CostModel',
    'FixedCommission',
    'PercentageCommission',
    'TieredCommission',
    'FixedSlippage',
    'SpreadSlippage',
    'LinearSlippage',
    'SquareRootSlippage',
    'AdaptiveSlippage',
    'create_default_cost_model',
    'get_broker_cost_model',
    
    # Analysis
    'PerformanceMetrics',
    'RiskAnalysis',
    'CorrelationMatrix',
    'SymbolSelector',
    'StrategyValidationSuite',
]