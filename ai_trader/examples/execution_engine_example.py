#!/usr/bin/env python3
"""
Execution Engine Integration Example

Demonstrates how to use the execution engine for trading operations.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from main.trading_engine.core.execution_engine import (
    ExecutionEngine, ExecutionMode, create_execution_engine
)
from main.trading_engine.core.trading_system import TradingMode
from main.models.common import Order, OrderSide, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_execution_example():
    """Basic example of using the execution engine"""
    
    # Configuration for paper trading
    config = {
        'brokers': {
            'alpaca': {
                'enabled': True,
                'api_key': 'your_api_key',
                'api_secret': 'your_api_secret',
                'base_url': 'https://paper-api.alpaca.markets'
            }
        },
        'execution': {
            'fast_path_enabled': True,
            'max_order_size': 10000,
            'max_position_size': 50000
        },
        'risk_management': {
            'max_drawdown': 0.1,
            'max_daily_loss': 0.05
        }
    }
    
    try:
        # Create and initialize execution engine
        logger.info("Creating execution engine...")
        engine = await create_execution_engine(
            config=config,
            trading_mode=TradingMode.PAPER,
            execution_mode=ExecutionMode.SEMI_AUTO
        )
        
        # Start trading operations
        logger.info("Starting trading operations...")
        await engine.start_trading()
        
        # Get system status
        status = await engine.get_comprehensive_status()
        logger.info(f"System status: {status['engine_status']}")
        logger.info(f"Active brokers: {status['active_brokers']}")
        
        # Submit a test order
        test_order = Order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        logger.info("Submitting test order...")
        order_id = await engine.submit_cross_system_order(test_order)
        
        if order_id:
            logger.info(f"Order submitted successfully: {order_id}")
        else:
            logger.warning("Order submission failed")
        
        # Wait a bit for order processing
        await asyncio.sleep(5)
        
        # Get updated metrics
        metrics = engine.session_metrics
        logger.info(f"Session metrics: {metrics}")
        
        # Pause trading
        logger.info("Pausing trading operations...")
        await engine.pause_trading()
        
        # Resume trading
        logger.info("Resuming trading operations...")
        await engine.resume_trading()
        
        # Clean shutdown
        logger.info("Shutting down execution engine...")
        await engine.shutdown()
        
    except Exception as e:
        logger.error(f"Error in execution example: {e}")


async def advanced_execution_example():
    """Advanced example with position management and risk controls"""
    
    config = {
        'brokers': {
            'alpaca': {
                'enabled': True,
                'api_key': 'your_api_key',
                'api_secret': 'your_api_secret'
            }
        },
        'risk_management': {
            'circuit_breaker': {
                'enabled': True,
                'max_loss_threshold': 0.02,
                'cooldown_period': 300
            },
            'drawdown_control': {
                'enabled': True,
                'max_drawdown': 0.1,
                'reduction_factor': 0.5
            }
        }
    }
    
    try:
        # Create execution engine
        engine = await create_execution_engine(
            config=config,
            trading_mode=TradingMode.PAPER,
            execution_mode=ExecutionMode.FULL_AUTO
        )
        
        # Add event handler for position updates
        def position_event_handler(event):
            logger.info(f"Position event: {event.event_type} for {event.symbol}")
        
        engine.add_event_handler(position_event_handler)
        
        # Start trading
        await engine.start_trading()
        
        # Submit multiple orders
        orders = [
            Order(symbol='AAPL', side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET),
            Order(symbol='GOOGL', side=OrderSide.BUY, quantity=5, order_type=OrderType.MARKET),
            Order(symbol='MSFT', side=OrderSide.BUY, quantity=15, order_type=OrderType.MARKET)
        ]
        
        for order in orders:
            order_id = await engine.submit_cross_system_order(order)
            logger.info(f"Submitted order for {order.symbol}: {order_id}")
            await asyncio.sleep(1)  # Small delay between orders
        
        # Monitor for a while
        for i in range(5):
            await asyncio.sleep(10)
            status = await engine.get_comprehensive_status()
            logger.info(f"Active positions: {status['session_metrics']['active_positions']}")
            logger.info(f"Total P&L: ${status['session_metrics']['total_realized_pnl']:.2f}")
        
        # Demonstrate emergency stop
        logger.warning("Triggering emergency stop...")
        await engine.emergency_stop()
        
        # Shutdown
        await engine.shutdown()
        
    except Exception as e:
        logger.error(f"Error in advanced example: {e}")


async def execution_manager_example():
    """Example using the ExecutionManager from orchestration layer"""
    
    from main.orchestration.managers.execution_manager import ExecutionManager
    
    # Mock orchestrator for this example
    class MockOrchestrator:
        def __init__(self):
            self.config = {
                'brokers': {'alpaca': {'enabled': True}},
                'execution': {'fast_path_enabled': True}
            }
            self.mode = TradingMode.PAPER
    
    orchestrator = MockOrchestrator()
    
    try:
        # Create execution manager
        exec_manager = ExecutionManager(orchestrator)
        
        # Initialize and start
        await exec_manager.initialize()
        await exec_manager.start()
        
        # Process a trading signal
        signal = {
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 100,
            'strategy': 'momentum',
            'confidence': 0.85,
            'order_type': 'market'
        }
        
        order_id = await exec_manager.process_signal(signal)
        
        if order_id:
            logger.info(f"Signal processed, order ID: {order_id}")
        
        # Get execution metrics
        metrics = await exec_manager.get_execution_metrics()
        logger.info(f"Execution metrics: Total orders: {metrics.total_orders}")
        
        # Get portfolio summary
        portfolio = await exec_manager.get_portfolio_summary()
        logger.info(f"Portfolio summary: {portfolio}")
        
        # Stop execution manager
        await exec_manager.stop()
        
    except Exception as e:
        logger.error(f"Error in execution manager example: {e}")


def main():
    """Run examples"""
    logger.info("=== Execution Engine Examples ===")
    
    # Run basic example
    logger.info("\n--- Basic Execution Example ---")
    asyncio.run(basic_execution_example())
    
    # Run advanced example
    logger.info("\n--- Advanced Execution Example ---")
    asyncio.run(advanced_execution_example())
    
    # Run execution manager example
    logger.info("\n--- Execution Manager Example ---")
    asyncio.run(execution_manager_example())


if __name__ == "__main__":
    main()