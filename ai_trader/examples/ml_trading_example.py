"""
ML Trading Example

This example demonstrates how to set up and run ML-based trading
using the trained AAPL model with the live trading system.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required components
from main.config.config_manager import get_config, merge_configs
from main.trading_engine.core.execution_engine import (
    create_execution_engine, TradingMode, ExecutionMode
)
from main.trading_engine.signals.unified_signal import UnifiedSignalHandler
from main.monitoring.alerts.alert_manager import AlertManager
from main.models.ml_trading_integration import create_ml_trading_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_ml_trading_example():
    """Run ML trading example with paper trading."""
    
    # Components
    execution_engine = None
    signal_handler = None
    alert_manager = None
    ml_integration = None
    
    try:
        logger.info("=== Starting ML Trading Example ===")
        
        # Load configuration
        config = get_config()
        
        # Load ML trading config
        ml_config_path = Path(__file__).parent.parent / "config" / "ml_trading_config.yaml"
        if ml_config_path.exists():
            import yaml
            with open(ml_config_path, 'r') as f:
                ml_config = yaml.safe_load(f)
            config = merge_configs(config, ml_config)
        
        # Create alert manager
        alert_manager = AlertManager(config)
        await alert_manager.initialize()
        
        # Create unified signal handler
        signal_handler = UnifiedSignalHandler(config)
        
        # Create execution engine in paper trading mode
        execution_engine = await create_execution_engine(
            config=config,
            trading_mode=TradingMode.PAPER,
            execution_mode=ExecutionMode.SEMI_AUTO
        )
        
        # Create ML trading integration
        ml_integration = await create_ml_trading_integration(
            execution_engine=execution_engine,
            signal_handler=signal_handler,
            alert_manager=alert_manager,
            config=config
        )
        
        # Get ML status
        ml_status = ml_integration.get_ml_status()
        logger.info(f"ML Trading Status: {ml_status}")
        
        # Start trading
        logger.info("Starting execution engine...")
        await execution_engine.start_trading()
        
        logger.info("Starting ML trading...")
        await ml_integration.start_ml_trading()
        
        # Run for a period (e.g., 5 minutes for demo)
        logger.info("ML trading is running. Press Ctrl+C to stop...")
        
        # Monitor for some time
        for i in range(10):  # Check every 30 seconds for 5 minutes
            await asyncio.sleep(30)
            
            # Get status updates
            engine_status = await execution_engine.get_comprehensive_status()
            ml_status = ml_integration.get_ml_status()
            
            logger.info(f"Iteration {i+1}/10:")
            logger.info(f"  Active brokers: {engine_status.get('active_brokers', [])}")
            logger.info(f"  Orders submitted: {engine_status['session_metrics']['total_orders_submitted']}")
            logger.info(f"  Active ML models: {ml_status.get('ml_service', {}).get('active_models', [])}")
            
            # Check for any signals in the queue
            if hasattr(signal_handler, 'signal_queue'):
                logger.info(f"  Signals in queue: {len(signal_handler.signal_queue)}")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    except Exception as e:
        logger.error(f"Error in ML trading example: {e}", exc_info=True)
    finally:
        logger.info("Shutting down ML trading example...")
        
        # Shutdown in reverse order
        if ml_integration:
            logger.info("Stopping ML trading...")
            await ml_integration.stop_ml_trading()
            await ml_integration.shutdown()
        
        if execution_engine:
            logger.info("Shutting down execution engine...")
            await execution_engine.shutdown()
        
        if alert_manager:
            await alert_manager.cleanup()
        
        logger.info("=== ML Trading Example Complete ===")


async def check_ml_model_status():
    """Check if ML models are properly registered and ready."""
    from main.models.inference.model_registry import ModelRegistry
    from main.models.inference.prediction_engine import PredictionEngine
    
    config = get_config()
    
    # Check for trained models
    models_dir = Path(config.get('paths', {}).get('models', 'models/trained'))
    
    # Initialize components
    prediction_engine = PredictionEngine(config)
    model_registry = ModelRegistry(
        models_dir=models_dir,
        prediction_engine=prediction_engine,
        config=config
    )
    
    # List available models
    models = model_registry.list_models()
    
    logger.info(f"Found {len(models)} registered models:")
    for model in models:
        logger.info(f"  - {model['model_id']} v{model['version']} ({model['status']})")
    
    # Check for AAPL model
    aapl_models = [m for m in models if 'aapl' in m['model_id'].lower()]
    if aapl_models:
        logger.info(f"AAPL models found: {len(aapl_models)}")
        
        # Check if any are in production
        production_models = [m for m in aapl_models if m['status'] == 'production']
        if production_models:
            logger.info("✓ AAPL model ready for trading")
            return True
        else:
            logger.warning("⚠ AAPL model found but not in production status")
            logger.info("You may need to deploy the model to production first")
    else:
        logger.warning("⚠ No AAPL model found in registry")
        logger.info("Please train and register an AAPL model first")
    
    return False


if __name__ == "__main__":
    # First check if models are ready
    logger.info("Checking ML model status...")
    
    async def main():
        model_ready = await check_ml_model_status()
        
        if model_ready:
            logger.info("\nStarting ML trading example...")
            await run_ml_trading_example()
        else:
            logger.error("\nCannot start ML trading - no production models available")
            logger.info("\nTo prepare a model for trading:")
            logger.info("1. Train a model using the training pipeline")
            logger.info("2. Register it with the ModelRegistry")
            logger.info("3. Deploy it with status='production'")
    
    asyncio.run(main())