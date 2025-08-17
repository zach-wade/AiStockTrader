#!/usr/bin/env python3
"""
Test ML Trading End-to-End

This script tests the complete ML trading flow:
1. Load trained AAPL model
2. Generate predictions
3. Convert to trading signals
4. Execute trades in paper mode
"""

# Standard library imports
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Local imports
# Setup logging
from main.utils.core import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Local imports
# Import components
from main.config.config_manager import get_config
from main.models.common import MLPrediction
from main.models.ml_signal_adapter import MLSignalAdapter
from main.models.registry.model_registry import ModelRegistry
from main.orchestration.ml_orchestrator import MLOrchestrator


async def test_ml_model_loading():
    """Test loading AAPL model from registry."""
    logger.info("\n=== Testing Model Loading ===")

    config = get_config()
    registry = ModelRegistry(config)

    # Check if AAPL model exists
    model_info = registry.get_model_info("aapl_xgboost")
    if model_info:
        logger.info(f"✅ Found AAPL model: {model_info['model_id']}")
        logger.info(f"   Model type: {model_info['model_type']}")
        logger.info(f"   Training date: {model_info['training_date']}")
        logger.info(f"   Performance: R² = {model_info['metrics']['r2']:.3f}")
        return True
    else:
        logger.error("❌ AAPL model not found in registry")
        logger.info("   Run 'python scripts/deploy_ml_model.py --deploy aapl_xgboost' first")
        return False


async def test_prediction_generation():
    """Test generating predictions from AAPL model."""
    logger.info("\n=== Testing Prediction Generation ===")

    config = get_config()

    # Create a mock prediction for testing
    prediction = MLPrediction(
        model_id="aapl_xgboost",
        symbol="AAPL",
        timestamp=datetime.now(),
        predicted_return=0.015,  # 1.5% predicted return
        confidence=0.75,
        prediction_horizon="1day",
        metadata={"current_price": 150.0, "features_used": 86, "model_version": "1.0"},
    )

    logger.info("✅ Generated test prediction:")
    logger.info(f"   Symbol: {prediction.symbol}")
    logger.info(f"   Predicted return: {prediction.predicted_return:.2%}")
    logger.info(f"   Confidence: {prediction.confidence:.2f}")

    return prediction


async def test_signal_conversion(prediction):
    """Test converting ML prediction to trading signal."""
    logger.info("\n=== Testing Signal Conversion ===")

    config = get_config()
    adapter = MLSignalAdapter(config.get("ml_trading", {}))

    # Convert prediction to signal
    signal = adapter.convert_prediction_to_signal(prediction)

    if signal:
        logger.info("✅ Created trading signal:")
        logger.info(f"   Signal ID: {signal.signal_id}")
        logger.info(f"   Symbol: {signal.symbol}")
        logger.info(f"   Side: {signal.side.value}")
        logger.info(f"   Strength: {signal.strength:.2f}")
        logger.info(f"   Position size: {signal.metadata['position_size_pct']:.2%}")
        return signal
    else:
        logger.error("❌ Failed to create signal from prediction")
        return None


async def test_ml_orchestrator():
    """Test ML orchestrator initialization and status."""
    logger.info("\n=== Testing ML Orchestrator ===")

    config = get_config()

    # Enable ML trading in config
    config.ml_trading.enabled = True
    config.broker.type = "paper"  # Use paper trading for test

    orchestrator = MLOrchestrator(config)

    try:
        # Initialize orchestrator
        await orchestrator.initialize()
        logger.info("✅ ML Orchestrator initialized")

        # Get system status
        status = await orchestrator.get_system_status()

        logger.info("\nSystem Status:")
        logger.info(f"   Orchestrator running: {status['orchestrator']['is_running']}")
        logger.info(f"   ML enabled: {status['orchestrator']['ml_enabled']}")
        logger.info(f"   Broker connected: {status['orchestrator']['broker_connected']}")
        logger.info(f"   Active models: {status['orchestrator']['active_models']}")

        # Test enabling trading
        await orchestrator.enable_trading()
        logger.info("✅ Trading enabled")

        return orchestrator

    except Exception as e:
        logger.error(f"❌ Error initializing orchestrator: {e}")
        return None


async def test_end_to_end_flow():
    """Test complete ML trading flow."""
    logger.info("\n=== Testing End-to-End ML Trading Flow ===")

    # Step 1: Check model availability
    model_available = await test_ml_model_loading()
    if not model_available:
        logger.error("Cannot proceed without AAPL model")
        return

    # Step 2: Generate test prediction
    prediction = await test_prediction_generation()

    # Step 3: Convert to signal
    signal = await test_signal_conversion(prediction)
    if not signal:
        logger.error("Cannot proceed without signal")
        return

    # Step 4: Test orchestrator
    orchestrator = await test_ml_orchestrator()
    if not orchestrator:
        logger.error("Cannot proceed without orchestrator")
        return

    logger.info("\n=== Simulating ML Trading ===")

    # Simulate signal processing
    if orchestrator.signal_handler:
        logger.info("Processing ML signal through trading system...")
        await orchestrator.signal_handler.process_signal(signal)
        logger.info("✅ Signal processed")

        # Wait a moment for processing
        await asyncio.sleep(2)

        # Check trading system status
        status = await orchestrator.get_system_status()
        trading_status = status.get("trading_system", {})

        logger.info("\nTrading System Status:")
        logger.info(f"   Status: {trading_status.get('status')}")
        logger.info(f"   Active orders: {trading_status.get('active_orders', 0)}")
        logger.info(f"   Metrics: {trading_status.get('metrics', {})}")

    # Cleanup
    await orchestrator.shutdown()
    logger.info("\n✅ End-to-end ML trading test completed successfully!")


async def test_continuous_trading(duration_minutes=5):
    """Test continuous ML trading for a specified duration."""
    logger.info(f"\n=== Testing Continuous ML Trading ({duration_minutes} minutes) ===")

    config = get_config()
    config.ml_trading.enabled = True
    config.ml_trading.prediction_interval_seconds = 30  # Generate predictions every 30s
    config.broker.type = "paper"

    orchestrator = MLOrchestrator(config)

    try:
        await orchestrator.initialize()
        await orchestrator.enable_trading()

        logger.info("Starting continuous trading simulation...")
        logger.info("Press Ctrl+C to stop")

        # Run for specified duration
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            # Get and log status every minute
            await asyncio.sleep(60)

            status = await orchestrator.get_system_status()
            logger.info("\nStatus Update:")
            logger.info(f"   Health: {status['health']}")
            logger.info(f"   Last prediction: {status['orchestrator']['last_prediction_time']}")
            logger.info(f"   Error count: {status['orchestrator']['error_count']}")

            # Show any active positions
            trading_status = status.get("trading_system", {})
            if "position_summary" in trading_status:
                positions = trading_status["position_summary"]
                if positions["position_count"] > 0:
                    logger.info(f"   Active positions: {positions['position_count']}")
                    logger.info(f"   Total P&L: ${positions['total_unrealized_pnl']:.2f}")

        logger.info("\n✅ Continuous trading test completed")

    except KeyboardInterrupt:
        logger.info("\nStopping continuous trading test...")
    finally:
        await orchestrator.shutdown()


async def main():
    """Main test function."""
    logger.info("=" * 80)
    logger.info("ML TRADING END-TO-END TEST")
    logger.info("=" * 80)

    # Run basic end-to-end test
    await test_end_to_end_flow()

    # Optionally run continuous test
    logger.info("\n" + "=" * 80)
    response = input("\nRun continuous trading test? (y/n): ")
    if response.lower() == "y":
        duration = input("Duration in minutes (default 5): ")
        duration = int(duration) if duration else 5
        await test_continuous_trading(duration)

    logger.info("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
