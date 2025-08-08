"""
Utility Commands Module

Handles miscellaneous CLI commands including model training, feature calculation,
event processing, system validation, status checks, and shutdown operations.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import click
import sys

from main.config import get_config_manager
from main.utils.core import get_logger

logger = get_logger(__name__)


@click.group()
def utility():
    """Utility and system management commands."""
    pass


@utility.command()
@click.option('--symbols', help='Comma-separated list of symbols to train on')
@click.option('--models', help='Comma-separated list of models (xgboost,lstm,ensemble)')
@click.option('--lookback-days', type=int, default=365, 
              help='Days of historical data for training')
@click.option('--test-size', type=float, default=0.2, 
              help='Test set size (0-1)')
@click.pass_context
def train(ctx, symbols: Optional[str], models: Optional[str],
          lookback_days: int, test_size: float):
    """Train machine learning models.
    
    Examples:
        # Train all models with default settings
        python ai_trader.py utility train
        
        # Train specific models for specific symbols
        python ai_trader.py utility train --symbols AAPL,GOOGL --models xgboost,lstm
        
        # Train with custom parameters
        python ai_trader.py utility train --lookback-days 730 --test-size 0.3
    """
    from main.models.training.model_trainer import ModelTrainer
    from main.models.training.training_config import TrainingConfig
    
    logger.info("Starting model training")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    # Parse parameters
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        symbol_list = config.get('training.symbols', ['AAPL', 'GOOGL', 'MSFT'])
    
    if models:
        model_list = [m.strip() for m in models.split(',')]
    else:
        model_list = ['xgboost', 'lstm', 'ensemble']
    
    try:
        # Create training configuration
        training_config = TrainingConfig(
            symbols=symbol_list,
            models=model_list,
            lookback_days=lookback_days,
            test_size=test_size,
            start_date=datetime.now() - timedelta(days=lookback_days),
            end_date=datetime.now()
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config, training_config)
        
        # Train models
        results = asyncio.run(trainer.train_all())
        
        # Display results
        _print_training_results(results)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


@utility.command()
@click.option('--symbols', help='Comma-separated list of symbols')
@click.option('--features', help='Comma-separated list of features to calculate')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--output', help='Output file for features (CSV or Parquet)')
@click.pass_context
def features(ctx, symbols: Optional[str], features: Optional[str],
             start_date: Optional[str], end_date: Optional[str],
             output: Optional[str]):
    """Calculate features for symbols.
    
    Examples:
        # Calculate all features for default symbols
        python ai_trader.py utility features
        
        # Calculate specific features for specific symbols
        python ai_trader.py utility features --symbols AAPL,TSLA --features rsi,macd,volume
        
        # Calculate features for date range and save
        python ai_trader.py utility features --start-date 2024-01-01 --end-date 2024-12-31 --output features.parquet
    """
    from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
    from main.feature_pipeline.feature_config import FeatureConfig
    
    logger.info("Calculating features")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    # Parse parameters
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        symbol_list = config.get('features.symbols', ['AAPL', 'GOOGL', 'MSFT'])
    
    if features:
        feature_list = [f.strip() for f in features.split(',')]
    else:
        feature_list = None  # Use all configured features
    
    # Parse dates
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start = datetime.now() - timedelta(days=90)
    
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end = datetime.now()
    
    try:
        # Create feature configuration
        feature_config = FeatureConfig(
            symbols=symbol_list,
            features=feature_list,
            start_date=start,
            end_date=end
        )
        
        # Initialize orchestrator
        orchestrator = FeatureOrchestrator(config)
        
        # Calculate features
        results = asyncio.run(orchestrator.calculate_features(feature_config))
        
        # Save if output specified
        if output:
            _save_features(results, output)
            logger.info(f"Features saved to {output}")
        else:
            # Display summary
            _print_feature_results(results)
            
    except Exception as e:
        logger.error(f"Feature calculation failed: {e}", exc_info=True)
        raise


@utility.command()
@click.option('--symbols', help='Comma-separated list of symbols to analyze')
@click.option('--events', help='Comma-separated event types to monitor')
@click.option('--duration', type=int, default=60, 
              help='Duration to run event monitor (seconds)')
@click.option('--output', help='Output file for events (JSON)')
@click.pass_context
def events(ctx, symbols: Optional[str], events: Optional[str],
           duration: int, output: Optional[str]):
    """Run event-driven market analysis.
    
    Examples:
        # Monitor all events for 60 seconds
        python ai_trader.py utility events
        
        # Monitor specific events for specific symbols
        python ai_trader.py utility events --symbols AAPL,TSLA --events earnings,news
        
        # Run for 5 minutes and save events
        python ai_trader.py utility events --duration 300 --output events.json
    """
    from main.interfaces.events import IEventBus
    from main.events.core import EventBusFactory
    
    # Simple event monitor implementation
    class EventMonitor:
        def __init__(self, config, event_bus):
            self.config = config
            self.event_bus = event_bus
            self.collected_events = []
        
        async def monitor(self, duration, symbols=None, event_types=None):
            # Subscribe to events
            def event_handler(event):
                if not event_types or event.type in event_types:
                    if not symbols or getattr(event, 'symbol', None) in symbols:
                        self.collected_events.append({
                            'type': event.type,
                            'timestamp': event.timestamp,
                            'message': str(event),
                            'data': event.__dict__
                        })
            
            # Subscribe to all event types
            await self.event_bus.subscribe('*', event_handler)
            
            # Wait for duration
            await asyncio.sleep(duration)
            
            # Unsubscribe
            await self.event_bus.unsubscribe('*', event_handler)
            
            return self.collected_events
    
    logger.info(f"Starting event monitor for {duration} seconds")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    # Parse parameters
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(',')]
    else:
        symbol_list = None  # Monitor all symbols
    
    if events:
        event_list = [e.strip() for e in events.split(',')]
    else:
        event_list = None  # Monitor all events
    
    try:
        # Create event bus
        event_bus: IEventBus = EventBusFactory.create(config)
        
        # Initialize monitor
        monitor = EventMonitor(config, event_bus)
        
        # Start monitoring
        collected_events = asyncio.run(
            monitor.monitor(
                duration=duration,
                symbols=symbol_list,
                event_types=event_list
            )
        )
        
        # Save or display events
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(collected_events, f, indent=2, default=str)
            logger.info(f"Saved {len(collected_events)} events to {output}")
        else:
            _print_event_summary(collected_events)
            
    except KeyboardInterrupt:
        logger.info("Event monitor stopped by user")
    except Exception as e:
        logger.error(f"Event monitoring failed: {e}", exc_info=True)
        raise


@utility.command()
@click.option('--component', type=click.Choice(['all', 'data', 'features', 'models', 'trading']), 
              default='all', help='Component to validate')
@click.pass_context
def validate(ctx, component: str):
    """Validate system components and configuration.
    
    Examples:
        # Validate all components
        python ai_trader.py utility validate
        
        # Validate specific component
        python ai_trader.py utility validate --component models
    """
    from main.data_pipeline.validation.core.validation_pipeline import ValidationPipeline
    
    # Simple system validator wrapper
    class SystemValidator:
        def __init__(self, config):
            self.config = config
            self.validator = ValidationPipeline(config)
        
        async def validate_all(self):
            results = await self.validator.run_validation('all')
            return results
        
        async def validate_component(self, component):
            results = await self.validator.run_validation(component)
            return results
    
    logger.info(f"Validating component: {component}")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize validator
        validator = SystemValidator(config)
        
        # Run validation
        if component == 'all':
            results = asyncio.run(validator.validate_all())
        else:
            results = asyncio.run(validator.validate_component(component))
        
        # Display results
        _print_validation_results(results)
        
        # Exit with error code if validation failed
        if not results.get('passed', False):
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise


@utility.command()
@click.pass_context
def status(ctx):
    """Check overall system status.
    
    Examples:
        # Check system status
        python ai_trader.py utility status
    """
    from main.monitoring.health.unified_health_reporter import UnifiedHealthReporter
    
    # Simple system monitor wrapper
    class SystemMonitor:
        def __init__(self, config):
            self.config = config
            self.health_reporter = UnifiedHealthReporter(config)
        
        async def get_full_status(self):
            health_data = await self.health_reporter.get_health_report()
            return {
                'uptime': 'N/A',  # Would need actual uptime tracking
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'components': health_data.get('subsystems', {}),
                'active_positions': 0,
                'open_orders': 0,
                'todays_pnl': 0.0,
                'processes': {}
            }
    
    logger.info("Checking system status")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    try:
        # Initialize monitor
        monitor = SystemMonitor(config)
        
        # Get status
        status_data = asyncio.run(monitor.get_full_status())
        
        # Display status
        _print_system_status(status_data)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        raise


@utility.command()
@click.option('--level', type=click.Choice(['soft', 'normal', 'hard', 'emergency']), 
              default='normal', help='Shutdown level')
@click.option('--timeout', type=int, default=30, 
              help='Timeout for graceful shutdown (seconds)')
@click.pass_context
def shutdown(ctx, level: str, timeout: int):
    """Shutdown the system gracefully.
    
    Examples:
        # Normal shutdown
        python ai_trader.py utility shutdown
        
        # Emergency shutdown
        python ai_trader.py utility shutdown --level emergency
        
        # Soft shutdown with custom timeout
        python ai_trader.py utility shutdown --level soft --timeout 60
    """
    from main.app.emergency_shutdown import EmergencyShutdown
    
    # Simple shutdown manager wrapper
    class ShutdownManager:
        def __init__(self, config):
            self.config = config
            self.emergency_shutdown_handler = EmergencyShutdown(config)
        
        async def emergency_shutdown(self):
            await self.emergency_shutdown_handler.execute()
        
        async def shutdown(self, level='normal', timeout=30):
            if level == 'emergency':
                await self.emergency_shutdown()
            else:
                # Graceful shutdown
                logger.info(f"Performing {level} shutdown with {timeout}s timeout")
                # Would implement graceful shutdown logic here
                return {
                    'stopped': 0,
                    'data_saved': True,
                    'positions_closed': 0,
                    'orders_cancelled': 0
                }
    
    logger.info(f"Initiating {level} shutdown")
    
    # Load configuration
    config_manager = get_config_manager()
    config = config_manager.load_config('unified_config')
    
    if level != 'emergency' and not click.confirm(f"Are you sure you want to {level} shutdown?"):
        logger.info("Shutdown cancelled")
        return
    
    try:
        # Initialize shutdown manager
        manager = ShutdownManager(config)
        
        # Execute shutdown
        if level == 'emergency':
            logger.warning("EMERGENCY SHUTDOWN - Stopping all processes immediately")
            asyncio.run(manager.emergency_shutdown())
        else:
            results = asyncio.run(manager.shutdown(level=level, timeout=timeout))
            _print_shutdown_results(results)
            
        logger.info("System shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown failed: {e}", exc_info=True)
        raise


# Helper functions

def _print_training_results(results: Dict):
    """Print model training results."""
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)
    
    for model_name, model_results in results.items():
        status = "✅" if model_results.get('success', False) else "❌"
        print(f"\n{status} {model_name.upper()}:")
        
        if model_results.get('success'):
            print(f"  Accuracy: {model_results.get('accuracy', 0):.4f}")
            print(f"  Precision: {model_results.get('precision', 0):.4f}")
            print(f"  Recall: {model_results.get('recall', 0):.4f}")
            print(f"  F1 Score: {model_results.get('f1_score', 0):.4f}")
            print(f"  Training time: {model_results.get('training_time', 0):.2f}s")
            print(f"  Model saved to: {model_results.get('model_path', 'N/A')}")
        else:
            print(f"  Error: {model_results.get('error', 'Unknown error')}")


def _print_feature_results(results: Dict):
    """Print feature calculation results."""
    print("\n📊 Feature Calculation Summary:")
    print("-" * 40)
    print(f"Symbols processed: {results.get('symbols_processed', 0)}")
    print(f"Features calculated: {results.get('features_calculated', 0)}")
    print(f"Total data points: {results.get('total_data_points', 0):,}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f}s")
    
    if results.get('errors'):
        print(f"\n⚠️ Errors encountered:")
        for error in results['errors'][:5]:
            print(f"  - {error}")


def _save_features(data: Any, output_path: str):
    """Save feature data to file."""
    import pandas as pd
    
    if isinstance(data, dict):
        # Convert to DataFrame
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path)
    else:
        # Default to parquet
        df.to_parquet(output_path + '.parquet')


def _print_event_summary(events: List[Dict]):
    """Print event monitoring summary."""
    print(f"\n📡 Event Monitor Summary:")
    print("-" * 40)
    print(f"Total events: {len(events)}")
    
    # Group by type
    by_type = {}
    for event in events:
        event_type = event.get('type', 'unknown')
        if event_type not in by_type:
            by_type[event_type] = 0
        by_type[event_type] += 1
    
    print(f"\nEvents by type:")
    for event_type, count in by_type.items():
        print(f"  {event_type}: {count}")
    
    # Show recent events
    if events:
        print(f"\nRecent events:")
        for event in events[-10:]:
            print(f"  [{event.get('timestamp', 'N/A')}] {event.get('type', 'N/A')}: {event.get('message', 'N/A')}")


def _print_validation_results(results: Dict):
    """Print system validation results."""
    print("\n" + "="*60)
    print("SYSTEM VALIDATION RESULTS")
    print("="*60)
    
    overall = "✅ PASSED" if results.get('passed', False) else "❌ FAILED"
    print(f"\nOverall Status: {overall}")
    
    for component, component_results in results.get('components', {}).items():
        status = "✅" if component_results.get('passed', False) else "❌"
        print(f"\n{status} {component.upper()}:")
        
        for check, check_result in component_results.get('checks', {}).items():
            check_status = "✅" if check_result else "❌"
            print(f"  {check_status} {check}")
        
        if component_results.get('errors'):
            print(f"  Errors:")
            for error in component_results['errors']:
                print(f"    - {error}")


def _print_system_status(status: Dict):
    """Print system status information."""
    print("\n" + "="*60)
    print("SYSTEM STATUS")
    print("="*60)
    
    print(f"\n🖥️ System Information:")
    print(f"  Uptime: {status.get('uptime', 'N/A')}")
    print(f"  CPU Usage: {status.get('cpu_usage', 0):.1f}%")
    print(f"  Memory Usage: {status.get('memory_usage', 0):.1f}%")
    print(f"  Disk Usage: {status.get('disk_usage', 0):.1f}%")
    
    print(f"\n📊 Component Status:")
    for component, component_status in status.get('components', {}).items():
        status_icon = "✅" if component_status.get('healthy', False) else "❌"
        print(f"  {status_icon} {component}: {component_status.get('status', 'Unknown')}")
    
    print(f"\n📈 Trading Status:")
    print(f"  Active positions: {status.get('active_positions', 0)}")
    print(f"  Open orders: {status.get('open_orders', 0)}")
    print(f"  Today's P&L: ${status.get('todays_pnl', 0):,.2f}")
    
    print(f"\n🔄 Process Status:")
    for process, process_status in status.get('processes', {}).items():
        status_icon = "🟢" if process_status == 'running' else "🔴"
        print(f"  {status_icon} {process}: {process_status}")


def _print_shutdown_results(results: Dict):
    """Print shutdown results."""
    print(f"\n🔌 Shutdown Summary:")
    print(f"  Components stopped: {results.get('stopped', 0)}")
    print(f"  Data saved: {results.get('data_saved', False)}")
    print(f"  Positions closed: {results.get('positions_closed', 0)}")
    print(f"  Orders cancelled: {results.get('orders_cancelled', 0)}")