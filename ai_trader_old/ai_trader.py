#!/usr/bin/env python3
"""
AI Trader - Unified Command Line Interface

Usage:
    python ai_trader.py <command> [options]

Commands:
    trade       Run the trading system
    backfill    Backfill historical data
    train       Train ML models
    features    Calculate features for symbols
    events      Run event-driven market analysis
    validate    Validate system components
    universe    Universe management and Layer 0-3 scanning
    status      Check system status
    shutdown    Shutdown the system
"""

# Standard library imports
# Debug mode disabled - use proper logging instead
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys

# Third-party imports
import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Standard library imports
# Set DATA_LAKE_PATH if running from parent directory
import os

if not os.path.exists("data_lake") and os.path.exists("ai_trader/data_lake"):
    os.environ["DATA_LAKE_PATH"] = os.path.abspath("ai_trader/data_lake")
    print(f"Auto-detected DATA_LAKE_PATH: {os.environ['DATA_LAKE_PATH']}")

# Local imports
# Load environment variables FIRST, before any other imports that might use config
from main.config.env_loader import ensure_environment_loaded

ensure_environment_loaded()

# Local imports
# Now import everything else
from main.config.config_manager import get_config
from main.utils.core import setup_logging

# Configure root logger
setup_logging()
logger = logging.getLogger(__name__)


def get_available_stages():
    """Get available backfill stages from configuration."""
    config = get_config()
    try:
        # Handle OmegaConf DictConfig objects
        stages = config.data_pipeline.resilience.stages
        stage_names = []
        for stage in stages:
            # Convert DictConfig to dict and get name
            stage_dict = dict(stage) if hasattr(stage, "__dict__") else stage
            if "name" in stage_dict:
                stage_names.append(stage_dict["name"])
        return stage_names + ["all"]  # Add 'all' as a special option
    except AttributeError:
        # Fallback if config structure is different
        logger.warning("Could not find stages in config, using defaults")
        return ["all"]


def validate_stage(ctx, param, value):
    """Validate stage parameter against available stages."""
    if value is None:
        return value

    available_stages = get_available_stages()
    if value not in available_stages:
        raise click.BadParameter(
            f"Invalid stage '{value}'. Available stages: {', '.join(available_stages)}\n"
            f"Use --list-stages to see details about each stage."
        )
    return value


@click.group()
@click.option(
    "--config",
    default=None,
    help="Path to configuration file (optional, uses modular config by default)",
)
@click.option(
    "--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"])
)
@click.option(
    "--environment", default=None, help="Environment override (paper, live, backtest, training)"
)
@click.pass_context
def cli(ctx, config, log_level, environment):
    """AI Trader - Algorithmic Trading System"""
    print("DEBUG: cli() function called", file=sys.stderr)
    print(
        f"DEBUG: ctx={ctx}, config={config}, log_level={log_level}, environment={environment}",
        file=sys.stderr,
    )

    # Load configuration using the new modular config system
    try:
        print("DEBUG: Loading configuration...", file=sys.stderr)
        if config:
            # If a specific config file is provided, we'd need to handle it
            # For now, log a warning and use the modular system
            logger.warning(
                f"Custom config file '{config}' specified but using modular config system"
            )

        # Use the modular configuration manager
        print("DEBUG: Calling get_config()...", file=sys.stderr)
        app_config = get_config()
        print("DEBUG: get_config() completed", file=sys.stderr)

        # Store in context
        print("DEBUG: Storing config in context...", file=sys.stderr)
        ctx.ensure_object(dict)
        ctx.obj["config"] = app_config
        ctx.obj["config_path"] = "modular_config_system"
        print("DEBUG: Config stored in context", file=sys.stderr)

        # Apply environment override if specified
        if environment:
            # Update the environment setting in the config
            app_config.system.environment = environment
            logger.info(f"Environment overridden to: {environment}")

        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level))

        logger.info(
            f"AI Trader initialized with modular config system (environment: {app_config.system.environment})"
        )
        print("DEBUG: cli() function initialization complete", file=sys.stderr)

    except Exception as e:
        print(f"DEBUG: Exception in cli initialization: {e}", file=sys.stderr)
        logger.error(f"Failed to initialize configuration: {e}")
        ctx.exit(1)


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["live", "paper", "backtest", "dev"]),
    default="paper",
    help="Trading mode",
)
@click.option("--symbols", help="Comma-separated list of symbols to trade")
@click.option("--strategy", help="Strategy to use (default: all enabled)")
@click.option("--enable-ml", is_flag=True, help="Enable ML trading models")
@click.option("--disable-monitoring", is_flag=True, help="Disable monitoring dashboards")
@click.option("--disable-streaming", is_flag=True, help="Disable real-time data streaming")
@click.option("--dashboard-port", type=int, default=8080, help="Port for trading dashboard")
@click.option("--websocket-port", type=int, default=8081, help="Port for WebSocket updates")
@click.option(
    "--enable-system-dashboard", is_flag=True, help="Enable system health dashboard (port 8052)"
)
@click.option("--trading-dashboard-only", is_flag=True, help="Only start trading dashboard")
@click.option("--system-dashboard-only", is_flag=True, help="Only start system dashboard")
@click.pass_context
def trade(
    ctx,
    mode,
    symbols,
    strategy,
    enable_ml,
    disable_monitoring,
    disable_streaming,
    dashboard_port,
    websocket_port,
    enable_system_dashboard,
    trading_dashboard_only,
    system_dashboard_only,
):
    """Run the trading system with integrated monitoring and real-time data"""
    enable_monitoring = not disable_monitoring
    enable_streaming = not disable_streaming

    # Determine which dashboards to start
    start_trading_dashboard = enable_monitoring and not system_dashboard_only
    start_system_dashboard = enable_monitoring and not trading_dashboard_only

    # If both only flags are set, that's an error
    if trading_dashboard_only and system_dashboard_only:
        logger.error("Cannot specify both --trading-dashboard-only and --system-dashboard-only")
        ctx.exit(1)

    logger.info("Starting imports for trade command...")

    logger.info("Importing managed_app_context...")
    # Local imports
    from main.utils.app.context import managed_app_context

    logger.info("✓ managed_app_context imported")

    logger.info("Importing MLOrchestrator...")
    # Local imports
    from main.orchestration.ml_orchestrator import MLOrchestrator

    logger.info("✓ MLOrchestrator imported")

    # DashboardServer removed - using V2 dashboard system

    logger.info("Importing StreamProcessor...")
    # Local imports
    from main.data_pipeline.stream_processor import StreamProcessor

    logger.info("✓ StreamProcessor imported")

    logger.info("Importing EventBusFactory...")
    # Local imports
    from main.events.core import EventBusFactory

    logger.info("✓ EventBusFactory imported")

    logger.info("Importing PaperBroker...")
    # Local imports
    from main.trading_engine.brokers.paper_broker import PaperBroker

    logger.info("✓ PaperBroker imported")

    logger.info("Importing signal...")
    # Standard library imports
    import signal

    logger.info("✓ All imports completed successfully")

    config = ctx.obj["config"]

    # Override config with CLI options
    if mode:
        config.system.environment = mode
        config.broker.type = "paper" if mode == "paper" else mode
        config.broker.paper_trading = mode == "paper"
        logger.info(f"Trading mode set to: {mode}")
    if symbols:
        config.trading.universe = symbols.split(",")
        logger.info(f"Trading symbols set to: {symbols}")
    if strategy:
        config.strategies.active = [strategy]
        logger.info(f"Strategy set to: {strategy}")
    elif not hasattr(config.strategies, "active") or not config.strategies.active:
        # Collect all enabled strategies
        enabled_strategies = []
        for strat_name in config.strategies:
            if strat_name != "active" and hasattr(config.strategies[strat_name], "enabled"):
                if config.strategies[strat_name].enabled:
                    enabled_strategies.append(strat_name)

        if enabled_strategies:
            config.strategies.active = enabled_strategies
            logger.info(f"Using all enabled strategies: {enabled_strategies}")
        else:
            logger.warning("No enabled strategies found")

    if enable_ml:
        # This flag is now redundant but kept for backward compatibility
        logger.info("ML strategies are already enabled by default in config")

    # Components to be initialized
    components = []
    app_context = None
    orchestrator = None
    stream_processor = None
    event_bus = None

    async def run_integrated_system():
        """Run the integrated trading system with all components"""
        print(f"🚀 DEBUG: run_integrated_system() CALLED at {datetime.now()}", file=sys.stderr)
        logger.info("🚀 ENTRY: run_integrated_system() function called")

        nonlocal app_context, orchestrator, stream_processor, event_bus

        try:
            # Initialize app context with all required components
            logger.info("Initializing integrated trading system...")
            print(
                f"🔍 DEBUG: About to enter managed_app_context at {datetime.now()}", file=sys.stderr
            )

            print(
                "🔍 DEBUG: Creating managed_app_context with components: ['database', 'dual_storage', 'data_sources', 'ingestion', 'processing']",
                file=sys.stderr,
            )
            async with managed_app_context(
                "ai_trader",
                components=["database", "dual_storage", "data_sources", "ingestion", "processing"],
            ) as app_context:
                print(f"✅ DEBUG: Entered managed_app_context at {datetime.now()}", file=sys.stderr)
                logger.info("✅ Successfully entered managed_app_context")
                print(f"🔍 DEBUG: app_context = {app_context}", file=sys.stderr)

                # Get event bus from context (initialized during dual_storage component)
                print(f"🔍 DEBUG: enable_streaming = {enable_streaming}", file=sys.stderr)
                event_bus = app_context.event_bus
                if not event_bus:
                    logger.info("Event bus not found in context, creating new one...")
                    event_bus = EventBusFactory.create()
                    await event_bus.start()
                    components.append(("event_bus", event_bus))
                else:
                    logger.info("Using event bus from context")
                    components.append(("event_bus", event_bus))

                # Check if dual storage is initialized
                if app_context.cold_storage:
                    logger.info("✅ Dual storage system already initialized by context")
                    # Local imports
                    from main.data_pipeline.storage.dual_storage_startup import (
                        get_dual_storage_manager,
                    )

                    components.append(("dual_storage", get_dual_storage_manager()))

                # Initialize ML orchestrator with app context dependencies
                logger.info("Initializing ML orchestrator...")
                orchestrator = MLOrchestrator(
                    config,
                    db_pool=app_context.db_pool,
                    data_source_manager=app_context.data_source_manager,
                    event_bus=event_bus,
                )
                await orchestrator.initialize()
                components.append(("orchestrator", orchestrator))

                # Initialize streaming processor if enabled
                if enable_streaming:
                    logger.info("Initializing stream processor...")
                    stream_processor = StreamProcessor(config)
                    await stream_processor.start()
                    components.append(("stream_processor", stream_processor))
                    logger.info("✅ Real-time data streaming activated")

                # Initialize monitoring dashboards if enabled
                logger.info(f"🔍 DEBUG: enable_monitoring={enable_monitoring}")
                if enable_monitoring:
                    logger.info("Initializing V2 monitoring dashboards...")
                    logger.info(f"🔍 DEBUG: start_trading_dashboard={start_trading_dashboard}")
                    logger.info(f"🔍 DEBUG: start_system_dashboard={start_system_dashboard}")

                    try:
                        # Import the new V2 dashboard system
                        # Local imports
                        from main.monitoring.dashboards.v2 import DashboardManager

                        logger.info("✓ Successfully imported V2 DashboardManager")

                        # Create database configuration for dashboard manager
                        db_config = {
                            "host": config.database.host,
                            "port": config.database.port,
                            "database": config.database.name,
                            "user": config.database.user,
                            "password": config.database.password,
                        }

                        # Create dashboard manager
                        dashboard_manager = DashboardManager(db_config)
                        logger.info("✓ Dashboard manager created")

                        # Start requested dashboards
                        dashboards_to_start = []
                        if start_trading_dashboard:
                            dashboards_to_start.append("trading")
                        if start_system_dashboard:
                            dashboards_to_start.append("system")

                        # Start dashboards
                        for dashboard_name in dashboards_to_start:
                            logger.info(f"Starting {dashboard_name} dashboard...")
                            await dashboard_manager.start_dashboard(dashboard_name)

                        # Store dashboard manager for cleanup
                        components.append(("dashboard_manager", dashboard_manager))

                        # Print dashboard status
                        dashboard_manager.print_status()

                        logger.info("✅ V2 Dashboards initialized successfully")
                        logger.info("=" * 60)
                        logger.info("🎯 Dashboard URLs:")
                        if start_trading_dashboard:
                            logger.info("   Trading Dashboard: http://localhost:8080")
                        if start_system_dashboard:
                            logger.info("   System Dashboard:  http://localhost:8052")
                        logger.info("=" * 60)

                    except Exception as e:
                        logger.error(f"❌ Failed to initialize V2 dashboards: {e}", exc_info=True)
                        logger.warning("Continuing without dashboards...")

                # Initialize paper broker if in paper mode
                if mode == "paper":
                    logger.info("Initializing paper trading broker...")
                    paper_broker = PaperBroker(config)
                    await paper_broker.connect()
                    orchestrator.set_broker(paper_broker)
                    logger.info("✅ Paper trading broker activated")

                # Log system status
                logger.info("=" * 60)
                logger.info("🚀 AI Trader System Started Successfully!")
                logger.info(f"   Mode: {mode.upper()}")
                logger.info(f"   Symbols: {config.trading.universe}")
                logger.info(
                    f"   ML Trading: {'Enabled' if config.get('ml_trading', {}).get('enabled', False) else 'Disabled'}"
                )
                logger.info(f"   Streaming: {'Enabled' if enable_streaming else 'Disabled'}")
                logger.info(
                    f"   Dashboard: {'http://localhost:' + str(dashboard_port) if enable_monitoring else 'Disabled'}"
                )
                logger.info("=" * 60)

                # Run the orchestrator
                logger.info("🚀 Starting orchestrator.run()...")
                await orchestrator.run()
                logger.info("✅ Orchestrator completed successfully")

        except Exception as e:
            logger.error(f"❌ System initialization error: {e}", exc_info=True)
            print(f"❌ ERROR in run_integrated_system: {e}", file=sys.stderr)
            raise

    async def shutdown_system():
        """Gracefully shutdown all components"""
        logger.info("Initiating graceful shutdown...")

        # Shutdown in reverse order of initialization
        for name, component in reversed(components):
            try:
                logger.info(f"Shutting down {name}...")
                if name == "dashboard_manager":
                    # Special handling for dashboard manager
                    await component.stop_all()
                elif hasattr(component, "shutdown"):
                    await component.shutdown()
                elif hasattr(component, "stop"):
                    await component.stop()
                elif hasattr(component, "close"):
                    await component.close()
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")

        logger.info("✅ Shutdown completed")

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        asyncio.create_task(shutdown_system())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info(f"Starting integrated trading system in {mode} mode...")
        print(
            f"🔍 DEBUG: About to call asyncio.run(run_integrated_system()) at {datetime.now()}",
            file=sys.stderr,
        )
        asyncio.run(run_integrated_system())
        print(
            f"🔍 DEBUG: asyncio.run(run_integrated_system()) completed at {datetime.now()}",
            file=sys.stderr,
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        asyncio.run(shutdown_system())
    except Exception as e:
        logger.error(f"Trading system error: {e}", exc_info=True)
        asyncio.run(shutdown_system())
        sys.exit(1)


@cli.command()
@click.option(
    "--stage",
    callback=validate_stage,
    default="all",
    help="Data collection stage (use --list-stages to see available options)",
)
@click.option("--list-stages", is_flag=True, help="List available backfill stages")
@click.option(
    "--symbols",
    help='Comma-separated symbols, "universe" for all, or "layer0/layer1/layer2/layer3" for scanner-qualified symbols',
)
@click.option(
    "--days", type=int, default=1825, help="Number of days to backfill (default: 5 years)"
)
@click.option("--force", is_flag=True, help="Force re-download of existing data")
@click.option("--test-mode", is_flag=True, help="Test with 10 known good symbols")
@click.option("--limit", type=int, help="Limit number of symbols to process")
@click.pass_context
def backfill(ctx, stage, list_stages, symbols, days, force, test_mode, limit):
    """Backfill historical data"""
    # Local imports
    from main.app.historical_backfill import run_historical_backfill

    config = ctx.obj["config"]

    # Handle list stages flag
    if list_stages:
        try:
            stages = config.data_pipeline.resilience.stages
        except AttributeError:
            logger.error("Could not find stages in configuration")
            return

        logger.info("\n=== Available Backfill Stages ===\n")

        # Add helpful header
        logger.info("For scanner pipeline data, use:")
        logger.info("  - scanner_daily: Daily bars for Layer 1-3 scanners (60 days)")
        logger.info("  - scanner_intraday: Minute bars for real-time scanners (7 days)")
        logger.info("\nFor historical analysis, use:")
        logger.info("  - long_term: Complete historical data to data lake (10 years)")
        logger.info("  - news_data, corporate_actions, etc.: Alternative data sources")
        logger.info("\n" + "=" * 50 + "\n")

        for stage_config in stages:
            # Convert DictConfig to dict
            stage_dict = dict(stage_config)
            name = stage_dict.get("name", "Unknown")
            desc = stage_dict.get("description", "No description")
            dest = stage_dict.get("destination", "Unknown")
            intervals = stage_dict.get("intervals", [])
            sources = stage_dict.get("sources", [])
            lookback = stage_dict.get("lookback_days", "Default")

            # Add emoji indicators for clarity
            emoji = "📊" if "scanner" in name else "💾" if dest == "data_lake" else "📰"
            dest_emoji = "🐘" if dest == "postgresql" else "🗄️"

            logger.info(f"{emoji} Stage: {name}")
            logger.info(f"  Description: {desc}")
            logger.info(f"  {dest_emoji} Destination: {dest}")
            logger.info(f"  Sources: {', '.join(sources) if sources else 'Default'}")
            logger.info(f"  Intervals: {', '.join(intervals) if intervals else 'N/A'}")
            logger.info(f"  Lookback Days: {lookback}")
            logger.info("")

        logger.info("Stage: all")
        logger.info("  Description: Run all available stages")
        logger.info("")
        return

    # Get all stage names for 'all' option
    all_stages = []
    if stage == "all":
        try:
            stages = config.data_pipeline.resilience.stages
            all_stages = [dict(s)["name"] for s in stages if "name" in dict(s)]
        except AttributeError:
            logger.warning("Could not find stages in config for 'all' option")
            all_stages = []

    # Create backfill config
    backfill_config = {
        "stages": all_stages if stage == "all" else [stage],
        "symbols": None if symbols == "universe" else (symbols.split(",") if symbols else None),
        "lookback_days": days,
        "force": force,
        "test_mode": test_mode,
        "limit": limit,
    }

    try:
        logger.info(f"Starting historical data backfill: {backfill_config}")
        result = asyncio.run(run_historical_backfill(backfill_config))

        if result["status"] == "success":
            logger.info("✅ Historical backfill completed successfully")
            logger.info(
                f"Processed {result['symbols_processed']} symbols in {result['duration_seconds']:.2f}s"
            )
        else:
            logger.error("❌ Historical backfill failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Historical backfill error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--symbols", help="Comma-separated list of symbols to train on")
@click.option("--models", help="Comma-separated list of models to train")
@click.option("--lookback-days", type=int, default=365, help="Days of historical data")
@click.option("--test-size", type=float, default=0.2, help="Test set size (0-1)")
@click.pass_context
def train(ctx, symbols, models, lookback_days, test_size):
    """Train ML models"""
    # Local imports
    from main.models.training.training_orchestrator import (
        ModelTrainingOrchestrator as TrainingOrchestrator,
    )

    config = ctx.obj["config"]

    # Prepare training parameters
    symbol_list = (
        symbols.split(",") if symbols else config.get("universe.symbols", ["AAPL", "MSFT"])
    )  # Default fallback
    model_types = models.split(",") if models else ["xgboost", "lightgbm", "random_forest"]
    fast_mode = True  # Use fast mode for CLI training

    # Run training
    orchestrator = TrainingOrchestrator(config)

    try:
        logger.info(
            f"Starting model training: symbols={symbol_list}, models={model_types}, fast_mode={fast_mode}"
        )
        results = asyncio.run(orchestrator.run_full_workflow(symbol_list, model_types, fast_mode))
        logger.info("✅ Training completed successfully")
        logger.info(f"Training results: {list(results.keys()) if results else 'No results'}")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--symbols", help="Comma-separated list of symbols to calculate features for")
@click.option(
    "--features",
    help="Comma-separated list of feature sets (technical, fundamental, sentiment, all)",
)
@click.option("--start-date", help="Start date for feature calculation (YYYY-MM-DD)")
@click.option("--end-date", help="End date for feature calculation (YYYY-MM-DD)")
@click.option("--output", help="Output file path (optional)")
@click.pass_context
def features(ctx, symbols, features, start_date, end_date, output):
    """Calculate features for symbols"""
    # Local imports
    from main.app.calculate_features import FeatureCalculationEngine

    config = ctx.obj["config"]

    # Create feature calculation config
    feature_config = {
        "symbols": symbols.split(",") if symbols else config.get("universe.symbols", []),
        "feature_sets": (
            features.split(",") if features else ["technical", "fundamental", "sentiment"]
        ),
        "start_date": start_date,
        "end_date": end_date,
        "output_file": output,
    }

    # Run feature calculation
    engine = FeatureCalculationEngine(config)

    try:
        logger.info(f"Starting feature calculation: {feature_config}")
        results = asyncio.run(engine.run(feature_config))
        logger.info("✅ Feature calculation completed successfully")

        # Print summary
        if results:
            logger.info(f"Features calculated for {len(results.get('symbols', []))} symbols")
            logger.info(f"Feature sets: {results.get('feature_sets', [])}")
            if output:
                logger.info(f"Results saved to: {output}")

    except Exception as e:
        logger.error(f"Feature calculation error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--symbols", help="Comma-separated list of symbols to analyze")
@click.option("--events", help="Comma-separated list of event types to monitor")
@click.option("--duration", type=int, default=60, help="Duration to run analysis (minutes)")
@click.option("--output", help="Output file path for results (optional)")
@click.pass_context
def events(ctx, symbols, events, duration, output):
    """Run event-driven market analysis"""
    # Local imports
    from main.events.handlers.event_driven_engine import EventDrivenEngine

    config = ctx.obj["config"]

    # Create event analysis config
    event_config = {
        "symbols": symbols.split(",") if symbols else config.get("universe.symbols", []),
        "event_types": (
            events.split(",") if events else ["earnings", "news", "price_movement", "volume_spike"]
        ),
        "duration_minutes": duration,
        "output_file": output,
    }

    # Run event analysis
    engine = EventDrivenEngine(config)

    try:
        logger.info(f"Starting event-driven analysis: {event_config}")
        results = asyncio.run(engine.run(event_config))
        logger.info("✅ Event analysis completed successfully")

        # Print summary
        if results:
            logger.info(f"Events analyzed for {len(results.get('symbols', []))} symbols")
            logger.info(f"Event types: {results.get('event_types', [])}")
            logger.info(f"Events detected: {results.get('events_detected', 0)}")
            if output:
                logger.info(f"Results saved to: {output}")

    except Exception as e:
        logger.error(f"Event analysis error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--component",
    type=click.Choice(["all", "data", "features", "models", "trading"]),
    default="all",
    help="Component to validate",
)
@click.pass_context
def validate(ctx, component):
    """Validate system components"""
    # Local imports
    from main.app.run_validation import ValidationRunner

    config = ctx.obj["config"]

    # Run validation
    runner = ValidationRunner(config)

    try:
        logger.info(f"Running validation for: {component}")
        results = asyncio.run(runner.validate(component))

        # Print results
        for comp, result in results.items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            logger.info(f"{comp}: {status}")
            if not result["passed"]:
                for error in result.get("errors", []):
                    logger.error(f"  - {error}")

        # Exit with error if any validation failed
        if not all(r["passed"] for r in results.values()):
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--data-type", default="all", help="Data type to process (market_data, news, all)")
@click.option("--limit", type=int, default=100, help="Maximum number of files to process")
@click.pass_context
def process_raw(ctx, data_type, limit):
    """Process raw data from data_lake/raw/ to data_lake/processed/"""
    # Local imports
    from main.app.process_raw_data import RawDataProcessor

    config = ctx.obj["config"]

    try:
        logger.info(f"Starting raw data processing: data_type={data_type}, limit={limit}")

        # Create raw data processor
        processor = RawDataProcessor(config)

        # Run processing pipeline
        result = asyncio.run(processor.process_files(data_type, limit))

        if result["files_processed"] > 0:
            logger.info(f"✅ Processing completed: {result['files_processed']} files processed")
        else:
            logger.info("No unprocessed files found")

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option("--model-path", help="Path to saved model (or use --model-type for latest)")
@click.option(
    "--model-type",
    type=click.Choice(["xgboost", "lightgbm", "random_forest"]),
    help="Use latest model of this type",
)
@click.option("--symbols", help="Comma-separated list of symbols to backtest")
@click.option("--start-date", help="Backtest start date (YYYY-MM-DD)")
@click.option("--end-date", help="Backtest end date (YYYY-MM-DD)")
@click.option("--initial-cash", type=float, default=100000, help="Starting capital")
@click.option("--list-models", is_flag=True, help="List available models and exit")
@click.option(
    "--compare", multiple=True, help="Compare multiple models (can be used multiple times)"
)
@click.option("--output", help="Save results to file (JSON format)")
@click.pass_context
def backtest(
    ctx,
    model_path,
    model_type,
    symbols,
    start_date,
    end_date,
    initial_cash,
    list_models,
    compare,
    output,
):
    """Run backtest on saved ML models"""
    # Standard library imports
    from datetime import datetime
    import json

    # Local imports
    from main.app.run_backtest import BacktestRunner, find_and_list_models, find_latest_model

    # Handle list models flag
    if list_models:
        find_and_list_models()
        return

    # Check required options for backtest
    if not symbols:
        logger.error("Must specify --symbols when running backtest")
        sys.exit(1)

    config = ctx.obj["config"]

    # Determine model path(s)
    model_paths = []

    if compare:
        # Multiple models to compare
        model_paths.extend(compare)
    elif model_path:
        # Specific model path provided
        model_paths.append(model_path)
    elif model_type:
        # Find latest model of given type
        latest = find_latest_model(model_type)
        if latest:
            model_paths.append(str(latest))
            logger.info(f"Using latest {model_type} model: {latest}")
        else:
            logger.error(f"No {model_type} models found")
            sys.exit(1)
    else:
        logger.error("Must specify either --model-path or --model-type")
        sys.exit(1)

    # Parse dates
    if start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(",")]

    # Create runner
    runner = BacktestRunner(config)

    try:
        if len(model_paths) > 1:
            # Run comparison
            logger.info(f"Running backtest comparison for {len(model_paths)} models")
            results = asyncio.run(
                runner.run_multiple_backtests(
                    model_paths=model_paths,
                    symbols=symbol_list,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash,
                )
            )

            # Print comparison summary
            _print_comparison_results(results)
        else:
            # Run single backtest
            logger.info(f"Running backtest for model: {model_paths[0]}")
            results = asyncio.run(
                runner.run_model_backtest(
                    model_path=model_paths[0],
                    symbols=symbol_list,
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=initial_cash,
                )
            )

            # Print results
            _print_backtest_results(results)

        # Save results if requested
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output}")

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        sys.exit(1)


def _print_backtest_results(results: dict):
    """Print formatted backtest results."""
    if not results.get("success", False):
        logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
        return

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Model info
    model_info = results.get("model_info", {})
    print(f"\nModel Type: {model_info.get('model_type', 'Unknown')}")
    print(f"Training Date: {model_info.get('training_date', 'Unknown')}")
    print(f"Features: {model_info.get('features_count', 0)}")

    # Performance metrics
    perf = results.get("performance_metrics", {})
    print("\nPerformance Metrics:")
    print(f"  Total Return: {perf.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2%}")
    print(f"  Volatility: {perf.get('volatility', 0):.2%}")
    print(f"  Final Equity: ${perf.get('final_equity', 0):,.2f}")

    # Trade metrics
    trade = results.get("trade_metrics", {})
    print("\nTrade Metrics:")
    print(f"  Total Trades: {trade.get('total_trades', 0)}")
    print(f"  Win Rate: {trade.get('win_rate', 0):.2%}")
    print(f"  Avg Win: ${trade.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss: ${trade.get('avg_loss', 0):,.2f}")
    print(f"  Trades/Day: {trade.get('trades_per_day', 0):.2f}")

    print("\n" + "=" * 80)


def _print_comparison_results(results: dict):
    """Print formatted comparison results."""
    comparison = results.get("comparison", {})

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    # Summary table
    summary = comparison.get("summary", [])
    if summary:
        # Third-party imports
        import pandas as pd

        df = pd.DataFrame(summary)

        # Select key columns
        display_cols = [
            "model_type",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_trades",
        ]
        display_cols = [col for col in display_cols if col in df.columns]

        print("\nSummary:")
        print(df[display_cols].to_string(index=False, float_format="%.3f"))

    # Best by metric
    best = comparison.get("best_by_metric", {})
    if best:
        print("\nBest Models by Metric:")
        for metric, info in best.items():
            print(f"  {metric}: {info['model_type']} ({info['value']:.3f})")

    print("\n" + "=" * 80)


@cli.command()
@click.option(
    "--level",
    type=click.Choice(["soft", "normal", "hard", "emergency"]),
    default="normal",
    help="Shutdown level",
)
@click.option("--timeout", type=int, default=30, help="Shutdown timeout in seconds")
@click.pass_context
def shutdown(ctx, level, timeout):
    """Shutdown the trading system"""
    # Local imports
    from main.app.emergency_shutdown import EmergencyShutdown

    config = ctx.obj["config"]

    # Run shutdown
    shutdown_handler = EmergencyShutdown(config)

    try:
        logger.info(f"Initiating {level} shutdown...")
        asyncio.run(shutdown_handler.execute(level, timeout))
        logger.info("✅ Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Check system status"""
    # Local imports
    from main.orchestration.ml_orchestrator import MLOrchestrator

    config = ctx.obj["config"]

    # Get status
    orchestrator = MLOrchestrator(config)

    try:
        # Initialize just enough to get status
        asyncio.run(orchestrator.initialize())
        status = asyncio.run(orchestrator.get_system_status())

        # Print status
        logger.info("=== AI Trader System Status ===")
        logger.info(f"Overall Health: {status.get('health', 'UNKNOWN')}")

        # Print orchestrator status
        if "orchestrator" in status:
            logger.info("\nOrchestrator:")
            for key, value in status["orchestrator"].items():
                logger.info(f"  {key}: {value}")

        # Print trading system status
        if "trading_system" in status:
            logger.info("\nTrading System:")
            ts = status["trading_system"]
            logger.info(f"  Status: {ts.get('status', 'unknown')}")
            logger.info(f"  Mode: {ts.get('mode', 'unknown')}")
            logger.info(f"  Trading Enabled: {ts.get('trading_enabled', False)}")
            logger.info(f"  Active Orders: {ts.get('active_orders', 0)}")

        # Print ML system status
        if "ml_system" in status:
            logger.info("\nML System:")
            ml = status["ml_system"]
            logger.info(f"  Models Loaded: {ml.get('models_loaded', 0)}")
            logger.info(f"  Active Models: {ml.get('active_models', [])}")

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        asyncio.run(orchestrator.shutdown())


@cli.command()
@click.option("--populate", is_flag=True, help="Populate universe with Layer 0 scan")
@click.option("--stats", is_flag=True, help="Show universe statistics")
@click.option("--health", is_flag=True, help="Check universe health")
@click.option(
    "--layer", type=click.Choice(["0", "1", "2", "3"]), default="0", help="Layer to query"
)
@click.option("--limit", type=int, help="Limit number of symbols returned")
@click.pass_context
def universe(ctx, populate, stats, health, layer, limit):
    """Universe management and Layer 0-3 scanning"""
    # Local imports
    from main.universe.universe_manager import UniverseManager

    config = ctx.obj["config"]

    # Initialize universe manager
    universe_manager = UniverseManager(config)

    try:
        if populate:
            logger.info("Starting universe population (Layer 0 scan)...")
            result = asyncio.run(universe_manager.populate_universe())

            if result["success"]:
                logger.info("✅ Universe populated successfully")
                logger.info(f"Assets discovered: {result['assets_discovered']}")
                logger.info(f"Companies in database: {result['companies_in_db']}")
                logger.info(f"Duration: {result['duration_seconds']:.2f} seconds")
            else:
                logger.error(f"❌ Universe population failed: {result['error']}")
                sys.exit(1)

        elif stats:
            logger.info("Getting universe statistics...")
            stats_data = asyncio.run(universe_manager.get_universe_stats())

            logger.info("=== Universe Statistics ===")
            logger.info(f"Total companies: {stats_data['total_companies']}")
            logger.info(f"Active companies: {stats_data['active_companies']}")
            logger.info(f"Active percentage: {stats_data['active_percentage']:.1f}%")
            logger.info(f"Layer 1 qualified: {stats_data['layer1_qualified']}")
            logger.info(f"Layer 2 qualified: {stats_data['layer2_qualified']}")
            logger.info(f"Layer 3 qualified: {stats_data['layer3_qualified']}")

        elif health:
            logger.info("Checking universe health...")
            health_data = asyncio.run(universe_manager.health_check())

            status = "✅ HEALTHY" if health_data["healthy"] else "❌ UNHEALTHY"
            logger.info(f"Universe Health: {status}")
            logger.info(f"Companies count: {health_data['companies_count']}")
            logger.info(f"Database accessible: {health_data['database_accessible']}")
            logger.info(f"Sufficient companies: {health_data['has_sufficient_companies']}")

            if not health_data["healthy"]:
                logger.warning(
                    "⚠️ Consider running 'python ai_trader.py universe --populate' to populate the universe"
                )

        else:
            # Default: show symbols for specified layer
            logger.info(f"Getting symbols for layer {layer}...")
            symbols = asyncio.run(universe_manager.get_qualified_symbols(layer, limit=limit))

            if symbols:
                logger.info(f"✅ Found {len(symbols)} symbols for layer {layer}")
                logger.info(f"Sample symbols: {symbols[:20]}{'...' if len(symbols) > 20 else ''}")

                # Show summary statistics
                if len(symbols) > 100:
                    logger.info(f"Total symbols: {len(symbols)}")
                    logger.info(f"First 10: {symbols[:10]}")
                    logger.info(f"Last 10: {symbols[-10:]}")
                else:
                    logger.info(f"All symbols: {symbols}")
            else:
                logger.warning(f"❌ No symbols found for layer {layer}")
                logger.info("Consider running 'python ai_trader.py universe --populate' first")

    except Exception as e:
        logger.error(f"Universe command error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        try:
            asyncio.run(universe_manager.close())
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
            # Ignore event loop closed errors during cleanup


@cli.command()
@click.option("--pipeline", is_flag=True, help="Run full scanner pipeline (Layer 0-3)")
@click.option("--layer", type=click.Choice(["0", "1", "2", "3"]), help="Run specific layer")
@click.option("--catalyst", help="Run specific catalyst scanner")
@click.option("--symbols", help="Comma-separated symbols to scan")
@click.option("--dry-run", is_flag=True, help="Test run without database updates")
@click.option("--show-alerts", is_flag=True, help="Display generated alerts")
@click.pass_context
def scan(ctx, pipeline, layer, catalyst, symbols, dry_run, show_alerts):
    """Run scanner pipeline for symbol qualification and catalyst detection"""
    config = ctx.obj["config"]

    try:
        if pipeline:
            # Run full scanner pipeline
            logger.info("Starting full scanner pipeline (Layer 0-3)...")
            # Local imports
            from main.scanners.scanner_pipeline import ScannerPipeline

            scanner_pipeline = ScannerPipeline(config, test_mode=dry_run)
            result = asyncio.run(scanner_pipeline.run())

            if result.success:
                logger.info("✅ Scanner pipeline completed successfully")
                logger.info(f"Final symbols qualified: {len(result.final_symbols)}")
                logger.info(f"Opportunities found: {len(result.final_opportunities)}")
                logger.info(f"Duration: {result.total_duration:.2f}s")
            else:
                logger.error(f"❌ Scanner pipeline failed: {', '.join(result.errors)}")
                sys.exit(1)

        elif layer is not None:
            # Run specific layer
            logger.info(f"Running Layer {layer} scanner...")
            # Local imports
            from main.scanners.layers import (
                Layer0StaticUniverseScanner,
                Layer1LiquidityFilter,
                Layer2CatalystOrchestrator,
                Layer3PreMarketScanner,
            )

            layer_map = {
                "0": Layer0StaticUniverseScanner,
                "1": Layer1LiquidityFilter,
                "2": Layer2CatalystOrchestrator,
                "3": Layer3PreMarketScanner,
            }

            scanner_class = layer_map[layer]
            scanner = scanner_class(config)

            # Get input symbols
            if symbols:
                symbol_list = symbols.split(",")
            else:
                # Get from previous layer or universe
                # Local imports
                from main.universe.universe_manager import UniverseManager

                um = UniverseManager(config)
                prev_layer = max(0, int(layer) - 1)
                symbol_list = asyncio.run(um.get_qualified_symbols(prev_layer))
                asyncio.run(um.close())

            # Run scanner
            if layer == "0":
                results = asyncio.run(scanner.scan())
            else:
                results = asyncio.run(scanner.scan(symbol_list))

            logger.info(f"✅ Layer {layer} scan complete")
            logger.info(
                f"Symbols qualified: {len(results) if isinstance(results, list) else 'N/A'}"
            )

        elif catalyst:
            # Run specific catalyst scanner
            logger.info(f"Running {catalyst} catalyst scanner...")
            # Local imports
            from main.scanners.scanner_factory_v2 import ScannerFactoryV2

            factory = ScannerFactoryV2(config)
            scanner = factory.create_scanner(catalyst)

            if not scanner:
                logger.error(f"Unknown catalyst scanner: {catalyst}")
                logger.info(
                    "Available scanners: volume, technical, news, earnings, options, insider, sector, social, intermarket"
                )
                sys.exit(1)

            # Get symbols
            if symbols:
                symbol_list = symbols.split(",")
            else:
                # Local imports
                from main.universe.universe_manager import UniverseManager

                um = UniverseManager(config)
                symbol_list = asyncio.run(um.get_qualified_symbols(2))  # Layer 2 for catalysts
                asyncio.run(um.close())

            # Run scanner
            alerts = asyncio.run(scanner.scan(symbol_list))

            logger.info(f"✅ {catalyst} scan complete")
            logger.info(f"Alerts generated: {len(alerts)}")

            if show_alerts and alerts:
                logger.info("\n=== Scanner Alerts ===")
                for alert in alerts[:10]:  # Show first 10
                    logger.info(
                        f"  {alert.symbol}: {alert.alert_type.value} (confidence: {alert.confidence:.2f})"
                    )
                if len(alerts) > 10:
                    logger.info(f"  ... and {len(alerts) - 10} more alerts")
        else:
            logger.error("Must specify --pipeline, --layer, or --catalyst")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Scanner error: {e}", exc_info=True)
        sys.exit(1)


@cli.command("scanner-status")
@click.option(
    "--layer",
    type=click.Choice(["0", "1", "2", "3", "all"]),
    default="all",
    help="Show status for specific layer or all",
)
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
@click.pass_context
def scanner_status(ctx, layer, detailed):
    """Show scanner layer qualifications and statistics"""
    # Local imports
    from main.data_pipeline.storage.database_factory import DatabaseFactory

    config = ctx.obj["config"]

    try:
        # Initialize components
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(config)
        # Local imports
        from main.data_pipeline.storage.repositories import get_repository_factory

        repo_factory = get_repository_factory()
        company_repo = repo_factory.create_company_repository(db_adapter)

        logger.info("=== Scanner Status ===\n")

        if layer == "all":
            # Show all layers
            for l in range(4):
                count = asyncio.run(company_repo.get_layer_count(l))
                logger.info(f"Layer {l}: {count} symbols qualified")

                if detailed and count > 0:
                    symbols = asyncio.run(company_repo.get_layer_symbols(l, limit=5))
                    logger.info(f"  Sample: {', '.join(symbols[:5])}")
        else:
            # Show specific layer
            l = int(layer)
            count = asyncio.run(company_repo.get_layer_count(l))
            logger.info(f"Layer {l}: {count} symbols qualified")

            if detailed:
                symbols = asyncio.run(company_repo.get_layer_symbols(l, limit=20))
                if symbols:
                    logger.info(f"\nSymbols: {', '.join(symbols)}")

        # Show recent scanner activity
        logger.info("\n=== Recent Scanner Activity ===")
        # This would query scanner_qualifications table for recent activity
        logger.info("(Scanner activity tracking coming soon)")

    except Exception as e:
        logger.error(f"Scanner status error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            asyncio.run(db_adapter.close())
        except Exception as e:
            logger.debug(f"Error closing database adapter: {e}")


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
        raise
    print("DEBUG: About to return from cli()", file=sys.stderr)
