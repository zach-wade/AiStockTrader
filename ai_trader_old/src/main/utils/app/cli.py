"""
CLI Application Utilities

Provides standardized CLI application creation and management utilities,
replacing the basic typer usage patterns found across app files.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
import signal
import sys

# Third-party imports
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger, setup_logging
from main.utils.monitoring import record_metric

# Removed get_config import to avoid circular dependency - not used in this file
from .context import StandardAppContext, managed_app_context

console = Console()
logger = get_logger(__name__)


@dataclass
class CLIAppConfig:
    """Configuration for CLI application creation."""

    name: str
    description: str = "AI Trader CLI Application"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_file: str | None = None
    json_logging: bool = False
    enable_monitoring: bool = True
    enable_error_handling: bool = True
    context_components: list[str] = field(default_factory=lambda: ["database", "data_sources"])
    graceful_shutdown: bool = True
    show_progress: bool = True


class StandardCLIHandler(ErrorHandlingMixin):
    """
    Standard CLI command handler with error handling, monitoring, and context management.
    """

    def __init__(self, config: CLIAppConfig):
        """
        Initialize StandardCLIHandler.

        Args:
            config: CLI application configuration
        """
        super().__init__()
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.name}")
        self.context: StandardAppContext | None = None
        self.shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        if config.graceful_shutdown:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    async def initialize(self) -> "StandardCLIHandler":
        """Initialize the CLI handler with app context."""
        try:
            # Setup logging
            setup_logging(
                level=self.config.log_level,
                log_file=self.config.log_file,
                json_logging=self.config.json_logging,
                component=self.config.name,
            )

            # Initialize app context
            self.context = await StandardAppContext(self.config.name).initialize(
                self.config.context_components
            )

            self.logger.info(f"✅ {self.config.name} CLI handler initialized successfully")

            # Record initialization metric
            if self.config.enable_monitoring:
                record_metric("cli_app_started", 1, tags={"app": self.config.name})

            return self

        except Exception as e:
            self.handle_error(e, f"initializing {self.config.name} CLI handler")
            raise

    async def shutdown(self):
        """Shutdown the CLI handler gracefully."""
        try:
            if self.context:
                await self.context.safe_shutdown()

            self.logger.info(f"✅ {self.config.name} CLI handler shutdown completed")

            # Record shutdown metric
            if self.config.enable_monitoring:
                record_metric("cli_app_stopped", 1, tags={"app": self.config.name})

        except Exception as e:
            self.handle_error(e, f"shutting down {self.config.name} CLI handler")

    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for managed CLI execution."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()


def create_cli_app(config: CLIAppConfig) -> typer.Typer:
    """
    Create a standardized CLI application with proper error handling and monitoring.

    Args:
        config: CLI application configuration

    Returns:
        Configured typer.Typer application
    """
    app = typer.Typer(name=config.name, help=config.description, add_completion=False)

    # Add version command
    @app.command()
    def version():
        """Show application version."""
        console.print(f"{config.name} version {config.version}")

    # Add status command
    @app.command()
    def status():
        """Show application status and health."""
        asyncio.run(_show_status(config))

    return app


async def _show_status(config: CLIAppConfig):
    """Show application status."""
    try:
        async with managed_app_context(config.name, config.context_components) as context:
            status = context.get_component_status()

            # Create status table
            table = Table(title=f"{config.name} Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="white")

            table.add_row("Application", "✅ Running", f"Uptime: {status['uptime_seconds']:.1f}s")
            table.add_row(
                "Database",
                "✅ Connected" if status["database_pool_active"] else "❌ Disconnected",
                "",
            )
            table.add_row(
                "Data Sources",
                "✅ Active" if status["data_source_manager_active"] else "❌ Inactive",
                "",
            )
            table.add_row(
                "Ingestion",
                "✅ Ready" if status["ingestion_orchestrator_active"] else "❌ Not Ready",
                "",
            )
            table.add_row(
                "Processing",
                "✅ Ready" if status["processing_manager_active"] else "❌ Not Ready",
                "",
            )
            table.add_row(
                "Errors",
                "⚠️ Issues" if status["error_count"] > 0 else "✅ None",
                f"Count: {status['error_count']}",
            )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Failed to get status: {e}", style="red")


def async_command(
    cli_handler: StandardCLIHandler, show_progress: bool = True, operation_name: str = "Operation"
):
    """
    Decorator for async CLI commands with standardized error handling and progress display.

    Args:
        cli_handler: CLI handler instance
        show_progress: Whether to show progress spinner
        operation_name: Name of the operation for progress display
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return asyncio.run(
                _execute_async_command(
                    func, cli_handler, show_progress, operation_name, *args, **kwargs
                )
            )

        return wrapper

    return decorator


async def _execute_async_command(
    func: Callable,
    cli_handler: StandardCLIHandler,
    show_progress: bool,
    operation_name: str,
    *args,
    **kwargs,
):
    """Execute async command with proper error handling and monitoring."""
    start_time = asyncio.get_event_loop().time()

    try:
        async with cli_handler.managed_execution():
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold blue]{operation_name}..."),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task(operation_name, total=None)
                    result = await func(cli_handler, *args, **kwargs)
            else:
                result = await func(cli_handler, *args, **kwargs)

            # Success message
            duration = asyncio.get_event_loop().time() - start_time
            console.print(
                f"✅ {operation_name} completed successfully in {duration:.2f}s", style="green"
            )

            # Record success metric
            if cli_handler.config.enable_monitoring:
                record_metric(
                    "cli_command_success",
                    1,
                    tags={
                        "app": cli_handler.config.name,
                        "command": func.__name__,
                        "duration": duration,
                    },
                )

            return result

    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        error_msg = f"❌ {operation_name} failed after {duration:.2f}s: {e}"

        console.print(error_msg, style="red")

        # Record error metric
        if cli_handler.config.enable_monitoring:
            record_metric(
                "cli_command_error",
                1,
                tags={
                    "app": cli_handler.config.name,
                    "command": func.__name__,
                    "error_type": type(e).__name__,
                },
            )

        # Handle error through CLI handler
        cli_handler.handle_error(e, f"executing {operation_name}")

        # Exit with error code
        sys.exit(1)


def create_data_pipeline_app(app_name: str, description: str = None) -> typer.Typer:
    """
    Create a standardized data pipeline CLI application.

    Args:
        app_name: Name of the application
        description: Optional description

    Returns:
        Configured typer.Typer application
    """
    config = CLIAppConfig(
        name=app_name,
        description=description or f"AI Trader {app_name.title()} Pipeline",
        context_components=["database", "data_sources", "ingestion"],
        enable_monitoring=True,
        show_progress=True,
    )

    return create_cli_app(config)


def create_training_app(app_name: str, description: str = None) -> typer.Typer:
    """
    Create a standardized training CLI application.

    Args:
        app_name: Name of the application
        description: Optional description

    Returns:
        Configured typer.Typer application
    """
    config = CLIAppConfig(
        name=app_name,
        description=description or f"AI Trader {app_name.title()} Training",
        context_components=["database", "data_sources"],
        enable_monitoring=True,
        show_progress=True,
        log_level="INFO",
    )

    return create_cli_app(config)


def create_validation_app(app_name: str, description: str = None) -> typer.Typer:
    """
    Create a standardized validation CLI application.

    Args:
        app_name: Name of the application
        description: Optional description

    Returns:
        Configured typer.Typer application
    """
    config = CLIAppConfig(
        name=app_name,
        description=description or f"AI Trader {app_name.title()} Validation",
        context_components=["database", "data_sources"],
        enable_monitoring=True,
        show_progress=True,
        log_level="INFO",
    )

    return create_cli_app(config)


# Convenience functions for common CLI patterns
def success_message(message: str, details: str | None = None):
    """Display a success message with optional details."""
    console.print(f"✅ {message}", style="green")
    if details:
        console.print(f"   {details}", style="dim")


def error_message(message: str, details: str | None = None):
    """Display an error message with optional details."""
    console.print(f"❌ {message}", style="red")
    if details:
        console.print(f"   {details}", style="dim")


def warning_message(message: str, details: str | None = None):
    """Display a warning message with optional details."""
    console.print(f"⚠️ {message}", style="yellow")
    if details:
        console.print(f"   {details}", style="dim")


def info_message(message: str, details: str | None = None):
    """Display an info message with optional details."""
    console.print(f"ℹ️ {message}", style="blue")
    if details:
        console.print(f"   {details}", style="dim")
