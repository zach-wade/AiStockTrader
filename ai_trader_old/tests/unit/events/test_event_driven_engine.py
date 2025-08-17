"""Unit tests for event_driven_engine module."""

# Standard library imports
import asyncio
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest

# Local imports
from main.events.handlers.event_driven_engine import EventDrivenEngine
from tests.fixtures.events.mock_configs import create_complete_test_config


class TestEventDrivenEngine:
    """Test EventDrivenEngine class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = create_complete_test_config()
        config.update(
            {
                "system": {"name": "test_system", "version": "1.0.0", "environment": "test"},
                "monitoring": {"enabled": True, "metrics_interval": 60},
            }
        )
        return config

    @pytest.fixture
    def engine(self, config):
        """Create EventDrivenEngine instance for testing."""
        with patch("main.events.handlers.event_driven_engine.setup_logging"):
            engine = EventDrivenEngine(config)
            return engine

    def test_initialization(self, config):
        """Test engine initialization."""
        with patch("main.events.handlers.event_driven_engine.setup_logging") as mock_logging:
            engine = EventDrivenEngine(config)

            assert engine.config == config
            assert engine.components == {}
            assert engine.monitors == {}
            assert engine._shutdown_event is not None

            # Logging should be setup
            mock_logging.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_components(self, engine):
        """Test component initialization."""
        with patch(
            "main.events.handlers.event_driven_engine.initialize_event_bus"
        ) as mock_init_bus:
            with patch(
                "main.events.handlers.event_driven_engine.initialize_scanner_feature_bridge"
            ) as mock_init_bridge:
                with patch(
                    "main.events.handlers.event_driven_engine.FeaturePipelineHandler"
                ) as mock_handler_class:
                    # Setup mocks
                    mock_bus = AsyncMock()
                    mock_bridge = AsyncMock()
                    mock_handler = AsyncMock()

                    mock_init_bus.return_value = mock_bus
                    mock_init_bridge.return_value = mock_bridge
                    mock_handler_class.return_value = mock_handler

                    # Initialize components
                    await engine._initialize_components()

                    # Check components created
                    assert "event_bus" in engine.components
                    assert "scanner_bridge" in engine.components
                    assert "feature_pipeline" in engine.components

                    # Check initialization calls
                    mock_init_bus.assert_called_once_with(engine.config)
                    mock_init_bridge.assert_called_once_with(engine.config)
                    mock_handler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_monitors(self, engine):
        """Test monitor initialization."""
        # Create mock components
        engine.components = {
            "event_bus": Mock(get_stats=Mock(return_value={"events": 10})),
            "scanner_bridge": Mock(get_stats=Mock(return_value={"alerts": 5})),
        }

        with patch("main.events.handlers.event_driven_engine.SystemHealthMonitor") as mock_health:
            with patch("main.events.handlers.event_driven_engine.PerformanceMonitor") as mock_perf:
                mock_health_instance = Mock()
                mock_perf_instance = Mock()

                mock_health.return_value = mock_health_instance
                mock_perf.return_value = mock_perf_instance

                # Initialize monitors
                await engine._initialize_monitors()

                assert "health" in engine.monitors
                assert "performance" in engine.monitors

    @pytest.mark.asyncio
    async def test_setup_signal_handlers(self, engine):
        """Test signal handler setup."""
        with patch("signal.signal") as mock_signal:
            engine._setup_signal_handlers()

            # Should register handlers for SIGINT and SIGTERM
            assert mock_signal.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_lifecycle(self, engine):
        """Test engine run lifecycle."""
        # Mock all initialization methods
        engine._initialize_components = AsyncMock()
        engine._initialize_monitors = AsyncMock()
        engine._setup_signal_handlers = Mock()
        engine._monitor_loop = AsyncMock()
        engine._shutdown = AsyncMock()

        # Set shutdown event to stop immediately
        engine._shutdown_event.set()

        # Run engine
        await engine.run()

        # Check initialization sequence
        engine._initialize_components.assert_called_once()
        engine._initialize_monitors.assert_called_once()
        engine._setup_signal_handlers.assert_called_once()
        engine._shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_loop(self, engine):
        """Test monitoring loop."""
        # Setup components with stats
        engine.components = {
            "event_bus": Mock(
                get_stats=Mock(return_value={"events_published": 100, "events_processed": 95})
            ),
            "scanner_bridge": Mock(get_stats=Mock(return_value={"alerts_received_total": 50})),
        }

        # Setup monitors
        engine.monitors = {
            "health": Mock(check_health=Mock(return_value={"status": "healthy", "uptime": 3600})),
            "performance": Mock(
                get_metrics=Mock(return_value={"cpu_usage": 45.5, "memory_usage": 60.2})
            ),
        }

        # Run monitor loop once
        engine._shutdown_event.set()  # Stop after one iteration
        await engine._monitor_loop()

        # Check stats were collected
        engine.components["event_bus"].get_stats.assert_called()
        engine.components["scanner_bridge"].get_stats.assert_called()
        engine.monitors["health"].check_health.assert_called()
        engine.monitors["performance"].get_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown(self, engine):
        """Test clean shutdown."""
        # Setup components
        mock_handler = AsyncMock()
        engine.components = {
            "feature_pipeline": mock_handler,
            "scanner_bridge": Mock(),
            "event_bus": Mock(),
        }

        with patch(
            "main.events.handlers.event_driven_engine.cleanup_scanner_feature_bridge"
        ) as mock_cleanup_bridge:
            with patch(
                "main.events.handlers.event_driven_engine.cleanup_event_bus"
            ) as mock_cleanup_bus:
                mock_cleanup_bridge.return_value = asyncio.Future()
                mock_cleanup_bridge.return_value.set_result(None)
                mock_cleanup_bus.return_value = asyncio.Future()
                mock_cleanup_bus.return_value.set_result(None)

                # Shutdown
                await engine._shutdown()

                # Check cleanup sequence
                mock_handler.stop.assert_called_once()
                mock_cleanup_bridge.assert_called_once()
                mock_cleanup_bus.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_initialization(self, engine):
        """Test error handling during initialization."""
        with patch("main.events.handlers.event_driven_engine.initialize_event_bus") as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            # Should handle error gracefully
            with pytest.raises(Exception):
                await engine._initialize_components()

    @pytest.mark.asyncio
    async def test_handle_exception(self, engine):
        """Test exception handler."""
        loop = asyncio.get_event_loop()
        context = {"exception": ValueError("Test error"), "message": "Test exception"}

        # Capture log output
        with patch("main.events.handlers.event_driven_engine.logger") as mock_logger:
            engine._handle_exception(loop, context)

            # Should log error
            mock_logger.error.assert_called()

            # Should set shutdown event
            assert engine._shutdown_event.is_set()

    def test_signal_handler(self, engine):
        """Test signal handler behavior."""
        # Standard library imports
        import signal

        # Call signal handler
        engine._signal_handler(signal.SIGINT, None)

        # Should set shutdown event
        assert engine._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_cli_mode(self, engine):
        """Test CLI mode operation."""
        # Mock CLI components
        with patch("main.events.handlers.event_driven_engine.EngineShell") as mock_shell_class:
            mock_shell = Mock()
            mock_shell.cmdloop = Mock()
            mock_shell_class.return_value = mock_shell

            # Enable CLI mode
            engine.config["cli_mode"] = True

            # Mock initialization
            engine._initialize_components = AsyncMock()
            engine._initialize_monitors = AsyncMock()

            # Run in CLI mode
            engine._shutdown_event.set()  # Stop immediately
            await engine.run()

            # Shell should be created and run
            mock_shell_class.assert_called_once_with(engine)
            mock_shell.cmdloop.assert_called_once()

    @pytest.mark.asyncio
    async def test_component_stats_aggregation(self, engine):
        """Test aggregating stats from all components."""
        # Setup components
        engine.components = {
            "event_bus": Mock(
                get_stats=Mock(return_value={"events_published": 1000, "queue_size": 5})
            ),
            "scanner_bridge": Mock(
                get_stats=Mock(return_value={"alerts_processed": 250, "batch_size": 10})
            ),
            "feature_pipeline": Mock(
                get_stats=Mock(return_value={"requests_processed": 200, "workers": 4})
            ),
        }

        # Get aggregated stats
        stats = engine.get_system_stats()

        assert "event_bus" in stats
        assert "scanner_bridge" in stats
        assert "feature_pipeline" in stats
        assert stats["event_bus"]["events_published"] == 1000
        assert stats["scanner_bridge"]["alerts_processed"] == 250
        assert stats["feature_pipeline"]["requests_processed"] == 200

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, engine):
        """Test system continues if non-critical component fails."""
        with patch("main.events.handlers.event_driven_engine.initialize_event_bus") as mock_bus:
            with patch(
                "main.events.handlers.event_driven_engine.initialize_scanner_feature_bridge"
            ) as mock_bridge:
                mock_bus.return_value = AsyncMock()
                mock_bridge.side_effect = Exception("Bridge init failed")

                # Initialize with one component failing
                await engine._initialize_components()

                # Event bus should still be initialized
                assert "event_bus" in engine.components
                # Bridge initialization failed but system continues
                assert "scanner_bridge" not in engine.components

    def test_version_info(self, engine):
        """Test version information methods."""
        version_info = engine.get_version_info()

        assert "system_name" in version_info
        assert "version" in version_info
        assert "environment" in version_info
        assert version_info["system_name"] == "test_system"
        assert version_info["version"] == "1.0.0"
        assert version_info["environment"] == "test"
