"""Unit tests for scanner_feature_bridge_initializer module."""

# Standard library imports
from unittest.mock import AsyncMock, Mock, patch

# Third-party imports
import pytest

# Local imports
from main.events.core.event_bus import EventBus

# scanner_feature_bridge_initializer removed - use ScannerFeatureBridge directly
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from tests.fixtures.events.mock_configs import MockConfig, create_scanner_bridge_config


class TestScannerFeatureBridgeInitializer:
    """Test scanner feature bridge initializer functions."""

    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        # Reset global state before each test
        scanner_feature_bridge_initializer._bridge_instance = None
        yield
        # Cleanup after test
        scanner_feature_bridge_initializer._bridge_instance = None

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MockConfig()
        config.update(create_scanner_bridge_config())
        return config

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = Mock(spec=EventBus)
        bus.subscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    def test_get_scanner_feature_bridge_uninitialized(self, setup, mock_event_bus):
        """Test getting bridge when not initialized."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                # Get bridge (should create with default config)
                bridge = scanner_feature_bridge_initializer.get_scanner_feature_bridge()

                assert bridge is not None
                assert isinstance(bridge, ScannerFeatureBridge)
                assert scanner_feature_bridge_initializer._bridge_instance == bridge

                # Getting again should return same instance
                bridge2 = scanner_feature_bridge_initializer.get_scanner_feature_bridge()
                assert bridge2 is bridge

    def test_get_bridge_thread_safety(self, setup, mock_event_bus):
        """Test thread-safe access to bridge."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                bridges = []

                def get_bridge():
                    bridges.append(scanner_feature_bridge_initializer.get_scanner_feature_bridge())

                # Create multiple threads trying to get bridge
                # Standard library imports
                import threading

                threads = []
                for _ in range(5):
                    t = threading.Thread(target=get_bridge)
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()

                # All should have gotten the same instance
                assert len(bridges) == 5
                assert all(bridge is bridges[0] for bridge in bridges)

    @pytest.mark.asyncio
    async def test_initialize_scanner_feature_bridge(self, setup, mock_event_bus):
        """Test initializing bridge with config."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                # Initialize bridge
                bridge = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )

                assert bridge is not None
                assert isinstance(bridge, ScannerFeatureBridge)
                assert bridge._running is True
                assert scanner_feature_bridge_initializer._bridge_instance == bridge

                # Clean up
                await bridge.stop()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_config(self, setup, mock_event_bus):
        """Test initializing with custom configuration."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            mock_get_bus.return_value = mock_event_bus

            custom_config = {
                "scanner_feature_bridge": {
                    "batch_size": 25,
                    "batch_timeout_seconds": 3.0,
                    "max_symbols_per_batch": 75,
                    "rate_limit_per_second": 15,
                    "dedup_window_seconds": 45,
                }
            }

            bridge = await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge(
                custom_config
            )

            assert bridge.batch_size == 25
            assert bridge.batch_timeout_seconds == 3.0
            assert bridge.max_symbols_per_batch == 75
            assert bridge.dedup_window_seconds == 45

            # Clean up
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, setup, mock_event_bus):
        """Test initializing when already initialized."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                # Initialize first time
                bridge1 = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )

                # Try to initialize again
                bridge2 = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )

                # Should return same instance and log warning
                assert bridge2 is bridge1

                # Clean up
                await bridge1.stop()

    @pytest.mark.asyncio
    async def test_cleanup_scanner_feature_bridge(self, setup, mock_event_bus):
        """Test cleaning up bridge."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                # Initialize bridge
                bridge = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )
                assert bridge._running is True

                # Mock get_stats
                bridge.get_stats = Mock(
                    return_value={
                        "alerts_received_total": 50,
                        "feature_requests_sent_total": 10,
                        "unique_symbols_processed_count": 25,
                    }
                )

                # Clean up
                await scanner_feature_bridge_initializer.cleanup_scanner_feature_bridge()

                # Should be stopped and cleared
                assert bridge._running is False
                assert scanner_feature_bridge_initializer._bridge_instance is None

                # Stats should have been retrieved
                bridge.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self, setup):
        """Test cleanup when not initialized."""
        # Should not raise error
        await scanner_feature_bridge_initializer.cleanup_scanner_feature_bridge()
        assert scanner_feature_bridge_initializer._bridge_instance is None

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self, setup, mock_event_bus):
        """Test cleanup handles errors gracefully."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                bridge = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )

                # Make stop raise an error
                bridge.stop = AsyncMock(side_effect=Exception("Stop failed"))

                # Cleanup should still complete
                await scanner_feature_bridge_initializer.cleanup_scanner_feature_bridge()

                # Instance should still be cleared
                assert scanner_feature_bridge_initializer._bridge_instance is None

    def test_is_bridge_running(self, setup):
        """Test checking if bridge is running."""
        # Not initialized
        assert scanner_feature_bridge_initializer.is_bridge_running() is False

        # Create mock bridge
        mock_bridge = Mock()
        mock_bridge._running = True
        scanner_feature_bridge_initializer._bridge_instance = mock_bridge

        assert scanner_feature_bridge_initializer.is_bridge_running() is True

        # Not running
        mock_bridge._running = False
        assert scanner_feature_bridge_initializer.is_bridge_running() is False

    @pytest.mark.asyncio
    async def test_restart_scanner_feature_bridge(self, setup, mock_event_bus):
        """Test restarting bridge."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                mock_get_bus.return_value = mock_event_bus
                mock_get_config.return_value = create_scanner_bridge_config()

                # Initialize first bridge
                bridge1 = (
                    await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                )
                bridge1_id = id(bridge1)

                # Restart with new config
                new_config = create_scanner_bridge_config()
                new_config["scanner_feature_bridge"]["batch_size"] = 100

                bridge2 = await scanner_feature_bridge_initializer.restart_scanner_feature_bridge(
                    new_config
                )

                # Should be different instance
                assert id(bridge2) != bridge1_id
                assert bridge2.batch_size == 100
                assert bridge1._running is False  # Old bridge should be stopped
                assert bridge2._running is True

                # Clean up
                await bridge2.stop()

    def test_get_bridge_stats(self, setup):
        """Test getting bridge statistics."""
        # Not initialized
        stats = scanner_feature_bridge_initializer.get_bridge_stats()
        assert stats == {}

        # Create mock bridge with stats
        mock_bridge = Mock()
        mock_bridge.get_stats = Mock(
            return_value={"alerts_received_total": 100, "feature_requests_sent_total": 20}
        )
        scanner_feature_bridge_initializer._bridge_instance = mock_bridge

        stats = scanner_feature_bridge_initializer.get_bridge_stats()
        assert stats["alerts_received_total"] == 100
        assert stats["feature_requests_sent_total"] == 20

        mock_bridge.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_with_metrics(self, setup, mock_event_bus):
        """Test that initialization records metrics."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            with patch(
                "main.events.scanner_feature_bridge_initializer.get_config"
            ) as mock_get_config:
                with patch(
                    "main.events.scanner_feature_bridge_initializer.record_metric"
                ) as mock_metric:
                    mock_get_bus.return_value = mock_event_bus
                    mock_get_config.return_value = create_scanner_bridge_config()

                    bridge = (
                        await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge()
                    )

                    # Should record success metric
                    mock_metric.assert_called_with("scanner_bridge_initializer.success", 1)

                    # Clean up
                    await bridge.stop()

    @pytest.mark.asyncio
    async def test_configuration_extraction(self, setup, mock_event_bus):
        """Test proper extraction of configuration values."""
        with patch("main.events.scanner_feature_bridge_initializer.get_event_bus") as mock_get_bus:
            mock_get_bus.return_value = mock_event_bus

            config = {
                "scanner_feature_bridge": {
                    "batch_size": 30,
                    "batch_timeout_seconds": 4.5,
                    "max_symbols_per_batch": 80,
                    "rate_limit_per_second": 12,
                    "dedup_window_seconds": 90,
                    "custom_setting": "ignored",  # Extra settings
                }
            }

            bridge = await scanner_feature_bridge_initializer.initialize_scanner_feature_bridge(
                config
            )

            # Check all settings properly extracted
            assert bridge.batch_size == 30
            assert bridge.batch_timeout_seconds == 4.5
            assert bridge.max_symbols_per_batch == 80
            assert bridge.dedup_window_seconds == 90

            # Config should be passed through
            assert bridge.config == config["scanner_feature_bridge"]

            # Clean up
            await bridge.stop()
