"""Unit tests for EventBusRegistry."""

# Standard library imports
from unittest.mock import AsyncMock, Mock

# Third-party imports
import pytest

# Local imports
from main.events.core import EventBusConfig, EventBusFactory, EventBusRegistry
from main.events.core.event_bus_registry import EventBusAlreadyExistsError, EventBusNotFoundError
from main.interfaces.events import IEventBus


@pytest.mark.unit
@pytest.mark.events
class TestEventBusRegistry:
    """Test EventBusRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create registry for testing."""
        return EventBusRegistry(auto_create=False)

    @pytest.fixture
    def auto_registry(self):
        """Create registry with auto-create enabled."""
        return EventBusRegistry(auto_create=True)

    def test_register_and_get(self, registry):
        """Test registering and retrieving event bus."""
        # Create and register
        bus = EventBusFactory.create_test_instance()
        registry.register_event_bus(bus, "test")

        # Retrieve
        retrieved = registry.get_event_bus("test")
        assert retrieved is bus

    def test_register_default(self, registry):
        """Test registering default (None name) event bus."""
        bus = EventBusFactory.create_test_instance()
        registry.register_event_bus(bus)

        retrieved = registry.get_event_bus()
        assert retrieved is bus

    def test_has_event_bus(self, registry):
        """Test checking if event bus exists."""
        assert not registry.has_event_bus("test")

        bus = EventBusFactory.create_test_instance()
        registry.register_event_bus(bus, "test")

        assert registry.has_event_bus("test")
        assert not registry.has_event_bus("other")

    def test_unregister(self, registry):
        """Test unregistering event bus."""
        bus = EventBusFactory.create_test_instance()
        registry.register_event_bus(bus, "test")

        registry.unregister_event_bus("test")
        assert not registry.has_event_bus("test")

    def test_register_duplicate_error(self, registry):
        """Test error when registering duplicate name."""
        bus1 = EventBusFactory.create_test_instance()
        bus2 = EventBusFactory.create_test_instance()

        registry.register_event_bus(bus1, "test")

        with pytest.raises(EventBusAlreadyExistsError):
            registry.register_event_bus(bus2, "test")

    def test_get_nonexistent_error(self, registry):
        """Test error when getting non-existent bus."""
        with pytest.raises(EventBusNotFoundError):
            registry.get_event_bus("nonexistent")

    def test_unregister_nonexistent_error(self, registry):
        """Test error when unregistering non-existent bus."""
        with pytest.raises(EventBusNotFoundError):
            registry.unregister_event_bus("nonexistent")

    def test_auto_create(self, auto_registry):
        """Test auto-creation of event buses."""
        # Should create automatically
        bus = auto_registry.get_event_bus("auto_test")
        assert isinstance(bus, IEventBus)
        assert auto_registry.has_event_bus("auto_test")

        # Should return same instance
        bus2 = auto_registry.get_event_bus("auto_test")
        assert bus is bus2

    def test_register_with_config(self, registry):
        """Test registering with configuration."""
        bus = EventBusFactory.create_test_instance()
        config = EventBusConfig(max_queue_size=123)

        registry.register_event_bus(bus, "test", config)

        # Config should be stored
        assert "test" in registry._configs
        assert registry._configs["test"].max_queue_size == 123

    def test_register_config_for_auto_create(self, auto_registry):
        """Test pre-registering config for auto-creation."""
        config = EventBusConfig(max_queue_size=999, max_workers=3)
        auto_registry.register_config("future", config)

        # When auto-created, should use the config
        bus = auto_registry.get_event_bus("future")
        assert isinstance(bus, IEventBus)

    def test_get_all_names(self, registry):
        """Test getting all registered names."""
        assert registry.get_all_names() == set()

        registry.register_event_bus(Mock(), "bus1")
        registry.register_event_bus(Mock(), "bus2")
        registry.register_event_bus(Mock(), None)  # default

        names = registry.get_all_names()
        assert names == {"bus1", "bus2", None}

    def test_clear(self, registry):
        """Test clearing all registrations."""
        registry.register_event_bus(Mock(), "bus1")
        registry.register_event_bus(Mock(), "bus2")
        registry.register_config("bus3", EventBusConfig())

        registry.clear()

        assert len(registry._instances) == 0
        assert len(registry._configs) == 0
        assert registry.get_all_names() == set()

    @pytest.mark.asyncio
    async def test_stop_all(self, registry):
        """Test stopping all event buses."""
        # Create mock buses
        bus1 = AsyncMock()
        bus1.is_running.return_value = True

        bus2 = AsyncMock()
        bus2.is_running.return_value = False

        bus3 = AsyncMock()
        bus3.is_running.return_value = True
        bus3.stop.side_effect = Exception("Stop failed")

        # Register them
        registry.register_event_bus(bus1, "bus1")
        registry.register_event_bus(bus2, "bus2")
        registry.register_event_bus(bus3, "bus3")

        # Stop all
        await registry.stop_all()

        # Verify calls
        bus1.stop.assert_called_once()
        bus2.stop.assert_not_called()  # Not running
        bus3.stop.assert_called_once()  # Should try despite error

    def test_thread_safety(self, registry):
        """Test that registry operations are thread-safe."""
        # Standard library imports
        import threading

        results = []

        def register_bus(name):
            try:
                bus = EventBusFactory.create_test_instance()
                registry.register_event_bus(bus, name)
                results.append(("success", name))
            except EventBusAlreadyExistsError:
                results.append(("duplicate", name))

        # Try to register same name from multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=register_bus, args=("concurrent",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Exactly one should succeed
        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 1
