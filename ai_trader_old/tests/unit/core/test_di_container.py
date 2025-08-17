"""Unit tests for DIContainer."""

# Standard library imports
from unittest.mock import Mock

# Third-party imports
import pytest

# Local imports
from main.core import DIContainer, Lifecycle
from main.interfaces.events import IEventBus, IEventBusProvider


@pytest.mark.unit
class TestDIContainer:
    """Test DIContainer functionality."""

    @pytest.fixture
    def container(self):
        """Create container for testing."""
        return DIContainer()

    def test_register_and_resolve_transient(self, container):
        """Test transient lifecycle."""

        # Register a class
        class TestService:
            pass

        container.register_transient(TestService, TestService)

        # Resolve multiple times
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)

        # Should be different instances
        assert isinstance(instance1, TestService)
        assert isinstance(instance2, TestService)
        assert instance1 is not instance2

    def test_register_and_resolve_singleton(self, container):
        """Test singleton lifecycle."""

        class TestService:
            pass

        container.register_singleton(TestService, TestService)

        # Resolve multiple times
        instance1 = container.resolve(TestService)
        instance2 = container.resolve(TestService)

        # Should be same instance
        assert instance1 is instance2

    def test_register_instance_as_singleton(self, container):
        """Test registering an instance directly."""
        instance = ["test", "instance"]
        container.register(list, instance)

        resolved = container.resolve(list)
        assert resolved is instance

    def test_register_factory(self, container):
        """Test factory registration."""
        counter = 0

        def factory():
            nonlocal counter
            counter += 1
            return f"instance_{counter}"

        container.register_factory(str, factory)

        # Each resolve should call factory
        instance1 = container.resolve(str)
        instance2 = container.resolve(str)

        assert instance1 == "instance_1"
        assert instance2 == "instance_2"

    def test_constructor_injection(self, container):
        """Test automatic constructor injection."""

        class DependencyA:
            def __init__(self):
                self.name = "A"

        class DependencyB:
            def __init__(self):
                self.name = "B"

        class ServiceWithDeps:
            def __init__(self, dep_a: DependencyA, dep_b: DependencyB):
                self.dep_a = dep_a
                self.dep_b = dep_b

        # Register dependencies
        container.register_singleton(DependencyA, DependencyA)
        container.register_singleton(DependencyB, DependencyB)
        container.register_transient(ServiceWithDeps, ServiceWithDeps)

        # Resolve service - dependencies should be injected
        service = container.resolve(ServiceWithDeps)

        assert isinstance(service, ServiceWithDeps)
        assert isinstance(service.dep_a, DependencyA)
        assert isinstance(service.dep_b, DependencyB)
        assert service.dep_a.name == "A"
        assert service.dep_b.name == "B"

    def test_optional_dependency_injection(self, container):
        """Test injection with Optional types."""

        class OptionalDep:
            pass

        class ServiceWithOptional:
            def __init__(self, dep: OptionalDep | None = None):
                self.dep = dep

        # Register service but not dependency
        container.register_transient(ServiceWithOptional, ServiceWithOptional)

        # Should resolve with None for optional dep
        service = container.resolve(ServiceWithOptional)
        assert service.dep is None

        # Now register the dependency
        container.register_singleton(OptionalDep, OptionalDep)

        # Should resolve with instance
        service2 = container.resolve(ServiceWithOptional)
        assert isinstance(service2.dep, OptionalDep)

    def test_factory_with_injection(self, container):
        """Test factory functions with dependency injection."""

        class Dependency:
            def __init__(self):
                self.value = 42

        def service_factory(dep: Dependency) -> dict:
            return {"dep_value": dep.value}

        container.register_singleton(Dependency, Dependency)
        container.register_factory(dict, service_factory)

        result = container.resolve(dict)
        assert result == {"dep_value": 42}

    def test_scoped_lifecycle(self, container):
        """Test scoped lifecycle."""

        class ScopedService:
            pass

        container.register(ScopedService, ScopedService, Lifecycle.SCOPED)

        # Within same scope, should be same instance
        instance1 = container.resolve(ScopedService)
        instance2 = container.resolve(ScopedService)
        assert instance1 is instance2

        # Clear scope
        container.clear_scoped()

        # New scope should have new instance
        instance3 = container.resolve(ScopedService)
        assert instance3 is not instance1

    def test_has_registration(self, container):
        """Test checking if type is registered."""
        assert not container.has_registration(str)

        container.register_singleton(str, "test")
        assert container.has_registration(str)

    def test_unregister(self, container):
        """Test unregistering types."""
        container.register_singleton(str, "test")
        assert container.has_registration(str)

        container.unregister(str)
        assert not container.has_registration(str)

    def test_clear(self, container):
        """Test clearing all registrations."""
        container.register_singleton(str, "test")
        container.register_transient(list, list)
        container.register(dict, dict, Lifecycle.SCOPED)

        container.clear()

        assert not container.has_registration(str)
        assert not container.has_registration(list)
        assert not container.has_registration(dict)

    def test_resolve_unregistered_error(self, container):
        """Test error when resolving unregistered type."""
        with pytest.raises(ValueError, match="not registered"):
            container.resolve(str)

    def test_duplicate_registration_error(self, container):
        """Test error when registering duplicate."""
        container.register_singleton(str, "test")

        with pytest.raises(ValueError, match="already registered"):
            container.register_singleton(str, "test2")

    def test_metadata_storage(self, container):
        """Test storing metadata with registrations."""
        container.register(str, "test", metadata_key="metadata_value", another_key=123)

        registration = container._registrations[str]
        assert registration.metadata["metadata_key"] == "metadata_value"
        assert registration.metadata["another_key"] == 123

    def test_real_world_scenario(self, container):
        """Test a real-world DI scenario."""
        # Simulate event bus registration
        # Local imports
        from main.events import EventBusFactory

        # Register event bus provider
        mock_provider = Mock(spec=IEventBusProvider)
        container.register_singleton(IEventBusProvider, mock_provider)

        # Register event bus factory
        container.register_factory(
            IEventBus, lambda: EventBusFactory.create_test_instance(), lifecycle=Lifecycle.SINGLETON
        )

        # Resolve
        provider = container.resolve(IEventBusProvider)
        event_bus = container.resolve(IEventBus)

        assert provider is mock_provider
        assert isinstance(event_bus, IEventBus)

        # Singleton should return same instance
        event_bus2 = container.resolve(IEventBus)
        assert event_bus is event_bus2
