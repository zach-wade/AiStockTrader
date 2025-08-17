"""
Test to verify test infrastructure is working correctly.

This test ensures pytest, fixtures, and basic functionality work.
"""

# Standard library imports
import asyncio
from datetime import datetime

# Third-party imports
import pytest


class TestInfrastructure:
    """Basic tests to verify test setup."""

    def test_basic_assertion(self):
        """Test that basic assertions work."""
        assert True
        assert 1 + 1 == 2
        assert "test" in "testing"

    def test_pytest_fixtures_work(self):
        """Test that pytest fixtures are available."""
        # This will fail if pytest isn't properly configured
        assert hasattr(pytest, "fixture")
        assert hasattr(pytest, "mark")

    @pytest.mark.asyncio
    async def test_async_support(self):
        """Test that async tests work."""
        await asyncio.sleep(0.01)
        result = await self._async_helper()
        assert result == "async works"

    async def _async_helper(self):
        """Helper async function."""
        return "async works"

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        # This test should only run when integration tests are selected
        assert True

    def test_imports_work(self):
        """Test that we can import main modules."""
        try:
            # Local imports
            from main.config.config_manager import get_config

            assert True
        except ImportError as e:
            pytest.skip(f"Import not available in test environment: {e}")

    @pytest.mark.parametrize("input,expected", [(1, 1), (2, 4), (3, 9), (4, 16)])
    def test_parametrized(self, input, expected):
        """Test parametrized tests work."""
        assert input**2 == expected

    def test_datetime_available(self):
        """Test standard libraries available."""
        now = datetime.now()
        assert now.year >= 2024

    @pytest.fixture
    def sample_data(self):
        """Test fixture."""
        return {"key": "value", "number": 42, "list": [1, 2, 3]}

    def test_fixture_usage(self, sample_data):
        """Test using fixtures."""
        assert sample_data["key"] == "value"
        assert sample_data["number"] == 42
        assert len(sample_data["list"]) == 3


@pytest.mark.slow
class TestSlowOperations:
    """Tests marked as slow."""

    def test_slow_operation(self):
        """Test that slow marker works."""
        # This should only run when slow tests are included
        # Standard library imports
        import time

        time.sleep(0.1)  # Simulate slow operation
        assert True
