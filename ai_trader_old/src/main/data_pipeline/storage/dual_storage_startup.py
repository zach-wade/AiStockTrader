"""
Dual Storage Startup Module

Provides minimal implementation for dual storage initialization.
This is a placeholder that allows the system to run without full
event-driven dual storage implementation.

Future enhancements can add:
- Event bus for async replication
- Cold storage consumer for automatic archiving
- Data lifecycle management
"""

# Standard library imports
from typing import Any

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class DualStorageManager:
    """
    Placeholder for dual storage management.

    Future implementation will coordinate between hot and cold storage
    with event-driven replication.
    """

    def __init__(self):
        """Initialize the dual storage manager."""
        self.hot_storage = None
        self.cold_storage = None
        self.event_bus = None
        self.cold_consumer = None
        self.is_running = False

    def get_metrics(self) -> dict:
        """
        Get dual storage metrics.

        Returns:
            Dictionary with placeholder metrics
        """
        return {
            "hot_storage": {"status": "active" if self.hot_storage else "inactive"},
            "cold_storage": {"status": "inactive"},
            "cold_consumer": {"status": "not_implemented"},
            "event_bus": {"status": "not_implemented"},
        }

    async def health_check(self) -> dict:
        """
        Check health of dual storage components.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": False,
            "reason": "Dual storage not implemented",
            "metrics": self.get_metrics(),
        }


# Global instance (singleton pattern)
_dual_storage_manager: DualStorageManager | None = None


def initialize_dual_storage(
    hot_storage: Any, enable_dual_storage: bool = True
) -> tuple[Any | None, Any | None]:
    """
    Initialize dual storage components.

    This is a minimal implementation that returns None for both
    event bus and cold storage, allowing the system to run without
    these components.

    Args:
        hot_storage: The hot storage adapter (PostgreSQL)
        enable_dual_storage: Whether to enable dual storage (ignored for now)

    Returns:
        Tuple of (event_bus, cold_storage) - both None in this implementation
    """
    global _dual_storage_manager

    logger.info("Initializing dual storage (minimal implementation)")

    # Create manager instance
    _dual_storage_manager = DualStorageManager()
    _dual_storage_manager.hot_storage = hot_storage

    # Log that dual storage is not fully implemented
    if enable_dual_storage:
        logger.warning(
            "Dual storage requested but not fully implemented. "
            "System will run with hot storage only."
        )

    # Return None for both components (system handles this gracefully)
    return None, None


async def start_dual_storage_consumer() -> None:
    """
    Start the cold storage consumer.

    This is a no-op in the minimal implementation.
    """
    logger.debug("start_dual_storage_consumer called (no-op)")

    if _dual_storage_manager:
        _dual_storage_manager.is_running = True


async def stop_dual_storage() -> None:
    """
    Stop dual storage components.

    This is a no-op in the minimal implementation.
    """
    logger.debug("stop_dual_storage called (no-op)")

    global _dual_storage_manager
    if _dual_storage_manager:
        _dual_storage_manager.is_running = False
        _dual_storage_manager = None


def get_dual_storage_manager() -> DualStorageManager | None:
    """
    Get the dual storage manager instance.

    Returns:
        DualStorageManager instance or None if not initialized
    """
    return _dual_storage_manager


# Future Implementation Notes:
#
# 1. Event Bus Integration:
#    - Create EventBus instance for async communication
#    - Publish events when data is written to hot storage
#    - Subscribe to events in cold storage consumer
#
# 2. Cold Storage Consumer:
#    - Background task that listens for events
#    - Automatically archives data to cold storage
#    - Manages data lifecycle (move old data to cold)
#
# 3. Data Lifecycle Management:
#    - Automatic migration of old data from hot to cold
#    - Configurable retention policies per data type
#    - Cleanup of expired data
#
# 4. Query Federation:
#    - Combine results from hot and cold storage
#    - Transparent access to all data regardless of location
#    - Caching layer for frequently accessed cold data
