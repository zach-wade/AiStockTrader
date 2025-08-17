"""
Global Universe Manager

Global instance management for the trading universe system.
"""

# Standard library imports
from collections.abc import Callable
import logging

from .manager import UniverseManager

logger = logging.getLogger(__name__)


# Global universe manager instance
_global_manager: UniverseManager | None = None


def get_global_manager() -> UniverseManager | None:
    """Get the global universe manager instance."""
    return _global_manager


def set_global_manager(manager: UniverseManager):
    """Set the global universe manager instance."""
    global _global_manager
    _global_manager = manager
    logger.info("Global universe manager set")


def init_global_manager(data_provider: Callable | None = None):
    """Initialize global universe manager."""
    global _global_manager
    _global_manager = UniverseManager(data_provider)
    logger.info("Global universe manager initialized")


def ensure_global_manager() -> UniverseManager:
    """Ensure global manager exists and return it."""
    global _global_manager
    if _global_manager is None:
        init_global_manager()
    return _global_manager


def reset_global_manager():
    """Reset the global universe manager."""
    global _global_manager
    _global_manager = None
    logger.info("Global universe manager reset")


def is_global_manager_initialized() -> bool:
    """Check if global manager is initialized."""
    return _global_manager is not None
