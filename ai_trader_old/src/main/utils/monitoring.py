"""
Monitoring utilities for the AI Trader system.

This module provides a unified interface to all monitoring utilities including:
- System metrics collection (CPU, memory, disk, network)
- Performance monitoring and function tracking
- Alert management and notification systems
- Memory profiling and leak detection
- Global monitoring coordination
- Metrics export and visualization support

This is the main interface module that imports all monitoring utilities from the
monitoring/ subdirectory for easy access throughout the system.
"""

# Import and explicitly re-export monitoring utilities

# Convenience aliases for common patterns

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"

# Default monitoring configuration
DEFAULT_COLLECTION_INTERVAL = 60  # seconds
DEFAULT_RETENTION_PERIOD = 24 * 60 * 60  # 24 hours in seconds
DEFAULT_ALERT_THRESHOLD_CPU = 80.0  # percent
DEFAULT_ALERT_THRESHOLD_MEMORY = 85.0  # percent
DEFAULT_ALERT_THRESHOLD_DISK = 90.0  # percent
