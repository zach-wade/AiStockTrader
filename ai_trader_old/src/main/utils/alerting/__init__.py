"""
Alerting Module

Provides integration with external alerting services for critical system events.
"""

from .alerting_service import AlertChannel, AlertingService, AlertPriority, get_alerting_service

__all__ = ["AlertingService", "AlertChannel", "AlertPriority", "get_alerting_service"]
