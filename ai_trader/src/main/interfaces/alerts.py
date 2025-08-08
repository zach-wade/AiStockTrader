"""
Alert System Interfaces

Defines contracts for alert system components to break circular dependencies
and provide clean abstractions for alert handling.
"""

from abc import abstractmethod
from typing import Protocol, Dict, Any, Optional, List, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert category types."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    DATA = "data"
    PERFORMANCE = "performance"
    ERROR = "error"
    CUSTOM = "custom"


# Alias for backward compatibility
AlertLevel = AlertSeverity


@dataclass
class Alert:
    """Alert message structure."""
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    alert_id: Optional[str] = None
    
    @property
    def level(self) -> AlertSeverity:
        """Alias for severity for backward compatibility."""
        return self.severity


@runtime_checkable
class IAlertChannel(Protocol):
    """
    Interface for alert delivery channels.
    
    All alert channels (email, SMS, Slack, etc.) must implement this interface
    to ensure consistent behavior across different notification methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the channel name."""
        ...
    
    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if the channel is enabled."""
        ...
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.
        
        Args:
            alert: The alert to send
            
        Returns:
            True if the alert was sent successfully, False otherwise
        """
        ...
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the channel is currently available.
        
        Returns:
            True if the channel can send alerts, False otherwise
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get channel statistics.
        
        Returns:
            Dictionary containing channel-specific statistics
        """
        ...


@runtime_checkable
class IAlertSystem(Protocol):
    """
    Interface for the main alert system.
    
    The alert system manages multiple channels and handles routing,
    rate limiting, and delivery of alerts.
    """
    
    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        level: AlertSeverity,
        category: AlertCategory = AlertCategory.SYSTEM,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> bool:
        """
        Send an alert through the alert system.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert severity level
            category: Alert category
            metadata: Additional metadata
            channels: Specific channels to use (None for default routing)
            
        Returns:
            True if alert was sent to at least one channel
        """
        ...
    
    @abstractmethod
    def add_channel(self, name: str, channel: IAlertChannel) -> None:
        """Add an alert channel to the system."""
        ...
    
    @abstractmethod
    def remove_channel(self, name: str) -> None:
        """Remove an alert channel from the system."""
        ...
    
    @abstractmethod
    def get_channels(self) -> Dict[str, IAlertChannel]:
        """Get all registered channels."""
        ...
    
    @abstractmethod
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        ...