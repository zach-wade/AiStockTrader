"""
Base Scanner Class

Provides the foundation for all scanner implementations, ensuring
consistent output format and behavior across the scanning pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from main.events.types import ScanAlert, AlertType
from main.interfaces.events import IEventBus, EventType, ScannerAlertEvent
from main.interfaces.events.time_utils import ensure_utc


class BaseScanner(ABC):
    """
    Abstract base class for all scanner implementations.
    
    Enforces a consistent interface and output format (ScanAlert objects)
    across all scanning components in the Intelligence Agency architecture.
    """
    
    def __init__(self, name: str):
        """
        Initialize scanner with a unique name.
        
        Args:
            name: Scanner identifier (e.g., 'VolumeScanner', 'TechnicalScanner')
        """
        self._name = name
        self.logger = logging.getLogger(f"main.scanners.{name}")
    
    @property
    def name(self) -> str:
        """Scanner name for identification and logging."""
        return self._name
    
    @abstractmethod
    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Perform scanning operation on the given symbols.
        
        Args:
            symbols: List of stock symbols to scan
            **kwargs: Additional scanner-specific parameters
            
        Returns:
            List of ScanAlert objects for detected signals
        """
        pass
    
    def create_alert(
        self,
        symbol: str,
        alert_type: str,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        message: Optional[str] = None
    ) -> ScanAlert:
        """
        Helper method to create a standardized ScanAlert.
        
        Args:
            symbol: Stock ticker symbol
            alert_type: Type of alert (use AlertType constants)
            score: Confidence score (0.0-1.0)
            metadata: Additional context data
            timestamp: Alert timestamp (defaults to now)
            message: Human-readable alert description
            
        Returns:
            Properly formatted ScanAlert object
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if message is None:
            # Generate default message based on alert type
            message = f"{alert_type} alert for {symbol}"
        
        if score is None:
            # Default to medium confidence
            score = 0.5
            
        return ScanAlert(
            symbol=symbol,
            alert_type=alert_type,
            timestamp=timestamp,
            score=score,
            message=message,
            metadata=metadata or {},
            source_scanner=self.name
        )
    
    def convert_legacy_signals(self, signals: Dict[str, List[Dict[str, Any]]]) -> List[ScanAlert]:
        """
        Convert legacy dictionary-based signals to ScanAlert objects.
        
        This helper method assists in migrating existing scanners to the
        new uniform output format.
        
        Args:
            signals: Legacy format {symbol: [signal_dicts]}
            
        Returns:
            List of ScanAlert objects
        """
        alerts = []
        
        for symbol, symbol_signals in signals.items():
            for signal in symbol_signals:
                # Extract common fields
                score = signal.get('score')
                alert_type = signal.get('signal_type', signal.get('alert_type', 'Unknown'))
                
                # Build metadata from remaining fields
                metadata = {
                    k: v for k, v in signal.items() 
                    if k not in ['score', 'signal_type', 'alert_type', 'timestamp']
                }
                
                # Add reason to metadata if present
                if 'reason' in signal:
                    metadata['reason'] = signal['reason']
                
                # Create alert
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=alert_type,
                    score=score,
                    metadata=metadata,
                    timestamp=signal.get('timestamp')
                )
                
                alerts.append(alert)
        
        return alerts
    
    def filter_alerts_by_score(self, alerts: List[ScanAlert], min_score: float) -> List[ScanAlert]:
        """
        Filter alerts by minimum score threshold.
        
        Args:
            alerts: List of ScanAlert objects
            min_score: Minimum score threshold (0.0-1.0)
            
        Returns:
            Filtered list of alerts
        """
        return [
            alert for alert in alerts 
            if alert.score is not None and alert.score >= min_score
        ]
    
    def deduplicate_alerts(self, alerts: List[ScanAlert]) -> List[ScanAlert]:
        """
        Remove duplicate alerts for the same symbol and alert type.
        
        Keeps the alert with the highest score when duplicates are found.
        
        Args:
            alerts: List of ScanAlert objects
            
        Returns:
            Deduplicated list of alerts
        """
        # Group by (symbol, alert_type)
        alert_map = {}
        
        for alert in alerts:
            key = (alert.symbol, alert.alert_type)
            
            if key not in alert_map:
                alert_map[key] = alert
            else:
                # Keep the one with higher score
                existing = alert_map[key]
                if alert.score and (not existing.score or alert.score > existing.score):
                    alert_map[key] = alert
        
        return list(alert_map.values())
    
    async def publish_alerts_to_event_bus(
        self,
        alerts: List[ScanAlert],
        event_bus: Optional[IEventBus]
    ) -> None:
        """
        Publish scanner alerts to the event bus.
        
        Wraps ScanAlert objects in ScannerAlertEvent for proper event handling.
        
        Args:
            alerts: List of alerts to publish
            event_bus: Event bus instance (optional)
        """
        if not event_bus or not alerts:
            return
            
        for alert in alerts:
            # Create ScannerAlertEvent from ScanAlert
            event = ScannerAlertEvent(
                symbol=alert.symbol,
                alert_type=str(alert.alert_type),
                score=alert.score or 0.0,
                scanner_name=self.name,
                metadata=alert.metadata or {},
                timestamp=ensure_utc(alert.timestamp),
                event_type=EventType.SCANNER_ALERT
            )
            
            # Publish the event
            await event_bus.publish(event)
            
        self.logger.debug(f"Published {len(alerts)} alerts to event bus")