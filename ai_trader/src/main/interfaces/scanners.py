"""
Scanner interfaces for the trading system.

This module defines the protocol interfaces for all scanner components,
ensuring consistency, testability, and proper integration with the
hot/cold storage architecture.
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from datetime import datetime
from dataclasses import dataclass

from main.events.types import ScanAlert
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.interfaces.events import IEventBus


@runtime_checkable
class IScanner(Protocol):
    """
    Base protocol for all scanner implementations.
    
    Scanners identify trading opportunities by analyzing market data,
    news, technical indicators, and other signals.
    """
    
    @property
    def name(self) -> str:
        """Scanner name for identification and logging."""
        ...
    
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
        ...
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize scanner resources and connections."""
        ...
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        ...


@runtime_checkable
class IScannerRepository(Protocol):
    """
    Protocol for scanner data access layer.
    
    Provides abstraction over data storage, automatically routing
    queries to hot or cold storage based on the StorageRouter logic.
    """
    
    @abstractmethod
    async def get_market_data(
        self,
        symbols: List[str],
        query_filter: QueryFilter,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve market data for scanning.
        
        Args:
            symbols: List of symbols to retrieve data for
            query_filter: Time range and other filters
            columns: Specific columns to retrieve (None for all)
            
        Returns:
            Dictionary with symbol as key and data as value
        """
        ...
    
    @abstractmethod
    async def get_technical_indicators(
        self,
        symbols: List[str],
        indicators: List[str],
        query_filter: QueryFilter
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve pre-calculated technical indicators.
        
        Args:
            symbols: List of symbols
            indicators: List of indicator names
            query_filter: Time range and filters
            
        Returns:
            Nested dict: {symbol: {indicator: values}}
        """
        ...
    
    @abstractmethod
    async def get_news_sentiment(
        self,
        symbols: List[str],
        query_filter: QueryFilter
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve news and sentiment data.
        
        Args:
            symbols: List of symbols
            query_filter: Time range and filters
            
        Returns:
            Dictionary with symbol as key and news items as value
        """
        ...
    
    @abstractmethod
    async def get_volume_statistics(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve volume statistics for relative volume calculations.
        
        Args:
            symbols: List of symbols
            lookback_days: Number of days for average calculation
            
        Returns:
            Dict with symbol as key and stats (avg_volume, std_volume) as value
        """
        ...
    
    @abstractmethod
    async def get_social_sentiment(
        self,
        symbols: List[str],
        query_filter: QueryFilter
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve social sentiment data.
        
        Args:
            symbols: List of symbols
            query_filter: Time range and filters
            
        Returns:
            Dictionary with symbol as key and social posts as value
        """
        ...


@dataclass
class ScannerConfig:
    """Configuration for scanner instances."""
    name: str
    enabled: bool = True
    priority: int = 5
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    batch_size: int = 100
    min_confidence: float = 0.5
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@runtime_checkable
class IScannerOrchestrator(Protocol):
    """
    Protocol for scanner orchestration and coordination.
    
    Manages multiple scanners, handles parallelization,
    and coordinates scan execution.
    """
    
    @abstractmethod
    async def register_scanner(
        self,
        scanner: IScanner,
        config: ScannerConfig
    ) -> None:
        """Register a scanner with the orchestrator."""
        ...
    
    @abstractmethod
    async def unregister_scanner(self, scanner_name: str) -> None:
        """Unregister a scanner by name."""
        ...
    
    @abstractmethod
    async def scan_universe(
        self,
        universe: List[str],
        scanner_names: Optional[List[str]] = None
    ) -> List[ScanAlert]:
        """
        Scan a universe of symbols using registered scanners.
        
        Args:
            universe: List of symbols to scan
            scanner_names: Specific scanners to use (None for all)
            
        Returns:
            Aggregated list of scan alerts
        """
        ...
    
    @abstractmethod
    async def get_scanner_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registered scanners."""
        ...


@runtime_checkable
class IScannerAdapter(Protocol):
    """
    Protocol for scanner system adapter.
    
    Bridges scanners with the trading engine, converting
    scan alerts into actionable trading signals.
    """
    
    @abstractmethod
    def register_scanner(self, scanner: IScanner, **kwargs) -> None:
        """Register a scanner with the adapter."""
        ...
    
    @abstractmethod
    async def start(self, universe: List[str]) -> None:
        """Start the scanner adapter with initial universe."""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the scanner adapter."""
        ...
    
    @abstractmethod
    async def scan_once(
        self,
        symbols: Optional[List[str]] = None
    ) -> List[Any]:  # Returns trading signals
        """Perform a single scan and return signals."""
        ...
    
    @abstractmethod
    def update_universe(self, universe: List[str]) -> None:
        """Update the scanning universe."""
        ...
    
    @abstractmethod
    def get_alert_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about buffered alerts."""
        ...


@runtime_checkable
class IScannerMetrics(Protocol):
    """Protocol for scanner performance metrics."""
    
    @abstractmethod
    def record_scan_duration(
        self,
        scanner_name: str,
        duration_ms: float,
        symbol_count: int
    ) -> None:
        """Record scan execution time."""
        ...
    
    @abstractmethod
    def record_alert_generated(
        self,
        scanner_name: str,
        alert_type: str,
        symbol: str,
        confidence: float
    ) -> None:
        """Record alert generation."""
        ...
    
    @abstractmethod
    def record_scan_error(
        self,
        scanner_name: str,
        error_type: str,
        error_message: str
    ) -> None:
        """Record scan error."""
        ...
    
    @abstractmethod
    def get_metrics_summary(
        self,
        scanner_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics summary for scanner(s)."""
        ...


@runtime_checkable
class IScannerCache(Protocol):
    """Protocol for scanner result caching."""
    
    @abstractmethod
    async def get_cached_result(
        self,
        scanner_name: str,
        symbol: str,
        cache_key: str
    ) -> Optional[Any]:
        """Retrieve cached scanner result."""
        ...
    
    @abstractmethod
    async def cache_result(
        self,
        scanner_name: str,
        symbol: str,
        cache_key: str,
        result: Any,
        ttl_seconds: int = 300
    ) -> None:
        """Cache scanner result."""
        ...
    
    @abstractmethod
    async def invalidate_cache(
        self,
        scanner_name: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> None:
        """Invalidate cached results."""
        ...