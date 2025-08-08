"""
Dashboard interfaces for the monitoring system.

This module defines the protocols for dashboards and dashboard management,
ensuring consistent behavior across different dashboard implementations.
"""

from typing import Protocol, Dict, Any, Optional, List, runtime_checkable
from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass


class DashboardStatus(Enum):
    """Dashboard status states."""
    INITIALIZED = "initialized"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class DashboardConfig:
    """Configuration for a dashboard instance."""
    name: str
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    startup_mode: str = "process"  # process, thread, async
    auto_restart: bool = True
    health_check_interval: int = 30  # seconds
    extra_config: Optional[Dict[str, Any]] = None


class IDashboard(Protocol):
    """
    Interface for all dashboard implementations.
    
    This protocol defines the contract that all dashboards must implement,
    ensuring consistent lifecycle management and monitoring capabilities.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the dashboard components.
        
        This method should prepare the dashboard for starting but not
        actually start serving requests. Use for setup tasks like:
        - Creating the Dash/Flask app instance
        - Setting up routes and callbacks
        - Initializing data connections
        
        Raises:
            Exception: If initialization fails
        """
        ...
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the dashboard server.
        
        This method should start the dashboard in a non-blocking manner,
        typically by launching it in a separate process or thread.
        
        Raises:
            Exception: If starting fails
        """
        ...
    
    @abstractmethod
    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the dashboard server gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
            
        Raises:
            Exception: If stopping fails
        """
        ...
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current dashboard status and health information.
        
        Returns:
            Dictionary containing:
            - state: Current DashboardStatus
            - health: Health check results
            - metrics: Performance metrics
            - config: Current configuration
            - error: Any error information (if applicable)
        """
        ...
    
    @abstractmethod
    def get_url(self) -> str:
        """
        Get the dashboard URL.
        
        Returns:
            The URL where the dashboard can be accessed
        """
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a health check on the dashboard.
        
        Returns:
            True if healthy, False otherwise
        """
        ...


class IDashboardManager(Protocol):
    """
    Interface for dashboard lifecycle management.
    
    The dashboard manager is responsible for managing multiple dashboard
    instances, handling their lifecycle, monitoring health, and providing
    centralized control.
    """
    
    @abstractmethod
    async def register_dashboard(
        self, 
        name: str, 
        dashboard: IDashboard,
        config: Optional[DashboardConfig] = None
    ) -> None:
        """
        Register a dashboard instance with the manager.
        
        Args:
            name: Unique identifier for the dashboard
            dashboard: The dashboard instance
            config: Optional configuration override
            
        Raises:
            ValueError: If a dashboard with the same name already exists
            Exception: If registration fails
        """
        ...
    
    @abstractmethod
    async def unregister_dashboard(self, name: str) -> None:
        """
        Unregister a dashboard from the manager.
        
        Args:
            name: The dashboard identifier
            
        Raises:
            KeyError: If dashboard not found
            Exception: If unregistration fails
        """
        ...
    
    @abstractmethod
    async def start_dashboard(self, name: str) -> None:
        """
        Start a specific dashboard.
        
        Args:
            name: The dashboard identifier
            
        Raises:
            KeyError: If dashboard not found
            Exception: If starting fails
        """
        ...
    
    @abstractmethod
    async def stop_dashboard(self, name: str, timeout: float = 30.0) -> None:
        """
        Stop a specific dashboard.
        
        Args:
            name: The dashboard identifier
            timeout: Maximum time to wait for graceful shutdown
            
        Raises:
            KeyError: If dashboard not found
            Exception: If stopping fails
        """
        ...
    
    @abstractmethod
    async def restart_dashboard(self, name: str) -> None:
        """
        Restart a specific dashboard.
        
        Args:
            name: The dashboard identifier
            
        Raises:
            KeyError: If dashboard not found
            Exception: If restart fails
        """
        ...
    
    @abstractmethod
    async def start_all(self) -> None:
        """
        Start all registered dashboards.
        
        Raises:
            Exception: If any dashboard fails to start
        """
        ...
    
    @abstractmethod
    async def stop_all(self, timeout: float = 30.0) -> None:
        """
        Stop all registered dashboards.
        
        Args:
            timeout: Maximum time to wait for each dashboard to stop
            
        Raises:
            Exception: If any dashboard fails to stop
        """
        ...
    
    @abstractmethod
    def get_dashboard_status(self, name: str) -> Dict[str, Any]:
        """
        Get the status of a specific dashboard.
        
        Args:
            name: The dashboard identifier
            
        Returns:
            Dashboard status information
            
        Raises:
            KeyError: If dashboard not found
        """
        ...
    
    @abstractmethod
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all registered dashboards.
        
        Returns:
            Dictionary mapping dashboard names to their status
        """
        ...
    
    @abstractmethod
    def list_dashboards(self) -> List[str]:
        """
        List all registered dashboard names.
        
        Returns:
            List of dashboard identifiers
        """
        ...
    
    @abstractmethod
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all dashboards.
        
        Returns:
            Dictionary mapping dashboard names to health status
        """
        ...
    
    @abstractmethod
    async def handle_failed_dashboard(self, name: str) -> None:
        """
        Handle a dashboard that has failed health checks.
        
        This method should implement the failure handling strategy,
        such as restarting the dashboard or alerting operators.
        
        Args:
            name: The dashboard identifier
        """
        ...


@runtime_checkable
class IMetricsCollector(Protocol):
    """Interface for metrics collectors."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the metrics collector."""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the metrics collector."""
        ...
    
    @abstractmethod
    def record_operation_start(self, operation_type: str, data_type: str, symbol: str) -> str:
        """Record the start of an operation."""
        ...
    
    @abstractmethod
    def record_operation_end(
        self,
        operation_id: str,
        operation_type: str,
        data_type: str,
        symbol: str,
        start_time: float,
        records_count: int,
        bytes_size: int,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record the end of an operation."""
        ...


@runtime_checkable
class IArchiveMetricsCollector(IMetricsCollector, Protocol):
    """Interface for archive-specific metrics collectors."""
    
    @abstractmethod
    async def get_storage_metrics(self) -> Any:
        """Get current storage metrics."""
        ...
    
    @abstractmethod
    def get_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get operation statistics."""
        ...
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics."""
        ...
    
    @abstractmethod
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """Get conditions that should trigger alerts."""
        ...