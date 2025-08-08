"""
Base Manager Class

Provides common functionality for all management components that coordinate
multiple processors, services, or operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone
from asyncio import Lock

from main.utils.core import get_logger, ensure_utc, process_in_batches, gather_with_exceptions, ensure_directory_exists
from main.utils.monitoring import MetricsCollector
from ..exceptions import DataPipelineError, convert_exception
from ..enums import DataLayer


class BaseManager(ABC):
    """
    Abstract base class for all manager components.
    
    Provides common functionality including:
    - Resource management and coordination
    - State tracking and monitoring
    - Layer-aware operations
    - Concurrent operation management
    - Health monitoring
    """
    
    def __init__(
        self,
        manager_name: str,
        config: Optional[Dict[str, Any]] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.manager_name = manager_name
        self.config = config or {}
        self.metrics_collector = metrics_collector
        self.logger = get_logger(f"data_pipeline.{manager_name}")
        
        # State management
        self._is_running = False
        self._active_operations = set()
        self._operation_lock = Lock()
        
        # Health and statistics
        self._health_status = "healthy"
        self._operations_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'active_operations': 0,
            'start_time': None,
            'last_operation': None
        }
        
        self.logger.debug(f"Initialized {manager_name} manager")
    
    async def start(self) -> None:
        """Start the manager and initialize resources."""
        if self._is_running:
            self.logger.warning(f"Manager {self.manager_name} is already running")
            return
        
        try:
            await self._start_manager()
            self._is_running = True
            self._operations_stats['start_time'] = ensure_utc(datetime.now(timezone.utc))
            self._health_status = "healthy"
            
            self.logger.info(f"Manager {self.manager_name} started successfully")
        except Exception as e:
            error = convert_exception(e, f"Failed to start manager {self.manager_name}")
            self._health_status = "unhealthy"
            self.logger.error(f"Manager start failed: {error}")
            raise error
    
    @abstractmethod
    async def _start_manager(self) -> None:
        """Manager-specific startup logic. Override in subclasses."""
        pass
    
    async def stop(self) -> None:
        """Stop the manager and cleanup resources."""
        if not self._is_running:
            self.logger.warning(f"Manager {self.manager_name} is not running")
            return
        
        try:
            # Wait for active operations to complete
            await self._wait_for_operations()
            
            # Stop manager-specific resources
            await self._stop_manager()
            
            self._is_running = False
            self._health_status = "stopped"
            
            self.logger.info(f"Manager {self.manager_name} stopped successfully")
        except Exception as e:
            error = convert_exception(e, f"Failed to stop manager {self.manager_name}")
            self._health_status = "error"
            self.logger.error(f"Manager stop failed: {error}")
            raise error
    
    @abstractmethod
    async def _stop_manager(self) -> None:
        """Manager-specific shutdown logic. Override in subclasses."""
        pass
    
    async def _wait_for_operations(self, timeout_seconds: int = 60) -> None:
        """Wait for active operations to complete."""
        import asyncio
        
        if not self._active_operations:
            return
        
        self.logger.info(f"Waiting for {len(self._active_operations)} active operations to complete")
        
        start_time = ensure_utc(datetime.now(timezone.utc))
        while self._active_operations and (ensure_utc(datetime.now(timezone.utc)) - start_time).seconds < timeout_seconds:
            await asyncio.sleep(1)
        
        if self._active_operations:
            self.logger.warning(f"Timed out waiting for operations: {self._active_operations}")
    
    async def execute_operation(
        self,
        operation_id: str,
        operation_func,
        *args,
        layer: Optional[DataLayer] = None,
        **kwargs
    ) -> Any:
        """
        Execute an operation with tracking and monitoring.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_func: Function to execute
            *args: Positional arguments for the function
            layer: Data layer for layer-aware operations
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
        """
        if not self._is_running:
            raise DataPipelineError(
                f"Manager {self.manager_name} is not running",
                component="manager",
                context={'operation_id': operation_id}
            )
        
        async with self._operation_lock:
            self._active_operations.add(operation_id)
            self._operations_stats['active_operations'] = len(self._active_operations)
        
        self.logger.debug(f"Starting operation: {operation_id} (layer: {layer})")
        
        try:
            # Pre-operation validation
            await self._validate_operation(operation_id, layer)
            
            # Execute the operation
            result = await operation_func(*args, **kwargs)
            
            # Update statistics
            self._update_operation_stats(success=True)
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_operation_success(
                    manager=self.manager_name,
                    operation=operation_id,
                    layer=layer.value if layer else None
                )
            
            self.logger.debug(f"Operation completed successfully: {operation_id}")
            return result
            
        except Exception as e:
            self._update_operation_stats(success=False)
            
            # Record metrics for failure
            if self.metrics_collector:
                self.metrics_collector.record_operation_failure(
                    manager=self.manager_name,
                    operation=operation_id,
                    error=str(e),
                    layer=layer.value if layer else None
                )
            
            error = convert_exception(
                e, 
                f"Operation failed in manager {self.manager_name}",
                component="manager"
            )
            self.logger.error(f"Operation failed: {operation_id} - {error}")
            raise error
            
        finally:
            async with self._operation_lock:
                self._active_operations.discard(operation_id)
                self._operations_stats['active_operations'] = len(self._active_operations)
    
    async def _validate_operation(self, operation_id: str, layer: Optional[DataLayer]) -> None:
        """Validate operation before execution. Override in subclasses."""
        pass
    
    def _update_operation_stats(self, success: bool) -> None:
        """Update operation statistics."""
        self._operations_stats['total_operations'] += 1
        if success:
            self._operations_stats['successful_operations'] += 1
        else:
            self._operations_stats['failed_operations'] += 1
        self._operations_stats['last_operation'] = ensure_utc(datetime.now(timezone.utc))
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get manager health status."""
        stats = self._operations_stats.copy()
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
        else:
            stats['success_rate'] = 1.0
        
        return {
            'manager_name': self.manager_name,
            'health_status': self._health_status,
            'is_running': self._is_running,
            'active_operations': list(self._active_operations),
            'statistics': stats
        }
    
    def is_healthy(self) -> bool:
        """Check if manager is healthy."""
        return self._health_status == "healthy" and self._is_running
    
    def get_layer_config(self, layer: DataLayer) -> Dict[str, Any]:
        """Get configuration for a specific layer."""
        layer_configs = self.config.get('layers', {})
        return layer_configs.get(str(layer.value), {})
    
    def is_layer_supported(self, layer: DataLayer) -> bool:
        """Check if a layer is supported by this manager."""
        supported_layers = self.config.get('supported_layers', list(range(4)))  # Default: all layers
        return layer.value in supported_layers
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.manager_name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.manager_name}, running={self._is_running}, health={self._health_status})"
    
    async def execute_batch_operations(
        self,
        operations: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        layer: Optional[DataLayer] = None
    ) -> List[Any]:
        """
        Execute multiple operations in parallel with concurrency control.
        
        Args:
            operations: List of operation definitions with 'id', 'func', 'args', 'kwargs'
            max_concurrent: Maximum concurrent operations (defaults to config)
            layer: Data layer for all operations
            
        Returns:
            List of results in the same order as operations
        """
        if not operations:
            return []
        
        max_concurrent = max_concurrent or self.config.get('max_concurrent_operations', 5)
        
        self.logger.info(
            f"Executing batch of {len(operations)} operations "
            f"(max concurrent: {max_concurrent})"
        )
        
        # Define function to execute single operation
        async def execute_one(op_def: Dict[str, Any]) -> Any:
            return await self.execute_operation(
                operation_id=op_def['id'],
                operation_func=op_def['func'],
                *op_def.get('args', []),
                layer=layer,
                **op_def.get('kwargs', {})
            )
        
        # Process operations in batches
        results = await process_in_batches(
            operations,
            execute_one,
            max_concurrent=max_concurrent
        )
        
        return results
    
    async def execute_operations_with_exceptions(
        self,
        operations: List[Dict[str, Any]],
        layer: Optional[DataLayer] = None
    ) -> Dict[str, Any]:
        """
        Execute multiple operations and gather results with exception handling.
        
        Args:
            operations: List of operation definitions
            layer: Data layer for all operations
            
        Returns:
            Dictionary with 'results' and 'errors'
        """
        tasks = []
        for op_def in operations:
            task = self.execute_operation(
                operation_id=op_def['id'],
                operation_func=op_def['func'],
                *op_def.get('args', []),
                layer=layer,
                **op_def.get('kwargs', {})
            )
            tasks.append(task)
        
        results = await gather_with_exceptions(*tasks)
        
        # Separate successful results from errors
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    'operation_id': operations[i]['id'],
                    'error': str(result)
                })
            else:
                successful_results.append({
                    'operation_id': operations[i]['id'],
                    'result': result
                })
        
        return {
            'results': successful_results,
            'errors': errors,
            'total': len(operations),
            'successful': len(successful_results),
            'failed': len(errors)
        }
    
    async def ensure_working_directory(self, path: str) -> None:
        """Ensure working directory exists for manager operations."""
        await ensure_directory_exists(path)
        self.logger.debug(f"Ensured working directory exists: {path}")