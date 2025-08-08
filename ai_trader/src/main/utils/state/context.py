"""
State Context Managers

Context managers for state management operations.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Dict


class StateContext:
    """Context managers for state operations."""
    
    def __init__(self, state_manager):
        """Initialize context manager."""
        self.state_manager = state_manager
        self._locks: Dict[str, asyncio.Lock] = {}
    
    @asynccontextmanager
    async def lock(self, resource: str, timeout: float = 10.0) -> AsyncContextManager[None]:
        """
        Acquire distributed lock for resource.
        
        Args:
            resource: Resource to lock
            timeout: Lock timeout in seconds
        """
        if resource not in self._locks:
            self._locks[resource] = asyncio.Lock()
        
        lock = self._locks[resource]
        
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"Failed to acquire lock for {resource} within {timeout}s")
        finally:
            if lock.locked():
                lock.release()
    
    @asynccontextmanager
    async def transaction(self, namespace: str = "default"):
        """
        Transaction context for atomic state operations.
        
        Args:
            namespace: Namespace for transaction
        """
        # This is a simplified transaction - in practice you'd implement
        # proper rollback capabilities
        checkpoint_id = None
        
        try:
            # Create checkpoint before transaction
            checkpoint_id = await self.state_manager.checkpoint(namespace)
            yield
            
        except Exception as e:
            # Rollback on error
            if checkpoint_id:
                await self.state_manager.restore(checkpoint_id, namespace)
            raise
        
        finally:
            # Cleanup checkpoint on success
            if checkpoint_id:
                await self.state_manager.delete_checkpoint(checkpoint_id)
    
    @asynccontextmanager
    async def batch_operations(self, namespace: str = "default"):
        """
        Batch operations context for performance optimization.
        
        Args:
            namespace: Namespace for batch operations
        """
        # This would implement batching for performance
        # For now, just yield
        yield
    
    def get_active_locks(self) -> Dict[str, bool]:
        """Get status of active locks."""
        return {
            resource: lock.locked()
            for resource, lock in self._locks.items()
        }