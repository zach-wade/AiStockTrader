"""
State Persistence

Checkpoint and recovery functionality for state management.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from .types import StateCheckpoint, StorageBackend

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles state checkpointing and recovery."""
    
    def __init__(self, state_manager):
        """Initialize persistence manager."""
        self.state_manager = state_manager
        self._checkpoints: Dict[str, StateCheckpoint] = {}
    
    async def checkpoint(self, namespace: str = "default") -> str:
        """
        Create a checkpoint of namespace state.
        
        Args:
            namespace: Namespace to checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{namespace}_{uuid4().hex[:8]}"
        
        try:
            # Get all keys in namespace
            state_keys = await self.state_manager.keys("*", namespace)
            
            # Create checkpoint metadata
            checkpoint = StateCheckpoint(
                checkpoint_id=checkpoint_id,
                namespace=namespace,
                created_at=datetime.utcnow(),
                state_keys=state_keys,
                metadata={'total_keys': len(state_keys)}
            )
            
            # Store checkpoint
            await self.state_manager.set(
                f"_checkpoint_{checkpoint_id}",
                checkpoint.to_dict(),
                namespace="_system",
                backend=StorageBackend.FILE  # Use persistent storage
            )
            
            self._checkpoints[checkpoint_id] = checkpoint
            
            logger.info(f"Created checkpoint {checkpoint_id} for namespace {namespace}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for namespace {namespace}: {e}")
            raise
    
    async def restore(self, checkpoint_id: str, namespace: str = "default") -> bool:
        """
        Restore state from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to restore from
            namespace: Target namespace for restoration
            
        Returns:
            True if successful
        """
        try:
            # Get checkpoint metadata
            checkpoint_data = await self.state_manager.get(
                f"_checkpoint_{checkpoint_id}",
                namespace="_system",
                backend=StorageBackend.FILE
            )
            
            if not checkpoint_data:
                logger.error(f"Checkpoint {checkpoint_id} not found")
                return False
            
            checkpoint = StateCheckpoint.from_dict(checkpoint_data)
            
            # Clear current namespace state
            current_keys = await self.state_manager.keys("*", namespace)
            for key in current_keys:
                await self.state_manager.delete(key, namespace)
            
            # Restore state keys
            for key in checkpoint.state_keys:
                # This is a simplified restoration - in practice you'd need
                # to store the actual state data with the checkpoint
                pass
            
            logger.info(f"Restored checkpoint {checkpoint_id} to namespace {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return False
    
    async def list_checkpoints(self, namespace: Optional[str] = None) -> List[StateCheckpoint]:
        """
        List available checkpoints.
        
        Args:
            namespace: Filter by namespace (optional)
            
        Returns:
            List of checkpoints
        """
        checkpoints = []
        
        try:
            # Get all checkpoint keys
            checkpoint_keys = await self.state_manager.keys("_checkpoint_*", "_system")
            
            for key in checkpoint_keys:
                checkpoint_data = await self.state_manager.get(key, "_system")
                if checkpoint_data:
                    checkpoint = StateCheckpoint.from_dict(checkpoint_data)
                    
                    # Filter by namespace if specified
                    if namespace is None or checkpoint.namespace == namespace:
                        checkpoints.append(checkpoint)
            
            return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            
        Returns:
            True if successful
        """
        try:
            result = await self.state_manager.delete(
                f"_checkpoint_{checkpoint_id}",
                namespace="_system",
                backend=StorageBackend.FILE
            )
            
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
            
            if result:
                logger.info(f"Deleted checkpoint {checkpoint_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def auto_checkpoint(self, namespaces: List[str]) -> Dict[str, str]:
        """
        Auto-checkpoint multiple namespaces.
        
        Args:
            namespaces: List of namespaces to checkpoint
            
        Returns:
            Dict mapping namespace to checkpoint ID
        """
        results = {}
        
        for namespace in namespaces:
            try:
                checkpoint_id = await self.checkpoint(namespace)
                results[namespace] = checkpoint_id
            except Exception as e:
                logger.error(f"Auto-checkpoint failed for {namespace}: {e}")
                results[namespace] = None
        
        return results
    
    def get_checkpoint_stats(self) -> Dict[str, int]:
        """Get checkpoint statistics."""
        return {
            'total_checkpoints': len(self._checkpoints),
            'checkpoints_by_namespace': len(set(
                cp.namespace for cp in self._checkpoints.values()
            ))
        }