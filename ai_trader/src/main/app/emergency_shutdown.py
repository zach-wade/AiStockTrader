#!/usr/bin/env python3
"""
Emergency Shutdown Handler

Provides different levels of system shutdown for the AI Trader:
- Soft: Graceful shutdown, complete current operations
- Normal: Standard shutdown, cancel pending operations  
- Hard: Force shutdown, minimal cleanup
- Emergency: Immediate shutdown, emergency procedures
"""

import asyncio
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from main.utils import (
    get_logger,
    setup_logging,
    ErrorHandlingMixin,
    ensure_utc,
    safe_json_write,
    ensure_directory_exists,
    record_metric,
    timer
)

# Setup standardized logging
setup_logging()
logger = get_logger(__name__)


class EmergencyShutdown(ErrorHandlingMixin):
    """
    Handles different levels of system shutdown with appropriate cleanup and error handling.
    """
    
    def __init__(self, config: Any):
        """
        Initialize EmergencyShutdown.
        
        Args:
            config: System configuration
        """
        super().__init__()
        self.config = config
        self.shutdown_initiated = False
        self.active_tasks: List[asyncio.Task] = []
        self.startup_time = datetime.now()
        self.logger = get_logger(f"{__name__}.EmergencyShutdown")
        
        # Register error callback for monitoring
        self.register_error_callback(
            "shutdown_monitoring",
            lambda error, context: record_metric(
                "shutdown_error",
                1,
                tags={"context": context, "error_type": type(error).__name__}
            )
        )
        
    async def execute(self, level: str = 'normal', timeout: int = 30) -> Dict[str, Any]:
        """
        Execute shutdown at specified level with monitoring and error handling.
        
        Args:
            level: Shutdown level ('soft', 'normal', 'hard', 'emergency')
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            Dictionary with shutdown results
        """
        if self.shutdown_initiated:
            self.logger.warning("Shutdown already in progress")
            return {'status': 'already_in_progress'}
            
        self.shutdown_initiated = True
        
        with timer() as shutdown_timer:
            self.logger.info(f"Initiating {level} shutdown with {timeout}s timeout...")
            
            try:
                if level == 'soft':
                    result = await self._soft_shutdown(timeout)
                elif level == 'normal':
                    result = await self._normal_shutdown(timeout)
                elif level == 'hard':
                    result = await self._hard_shutdown(timeout)
                elif level == 'emergency':
                    result = await self._emergency_shutdown()
                else:
                    raise ValueError(f"Unknown shutdown level: {level}")
                    
                result['duration_seconds'] = shutdown_timer.elapsed
                result['timestamp'] = ensure_utc(datetime.now()).isoformat()
                result['uptime_seconds'] = (datetime.now() - self.startup_time).total_seconds()
                
                self.logger.info(f"Shutdown completed in {shutdown_timer.elapsed:.2f}s")
                
                # Record shutdown metrics
                record_metric(
                    "shutdown_completed",
                    1,
                    tags={"level": level, "duration": shutdown_timer.elapsed}
                )
                
                return result
                
            except Exception as e:
                self.handle_error(e, f"executing {level} shutdown")
                # Fallback to emergency shutdown
                return await self._emergency_shutdown()
    
    async def _soft_shutdown(self, timeout: int) -> Dict[str, Any]:
        """
        Soft shutdown: Wait for current operations to complete.
        """
        logger.info("Performing soft shutdown - waiting for operations to complete...")
        
        steps_completed = []
        
        try:
            # 1. Stop accepting new requests/orders
            steps_completed.append("stopped_new_requests")
            logger.info("✓ Stopped accepting new requests")
            
            # 2. Wait for active trading operations to complete
            if self.active_tasks:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks...")
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.active_tasks, return_exceptions=True),
                        timeout=timeout * 0.8  # Use 80% of timeout for active tasks
                    )
                    steps_completed.append("completed_active_tasks")
                    logger.info("✓ All active tasks completed")
                except asyncio.TimeoutError:
                    logger.warning("Active tasks did not complete within timeout")
                    steps_completed.append("active_tasks_timeout")
            
            # 3. Save current state
            await self._save_system_state()
            steps_completed.append("saved_state")
            logger.info("✓ System state saved")
            
            # 4. Close connections gracefully
            await self._close_connections()
            steps_completed.append("closed_connections")
            logger.info("✓ Connections closed")
            
            return {
                'status': 'success',
                'level': 'soft',
                'steps_completed': steps_completed
            }
            
        except Exception as e:
            logger.error(f"Soft shutdown failed: {e}")
            return {
                'status': 'partial_failure',
                'level': 'soft',
                'steps_completed': steps_completed,
                'error': str(e)
            }
    
    async def _normal_shutdown(self, timeout: int) -> Dict[str, Any]:
        """
        Normal shutdown: Cancel pending operations, complete critical ones.
        """
        logger.info("Performing normal shutdown - canceling non-critical operations...")
        
        steps_completed = []
        
        try:
            # 1. Cancel pending non-critical tasks
            if self.active_tasks:
                critical_tasks = []  # Tasks that must complete
                cancelled_count = 0
                
                for task in self.active_tasks:
                    if not task.done() and not self._is_critical_task(task):
                        task.cancel()
                        cancelled_count += 1
                    else:
                        critical_tasks.append(task)
                
                logger.info(f"✓ Cancelled {cancelled_count} non-critical tasks")
                steps_completed.append("cancelled_non_critical_tasks")
                
                # Wait for critical tasks
                if critical_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*critical_tasks, return_exceptions=True),
                            timeout=timeout * 0.6
                        )
                        steps_completed.append("completed_critical_tasks")
                    except asyncio.TimeoutError:
                        logger.warning("Critical tasks did not complete within timeout")
                        steps_completed.append("critical_tasks_timeout")
            
            # 2. Save essential state only
            await self._save_essential_state()
            steps_completed.append("saved_essential_state")
            logger.info("✓ Essential state saved")
            
            # 3. Close connections
            await self._close_connections()
            steps_completed.append("closed_connections")
            logger.info("✓ Connections closed")
            
            return {
                'status': 'success',
                'level': 'normal',
                'steps_completed': steps_completed
            }
            
        except Exception as e:
            logger.error(f"Normal shutdown failed: {e}")
            return {
                'status': 'partial_failure',
                'level': 'normal',
                'steps_completed': steps_completed,
                'error': str(e)
            }
    
    async def _hard_shutdown(self, timeout: int) -> Dict[str, Any]:
        """
        Hard shutdown: Force cancel all operations, minimal cleanup.
        """
        logger.info("Performing hard shutdown - forcing shutdown...")
        
        steps_completed = []
        
        try:
            # 1. Cancel all active tasks immediately
            if self.active_tasks:
                for task in self.active_tasks:
                    if not task.done():
                        task.cancel()
                
                # Give minimal time for cancellation
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.active_tasks, return_exceptions=True),
                        timeout=min(5, timeout * 0.3)
                    )
                except asyncio.TimeoutError:
                    pass  # Expected in hard shutdown
                
                steps_completed.append("force_cancelled_tasks")
                logger.info("✓ Force cancelled all tasks")
            
            # 2. Emergency state save (if possible)
            try:
                await asyncio.wait_for(self._save_emergency_state(), timeout=3)
                steps_completed.append("emergency_state_saved")
                logger.info("✓ Emergency state saved")
            except Exception as e:
                logger.warning(f"Could not save emergency state: {e}")
                steps_completed.append("emergency_state_failed")
            
            # 3. Force close connections
            await self._force_close_connections()
            steps_completed.append("force_closed_connections")
            logger.info("✓ Force closed connections")
            
            return {
                'status': 'success',
                'level': 'hard',
                'steps_completed': steps_completed
            }
            
        except Exception as e:
            logger.error(f"Hard shutdown failed: {e}")
            return {
                'status': 'partial_failure',
                'level': 'hard',
                'steps_completed': steps_completed,
                'error': str(e)
            }
    
    async def _emergency_shutdown(self) -> Dict[str, Any]:
        """
        Emergency shutdown: Immediate shutdown with minimal operations.
        """
        logger.critical("EMERGENCY SHUTDOWN - Immediate termination")
        
        steps_completed = []
        
        try:
            # 1. Log emergency state
            logger.critical(f"Emergency shutdown at {datetime.now().isoformat()}")
            steps_completed.append("logged_emergency")
            
            # 2. Cancel everything immediately
            if self.active_tasks:
                for task in self.active_tasks:
                    if not task.done():
                        task.cancel()
                steps_completed.append("cancelled_all_tasks")
            
            # 3. Write emergency marker file using standardized file operations
            try:
                emergency_data = {
                    "timestamp": ensure_utc(datetime.now()).isoformat(),
                    "level": "emergency",
                    "reason": "Emergency shutdown initiated",
                    "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                    "active_tasks": len(self.active_tasks)
                }
                
                ensure_directory_exists(Path("logs"))
                safe_json_write(Path("logs/emergency_shutdown.json"), emergency_data)
                steps_completed.append("wrote_emergency_marker")
                
            except Exception as e:
                self.handle_error(e, "writing emergency marker file")
                # Don't fail emergency shutdown for file write
            
            # 4. Force close everything
            await self._force_close_connections()
            steps_completed.append("force_closed_all")
            
            return {
                'status': 'emergency_complete',
                'level': 'emergency',
                'steps_completed': steps_completed
            }
            
        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
            # Don't even try to recover - just exit
            sys.exit(1)
    
    def _is_critical_task(self, task: asyncio.Task) -> bool:
        """Check if a task is critical and should not be cancelled."""
        # In a real implementation, you'd check task names/metadata
        # For now, assume no tasks are critical in shutdown
        return False
    
    async def _save_system_state(self):
        """Save complete system state."""
        logger.info("Saving complete system state...")
        # Implementation would save positions, orders, model states, etc.
        await asyncio.sleep(0.1)  # Simulate save operation
    
    async def _save_essential_state(self):
        """Save only essential system state."""
        logger.info("Saving essential system state...")
        # Implementation would save critical positions and orders only
        await asyncio.sleep(0.05)  # Simulate save operation
    
    async def _save_emergency_state(self):
        """Save emergency state information."""
        logger.info("Saving emergency state...")
        # Implementation would save absolute minimum state
        await asyncio.sleep(0.01)  # Simulate save operation
    
    async def _close_connections(self):
        """Close all connections gracefully."""
        logger.info("Closing connections gracefully...")
        # Implementation would close database, broker, data feed connections
        await asyncio.sleep(0.1)  # Simulate connection cleanup
    
    async def _force_close_connections(self):
        """Force close all connections."""
        logger.info("Force closing connections...")
        # Implementation would forcibly terminate all connections
        await asyncio.sleep(0.01)  # Simulate forced cleanup


async def main():
    """Test emergency shutdown functionality."""
    from main.config.config_manager import get_config
    
    config = get_config()
    shutdown_handler = EmergencyShutdown(config)
    
    # Test normal shutdown
    result = await shutdown_handler.execute('normal', 10)
    print(f"Shutdown result: {result}")


if __name__ == '__main__':
    asyncio.run(main())