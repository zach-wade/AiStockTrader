"""
Dashboard Manager for V2 dashboards - Handles lifecycle management.

This module provides a manager that starts dashboards in separate processes
to avoid blocking issues and provides centralized control.
"""

import asyncio
import subprocess
import logging
import signal
import time
import json
import os
import sys
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DashboardState(Enum):
    """Dashboard states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class DashboardInfo:
    """Information about a dashboard instance."""
    name: str
    port: int
    process: Optional[subprocess.Popen] = None
    state: DashboardState = DashboardState.STOPPED
    pid: Optional[int] = None
    start_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None


class DashboardManager:
    """
    Manages the lifecycle of V2 dashboards.
    
    Features:
    - Starts dashboards in separate processes using subprocess
    - Monitors dashboard health
    - Handles graceful shutdown
    - Provides status information
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize dashboard manager.
        
        Args:
            db_config: Database configuration dict with host, port, database, user, password
        """
        self.db_config = db_config
        self.dashboards: Dict[str, DashboardInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Find the dashboard scripts
        self._scripts_dir = Path(__file__).parent
        self._trading_script = self._scripts_dir / "run_trading_dashboard.py"
        self._system_script = self._scripts_dir / "run_system_dashboard.py"
        
        # Register dashboards
        self.register_dashboard("trading", 8080)
        self.register_dashboard("system", 8052)
    
    def register_dashboard(self, name: str, port: int):
        """Register a dashboard."""
        if name in self.dashboards:
            logger.warning(f"Dashboard {name} already registered")
            return
        
        self.dashboards[name] = DashboardInfo(name=name, port=port)
        logger.info(f"Registered dashboard: {name} on port {port}")
    
    async def start_all(self):
        """Start all registered dashboards."""
        logger.info("Starting all dashboards...")
        
        for name in self.dashboards:
            await self.start_dashboard(name)
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_dashboards())
        
        logger.info("All dashboards started")
    
    async def start_dashboard(self, name: str):
        """Start a specific dashboard."""
        logger.info(f"=== Starting {name} dashboard ===")
        
        if name not in self.dashboards:
            logger.error(f"Dashboard {name} not registered")
            return
        
        info = self.dashboards[name]
        logger.info(f"Dashboard {name} info: port={info.port}, state={info.state}")
        
        if info.state == DashboardState.RUNNING:
            logger.warning(f"Dashboard {name} already running")
            return
        
        logger.info(f"Starting {name} dashboard on port {info.port}")
        info.state = DashboardState.STARTING
        
        try:
            # Determine which script to use
            if name == "trading":
                script_path = self._trading_script
            elif name == "system":
                script_path = self._system_script
            else:
                logger.error(f"Unknown dashboard type: {name}")
                info.state = DashboardState.ERROR
                return
            
            # Verify script exists
            if not script_path.exists():
                logger.error(f"Dashboard script not found: {script_path}")
                info.state = DashboardState.ERROR
                info.last_error = f"Script not found: {script_path}"
                return
            
            # Prepare database config as JSON
            db_config_json = json.dumps(self.db_config)
            
            # Build command
            cmd = [
                sys.executable,  # Use the same Python interpreter
                str(script_path),
                "--config", db_config_json,
                "--port", str(info.port)
            ]
            
            # Set up environment
            env = os.environ.copy()
            
            # Start the process
            logger.info(f"Starting process: {' '.join(cmd)}")
            logger.info(f"Python executable: {sys.executable}")
            logger.info(f"Script path: {script_path}")
            logger.info(f"Working directory: {os.getcwd()}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Start a task to read output
            asyncio.create_task(self._read_dashboard_output(name, process))
            
            info.process = process
            info.pid = process.pid
            info.start_time = time.time()
            info.state = DashboardState.RUNNING
            
            logger.info(f"Started {name} dashboard with PID {info.pid}")
            
            # Give it a moment to initialize
            await asyncio.sleep(2)
            
            # Check if it's still running
            if process.poll() is not None:
                # Process died
                stdout, stderr = process.communicate()
                logger.error(f"Dashboard {name} failed to start")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                info.state = DashboardState.ERROR
                info.error_count += 1
                info.last_error = f"Process died immediately: {stderr}"
            
        except Exception as e:
            logger.error(f"Error starting {name} dashboard: {e}", exc_info=True)
            info.state = DashboardState.ERROR
            info.error_count += 1
            info.last_error = str(e)
    
    async def stop_all(self):
        """Stop all dashboards."""
        logger.info("Stopping all dashboards...")
        
        # Cancel monitoring
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop dashboards
        tasks = []
        for name in self.dashboards:
            tasks.append(self.stop_dashboard(name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All dashboards stopped")
    
    async def stop_dashboard(self, name: str):
        """Stop a specific dashboard."""
        if name not in self.dashboards:
            logger.error(f"Dashboard {name} not registered")
            return
        
        info = self.dashboards[name]
        
        if info.state != DashboardState.RUNNING:
            logger.warning(f"Dashboard {name} not running")
            return
        
        logger.info(f"Stopping {name} dashboard")
        info.state = DashboardState.STOPPING
        
        if info.process and info.process.poll() is None:
            try:
                # Try graceful termination first
                info.process.terminate()
                
                # Wait up to 5 seconds for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(info.process)),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # Force kill if still running
                    logger.warning(f"Force killing {name} dashboard")
                    info.process.kill()
                    await asyncio.create_task(self._wait_for_process(info.process))
                
            except Exception as e:
                logger.error(f"Error stopping {name} dashboard: {e}")
                info.error_count += 1
                info.last_error = str(e)
        
        info.process = None
        info.pid = None
        info.state = DashboardState.STOPPED
        logger.info(f"Stopped {name} dashboard")
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for a process to terminate."""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def restart_dashboard(self, name: str):
        """Restart a specific dashboard."""
        logger.info(f"Restarting {name} dashboard")
        await self.stop_dashboard(name)
        await asyncio.sleep(1)  # Brief pause
        await self.start_dashboard(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all dashboards."""
        status = {}
        
        for name, info in self.dashboards.items():
            # Check if process is actually alive
            if info.process and info.process.poll() is not None:
                info.state = DashboardState.ERROR
                info.last_error = "Process died unexpectedly"
            
            status[name] = {
                'state': info.state.value,
                'port': info.port,
                'pid': info.pid,
                'uptime': time.time() - info.start_time if info.start_time else 0,
                'error_count': info.error_count,
                'last_error': info.last_error,
                'url': f"http://localhost:{info.port}" if info.state == DashboardState.RUNNING else None
            }
        
        return status
    
    async def _monitor_dashboards(self):
        """Monitor dashboard health and restart if needed."""
        while True:
            try:
                for name, info in self.dashboards.items():
                    if info.state == DashboardState.RUNNING:
                        if info.process and info.process.poll() is not None:
                            logger.error(f"Dashboard {name} died unexpectedly")
                            info.state = DashboardState.ERROR
                            info.error_count += 1
                            info.last_error = "Process died unexpectedly"
                            
                            # Auto-restart if not too many errors
                            if info.error_count < 3:
                                logger.info(f"Auto-restarting {name} dashboard")
                                await self.start_dashboard(name)
                            else:
                                logger.error(f"Dashboard {name} has failed too many times, not restarting")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard monitor: {e}")
                await asyncio.sleep(5)
    
    async def _read_dashboard_output(self, name: str, process: subprocess.Popen):
        """Read and log dashboard output."""
        async def read_stream(stream, prefix):
            try:
                while True:
                    line = await asyncio.get_event_loop().run_in_executor(None, stream.readline)
                    if not line:
                        if process.poll() is not None:
                            # Process has ended
                            break
                        await asyncio.sleep(0.01)
                        continue
                    
                    line = line.strip()
                    if line:
                        logger.info(f"[{name}:{prefix}] {line}")
                        print(f"[{name}:{prefix}] {line}")  # Also print to console
                        
                # Read any remaining output after process ends
                remaining = stream.read()
                if remaining:
                    for line in remaining.strip().split('\n'):
                        if line:
                            logger.info(f"[{name}:{prefix}] {line}")
                            print(f"[{name}:{prefix}] {line}")
                            
            except Exception as e:
                logger.error(f"Error reading {name} {prefix}: {e}")
        
        # Read both stdout and stderr
        await asyncio.gather(
            read_stream(process.stdout, "OUT"),
            read_stream(process.stderr, "ERR"),
            return_exceptions=True
        )
    
    def print_status(self):
        """Print dashboard status to console."""
        status = self.get_status()
        
        print("\n=== Dashboard Status ===")
        for name, info in status.items():
            state_symbol = {
                'running': '✅',
                'stopped': '⭕',
                'error': '❌',
                'starting': '🔄',
                'stopping': '🔄'
            }.get(info['state'], '❓')
            
            print(f"\n{state_symbol} {name.capitalize()} Dashboard:")
            print(f"   State: {info['state']}")
            print(f"   Port: {info['port']}")
            if info['url']:
                print(f"   URL: {info['url']}")
            if info['pid']:
                print(f"   PID: {info['pid']}")
            if info['uptime'] > 0:
                print(f"   Uptime: {info['uptime']:.0f}s")
            if info['error_count'] > 0:
                print(f"   Errors: {info['error_count']}")
                if info['last_error']:
                    print(f"   Last Error: {info['last_error']}")
        print("")