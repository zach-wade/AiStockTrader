"""
System Metrics Collectors

System resource collection and monitoring utilities.
"""

# Standard library imports
from datetime import datetime
import logging

# Third-party imports
import psutil

from .types import SystemResources

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Collects system resource metrics."""

    def __init__(self):
        """Initialize system metrics collector."""
        self._network_baseline = None
        logger.debug("System metrics collector initialized")

    def collect_system_metrics(self) -> SystemResources | None:
        """
        Collect current system resource metrics.

        Returns:
            SystemResources snapshot or None if collection fails
        """
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Disk usage
            disk_usage = psutil.disk_usage("/")

            # Network I/O
            network_io = psutil.net_io_counters()

            # Calculate network delta if we have a baseline
            if self._network_baseline is None:
                self._network_baseline = network_io
                return None  # Skip first collection to establish baseline

            bytes_sent_delta = network_io.bytes_sent - self._network_baseline.bytes_sent
            bytes_recv_delta = network_io.bytes_recv - self._network_baseline.bytes_recv
            self._network_baseline = network_io

            # Create system resource snapshot
            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk_usage.percent,
                disk_free_gb=disk_usage.free / 1024 / 1024 / 1024,
                network_bytes_sent=bytes_sent_delta,
                network_bytes_recv=bytes_recv_delta,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None

    def get_cpu_info(self) -> dict:
        """Get CPU information."""
        try:
            return {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "per_cpu_percent": psutil.cpu_percent(percpu=True),
                "cpu_times": psutil.cpu_times()._asdict(),
            }
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return {}

    def get_memory_info(self) -> dict:
        """Get detailed memory information."""
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            return {
                "virtual_memory": virtual_mem._asdict(),
                "swap_memory": swap_mem._asdict(),
                "memory_percent": virtual_mem.percent,
                "swap_percent": swap_mem.percent,
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def get_disk_info(self) -> dict:
        """Get disk usage information."""
        try:
            disk_usage = psutil.disk_usage("/")
            disk_io = psutil.disk_io_counters()

            return {
                "disk_usage": disk_usage._asdict(),
                "disk_io": disk_io._asdict() if disk_io else None,
                "disk_partitions": [p._asdict() for p in psutil.disk_partitions()],
            }
        except Exception as e:
            logger.error(f"Error getting disk info: {e}")
            return {}

    def get_network_info(self) -> dict:
        """Get network information."""
        try:
            net_io = psutil.net_io_counters()
            net_connections = psutil.net_connections()

            return {
                "net_io": net_io._asdict() if net_io else None,
                "connection_count": len(net_connections),
                "net_if_addrs": {
                    k: [addr._asdict() for addr in v] for k, v in psutil.net_if_addrs().items()
                },
                "net_if_stats": {k: v._asdict() for k, v in psutil.net_if_stats().items()},
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {}

    def get_process_info(self) -> dict:
        """Get current process information."""
        try:
            process = psutil.Process()

            return {
                "pid": process.pid,
                "name": process.name(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": process.memory_info()._asdict(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time(),
                "status": process.status(),
            }
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return {}

    def get_system_load(self) -> dict:
        """Get system load information."""
        try:
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            boot_time = psutil.boot_time()

            return {
                "load_avg": load_avg,
                "boot_time": boot_time,
                "uptime_seconds": datetime.now().timestamp() - boot_time,
                "process_count": len(psutil.pids()),
            }
        except Exception as e:
            logger.error(f"Error getting system load: {e}")
            return {}

    def reset_network_baseline(self):
        """Reset network I/O baseline."""
        self._network_baseline = None
        logger.debug("Network baseline reset")

    def get_comprehensive_snapshot(self) -> dict:
        """Get comprehensive system snapshot."""
        return {
            "cpu_info": self.get_cpu_info(),
            "memory_info": self.get_memory_info(),
            "disk_info": self.get_disk_info(),
            "network_info": self.get_network_info(),
            "process_info": self.get_process_info(),
            "system_load": self.get_system_load(),
            "timestamp": datetime.now().isoformat(),
        }
