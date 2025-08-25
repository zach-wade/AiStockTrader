"""
System monitoring utilities.

Provides system resource monitoring capabilities including
CPU, memory, disk, and process statistics.
"""

import sys
from typing import Any

import psutil


class SystemMonitor:
    """System resource monitoring utilities."""

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get current system information."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "python_version": sys.version,
                "process_memory_rss": process.memory_info().rss,
                "process_cpu_percent": process.cpu_percent(),
                "process_threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def capture_baseline_metrics() -> dict[str, Any]:
        """Capture baseline system metrics."""
        try:
            process = psutil.Process()
            return {
                "memory_rss": process.memory_info().rss,
                "memory_vms": process.memory_info().vms,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
                "threads": process.num_threads(),
            }
        except Exception:
            return {}

    @staticmethod
    def get_current_memory_usage() -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return int(process.memory_info().rss)
        except Exception:
            return 0

    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return float(cpu_percent)
        except Exception:
            return 0.0

    @staticmethod
    def get_memory_usage() -> dict[str, Any]:
        """Get memory usage statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            }
        except Exception:
            return {}

    @staticmethod
    def get_disk_usage() -> dict[str, Any]:
        """Get disk usage statistics."""
        try:
            disk = psutil.disk_usage("/")
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            }
        except Exception:
            return {}
