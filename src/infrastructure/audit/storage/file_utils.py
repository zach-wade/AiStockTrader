"""
File storage utility functions.

Provides utility functions for file operations, integrity checking,
and file management operations used by file storage backends.
"""

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class FileUtils:
    """Utility functions for file storage operations."""

    @staticmethod
    def read_events_from_file(file_path: Path) -> Iterator[dict[str, Any]]:
        """Read events from a log file."""
        try:
            if file_path.suffix == ".gz":
                f = gzip.open(file_path, "rt", encoding="utf-8")
            else:
                f = open(file_path, encoding="utf-8")

            with f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            # Log warning but continue processing
                            continue

        except Exception:
            # Silently handle file read errors
            return

    @staticmethod
    def search_file_for_event(file_path: Path, event_id: str) -> dict[str, Any] | None:
        """Search for specific event ID in a log file."""
        for event in FileUtils.read_events_from_file(file_path):
            if event.get("event_id") == event_id:
                return event
        return None

    @staticmethod
    def get_all_log_files(storage_path: Path, compression_enabled: bool) -> Iterator[Path]:
        """Get all audit log files."""
        patterns = ["audit_*.jsonl"]
        if compression_enabled:
            patterns.append("audit_*.jsonl.gz")

        for pattern in patterns:
            yield from storage_path.glob(pattern)

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        hash_obj = hashlib.sha256()

        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_obj.update(chunk)
            else:
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_obj.update(chunk)
        except Exception:
            return ""

        return hash_obj.hexdigest()

    @staticmethod
    def write_to_file(file_handle: Any, content: str, is_compressed: bool) -> int:
        """Write content to file handle and return bytes written."""
        if is_compressed and isinstance(file_handle, gzip.GzipFile):
            file_handle.write(content.encode("utf-8"))
            file_handle.flush()
            # Estimate compressed size
            return len(content.encode("utf-8")) // 2
        else:
            file_handle.write(content)
            file_handle.flush()
            return len(content.encode("utf-8"))

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query by removing specific values."""
        import re

        # Remove string literals
        normalized = re.sub(r"'[^']*'", "'?'", query)

        # Remove numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        # Remove whitespace variations
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()
