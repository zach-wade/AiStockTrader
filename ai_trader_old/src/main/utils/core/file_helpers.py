# File: ai_trader/utils/file_helpers.py

# Standard library imports
import json
import logging
from pathlib import Path
import shutil
from typing import Any

# Third-party imports
import aiofiles
import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(path: str) -> dict[Any, Any]:
    """Load YAML configuration file"""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path as string or Path object

    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_delete_file(file_path: str | Path) -> bool:
    """
    Safely delete a file if it exists.

    Args:
        file_path: Path to file to delete

    Returns:
        True if file was deleted, False if it didn't exist
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def read_json_file(file_path: str | Path) -> dict[str, Any] | None:
    """
    Read JSON file and return contents.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary with JSON contents or None if error
    """
    try:
        path = Path(file_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None


def write_json_file(file_path: str | Path, data: dict[str, Any], pretty: bool = True) -> bool:
    """
    Write dictionary to JSON file.

    Args:
        file_path: Path to write to
        data: Dictionary to write
        pretty: Whether to pretty-print the JSON

    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        ensure_directory_exists(path.parent)

        with open(path, "w") as f:
            if pretty:
                json.dump(data, f, indent=2, default=str)
            else:
                json.dump(data, f, default=str)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


async def safe_json_write(file_path: str | Path, data: dict[str, Any]) -> bool:
    """
    Async version of JSON write with atomic operation.
    Writes to temp file then moves to ensure atomicity.

    Args:
        file_path: Path to write to
        data: Dictionary to write

    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        ensure_directory_exists(path.parent)

        # Write to temp file first
        temp_path = path.with_suffix(".tmp")

        async with aiofiles.open(temp_path, "w") as f:
            await f.write(json.dumps(data, indent=2, default=str))

        # Atomic move
        temp_path.replace(path)
        return True

    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


def get_file_size(file_path: str | Path) -> int | None:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        Size in bytes or None if file doesn't exist
    """
    try:
        path = Path(file_path)
        if path.exists():
            return path.stat().st_size
        return None
    except Exception as e:
        logger.error(f"Error getting file size {file_path}: {e}")
        return None


def get_file_size_human(file_path: str | Path) -> str:
    """
    Get file size in human-readable format.

    Args:
        file_path: Path to file

    Returns:
        Human-readable size string (e.g., "1.5 MB")
    """
    size = get_file_size(file_path)
    if size is None:
        return "File not found"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def clean_old_files(directory: str | Path, days_old: int, pattern: str = "*") -> int:
    """
    Remove files older than specified days.

    Args:
        directory: Directory to clean
        days_old: Remove files older than this many days
        pattern: File pattern to match (default: all files)

    Returns:
        Number of files deleted
    """
    # Standard library imports
    import time

    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 3600)
        deleted_count = 0

        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_time = file_path.stat().st_mtime
                if file_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old file: {file_path}")

        return deleted_count

    except Exception as e:
        logger.error(f"Error cleaning old files in {directory}: {e}")
        return 0


def copy_with_backup(src: str | Path, dst: str | Path, backup_suffix: str = ".backup") -> bool:
    """
    Copy file with automatic backup of destination if it exists.

    Args:
        src: Source file path
        dst: Destination file path
        backup_suffix: Suffix for backup file

    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file {src} does not exist")
            return False

        # Backup existing destination
        if dst_path.exists():
            backup_path = dst_path.with_suffix(dst_path.suffix + backup_suffix)
            shutil.copy2(dst_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Ensure destination directory exists
        ensure_directory_exists(dst_path.parent)

        # Copy file
        shutil.copy2(src_path, dst_path)
        return True

    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False


def list_files(directory: str | Path, pattern: str = "*", recursive: bool = False) -> list[Path]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search subdirectories

    Returns:
        List of Path objects for matching files
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))

    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []


# Async version for reading large files
async def read_file_chunks(file_path: str | Path, chunk_size: int = 8192):
    """
    Async generator to read file in chunks.

    Args:
        file_path: Path to file
        chunk_size: Size of each chunk in bytes

    Yields:
        Chunks of file content
    """
    async with aiofiles.open(file_path, "rb") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk
