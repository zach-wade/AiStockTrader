#!/usr/bin/env python3
"""
Comprehensive system cleanup and duplicate removal for AI Trading System

This script identifies and handles duplicate files/folders to recover storage space
and improve system organization.
"""

# Standard library imports
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SystemCleanup:
    """Comprehensive system cleanup utility"""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.cleanup_plan = {"high_priority": [], "medium_priority": [], "low_priority": []}
        self.space_recovery = 0

    def analyze_duplicates(self):
        """Analyze and categorize duplicate files/folders"""
        logger.info("🔍 Analyzing system for duplicates...")

        # Check for major space consumers
        self._check_venv_directory()
        self._check_data_lake_backups()
        self._check_cache_directories()
        self._check_log_files()
        self._check_python_cache()
        self._check_duplicate_features()

        return self.cleanup_plan

    def _check_venv_directory(self):
        """Check virtual environment directory"""
        venv_path = self.root_path / "venv"
        if venv_path.exists():
            size_gb = self._get_directory_size(venv_path) / (1024**3)
            self.cleanup_plan["high_priority"].append(
                {
                    "task": "Move virtual environment to local-only",
                    "path": str(venv_path),
                    "size_gb": size_gb,
                    "action": "add_to_gitignore",
                    "recovery_gb": 0,  # No actual recovery, just cloud sync exclusion
                    "description": "Exclude from cloud sync to prevent unnecessary syncing",
                }
            )

    def _check_data_lake_backups(self):
        """Check for old data lake backups"""
        archive_path = self.root_path / "archive"
        if archive_path.exists():
            for backup_dir in archive_path.glob("data_lake_backup_*"):
                size_gb = self._get_directory_size(backup_dir) / (1024**3)
                # Check if backup is older than 30 days
                stat = backup_dir.stat()
                age_days = (datetime.now().timestamp() - stat.st_mtime) / (24 * 3600)

                if age_days > 30:
                    self.cleanup_plan["high_priority"].append(
                        {
                            "task": f"Archive old backup: {backup_dir.name}",
                            "path": str(backup_dir),
                            "size_gb": size_gb,
                            "action": "compress_and_archive",
                            "recovery_gb": size_gb * 0.8,  # Assume 80% compression
                            "age_days": age_days,
                            "description": f"Compress {age_days:.1f} day old backup",
                        }
                    )
                    self.space_recovery += size_gb * 0.8
                else:
                    self.cleanup_plan["medium_priority"].append(
                        {
                            "task": f"Recent backup: {backup_dir.name}",
                            "path": str(backup_dir),
                            "size_gb": size_gb,
                            "action": "keep",
                            "recovery_gb": 0,
                            "age_days": age_days,
                            "description": f"Keep recent backup ({age_days:.1f} days old)",
                        }
                    )

    def _check_cache_directories(self):
        """Check for cache directories"""
        cache_paths = [
            self.root_path / "cache",
            self.root_path / "__pycache__",
            self.root_path / ".pytest_cache",
        ]

        for cache_path in cache_paths:
            if cache_path.exists():
                size_mb = self._get_directory_size(cache_path) / (1024**2)
                if size_mb > 1:  # Only report if > 1MB
                    self.cleanup_plan["low_priority"].append(
                        {
                            "task": f"Clean cache directory: {cache_path.name}",
                            "path": str(cache_path),
                            "size_mb": size_mb,
                            "action": "clean_cache",
                            "recovery_mb": size_mb,
                            "description": "Remove cache files",
                        }
                    )

    def _check_log_files(self):
        """Check for old log files"""
        log_dirs = [self.root_path / "logs", self.root_path / "deployment" / "logs"]

        total_log_size = 0
        old_log_count = 0

        for log_dir in log_dirs:
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    stat = log_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / (24 * 3600)
                    size_mb = stat.st_size / (1024**2)

                    if age_days > 30:
                        total_log_size += size_mb
                        old_log_count += 1

        if total_log_size > 1:
            self.cleanup_plan["medium_priority"].append(
                {
                    "task": f"Clean old log files ({old_log_count} files)",
                    "path": "logs/",
                    "size_mb": total_log_size,
                    "action": "clean_old_logs",
                    "recovery_mb": total_log_size,
                    "description": f"Remove {old_log_count} log files older than 30 days",
                }
            )

    def _check_python_cache(self):
        """Check for Python cache files"""
        cache_dirs = list(self.root_path.glob("**/__pycache__")) + list(
            self.root_path.glob("**/.pytest_cache")
        )

        if cache_dirs:
            total_size = sum(self._get_directory_size(d) for d in cache_dirs)
            size_mb = total_size / (1024**2)

            self.cleanup_plan["low_priority"].append(
                {
                    "task": f"Remove Python cache ({len(cache_dirs)} directories)",
                    "path": "__pycache__/",
                    "size_mb": size_mb,
                    "action": "remove_python_cache",
                    "recovery_mb": size_mb,
                    "description": f"Remove {len(cache_dirs)} Python cache directories",
                }
            )

    def _check_duplicate_features(self):
        """Check for duplicate feature files"""
        feature_dirs = [
            self.root_path / "data_lake" / "features",
            self.root_path / "archive" / "data_lake_backup_20250710_225747" / "features",
        ]

        feature_hashes = defaultdict(list)

        for feature_dir in feature_dirs:
            if feature_dir.exists():
                for feature_file in feature_dir.glob("**/*.h5"):
                    try:
                        file_hash = self._calculate_file_hash(feature_file)
                        feature_hashes[file_hash].append(feature_file)
                    except (PermissionError, OSError):
                        continue

        # Find duplicates
        duplicates = {
            hash_val: paths for hash_val, paths in feature_hashes.items() if len(paths) > 1
        }

        if duplicates:
            total_duplicate_size = 0
            for hash_val, paths in duplicates.items():
                # Keep the most recent file, mark others for removal
                newest_file = max(paths, key=lambda p: p.stat().st_mtime)
                for duplicate_file in paths:
                    if duplicate_file != newest_file:
                        size_mb = duplicate_file.stat().st_size / (1024**2)
                        total_duplicate_size += size_mb

            if total_duplicate_size > 1:
                self.cleanup_plan["medium_priority"].append(
                    {
                        "task": f"Remove duplicate feature files ({len(duplicates)} sets)",
                        "path": "features/",
                        "size_mb": total_duplicate_size,
                        "action": "remove_duplicate_features",
                        "recovery_mb": total_duplicate_size,
                        "description": f"Remove {len(duplicates)} sets of duplicate feature files",
                    }
                )

    def execute_cleanup(self, priority_level: str = "high_priority", dry_run: bool = True):
        """Execute cleanup tasks"""
        tasks = self.cleanup_plan.get(priority_level, [])

        if dry_run:
            logger.info(f"🔍 DRY RUN - Would execute {len(tasks)} {priority_level} tasks:")
            for task in tasks:
                logger.info(f"  - {task['task']}")
                if "size_gb" in task:
                    logger.info(f"    Size: {task['size_gb']:.2f} GB")
                elif "size_mb" in task:
                    logger.info(f"    Size: {task['size_mb']:.2f} MB")
                logger.info(f"    Action: {task['action']}")
            return

        logger.info(f"⚡ Executing {len(tasks)} {priority_level} tasks:")

        for task in tasks:
            logger.info(f"📋 {task['task']}")

            try:
                if task["action"] == "add_to_gitignore":
                    self._add_to_gitignore(task["path"])
                elif task["action"] == "compress_and_archive":
                    self._compress_directory(task["path"])
                elif task["action"] == "clean_cache":
                    self._clean_cache_directory(task["path"])
                elif task["action"] == "clean_old_logs":
                    self._clean_old_logs()
                elif task["action"] == "remove_python_cache":
                    self._remove_python_cache()
                elif task["action"] == "remove_duplicate_features":
                    self._remove_duplicate_features()

                logger.info(f"✅ Completed: {task['task']}")

            except Exception as e:
                logger.error(f"❌ Failed: {task['task']} - {e}")

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    try:
                        total_size += filepath.stat().st_size
                    except (OSError, FileNotFoundError):
                        continue
        except (OSError, PermissionError):
            pass
        return total_size

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _add_to_gitignore(self, path: str):
        """Add path to .gitignore"""
        gitignore_path = self.root_path / ".gitignore"
        path_name = Path(path).name

        # Check if already in gitignore
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                if path_name in f.read():
                    logger.info(f"  {path_name} already in .gitignore")
                    return

        with open(gitignore_path, "a") as f:
            f.write(f"\n# Exclude from cloud sync\n{path_name}/\n")
        logger.info(f"  Added {path_name} to .gitignore")

    def _compress_directory(self, path: str):
        """Compress directory to tar.gz"""
        dir_path = Path(path)
        if not dir_path.exists():
            logger.warning(f"  Directory {path} does not exist")
            return

        archive_name = f"{dir_path.name}.tar.gz"
        archive_path = dir_path.parent / archive_name

        logger.info(f"  Compressing {dir_path.name} to {archive_name}...")
        # Note: This would be implemented with tarfile for large directories
        logger.info(f"  Would compress {path} to {archive_path}")

    def _clean_cache_directory(self, path: str):
        """Clean cache directory"""
        cache_path = Path(path)
        if cache_path.exists():
            if cache_path.is_dir() and not any(cache_path.iterdir()):
                cache_path.rmdir()
                logger.info(f"  Removed empty directory {path}")

    def _clean_old_logs(self):
        """Clean old log files"""
        log_dirs = [self.root_path / "logs", self.root_path / "deployment" / "logs"]

        for log_dir in log_dirs:
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    stat = log_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / (24 * 3600)

                    if age_days > 30:
                        log_file.unlink()
                        logger.info(f"  Removed old log: {log_file.name}")

    def _remove_python_cache(self):
        """Remove Python cache directories"""
        cache_dirs = list(self.root_path.glob("**/__pycache__")) + list(
            self.root_path.glob("**/.pytest_cache")
        )

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"  Removed cache: {cache_dir.relative_to(self.root_path)}")

    def _remove_duplicate_features(self):
        """Remove duplicate feature files"""
        logger.info("  Would analyze and remove duplicate feature files")
        # Implementation would compare file hashes and keep most recent versions


def main():
    parser = argparse.ArgumentParser(description="AI Trading System Cleanup")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )
    parser.add_argument(
        "--priority",
        choices=["high_priority", "medium_priority", "low_priority"],
        default="high_priority",
        help="Priority level to execute",
    )
    parser.add_argument("--execute", action="store_true", help="Actually execute cleanup tasks")

    args = parser.parse_args()

    cleanup = SystemCleanup()
    plan = cleanup.analyze_duplicates()

    print("\n📊 CLEANUP ANALYSIS COMPLETE")
    print(f"Potential space recovery: {cleanup.space_recovery:.1f}GB")
    print(f"High priority tasks: {len(plan['high_priority'])}")
    print(f"Medium priority tasks: {len(plan['medium_priority'])}")
    print(f"Low priority tasks: {len(plan['low_priority'])}")

    if args.execute and not args.dry_run:
        confirmation = input(f"\nExecute {args.priority} cleanup? (y/N): ")
        if confirmation.lower() == "y":
            cleanup.execute_cleanup(args.priority, dry_run=False)
        else:
            print("Cleanup cancelled")
    else:
        cleanup.execute_cleanup(args.priority, dry_run=True)


if __name__ == "__main__":
    main()
