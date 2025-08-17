"""
Configuration Persistence

Configuration saving, loading, and auto-reload functionality.
"""

# Standard library imports
import asyncio
from datetime import datetime
import json
import logging
import os
from typing import Any

# Third-party imports
import yaml

from .loaders import load_from_env, load_from_file
from .sources import ConfigFormat, ConfigSourceType, detect_config_format

logger = logging.getLogger(__name__)


class ConfigPersistence:
    """Handles configuration persistence and auto-reload."""

    def __init__(self, wrapper):
        """Initialize with configuration wrapper."""
        self.wrapper = wrapper
        self._reload_task: asyncio.Task | None = None
        self._reload_active = False

    def save_to_file(self, file_path: str, format: ConfigFormat | None = None):
        """
        Save configuration to file.

        Args:
            file_path: Target file path
            format: Configuration format (auto-detected if None)
        """
        if format is None:
            format = detect_config_format(file_path)

        with self.wrapper._lock:
            config = self.wrapper.to_dict()

        if format == ConfigFormat.JSON:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        elif format == ConfigFormat.YAML:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported save format: {format}")

        logger.info(f"Configuration saved to {file_path}")

    async def start_auto_reload(self):
        """Start automatic configuration reloading."""
        if self._reload_active:
            return

        self._reload_active = True
        self._reload_task = asyncio.create_task(self._reload_loop())
        logger.info("Started automatic configuration reloading")

    async def stop_auto_reload(self):
        """Stop automatic configuration reloading."""
        self._reload_active = False

        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped automatic configuration reloading")

    async def _reload_loop(self):
        """Automatic reload loop."""
        while self._reload_active:
            try:
                await asyncio.sleep(self.wrapper.reload_interval)

                # Check if any file sources have changed
                for source in self.wrapper._sources:
                    if source.source_type == ConfigSourceType.FILE:
                        if os.path.exists(source.location):
                            mtime = datetime.fromtimestamp(os.path.getmtime(source.location))
                            if source.last_modified is None or mtime > source.last_modified:
                                logger.info(f"Configuration file changed: {source.location}")
                                await self.reload_configuration()
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reload loop: {e}")

    async def reload_configuration(self):
        """Reload configuration from sources."""
        if not self.wrapper._sources:
            return

        for source in self.wrapper._sources:
            try:
                if source.source_type == ConfigSourceType.FILE:
                    config = load_from_file(source.location, source.format)
                    source.last_modified = datetime.now()
                elif source.source_type == ConfigSourceType.ENVIRONMENT:
                    config = load_from_env(source.location)
                else:
                    continue

                with self.wrapper._lock:
                    self.wrapper._config.update(config)
                    if self.wrapper.schema:
                        self.wrapper.schema.validate(self.wrapper._config)
                    self.wrapper._notify_watchers()

                logger.info(f"Configuration reloaded from {source.location}")

            except Exception as e:
                logger.error(f"Error reloading configuration from {source.location}: {e}")

    def backup_configuration(self, backup_path: str, format: ConfigFormat | None = None):
        """
        Create a backup of current configuration.

        Args:
            backup_path: Path for backup file
            format: Backup format (auto-detected if None)
        """
        if format is None:
            format = detect_config_format(backup_path)

        with self.wrapper._lock:
            config = self.wrapper.to_dict()

        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "sources": [
                {
                    "type": source.source_type.value,
                    "location": source.location,
                    "format": source.format.value,
                }
                for source in self.wrapper._sources
            ],
        }

        if format == ConfigFormat.JSON:
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
        elif format == ConfigFormat.YAML:
            with open(backup_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(backup_data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported backup format: {format}")

        logger.info(f"Configuration backup created: {backup_path}")

    def restore_from_backup(self, backup_path: str, format: ConfigFormat | None = None):
        """
        Restore configuration from backup.

        Args:
            backup_path: Path to backup file
            format: Backup format (auto-detected if None)
        """
        if format is None:
            format = detect_config_format(backup_path)

        if format == ConfigFormat.JSON:
            with open(backup_path, encoding="utf-8") as f:
                backup_data = json.load(f)
        elif format == ConfigFormat.YAML:
            with open(backup_path, encoding="utf-8") as f:
                backup_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported backup format: {format}")

        config = backup_data.get("config", {})

        with self.wrapper._lock:
            self.wrapper._config.clear()
            self.wrapper._config.update(config)
            self.wrapper._original_config = self.wrapper._config.copy()

            if self.wrapper.schema:
                self.wrapper.schema.validate(self.wrapper._config)

            self.wrapper._notify_watchers()

        logger.info(f"Configuration restored from backup: {backup_path}")

    def export_config(
        self, export_path: str, format: ConfigFormat | None = None, include_metadata: bool = True
    ):
        """
        Export configuration with optional metadata.

        Args:
            export_path: Export file path
            format: Export format (auto-detected if None)
            include_metadata: Whether to include metadata
        """
        if format is None:
            format = detect_config_format(export_path)

        with self.wrapper._lock:
            config = self.wrapper.to_dict()

        if include_metadata:
            export_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                    "source_count": len(self.wrapper._sources),
                },
                "config": config,
            }
        else:
            export_data = config

        if format == ConfigFormat.JSON:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif format == ConfigFormat.YAML:
            with open(export_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(export_data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Configuration exported to: {export_path}")

    def import_config(
        self, import_path: str, format: ConfigFormat | None = None, merge: bool = True
    ):
        """
        Import configuration from file.

        Args:
            import_path: Import file path
            format: Import format (auto-detected if None)
            merge: Whether to merge with existing config
        """
        if format is None:
            format = detect_config_format(import_path)

        if format == ConfigFormat.JSON:
            with open(import_path, encoding="utf-8") as f:
                import_data = json.load(f)
        elif format == ConfigFormat.YAML:
            with open(import_path, encoding="utf-8") as f:
                import_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported import format: {format}")

        # Extract config from import data
        if "config" in import_data:
            config = import_data["config"]
        else:
            config = import_data

        with self.wrapper._lock:
            if merge:
                self.wrapper._config.update(config)
            else:
                self.wrapper._config.clear()
                self.wrapper._config.update(config)

            if self.wrapper.schema:
                self.wrapper.schema.validate(self.wrapper._config)

            self.wrapper._notify_watchers()

        logger.info(f"Configuration imported from: {import_path}")

    def get_file_status(self) -> dict[str, Any]:
        """Get status of configuration files."""
        status = {}

        for source in self.wrapper._sources:
            if source.source_type == ConfigSourceType.FILE:
                file_path = source.location

                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    status[file_path] = {
                        "exists": True,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "readable": os.access(file_path, os.R_OK),
                        "writable": os.access(file_path, os.W_OK),
                    }
                else:
                    status[file_path] = {"exists": False}

        return status
