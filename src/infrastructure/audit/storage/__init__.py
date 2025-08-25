"""
Audit storage components.

This module provides a refactored architecture for audit storage,
breaking down the large storage module into focused components.
"""

from .base import AuditStorage
from .database_storage import DatabaseStorage
from .external_storage import ExternalStorage
from .file_storage import FileStorage
from .multi_storage import MultiStorage

__all__ = [
    "AuditStorage",
    "FileStorage",
    "DatabaseStorage",
    "ExternalStorage",
    "MultiStorage",
]
