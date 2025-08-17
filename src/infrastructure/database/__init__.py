"""
Database Infrastructure Module

This module provides PostgreSQL database access for the AI Trading System.
Implements the infrastructure layer for data persistence using asyncpg.
"""

from .adapter import PostgreSQLAdapter
from .connection import ConnectionFactory, DatabaseConfig, DatabaseConnection
from .migrations import MigrationManager

__all__ = [
    "PostgreSQLAdapter",
    "ConnectionFactory",
    "DatabaseConnection",
    "DatabaseConfig",
    "MigrationManager",
]
