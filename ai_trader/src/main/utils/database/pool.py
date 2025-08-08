
"""Database connection pooling for improved performance"""
from sqlalchemy import create_engine, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
import os
import time
import threading
from typing import Optional, Dict, Any, Union, List
from main.config.validation_models import AITraderConfig
from omegaconf import DictConfig
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from .helpers import (
    ConnectionPoolMetrics, ConnectionHealthStatus, 
    PoolHealthMonitor, QueryPerformanceTracker
)
from .helpers.connection_metrics import MetricsCollector

logger = logging.getLogger(__name__)


class DatabasePool:
    """Singleton database connection pool with comprehensive monitoring"""
    _instance: Optional['DatabasePool'] = None
    _engine = None
    _session_factory = None
    _database_url = None
    
    def __init__(self):
        # Use modular components
        self.metrics_collector = MetricsCollector()
        self.query_tracker = QueryPerformanceTracker()
        self.health_monitor = PoolHealthMonitor(self.metrics_collector)
        
        self._monitoring_enabled = True
        self._start_time = datetime.now()
        
        # Thread-safe event tracking
        self._lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(
    self,
    database_url: Optional[str] = None,
    config: Optional[Union[Dict[str, Any], DictConfig, AITraderConfig]] = None
    ):

        """
        Initialize database connection pool.
        
        Args:
            database_url: Direct database URL (takes precedence)
            config: Configuration dict with database settings
        """
        if self._engine is not None:
            logger.debug("Database pool already initialized")
            return
        
        # Determine database URL from various sources
        if database_url:
            self._database_url = database_url
        elif config:
            self._database_url = self._build_url_from_config(config)
        else:
            # Fallback to environment variable
            self._database_url = os.getenv('DATABASE_URL')
            if not self._database_url:
                # Try to build from individual environment variables
                self._database_url = self._build_url_from_env()
        
        if not self._database_url:
            raise ValueError(
                "Database URL not provided. Set DATABASE_URL environment variable "
                "or provide database_url/config parameter"
            )
        
        # Create engine with optimized connection pooling
        self._engine = create_engine(
            self._database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_timeout=30,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections every hour
            echo=False
        )
        
        # Create session factory with consistent settings
        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False
        )
        
        logger.info(f"Database connection pool initialized: {self._mask_password(self._database_url)}")
        
        # Set up monitoring event listeners
        self._setup_monitoring_events()
    
    def _setup_monitoring_events(self):
        """Set up SQLAlchemy event listeners for monitoring"""
        if not self._engine or not self._monitoring_enabled:
            return
        
        @event.listens_for(self._engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()
        
        @event.listens_for(self._engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                execution_time = time.time() - context._query_start_time
                
                # Record query performance
                self.metrics_collector.record_query(execution_time)
                
                # Analyze query if slow
                analysis = self.query_tracker.analyze_query(statement, execution_time)
                if analysis:
                    logger.warning(f"Slow query detected: {execution_time:.3f}s - {statement[:100]}...")
        
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self.metrics_collector.record_connection_event('connect')
        
        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            self.metrics_collector.record_connection_event('checkout')
        
        @event.listens_for(self._engine, "checkin")
        def on_checkin(dbapi_connection, connection_record):
            self.metrics_collector.record_connection_event('checkin')
        
        logger.info("Database monitoring events set up successfully")
    
    def _build_url_from_config(self, config) -> str:
        """Build database URL from config (dict or Pydantic model)."""
        # Handle both dict and Pydantic config
        if hasattr(config, 'database') and hasattr(config, '__dict__'):
            # Pydantic config object (AITraderConfig)
            db_config = config.database
            host = getattr(db_config, 'host', os.getenv('DB_HOST'))
            port = getattr(db_config, 'port', os.getenv('DB_PORT'))
            name = getattr(db_config, 'name', os.getenv('DB_NAME'))
            user = getattr(db_config, 'user', os.getenv('DB_USER'))
            password = getattr(db_config, 'password', os.getenv('DB_PASSWORD'))
        elif hasattr(config, 'get'):
            # Dict config (fallback)
            db_config = config.get('database', {})
            host = db_config.get('host', os.getenv('DB_HOST'))
            port = db_config.get('port', os.getenv('DB_PORT'))
            name = db_config.get('name', os.getenv('DB_NAME'))
            user = db_config.get('user', os.getenv('DB_USER'))
            password = db_config.get('password', os.getenv('DB_PASSWORD'))
        else:
            # Direct environment variable fallback
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT')
            name = os.getenv('DB_NAME')
            user = os.getenv('DB_USER')
            password = os.getenv('DB_PASSWORD')
        
        # Raise error if required core fields are missing (password optional for localhost)
        if not all([host, port, name, user]):
            missing_fields = []
            if not host: missing_fields.append('DB_HOST')
            if not port: missing_fields.append('DB_PORT') 
            if not name: missing_fields.append('DB_NAME')
            if not user: missing_fields.append('DB_USER')
            
            raise ValueError(
                f"Missing required database configuration: {', '.join(missing_fields)}. "
                "Please set environment variables or config."
            )
        
        # Convert port to int if it's a string
        if isinstance(port, str):
            port = int(port) if port and port != 'None' else 5432
        
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            return f"postgresql://{user}@{host}:{port}/{name}"
    
    def _build_url_from_env(self) -> Optional[str]:
        """Build database URL from environment variables."""
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT')
        name = os.getenv('DB_NAME')
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        
        # All required fields must be present to build URL
        if not all([host, port, name, user]):
            return None
        
        # Build URL with or without password (for local development)
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            return f"postgresql://{user}@{host}:{port}/{name}"
    
    def _mask_password(self, url: str) -> str:
        """Mask password in URL for logging."""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', url)
    
    def get_connection(self):
        """Get a raw database connection"""
        if self._engine is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._engine.connect()
    
    def get_session(self):
        """Get a new database session"""
        if self._session_factory is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._session_factory()
    
    def get_engine(self):
        """Get the SQLAlchemy engine"""
        if self._engine is None:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        return self._engine
    
    def get_url(self) -> Optional[str]:
        """Get the database URL"""
        return self._database_url
    
    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pool metrics"""
        if not self._engine:
            return {'error': 'Pool not initialized'}
        
        pool = self._engine.pool
        
        # Update current pool status
        pool_info = {
            'pool_size': pool.size(),
            'max_overflow': getattr(pool, '_max_overflow', 0),
            'active': pool.checkedout(),
            'idle': pool.checkedin(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }
        
        self.metrics_collector.update_pool_status(pool_info)
        
        # Get metrics snapshot
        metrics = self.metrics_collector.get_metrics_snapshot()
        
        # Add pool-specific information
        metrics['pool_status']['pool_size'] = pool_info['pool_size']
        metrics['pool_status']['max_overflow'] = pool_info['max_overflow']
        
        return metrics
    
    def get_health_status(self) -> ConnectionHealthStatus:
        """Get comprehensive health status of the connection pool"""
        if not self._engine:
            return ConnectionHealthStatus(
                is_healthy=False,
                pool_utilization=0.0,
                avg_response_time=0.0,
                error_rate=100.0,
                warnings=['Pool not initialized'],
                recommendations=['Initialize database pool']
            )
        
        pool = self._engine.pool
        pool_info = {
            'pool_size': pool.size(),
            'max_overflow': getattr(pool, '_max_overflow', 0)
        }
        
        return self.health_monitor.assess_health(pool_info)
    
    def get_slow_queries_report(self) -> Dict[str, Any]:
        """Get detailed report of slow queries"""
        return self.query_tracker.get_slow_query_summary()
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing or periodic resets)"""
        self.metrics_collector.reset_metrics()
        self.query_tracker.clear_history()
        logger.info("Database pool metrics reset")
    
    def enable_monitoring(self):
        """Enable performance monitoring"""
        self._monitoring_enabled = True
        if self._engine:
            self._setup_monitoring_events()
        logger.info("Database monitoring enabled")
    
    def disable_monitoring(self):
        """Disable performance monitoring"""
        self._monitoring_enabled = False
        logger.info("Database monitoring disabled")
    
    def get_connection_leak_report(self) -> Dict[str, Any]:
        """Detect potential connection leaks"""
        if not self._engine:
            return {'error': 'Pool not initialized'}
        
        pool = self._engine.pool
        pool_info = {
            'pool_size': pool.size(),
            'max_overflow': getattr(pool, '_max_overflow', 0)
        }
        
        return self.health_monitor.check_connection_leaks(pool_info)
    
    def dispose(self):
        """Dispose of the connection pool"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._database_url = None
            logger.info("Database connection pool disposed")
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a database connection for async operations.
        
        This provides compatibility with async database adapters.
        Returns a connection-like object that supports async operations.
        """
        import asyncpg
        
        # If we have a PostgreSQL URL, use asyncpg directly
        if self._database_url and self._database_url.startswith('postgresql://'):
            # Create an asyncpg connection
            conn = await asyncpg.connect(self._database_url)
            try:
                yield conn
            finally:
                await conn.close()
        else:
            # Fallback to sync session wrapped in async context
            session = self.get_session()
            try:
                # Create a simple wrapper to make sync session work in async context
                class AsyncSessionWrapper:
                    def __init__(self, session):
                        self.session = session
                    
                    async def execute(self, query, *args, **kwargs):
                        return self.session.execute(query, *args, **kwargs)
                    
                    async def fetch(self, query, *args, **kwargs):
                        result = self.session.execute(query, *args, **kwargs)
                        return result.fetchall()
                    
                    async def fetchrow(self, query, *args, **kwargs):
                        result = self.session.execute(query, *args, **kwargs)
                        return result.fetchone()
                    
                    async def close(self):
                        self.session.close()
                
                yield AsyncSessionWrapper(session)
            finally:
                session.close()
    
    def reset(self):
        """Reset the singleton (useful for testing)"""
        self.dispose()
        DatabasePool._instance = None


# Convenience function
def get_db_pool() -> DatabasePool:
    """Get the database connection pool instance"""
    return DatabasePool()
