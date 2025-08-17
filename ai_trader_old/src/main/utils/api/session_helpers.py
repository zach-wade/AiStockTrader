# File: utils/session_helpers.py

"""
Utility functions to help manage aiohttp ClientSession and httpx AsyncClient lifecycle.
Provides context managers and cleanup utilities to prevent unclosed session warnings.
"""

# Standard library imports
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from typing import Any
import warnings

logger = logging.getLogger(__name__)

# Track active sessions for debugging
_active_sessions = set()


@asynccontextmanager
async def managed_aiohttp_session(
    connector_kwargs: dict[str, Any] | None = None, **session_kwargs
) -> AsyncGenerator:
    """
    Context manager for aiohttp ClientSession that ensures proper cleanup.

    Args:
        connector_kwargs: Optional connector configuration
        **session_kwargs: Additional session configuration

    Yields:
        aiohttp.ClientSession: Configured session

    Example:
        async with managed_aiohttp_session() as session:
            async with session.get('https://api.example.com') as response:
                data = await response.json()
    """
    try:
        # Third-party imports
        import aiohttp
    except ImportError:
        raise ImportError("aiohttp is required for managed_aiohttp_session")

    session = None
    try:
        # Create connector if specified
        connector = None
        if connector_kwargs:
            connector = aiohttp.TCPConnector(**connector_kwargs)

        # Create session
        session = aiohttp.ClientSession(connector=connector, **session_kwargs)
        _active_sessions.add(id(session))

        logger.debug(f"Created aiohttp session {id(session)}")
        yield session

    except Exception as e:
        logger.error(f"Error in managed aiohttp session: {e}")
        raise
    finally:
        if session and not session.closed:
            await session.close()
            _active_sessions.discard(id(session))
            logger.debug(f"Closed aiohttp session {id(session)}")


@asynccontextmanager
async def managed_httpx_client(**client_kwargs) -> AsyncGenerator:
    """
    Context manager for httpx AsyncClient that ensures proper cleanup.

    Args:
        **client_kwargs: Client configuration

    Yields:
        httpx.AsyncClient: Configured client

    Example:
        async with managed_httpx_client() as client:
            response = await client.get('https://api.example.com')
            data = response.json()
    """
    try:
        # Third-party imports
        import httpx
    except ImportError:
        raise ImportError("httpx is required for managed_httpx_client")

    client = None
    try:
        client = httpx.AsyncClient(**client_kwargs)
        _active_sessions.add(id(client))

        logger.debug(f"Created httpx client {id(client)}")
        yield client

    except Exception as e:
        logger.error(f"Error in managed httpx client: {e}")
        raise
    finally:
        if client and not client.is_closed:
            await client.aclose()
            _active_sessions.discard(id(client))
            logger.debug(f"Closed httpx client {id(client)}")


class SessionManager:
    """
    Manages multiple HTTP sessions/clients with automatic cleanup.
    Useful for services that need to maintain multiple persistent connections.
    """

    def __init__(self):
        self.aiohttp_sessions: dict[str, Any] = {}
        self.httpx_clients: dict[str, Any] = {}

    async def get_aiohttp_session(
        self, name: str, connector_kwargs: dict[str, Any] | None = None, **session_kwargs
    ) -> Any:
        """Get or create a named aiohttp session."""
        if name not in self.aiohttp_sessions:
            try:
                # Third-party imports
                import aiohttp

                connector = None
                if connector_kwargs:
                    connector = aiohttp.TCPConnector(**connector_kwargs)

                session = aiohttp.ClientSession(connector=connector, **session_kwargs)
                self.aiohttp_sessions[name] = session
                _active_sessions.add(id(session))

                logger.debug(f"Created named aiohttp session '{name}' {id(session)}")

            except ImportError:
                raise ImportError("aiohttp is required for aiohttp session management")

        return self.aiohttp_sessions[name]

    async def get_httpx_client(self, name: str, **client_kwargs) -> Any:
        """Get or create a named httpx client."""
        if name not in self.httpx_clients:
            try:
                # Third-party imports
                import httpx

                client = httpx.AsyncClient(**client_kwargs)
                self.httpx_clients[name] = client
                _active_sessions.add(id(client))

                logger.debug(f"Created named httpx client '{name}' {id(client)}")

            except ImportError:
                raise ImportError("httpx is required for httpx client management")

        return self.httpx_clients[name]

    async def close_session(self, name: str, session_type: str = "aiohttp"):
        """Close a specific named session."""
        if session_type == "aiohttp" and name in self.aiohttp_sessions:
            session = self.aiohttp_sessions.pop(name)
            if not session.closed:
                await session.close()
                _active_sessions.discard(id(session))
                logger.debug(f"Closed named aiohttp session '{name}' {id(session)}")

        elif session_type == "httpx" and name in self.httpx_clients:
            client = self.httpx_clients.pop(name)
            if not client.is_closed:
                await client.aclose()
                _active_sessions.discard(id(client))
                logger.debug(f"Closed named httpx client '{name}' {id(client)}")

    async def close_all(self):
        """Close all managed sessions."""
        # Close aiohttp sessions
        for name in list(self.aiohttp_sessions.keys()):
            await self.close_session(name, "aiohttp")

        # Close httpx clients
        for name in list(self.httpx_clients.keys()):
            await self.close_session(name, "httpx")

        logger.info("Closed all managed sessions")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_all()


async def cleanup_orphaned_sessions():
    """
    Utility function to detect and warn about potentially unclosed sessions.
    This can be called during application shutdown to identify cleanup issues.
    """
    if _active_sessions:
        logger.warning(
            f"Found {len(_active_sessions)} potentially unclosed sessions: {_active_sessions}"
        )
        logger.warning("This may indicate missing cleanup in your code")
    else:
        logger.info("No orphaned sessions detected")


def suppress_aiohttp_warnings():
    """
    Temporarily suppress aiohttp connection pool warnings.
    Use sparingly and only when you're certain sessions are being cleaned up properly.
    """
    warnings.filterwarnings("ignore", message=".*Unclosed client session.*")
    warnings.filterwarnings("ignore", message=".*Unclosed connector.*")
    logger.warning("Suppressed aiohttp unclosed session warnings - ensure proper cleanup!")


def restore_aiohttp_warnings():
    """Restore aiohttp warnings to default behavior."""
    warnings.filterwarnings("default", message=".*Unclosed client session.*")
    warnings.filterwarnings("default", message=".*Unclosed connector.*")
    logger.info("Restored aiohttp warnings to default behavior")


async def create_managed_session(**session_kwargs) -> Any:
    """
    Create a properly configured aiohttp ClientSession.

    This is a convenience function that creates a session with
    good defaults for production use.

    Args:
        **session_kwargs: Additional session configuration

    Returns:
        aiohttp.ClientSession: Configured session

    Note:
        The caller is responsible for closing the session.
        Consider using managed_aiohttp_session() context manager instead.
    """
    try:
        # Third-party imports
        import aiohttp
    except ImportError:
        raise ImportError("aiohttp is required for session creation")

    # Default configuration
    default_kwargs = {
        "timeout": aiohttp.ClientTimeout(total=30, connect=10),
        "headers": {"User-Agent": "AI-Trader/1.0"},
        "connector": aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        ),
    }

    # Merge with provided kwargs
    final_kwargs = {**default_kwargs, **session_kwargs}

    session = aiohttp.ClientSession(**final_kwargs)
    _active_sessions.add(id(session))

    logger.debug(f"Created managed session {id(session)}")

    return session


# Usage examples and best practices
"""
BEST PRACTICES FOR SESSION MANAGEMENT:

1. Always use context managers when possible:

   async with managed_aiohttp_session() as session:
       async with session.get(url) as response:
           data = await response.json()

2. For long-running services, use SessionManager:

   async with SessionManager() as sessions:
       session = await sessions.get_aiohttp_session('api_client')
       # Use session...
       # Automatic cleanup on exit

3. In classes, implement proper cleanup:

   class MyAPIClient:
       def __init__(self):
           self.session = None

       async def connect(self):
           if not self.session:
               self.session = aiohttp.ClientSession()

       async def disconnect(self):
           if self.session and not self.session.closed:
               await self.session.close()
               self.session = None

       async def __aenter__(self):
           await self.connect()
           return self

       async def __aexit__(self, exc_type, exc_val, exc_tb):
           await self.disconnect()

4. For cleanup verification during testing:

   # At end of tests
   await cleanup_orphaned_sessions()

5. Only suppress warnings if you're absolutely sure about cleanup:

   suppress_aiohttp_warnings()
   try:
       # Your code that properly manages sessions
       pass
   finally:
       restore_aiohttp_warnings()
"""
