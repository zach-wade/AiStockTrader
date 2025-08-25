"""
Audit middleware for API request and response logging.

This module provides middleware components for automatically capturing
and logging audit events for API requests and responses, including
authentication, authorization, and data access patterns.
"""

import json
import time
import uuid
from typing import Any

from .events import AuthenticationEvent, EventSeverity
from .logger import AuditContext, AuditLogger


class AuditMiddleware:
    """
    Base audit middleware for web frameworks.

    Provides core functionality for capturing API requests and responses
    for audit logging with minimal performance impact.
    """

    def __init__(
        self,
        logger: AuditLogger,
        include_request_body: bool = True,
        include_response_body: bool = False,
        include_headers: bool = True,
        exclude_headers: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        max_body_size: int = 10000,
        sensitive_fields: list[str] | None = None,
    ):
        """
        Initialize audit middleware.

        Args:
            logger: Audit logger instance
            include_request_body: Whether to include request body in audit
            include_response_body: Whether to include response body in audit
            include_headers: Whether to include HTTP headers
            exclude_headers: List of header names to exclude
            exclude_paths: List of URL paths to exclude from auditing
            max_body_size: Maximum size of request/response body to log
            sensitive_fields: List of field names to mask in logs
        """
        self.logger = logger
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.include_headers = include_headers
        self.exclude_headers = exclude_headers or ["authorization", "cookie", "x-api-key"]
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
        self.max_body_size = max_body_size
        self.sensitive_fields = sensitive_fields or ["password", "token", "secret", "key"]

    def should_audit_request(self, path: str, method: str) -> bool:
        """
        Determine if request should be audited.

        Args:
            path: Request path
            method: HTTP method

        Returns:
            True if request should be audited
        """
        # Skip excluded paths
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return False

        # Skip OPTIONS requests (CORS preflight)
        if method.upper() == "OPTIONS":
            return False

        return True

    def extract_user_context(self, headers: dict[str, str], **kwargs: Any) -> AuditContext:
        """
        Extract user context from request.

        Args:
            headers: HTTP headers
            **kwargs: Additional context information

        Returns:
            Audit context
        """
        user_id = None
        session_id = None
        request_id = str(uuid.uuid4())

        # Extract user ID from various sources
        if "x-user-id" in headers:
            user_id = headers["x-user-id"]
        elif "authorization" in headers:
            # Could parse JWT token here for user ID
            pass

        # Extract session ID
        if "x-session-id" in headers:
            session_id = headers["x-session-id"]

        # Extract or generate request ID
        if "x-request-id" in headers:
            request_id = headers["x-request-id"]

        context_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "request_id": request_id,
            "ip_address": kwargs.get("remote_addr"),
            "user_agent": headers.get("user-agent"),
            "metadata": {
                "headers": self._filter_headers(headers) if self.include_headers else {},
                **kwargs,
            },
        }
        if session_id is not None:
            context_kwargs["session_id"] = session_id
        return AuditContext(**context_kwargs)

    def _filter_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out sensitive headers."""
        filtered = {}
        for name, value in headers.items():
            if name.lower() in [h.lower() for h in self.exclude_headers]:
                filtered[name] = "***REDACTED***"
            else:
                filtered[name] = value
        return filtered

    def _mask_sensitive_data(self, data: Any) -> Any:
        """Recursively mask sensitive data in request/response."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if key.lower() in [f.lower() for f in self.sensitive_fields]:
                    masked[key] = "***MASKED***"
                else:
                    masked[key] = self._mask_sensitive_data(value)
            return masked
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data

    def _serialize_body(self, body: Any) -> Any:
        """Serialize request/response body for logging."""
        if body is None:
            return None

        # Convert bytes to string
        if isinstance(body, bytes):
            try:
                body = body.decode("utf-8")
            except UnicodeDecodeError:
                return "<binary data>"

        # Limit body size
        if isinstance(body, str) and len(body) > self.max_body_size:
            return body[: self.max_body_size] + "...<truncated>"

        # Try to parse JSON
        if isinstance(body, str):
            try:
                parsed_json = json.loads(body)
                return self._mask_sensitive_data(parsed_json)
            except (json.JSONDecodeError, ValueError):
                pass

        # Mask sensitive data in dict/list
        return self._mask_sensitive_data(body)


class FlaskAuditMiddleware(AuditMiddleware):
    """Audit middleware for Flask applications."""

    def __init__(self, app: Any, logger: AuditLogger, **kwargs: Any) -> None:
        """
        Initialize Flask audit middleware.

        Args:
            app: Flask application instance
            logger: Audit logger instance
            **kwargs: Additional middleware configuration
        """
        super().__init__(logger, **kwargs)
        self.app = app

        # Register Flask hooks
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_request(self._teardown_request)

    def _before_request(self) -> None:
        """Flask before_request handler."""
        from flask import g, request

        # Check if request should be audited
        if not self.should_audit_request(request.path, request.method):
            return

        # Store request start time
        g.audit_start_time = time.time()
        g.audit_request_id = str(uuid.uuid4())

        # Extract audit context
        headers = dict(request.headers)
        g.audit_context = self.extract_user_context(
            headers=headers,
            remote_addr=request.remote_addr,
            method=request.method,
            path=request.path,
            query_string=str(request.query_string, "utf-8") if request.query_string else None,
        )

        # Log request event
        self._log_request_event(request, g.audit_context)

    def _after_request(self, response: Any) -> Any:
        """Flask after_request handler."""
        from flask import g, request

        if not hasattr(g, "audit_context"):
            return response

        # Calculate request duration
        duration_ms = (time.time() - g.audit_start_time) * 1000

        # Log response event
        self._log_response_event(request, response, g.audit_context, duration_ms)

        return response

    def _teardown_request(self, exception: Any) -> None:
        """Flask teardown_request handler."""
        from flask import g

        if not hasattr(g, "audit_context"):
            return

        # Log error event if exception occurred
        if exception:
            self._log_error_event(exception, g.audit_context)

    def _log_request_event(self, request: Any, context: AuditContext) -> None:
        """Log HTTP request event."""
        try:
            event_data = {
                "event_type": "http_request",
                "resource_type": "api_endpoint",
                "resource_id": f"{request.method} {request.path}",
                "action": "request",
                "http_method": request.method,
                "url_path": request.path,
                "query_parameters": dict(request.args) if request.args else None,
                "content_type": request.content_type,
                "content_length": request.content_length,
                "severity": EventSeverity.LOW,
                "is_critical": False,
            }

            # Add request body if configured
            if self.include_request_body and request.data:
                event_data["request_body"] = self._serialize_body(request.data)

            # Create generic audit event
            from .events import AuditEvent

            class HTTPRequestEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPRequestEvent(
                event_type="http_request",
                resource_type="api_endpoint",
                resource_id=event_data["resource_id"],
                action="request",
                severity=EventSeverity.LOW,
                metadata=event_data,
            )

            self.logger.log_event(event, context)

        except Exception as e:
            # Don't let audit failures break the request
            self.logger._logger.error(f"Failed to log request event: {e}")

    def _log_response_event(
        self, request: Any, response: Any, context: AuditContext, duration_ms: float
    ) -> None:
        """Log HTTP response event."""
        try:
            event_data = {
                "event_type": "http_response",
                "resource_type": "api_endpoint",
                "resource_id": f"{request.method} {request.path}",
                "action": "response",
                "http_method": request.method,
                "url_path": request.path,
                "status_code": response.status_code,
                "response_size": len(response.get_data()) if response.get_data() else 0,
                "duration_ms": round(duration_ms, 3),
                "severity": self._get_response_severity(response.status_code),
                "is_critical": response.status_code >= 500,
            }

            # Add response body if configured
            if self.include_response_body:
                event_data["response_body"] = self._serialize_body(response.get_data())

            # Create generic audit event
            from .events import AuditEvent

            class HTTPResponseEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPResponseEvent(
                event_type="http_response",
                resource_type="api_endpoint",
                resource_id=event_data["resource_id"],
                action="response",
                severity=event_data["severity"],
                is_critical=event_data["is_critical"],
                metadata=event_data,
            )

            self.logger.log_event(event, context)

        except Exception as e:
            self.logger._logger.error(f"Failed to log response event: {e}")

    def _log_error_event(self, exception: Exception, context: AuditContext) -> Any:
        """Log HTTP error event."""
        try:
            event_data = {
                "event_type": "http_error",
                "resource_type": "api_endpoint",
                "resource_id": "error_handler",
                "action": "error",
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "severity": EventSeverity.CRITICAL,
                "is_critical": True,
            }

            from .events import AuditEvent

            class HTTPErrorEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPErrorEvent(
                event_type="http_error",
                resource_type="api_endpoint",
                resource_id="error_handler",
                action="error",
                severity=EventSeverity.CRITICAL,
                is_critical=True,
                metadata=event_data,
            )

            self.logger.log_event(event, context)

        except Exception as e:
            self.logger._logger.error(f"Failed to log error event: {e}")

    def _get_response_severity(self, status_code: int) -> EventSeverity:
        """Get event severity based on HTTP status code."""
        if status_code < 300 or status_code < 400:
            return EventSeverity.LOW
        elif status_code < 500:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.HIGH


class FastAPIAuditMiddleware:
    """Audit middleware for FastAPI applications."""

    def __init__(self, logger: AuditLogger, **kwargs: Any) -> None:
        """
        Initialize FastAPI audit middleware.

        Args:
            logger: Audit logger instance
            **kwargs: Additional middleware configuration
        """
        self.audit_middleware = AuditMiddleware(logger, **kwargs)

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """FastAPI middleware callable."""
        # Check if request should be audited
        if not self.audit_middleware.should_audit_request(request.url.path, request.method):
            return await call_next(request)

        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Extract audit context
        headers = dict(request.headers)
        audit_context = self.audit_middleware.extract_user_context(
            headers=headers,
            remote_addr=str(request.client.host) if request.client else None,
            method=request.method,
            path=request.url.path,
            query_string=str(request.url.query) if request.url.query else None,
        )

        # Log request event
        await self._log_request_event(request, audit_context)

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response event
            await self._log_response_event(request, response, audit_context, duration_ms)

            return response

        except Exception as e:
            # Log error event
            await self._log_error_event(e, audit_context)
            raise

    async def _log_request_event(self, request: Any, context: AuditContext) -> None:
        """Log FastAPI request event."""
        try:
            # Read request body if needed
            request_body = None
            if self.audit_middleware.include_request_body:
                body_bytes = await request.body()
                if body_bytes:
                    request_body = self.audit_middleware._serialize_body(body_bytes)

            event_data = {
                "event_type": "http_request",
                "resource_type": "api_endpoint",
                "resource_id": f"{request.method} {request.url.path}",
                "action": "request",
                "http_method": request.method,
                "url_path": request.url.path,
                "query_parameters": dict(request.query_params) if request.query_params else None,
                "request_body": request_body,
                "severity": EventSeverity.LOW,
                "is_critical": False,
            }

            from .events import AuditEvent

            class HTTPRequestEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPRequestEvent(
                event_type="http_request",
                resource_type="api_endpoint",
                resource_id=event_data["resource_id"],
                action="request",
                severity=EventSeverity.LOW,
                metadata=event_data,
            )

            self.audit_middleware.logger.log_event(event, context)

        except Exception as e:
            self.audit_middleware.logger._logger.error(f"Failed to log FastAPI request event: {e}")

    async def _log_response_event(
        self, request: Any, response: Any, context: AuditContext, duration_ms: float
    ) -> None:
        """Log FastAPI response event."""
        try:
            event_data = {
                "event_type": "http_response",
                "resource_type": "api_endpoint",
                "resource_id": f"{request.method} {request.url.path}",
                "action": "response",
                "http_method": request.method,
                "url_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 3),
                "severity": self._get_response_severity(response.status_code),
                "is_critical": response.status_code >= 500,
            }

            from .events import AuditEvent

            class HTTPResponseEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPResponseEvent(
                event_type="http_response",
                resource_type="api_endpoint",
                resource_id=event_data["resource_id"],
                action="response",
                severity=event_data["severity"],
                is_critical=event_data["is_critical"],
                metadata=event_data,
            )

            self.audit_middleware.logger.log_event(event, context)

        except Exception as e:
            self.audit_middleware.logger._logger.error(f"Failed to log FastAPI response event: {e}")

    async def _log_error_event(self, exception: Exception, context: AuditContext) -> Any:
        """Log FastAPI error event."""
        try:
            event_data = {
                "event_type": "http_error",
                "resource_type": "api_endpoint",
                "resource_id": "error_handler",
                "action": "error",
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "severity": EventSeverity.CRITICAL,
                "is_critical": True,
            }

            from .events import AuditEvent

            class HTTPErrorEvent(AuditEvent):
                def get_resource_details(self) -> dict[str, Any]:
                    return event_data

                def _validate_resource_data(self) -> None:
                    pass

            event = HTTPErrorEvent(
                event_type="http_error",
                resource_type="api_endpoint",
                resource_id="error_handler",
                action="error",
                severity=EventSeverity.CRITICAL,
                is_critical=True,
                metadata=event_data,
            )

            self.audit_middleware.logger.log_event(event, context)

        except Exception as e:
            self.audit_middleware.logger._logger.error(f"Failed to log FastAPI error event: {e}")

    def _get_response_severity(self, status_code: int) -> EventSeverity:
        """Get event severity based on HTTP status code."""
        if status_code < 300 or status_code < 400:
            return EventSeverity.LOW
        elif status_code < 500:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.HIGH


class AuthenticationAuditMiddleware:
    """
    Specialized middleware for authentication and authorization auditing.

    Captures login attempts, token validation, permission checks, and
    security-related events for compliance and security monitoring.
    """

    def __init__(self, logger: AuditLogger) -> None:
        """
        Initialize authentication audit middleware.

        Args:
            logger: Audit logger instance
        """
        self.logger = logger

    def log_login_attempt(
        self,
        user_id: str,
        auth_method: str,
        success: bool,
        ip_address: str | None = None,
        user_agent: str | None = None,
        failure_reason: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Log authentication attempt.

        Args:
            user_id: User identifier
            auth_method: Authentication method used
            success: Whether login was successful
            ip_address: Client IP address
            user_agent: Client user agent
            failure_reason: Reason for failure if unsuccessful
            **kwargs: Additional context

        Returns:
            Event ID
        """
        event = AuthenticationEvent(
            event_type="authentication_attempt",
            resource_type="user",
            resource_id=user_id,
            action="login",
            user_id=user_id,
            auth_method=auth_method,
            ip_address=ip_address,
            user_agent=user_agent,
            login_success=success,
            failure_reason=failure_reason,
            severity=EventSeverity.CRITICAL if not success else EventSeverity.MEDIUM,
            is_critical=not success,
            metadata=kwargs,
        )

        context = AuditContext(user_id=user_id, ip_address=ip_address, user_agent=user_agent)

        return self.logger.log_event(event, context)

    def log_logout(
        self, user_id: str, session_id: str, session_duration: int | None = None, **kwargs: Any
    ) -> str:
        """
        Log user logout.

        Args:
            user_id: User identifier
            session_id: Session identifier
            session_duration: Session duration in seconds
            **kwargs: Additional context

        Returns:
            Event ID
        """
        event = AuthenticationEvent(
            event_type="authentication_logout",
            resource_type="user",
            resource_id=user_id,
            action="logout",
            user_id=user_id,
            session_duration=session_duration,
            login_success=True,
            severity=EventSeverity.LOW,
            metadata=kwargs,
        )

        context = AuditContext(user_id=user_id, session_id=session_id)

        return self.logger.log_event(event, context)

    def log_permission_check(
        self, user_id: str, resource: str, permission: str, granted: bool, **kwargs: Any
    ) -> str:
        """
        Log permission check.

        Args:
            user_id: User identifier
            resource: Resource being accessed
            permission: Permission being checked
            granted: Whether permission was granted
            **kwargs: Additional context

        Returns:
            Event ID
        """
        event = AuthenticationEvent(
            event_type="permission_check",
            resource_type="permission",
            resource_id=resource,
            action="authorize",
            user_id=user_id,
            login_success=granted,
            failure_reason=None if granted else "insufficient_permissions",
            permissions_granted=[permission] if granted else [],
            severity=EventSeverity.MEDIUM,
            is_critical=not granted,
            metadata={"resource": resource, "permission": permission, **kwargs},
        )

        context = AuditContext(user_id=user_id)
        return self.logger.log_event(event, context)


class DatabaseAuditMiddleware:
    """
    Middleware for auditing database operations.

    Captures database queries, modifications, and access patterns
    for compliance and security monitoring.
    """

    def __init__(self, logger: AuditLogger) -> None:
        """
        Initialize database audit middleware.

        Args:
            logger: Audit logger instance
        """
        self.logger = logger

    def log_database_query(
        self,
        query: str,
        user_id: str | None = None,
        database: str | None = None,
        table: str | None = None,
        operation: str = "query",
        duration_ms: float | None = None,
        rows_affected: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Log database query execution.

        Args:
            query: SQL query (sanitized)
            user_id: User executing the query
            database: Database name
            table: Primary table accessed
            operation: Type of operation (SELECT, INSERT, UPDATE, DELETE)
            duration_ms: Query execution time
            rows_affected: Number of rows affected
            **kwargs: Additional context

        Returns:
            Event ID
        """
        # Sanitize query (remove potential sensitive data)
        sanitized_query = self._sanitize_query(query)

        event_data = {
            "event_type": "database_operation",
            "resource_type": "database",
            "resource_id": table or database or "unknown",
            "action": operation.lower(),
            "database": database,
            "table": table,
            "query": sanitized_query,
            "duration_ms": duration_ms,
            "rows_affected": rows_affected,
            "severity": self._get_query_severity(operation),
            "is_critical": operation.upper() in ["DELETE", "DROP", "TRUNCATE"],
            **kwargs,
        }

        from .events import AuditEvent

        class DatabaseEvent(AuditEvent):
            def get_resource_details(self) -> dict[str, Any]:
                return event_data

            def _validate_resource_data(self) -> None:
                pass

        event = DatabaseEvent(
            event_type="database_operation",
            resource_type="database",
            resource_id=event_data["resource_id"],
            action=operation.lower(),
            user_id=user_id,
            severity=event_data["severity"],
            is_critical=event_data["is_critical"],
            metadata=event_data,
        )

        context = AuditContext(user_id=user_id)
        return self.logger.log_event(event, context)

    def _sanitize_query(self, query: str) -> str:
        """Sanitize SQL query for logging."""
        # Remove potential sensitive data patterns
        import re

        # Replace string literals with placeholders
        sanitized = re.sub(r"'[^']*'", "'***'", query)

        # Replace numeric literals in WHERE clauses (potential IDs)
        sanitized = re.sub(r"\b(\w+\s*=\s*)\d+\b", r"\1***", sanitized)

        # Limit query length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "...<truncated>"

        return sanitized

    def _get_query_severity(self, operation: str) -> EventSeverity:
        """Get severity based on database operation type."""
        operation = operation.upper()
        if operation in ["SELECT"]:
            return EventSeverity.LOW
        elif operation in ["INSERT", "UPDATE"]:
            return EventSeverity.MEDIUM
        elif operation in ["DELETE", "DROP", "TRUNCATE"]:
            return EventSeverity.HIGH
        else:
            return EventSeverity.MEDIUM
