"""
Middleware for rate limiting in web frameworks.

Provides middleware implementations for Flask, FastAPI, and Django
to automatically apply rate limiting to API endpoints.
"""

from collections.abc import Callable
from typing import Any

from .config import RateLimitConfig, RateLimitTier
from .exceptions import RateLimitExceeded
from .manager import RateLimitContext, RateLimitManager


class RateLimitMiddleware:
    """
    Generic rate limiting middleware.

    Can be adapted for different web frameworks.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        get_user_id: Callable | None = None,
        get_api_key: Callable | None = None,
        get_user_tier: Callable | None = None,
        exempt_paths: list[str] | None = None,
        rate_limit_headers: bool = True,
    ):
        self.manager = RateLimitManager(config)
        self.config = config
        self.get_user_id = get_user_id or self._default_get_user_id
        self.get_api_key = get_api_key or self._default_get_api_key
        self.get_user_tier = get_user_tier or self._default_get_user_tier
        self.exempt_paths = exempt_paths or []
        self.rate_limit_headers = rate_limit_headers

    def _default_get_user_id(self, request: Any) -> str | None:
        """Default method to extract user ID from request."""
        # Try common authentication patterns
        if hasattr(request, "user") and hasattr(request.user, "id"):
            return str(request.user.id)
        elif hasattr(request, "headers"):
            return request.headers.get("X-User-ID")
        return None

    def _default_get_api_key(self, request: Any) -> str | None:
        """Default method to extract API key from request."""
        if hasattr(request, "headers"):
            return (
                request.headers.get("X-API-Key")
                or request.headers.get("Authorization", "").replace("Bearer ", "")
                or request.headers.get("Api-Key")
            )
        return None

    def _default_get_user_tier(self, request: Any) -> "RateLimitTier":
        """Default method to determine user tier."""
        # Default to basic tier, override in production
        return RateLimitTier.BASIC

    def _build_context(self, request: Any) -> "RateLimitContext":
        """Build rate limit context from request."""
        context = RateLimitContext()

        # Extract user information
        context.user_id = self.get_user_id(request)
        context.api_key = self.get_api_key(request)
        context.user_tier = self.get_user_tier(request)

        # Extract request information
        context.ip_address = self._get_ip_address(request)
        context.endpoint = self._get_endpoint(request)
        context.method = self._get_method(request)

        return context

    def _get_ip_address(self, request: Any) -> str | None:
        """Extract IP address from request."""
        if hasattr(request, "remote_addr"):
            return request.remote_addr
        elif hasattr(request, "META"):  # Django
            return (
                request.META.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
                or request.META.get("HTTP_X_REAL_IP")
                or request.META.get("REMOTE_ADDR")
            )
        elif hasattr(request, "headers"):  # FastAPI
            return (
                request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or request.headers.get("X-Real-IP")
                or getattr(request.client, "host", None)
                if hasattr(request, "client")
                else None
            )
        return None

    def _get_endpoint(self, request: Any) -> str | None:
        """Extract endpoint from request."""
        if hasattr(request, "path"):
            return request.path
        elif hasattr(request, "url"):
            return str(request.url.path) if hasattr(request.url, "path") else str(request.url)
        return None

    def _get_method(self, request: Any) -> str | None:
        """Extract HTTP method from request."""
        if hasattr(request, "method"):
            return request.method
        return None

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        for exempt_path in self.exempt_paths:
            if path.startswith(exempt_path):
                return True
        return False

    def _create_rate_limit_headers(self, statuses: list[Any]) -> dict[str, str]:
        """Create rate limit headers for response."""
        if not self.rate_limit_headers or not statuses:
            return {}

        # Use the most restrictive rate limit for headers
        most_restrictive = min(statuses, key=lambda s: s.remaining)

        headers = {}
        prefix = self.config.rate_limit_header_prefix

        headers[f"{prefix}-Limit"] = str(most_restrictive.limit)
        headers[f"{prefix}-Remaining"] = str(most_restrictive.remaining)

        if most_restrictive.reset_time:
            headers[f"{prefix}-Reset"] = str(int(most_restrictive.reset_time.timestamp()))

        if hasattr(most_restrictive, "retry_after") and most_restrictive.retry_after:
            headers["Retry-After"] = str(most_restrictive.retry_after)

        return headers

    def _create_error_response(self, exception: "RateLimitExceeded") -> dict[str, Any]:
        """Create error response for rate limit exceeded."""
        response_data = exception.to_dict()

        # Add rate limit headers
        headers = {}
        if exception.retry_after:
            headers["Retry-After"] = str(exception.retry_after)

        prefix = self.config.rate_limit_header_prefix
        headers[f"{prefix}-Limit"] = str(exception.limit)
        headers[f"{prefix}-Remaining"] = "0"

        return {"status_code": 429, "body": response_data, "headers": headers}


class FlaskRateLimitMiddleware(RateLimitMiddleware):
    """Flask-specific rate limiting middleware."""

    def __init__(self, app: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app = app
        if app:
            self.init_app(app)

    def init_app(self, app: Any) -> None:
        """Initialize Flask app with rate limiting."""
        self.app = app
        app.before_request(self.before_request)
        app.after_request(self.after_request)

        # Add error handler for rate limit exceptions
        @app.errorhandler(RateLimitExceeded)
        def handle_rate_limit_exceeded(e: "RateLimitExceeded") -> Any:
            from flask import jsonify

            error_response = self._create_error_response(e)
            response = jsonify(error_response["body"])
            response.status_code = error_response["status_code"]

            for header, value in error_response["headers"].items():
                response.headers[header] = value

            return response

    def before_request(self) -> Any | None:
        """Flask before_request handler."""
        from flask import g, request

        # Skip exempt paths
        if self._is_exempt_path(request.path):
            return

        # Build context and check rate limits
        context = self._build_context(request)

        try:
            statuses = self.manager.check_rate_limit(context)
            g.rate_limit_statuses = statuses
        except RateLimitExceeded:
            # Let the error handler deal with it
            raise

    def after_request(self, response) -> Any:
        """Flask after_request handler."""
        from flask import g

        # Add rate limit headers if available
        if hasattr(g, "rate_limit_statuses"):
            headers = self._create_rate_limit_headers(g.rate_limit_statuses)
            for header, value in headers.items():
                response.headers[header] = value

        return response


class FastAPIRateLimitMiddleware(RateLimitMiddleware):
    """FastAPI-specific rate limiting middleware."""

    async def __call__(self, request, call_next) -> Any:
        """FastAPI middleware call."""
        # Skip exempt paths
        if self._is_exempt_path(str(request.url.path)):
            return await call_next(request)

        # Build context and check rate limits
        context = self._build_context(request)

        try:
            statuses = self.manager.check_rate_limit(context)
        except RateLimitExceeded as e:
            from fastapi.responses import JSONResponse

            error_response = self._create_error_response(e)
            return JSONResponse(
                status_code=error_response["status_code"],
                content=error_response["body"],
                headers=error_response["headers"],
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        headers = self._create_rate_limit_headers(statuses)
        for header, value in headers.items():
            response.headers[header] = value

        return response


class DjangoRateLimitMiddleware(RateLimitMiddleware):
    """Django-specific rate limiting middleware."""

    def __init__(self, get_response, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.get_response = get_response

    def __call__(self, request) -> Any:
        """Django middleware call."""
        # Skip exempt paths
        if self._is_exempt_path(request.path):
            return self.get_response(request)

        # Build context and check rate limits
        context = self._build_context(request)

        try:
            statuses = self.manager.check_rate_limit(context)
        except RateLimitExceeded as e:
            from django.http import JsonResponse

            error_response = self._create_error_response(e)
            response = JsonResponse(error_response["body"], status=error_response["status_code"])

            for header, value in error_response["headers"].items():
                response[header] = value

            return response

        # Process request
        response = self.get_response(request)

        # Add rate limit headers
        headers = self._create_rate_limit_headers(statuses)
        for header, value in headers.items():
            response[header] = value

        return response


class CustomFrameworkMiddleware(RateLimitMiddleware):
    """
    Custom framework middleware template.

    Extend this class for other web frameworks.
    """

    def process_request(self, request) -> Any:
        """Process incoming request."""
        # Skip exempt paths
        if self._is_exempt_path(self._get_endpoint(request)):
            return None

        # Build context and check rate limits
        context = self._build_context(request)

        try:
            statuses = self.manager.check_rate_limit(context)
            # Store statuses for use in response processing
            request._rate_limit_statuses = statuses
        except RateLimitExceeded as e:
            # Return error response
            return self._create_error_response(e)

        return None

    def process_response(self, request, response) -> Any:
        """Process outgoing response."""
        # Add rate limit headers if available
        if hasattr(request, "_rate_limit_statuses"):
            headers = self._create_rate_limit_headers(request._rate_limit_statuses)
            for header, value in headers.items():
                response.headers[header] = value

        return response


def create_middleware(framework: str, config: RateLimitConfig, **kwargs) -> RateLimitMiddleware:
    """
    Factory function to create framework-specific middleware.

    Args:
        framework: Web framework name ("flask", "fastapi", "django")
        config: Rate limiting configuration
        **kwargs: Additional middleware options
    """
    framework = framework.lower()

    if framework == "flask":
        return FlaskRateLimitMiddleware(config=config, **kwargs)
    elif framework == "fastapi":
        return FastAPIRateLimitMiddleware(config=config, **kwargs)
    elif framework == "django":
        return DjangoRateLimitMiddleware(config=config, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Example usage functions for different frameworks


def setup_flask_rate_limiting(app, config: RateLimitConfig, **kwargs: Any) -> Any:
    """Setup rate limiting for Flask app."""
    middleware = FlaskRateLimitMiddleware(app=app, config=config, **kwargs)
    return middleware


def setup_fastapi_rate_limiting(app, config: RateLimitConfig, **kwargs: Any) -> Any:
    """Setup rate limiting for FastAPI app."""
    middleware = FastAPIRateLimitMiddleware(config=config, **kwargs)
    app.add_middleware(lambda request, call_next: middleware(request, call_next))
    return middleware


def setup_django_rate_limiting(config: RateLimitConfig, **kwargs: Any) -> Any:
    """Setup rate limiting for Django (add to MIDDLEWARE setting)."""

    # Return the middleware class that Django will instantiate
    def middleware_factory(get_response) -> Any:
        return DjangoRateLimitMiddleware(get_response, config=config, **kwargs)

    return middleware_factory
