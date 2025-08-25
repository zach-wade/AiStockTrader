"""
Decorators for rate limiting in the AI Trading System.

Provides easy-to-use decorators for applying rate limits to functions,
methods, and API endpoints.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from .config import RateLimitAlgorithm, RateLimitConfig, RateLimitRule, RateLimitTier, TimeWindow
from .exceptions import RateLimitExceeded
from .manager import RateLimitContext, RateLimitManager

# Global rate limit manager instance
_rate_limit_manager: RateLimitManager | None = None


def initialize_rate_limiting(config: RateLimitConfig) -> None:
    """Initialize the global rate limit manager."""
    global _rate_limit_manager
    _rate_limit_manager = RateLimitManager(config)


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager."""
    if _rate_limit_manager is None:
        # Initialize with default config if not already initialized
        default_config = RateLimitConfig.from_env()
        initialize_rate_limiting(default_config)
    if _rate_limit_manager is None:
        raise RuntimeError("Rate limit manager not initialized")
    return _rate_limit_manager


def rate_limit(
    limit: int,
    window: str | int,
    algorithm: str = "token_bucket",
    per: str = "user",
    burst_allowance: int | None = None,
    key_func: Callable[..., str] | None = None,
    error_message: str | None = None,
    rule_types: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    General-purpose rate limiting decorator.

    Args:
        limit: Number of requests allowed
        window: Time window (e.g., "1min", "1h", 60)
        algorithm: Rate limiting algorithm ("token_bucket", "sliding_window", "fixed_window")
        per: Rate limit scope ("user", "ip", "api_key", "global")
        burst_allowance: Extra requests allowed in burst
        key_func: Custom function to generate rate limit key
        error_message: Custom error message
        rule_types: Specific rule types to check
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limit_manager()

            # Build rate limit context
            context = _build_context_from_function(func, args, kwargs, per, key_func)

            # Create custom rule if needed
            if not rule_types:
                # Use the rate limit parameters to create a temporary rule
                from .config import RateLimitAlgorithm

                algorithm_map = {
                    "token_bucket": RateLimitAlgorithm.TOKEN_BUCKET,
                    "sliding_window": RateLimitAlgorithm.SLIDING_WINDOW,
                    "fixed_window": RateLimitAlgorithm.FIXED_WINDOW,
                }

                custom_rule = RateLimitRule(
                    limit=limit,
                    window=TimeWindow(window),
                    algorithm=algorithm_map.get(algorithm.lower(), RateLimitAlgorithm.TOKEN_BUCKET),
                    burst_allowance=burst_allowance,
                    identifier=f"custom:{func.__name__}",
                )

                # Temporarily add this rule to the manager
                limiter_id = f"custom:{func.__name__}"
                from .algorithms import create_rate_limiter

                manager._limiters[limiter_id] = create_rate_limiter(custom_rule)

                # Check the custom rate limit
                identifier = _build_custom_identifier(context, per, func.__name__)
                result = manager._check_limiter(limiter_id, identifier, 1)

                if not result.allowed:
                    message = error_message or f"Rate limit exceeded for {func.__name__}"
                    raise RateLimitExceeded(
                        message=message,
                        limit=limit,
                        window_size=str(TimeWindow(window)),
                        current_count=result.current_count,
                        retry_after=result.retry_after,
                    )
            else:
                # Use existing rule types
                statuses = manager.check_rate_limit(context, 1, rule_types)
                # If we get here, all rate limits passed

            return func(*args, **kwargs)

        return wrapper

    return decorator


def trading_rate_limit(
    action: str,
    symbol: str | None = None,
    per_user: bool = True,
    per_symbol: bool = False,
    error_message: str | None = None,
) -> Callable[..., Any]:
    """
    Trading-specific rate limiting decorator.

    Args:
        action: Trading action (e.g., "submit_order", "cancel_order")
        symbol: Optional symbol for symbol-specific limiting
        per_user: Apply rate limit per user
        per_symbol: Apply rate limit per symbol
        error_message: Custom error message
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limit_manager()

            # Build trading context
            context = _build_trading_context(func, args, kwargs, action, symbol)

            # Check trading rate limits
            try:
                statuses = manager.check_rate_limit(context, 1, [f"trading:{action}"])
                # If we get here, rate limit passed
            except RateLimitExceeded as e:
                if error_message:
                    e.message = error_message
                raise

            return func(*args, **kwargs)

        return wrapper

    return decorator


def api_rate_limit(
    tier: str | RateLimitTier = RateLimitTier.BASIC,
    rule_types: list[str] | None = None,
    require_auth: bool = True,
    error_message: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    API endpoint rate limiting decorator.

    Args:
        tier: User tier for rate limiting
        rule_types: Specific rule types to check
        require_auth: Whether authentication is required
        error_message: Custom error message
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limit_manager()

            # Build API context
            context = _build_api_context(func, args, kwargs, tier, require_auth)

            # Check API rate limits
            check_rule_types = rule_types or ["api_requests"]

            try:
                statuses = manager.check_rate_limit(context, 1, check_rule_types)
                # If we get here, rate limit passed
            except RateLimitExceeded as e:
                if error_message:
                    e.message = error_message
                raise

            return func(*args, **kwargs)

        return wrapper

    return decorator


def ip_rate_limit(
    limit: int,
    window: str | int,
    algorithm: str = "fixed_window",
    error_message: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    IP-based rate limiting decorator.

    Args:
        limit: Number of requests allowed per IP
        window: Time window
        algorithm: Rate limiting algorithm
        error_message: Custom error message
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_rate_limit_manager()

            # Build IP context
            context = _build_ip_context(func, args, kwargs)

            # Create IP-specific rule

            algorithm_map = {
                "token_bucket": RateLimitAlgorithm.TOKEN_BUCKET,
                "sliding_window": RateLimitAlgorithm.SLIDING_WINDOW,
                "fixed_window": RateLimitAlgorithm.FIXED_WINDOW,
            }

            custom_rule = RateLimitRule(
                limit=limit,
                window=TimeWindow(window),
                algorithm=algorithm_map.get(algorithm.lower(), RateLimitAlgorithm.TOKEN_BUCKET),
                identifier=f"ip:{func.__name__}",
            )

            # Temporarily add this rule
            limiter_id = f"ip:{func.__name__}"
            from .algorithms import create_rate_limiter

            manager._limiters[limiter_id] = create_rate_limiter(custom_rule)

            # Check rate limit
            identifier = f"ip:{context.ip_address}:{func.__name__}"
            result = manager._check_limiter(limiter_id, identifier, 1)

            if not result.allowed:
                message = error_message or f"IP rate limit exceeded for {func.__name__}"
                raise RateLimitExceeded(
                    message=message,
                    limit=limit,
                    window_size=str(TimeWindow(window)),
                    current_count=result.current_count,
                    retry_after=result.retry_after,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def no_rate_limit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to bypass rate limiting for specific functions.
    Useful for admin or system operations.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Simply call the function without any rate limiting
        return func(*args, **kwargs)

    wrapper._no_rate_limit = True  # Mark function as exempt
    return wrapper


def _build_context_from_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    per: str,
    key_func: Callable[..., str] | None,
) -> RateLimitContext:
    """Build rate limit context from function parameters."""
    context = RateLimitContext()

    # Try to extract context from function parameters
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # Look for common parameter names
    for param_name, value in bound_args.arguments.items():
        if param_name in ["user_id", "user"]:
            context.user_id = str(value) if value else None
        elif param_name in ["api_key", "key"]:
            context.api_key = str(value) if value else None
        elif param_name in ["ip_address", "ip", "remote_addr"]:
            context.ip_address = str(value) if value else None
        elif param_name in ["request"]:
            # Flask/FastAPI request object
            if hasattr(value, "remote_addr"):
                context.ip_address = value.remote_addr
            if hasattr(value, "headers"):
                context.api_key = value.headers.get("X-API-Key")

    # Use custom key function if provided
    if key_func:
        custom_key = key_func(*args, **kwargs)
        if per == "user":
            context.user_id = str(custom_key)
        elif per == "ip":
            context.ip_address = str(custom_key)
        elif per == "api_key":
            context.api_key = str(custom_key)

    return context


def _build_trading_context(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    action: str,
    symbol: str | None,
) -> RateLimitContext:
    """Build trading-specific rate limit context."""
    context = _build_context_from_function(func, args, kwargs, "user", None)
    context.trading_action = action

    # Extract symbol if not provided
    if not symbol:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, value in bound_args.arguments.items():
            if param_name in ["symbol", "ticker", "instrument"]:
                context.symbol = str(value) if value else None
                break
    else:
        context.symbol = symbol

    return context


def _build_api_context(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    tier: str | RateLimitTier,
    require_auth: bool,
) -> RateLimitContext:
    """Build API-specific rate limit context."""
    context = _build_context_from_function(func, args, kwargs, "user", None)

    # Set tier
    if isinstance(tier, str):
        context.user_tier = RateLimitTier(tier.lower())
    else:
        context.user_tier = tier

    # Set endpoint information
    context.endpoint = func.__name__
    context.method = "POST"  # Default, could be extracted from request object

    return context


def _build_ip_context(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> RateLimitContext:
    """Build IP-specific rate limit context."""
    context = _build_context_from_function(func, args, kwargs, "ip", None)

    # If no IP found in parameters, try to get from common sources
    if not context.ip_address:
        # Check if first argument looks like a request object
        if args and hasattr(args[0], "remote_addr"):
            context.ip_address = args[0].remote_addr
        elif args and hasattr(args[0], "META"):  # Django request
            context.ip_address = args[0].META.get("REMOTE_ADDR")

    return context


def _build_custom_identifier(context: RateLimitContext, per: str, func_name: str) -> str:
    """Build custom identifier for rate limiting."""
    parts = [func_name]

    if per == "user" and context.user_id:
        parts.append(context.user_id)
    elif per == "ip" and context.ip_address:
        parts.append(context.ip_address)
    elif per == "api_key" and context.api_key:
        parts.append(context.api_key)
    elif per == "global":
        parts.append("global")

    return ":".join(parts)


# Convenience aliases
limit = rate_limit  # Alias for simpler usage
trading_limit = trading_rate_limit
api_limit = api_rate_limit
ip_limit = ip_rate_limit
