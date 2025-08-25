"""
Security module for the AI Trading System.

Provides comprehensive security infrastructure including:
- Secure credential management with encryption and rotation
- Input validation and sanitization
- Security hardening with rate limiting and throttling
- Request signing and authentication
- Security headers and defense mechanisms
"""

from .hardening import (
    InvalidTokenError,
    RateLimitExceeded,
    RateLimitRule,
    RequestSigner,
    SecurityConfig,
    SecurityError,
    SecurityHardening,
    SecurityHeaders,
    ThrottlingError,
    create_trading_security_config,
    secure_endpoint,
)
from .input_sanitizer import InputSanitizer, SanitizationError
from .secrets import (
    SecretConfig,
    SecretEncryptionError,
    SecretNotFoundError,
    SecretProvider,
    SecretProviderError,
    SecretRateLimitError,
    SecretsManager,
)
from .validation import (
    SchemaValidationError,
    SchemaValidator,
    SecurityValidationError,
    SecurityValidator,
    TradingInputValidator,
    ValidationError,
    check_required,
    sanitize_input,
)

__all__ = [
    # Secrets management
    "SecretsManager",
    "SecretConfig",
    "SecretProvider",
    "SecretNotFoundError",
    "SecretProviderError",
    "SecretEncryptionError",
    "SecretRateLimitError",
    # Input validation
    "ValidationError",
    "SchemaValidationError",
    "SecurityValidationError",
    "sanitize_input",
    "check_required",
    "SecurityValidator",
    "SchemaValidator",
    "TradingInputValidator",
    # Input sanitization
    "InputSanitizer",
    "SanitizationError",
    # Security hardening
    "SecurityHardening",
    "SecurityConfig",
    "RateLimitRule",
    "SecurityError",
    "RateLimitExceeded",
    "ThrottlingError",
    "InvalidTokenError",
    "SecurityHeaders",
    "RequestSigner",
    "secure_endpoint",
    "create_trading_security_config",
]
