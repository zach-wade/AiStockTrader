"""
Security middleware for the AI Trading System.

Provides comprehensive security middleware including:
- HTTPS/TLS enforcement
- Security headers
- Request validation
- CORS handling
- IP filtering
"""

from .https_enforcement import (
    HTTPSEnforcementMiddleware,
    HTTPSRedirectResponse,
    TLSConfigurationError,
    TLSValidationError,
)
from .security_hardening import (
    SecurityHardeningMiddleware,
    SecurityHeadersConfig,
    SecurityViolation,
)

__all__ = [
    # HTTPS enforcement
    "HTTPSEnforcementMiddleware",
    "HTTPSRedirectResponse",
    "TLSConfigurationError",
    "TLSValidationError",
    # Security hardening
    "SecurityHardeningMiddleware",
    "SecurityHeadersConfig",
    "SecurityViolation",
]
