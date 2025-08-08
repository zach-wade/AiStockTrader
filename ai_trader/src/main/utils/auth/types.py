"""
Authentication Types

Data classes and enums for authentication and credential validation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class CredentialType(Enum):
    """Types of credentials that can be validated."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH_TOKEN = "oauth_token"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    WEBHOOK_SECRET = "webhook_secret"
    PRIVATE_KEY = "private_key"
    PUBLIC_KEY = "public_key"
    CERTIFICATE = "certificate"
    CUSTOM = "custom"


class ValidationResult(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    WEAK = "weak"
    MALFORMED = "malformed"
    UNKNOWN = "unknown"


@dataclass
class CredentialValidation:
    """Result of credential validation."""
    credential_type: CredentialType
    status: ValidationResult
    is_valid: bool
    strength_score: int  # 0-100
    issues: List[str]
    recommendations: List[str]
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'credential_type': self.credential_type.value,
            'status': self.status.value,
            'is_valid': self.is_valid,
            'strength_score': self.strength_score,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata or {}
        }