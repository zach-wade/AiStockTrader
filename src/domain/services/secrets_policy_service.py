"""
Secrets Management Policy Service - Domain Layer

This service contains all business logic related to secrets management,
including access policies, rotation rules, and security requirements.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class SecretType(Enum):
    """Types of secrets in the system."""

    API_KEY = "api_key"
    API_SECRET = "api_secret"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    WEBHOOK_SECRET = "webhook_secret"
    SERVICE_TOKEN = "service_token"
    CERTIFICATE = "certificate"


class SecretAccessLevel(Enum):
    """Access levels for secrets."""

    PUBLIC = 0  # Not really a secret
    BASIC = 1  # Basic authentication
    STANDARD = 2  # Standard security
    ELEVATED = 3  # Elevated security requirements
    CRITICAL = 4  # Critical secrets requiring maximum security


class RotationFrequency(Enum):
    """Secret rotation frequencies."""

    NEVER = -1
    DAILY = 1
    WEEKLY = 7
    MONTHLY = 30
    QUARTERLY = 90
    ANNUALLY = 365


@dataclass
class SecretMetadata:
    """Metadata for a secret."""

    name: str
    type: SecretType
    created_at: datetime
    last_rotated: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0
    environment: str = "production"
    owner: str | None = None
    expires_at: datetime | None = None


@dataclass
class SecretAccessContext:
    """Context for secret access requests."""

    requester_id: str
    requester_role: str
    purpose: str
    environment: str
    source_ip: str | None = None
    timestamp: datetime | None = None


@dataclass
class SecretPolicy:
    """Policy for a specific secret."""

    access_level: SecretAccessLevel
    rotation_frequency: RotationFrequency
    allowed_roles: set[str]
    allowed_environments: set[str]
    require_mfa: bool
    audit_access: bool
    auto_rotate: bool
    max_age_days: int | None = None


class SecretsManagementPolicy:
    """
    Domain service for secrets management policies.

    Contains all business logic for secret access control, rotation policies,
    and security requirements extracted from infrastructure layer.
    """

    # Business rules for secret types
    SECRET_POLICIES = {
        SecretType.API_KEY: SecretPolicy(
            access_level=SecretAccessLevel.STANDARD,
            rotation_frequency=RotationFrequency.QUARTERLY,
            allowed_roles={"developer", "service", "admin"},
            allowed_environments={"development", "staging", "production"},
            require_mfa=False,
            audit_access=True,
            auto_rotate=True,
            max_age_days=90,
        ),
        SecretType.API_SECRET: SecretPolicy(
            access_level=SecretAccessLevel.ELEVATED,
            rotation_frequency=RotationFrequency.MONTHLY,
            allowed_roles={"service", "admin"},
            allowed_environments={"production"},
            require_mfa=True,
            audit_access=True,
            auto_rotate=True,
            max_age_days=30,
        ),
        SecretType.DATABASE_PASSWORD: SecretPolicy(
            access_level=SecretAccessLevel.CRITICAL,
            rotation_frequency=RotationFrequency.MONTHLY,
            allowed_roles={"service", "admin", "database_admin"},
            allowed_environments={"production", "staging"},
            require_mfa=True,
            audit_access=True,
            auto_rotate=True,
            max_age_days=30,
        ),
        SecretType.ENCRYPTION_KEY: SecretPolicy(
            access_level=SecretAccessLevel.CRITICAL,
            rotation_frequency=RotationFrequency.ANNUALLY,
            allowed_roles={"admin", "security_admin"},
            allowed_environments={"production"},
            require_mfa=True,
            audit_access=True,
            auto_rotate=False,
            max_age_days=365,
        ),
        SecretType.JWT_SECRET: SecretPolicy(
            access_level=SecretAccessLevel.CRITICAL,
            rotation_frequency=RotationFrequency.QUARTERLY,
            allowed_roles={"service", "admin"},
            allowed_environments={"production", "staging"},
            require_mfa=True,
            audit_access=True,
            auto_rotate=True,
            max_age_days=90,
        ),
    }

    # High-risk operations requiring additional checks
    HIGH_RISK_OPERATIONS = {"rotate_secret", "delete_secret", "export_secret", "share_secret"}

    def determine_secret_access_level(
        self, secret_type: SecretType, context: dict[str, Any]
    ) -> SecretAccessLevel:
        """
        Determine access requirements for secrets based on type and context.

        Args:
            secret_type: Type of secret
            context: Additional context for access determination

        Returns:
            Required access level for the secret
        """
        base_policy = self.SECRET_POLICIES.get(
            secret_type,
            SecretPolicy(
                access_level=SecretAccessLevel.STANDARD,
                rotation_frequency=RotationFrequency.QUARTERLY,
                allowed_roles={"admin"},
                allowed_environments={"production"},
                require_mfa=True,
                audit_access=True,
                auto_rotate=False,
                max_age_days=90,
            ),
        )

        # Elevate access level based on context
        if context.get("environment") == "production":
            if base_policy.access_level == SecretAccessLevel.BASIC:
                return SecretAccessLevel.STANDARD
            elif base_policy.access_level == SecretAccessLevel.STANDARD:
                return SecretAccessLevel.ELEVATED

        # Additional elevation for sensitive operations
        if context.get("operation") in self.HIGH_RISK_OPERATIONS:
            return SecretAccessLevel.CRITICAL

        return base_policy.access_level

    def evaluate_secret_rotation_need(self, metadata: SecretMetadata) -> bool:
        """
        Evaluate if secret needs rotation based on business rules.

        Args:
            metadata: Secret metadata

        Returns:
            True if secret should be rotated
        """
        policy = self.SECRET_POLICIES.get(metadata.type)
        if not policy or not policy.auto_rotate:
            return False

        # Check if secret has never been rotated
        if metadata.last_rotated is None:
            days_since_creation = (datetime.now() - metadata.created_at).days
            return days_since_creation >= policy.rotation_frequency.value

        # Check rotation frequency
        days_since_rotation = (datetime.now() - metadata.last_rotated).days
        if days_since_rotation >= policy.rotation_frequency.value:
            return True

        # Check max age
        if policy.max_age_days:
            secret_age = (datetime.now() - metadata.created_at).days
            if secret_age >= policy.max_age_days:
                return True

        # Check for expiration
        if metadata.expires_at and datetime.now() >= metadata.expires_at:
            return True

        # Check for suspicious access patterns
        if self._has_suspicious_access_pattern(metadata):
            return True

        return False

    def can_access_secret(
        self, secret_metadata: SecretMetadata, context: SecretAccessContext
    ) -> bool:
        """
        Determine if secret access should be allowed.

        Args:
            secret_metadata: Metadata of the secret
            context: Access request context

        Returns:
            True if access should be allowed
        """
        policy = self.SECRET_POLICIES.get(secret_metadata.type)
        if not policy:
            # Default deny for unknown secret types
            return False

        # Check role authorization
        if context.requester_role not in policy.allowed_roles:
            return False

        # Check environment authorization
        if context.environment not in policy.allowed_environments:
            return False

        # Check if secret is expired
        if secret_metadata.expires_at and datetime.now() >= secret_metadata.expires_at:
            return False

        # Check ownership for user-specific secrets
        if secret_metadata.owner and secret_metadata.owner != context.requester_id:
            # Allow admins to access any secret
            if context.requester_role != "admin":
                return False

        # Additional checks for production environment
        if context.environment == "production":
            # Require specific purpose for production access
            valid_purposes = ["service_authentication", "deployment", "emergency_access"]
            if context.purpose not in valid_purposes:
                return False

        return True

    def get_rotation_schedule(self, secret_type: SecretType) -> timedelta:
        """
        Get the rotation schedule for a secret type.

        Args:
            secret_type: Type of secret

        Returns:
            Timedelta representing rotation interval
        """
        policy = self.SECRET_POLICIES.get(secret_type)
        if not policy or policy.rotation_frequency == RotationFrequency.NEVER:
            return timedelta(days=365 * 10)  # 10 years = effectively never

        return timedelta(days=policy.rotation_frequency.value)

    def requires_mfa(self, secret_type: SecretType, operation: str) -> bool:
        """
        Determine if MFA is required for secret access.

        Args:
            secret_type: Type of secret
            operation: Operation being performed

        Returns:
            True if MFA is required
        """
        policy = self.SECRET_POLICIES.get(secret_type)
        if not policy:
            # Default to requiring MFA for unknown types
            return True

        # Always require MFA for high-risk operations
        if operation in self.HIGH_RISK_OPERATIONS:
            return True

        return policy.require_mfa

    def should_audit_access(self, secret_type: SecretType) -> bool:
        """
        Determine if secret access should be audited.

        Args:
            secret_type: Type of secret

        Returns:
            True if access should be audited
        """
        policy = self.SECRET_POLICIES.get(secret_type)
        return policy.audit_access if policy else True

    def get_secret_complexity_requirements(self, secret_type: SecretType) -> dict[str, Any]:
        """
        Get complexity requirements for generating new secrets.

        Args:
            secret_type: Type of secret

        Returns:
            Dictionary of complexity requirements
        """
        base_requirements: dict[str, Any] = {
            "min_length": 32,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special": True,
            "avoid_ambiguous": True,
        }

        # Specific requirements by type
        if secret_type == SecretType.API_KEY:
            base_requirements["min_length"] = 40
            base_requirements["format"] = "alphanumeric"
        elif secret_type == SecretType.DATABASE_PASSWORD:
            base_requirements["min_length"] = 20
            base_requirements["max_length"] = 128
        elif secret_type == SecretType.ENCRYPTION_KEY:
            base_requirements["min_length"] = 256
            base_requirements["format"] = "base64"
        elif secret_type == SecretType.JWT_SECRET:
            base_requirements["min_length"] = 512
            base_requirements["format"] = "base64url"

        return base_requirements

    def _has_suspicious_access_pattern(self, metadata: SecretMetadata) -> bool:
        """
        Check for suspicious access patterns that might indicate compromise.

        Args:
            metadata: Secret metadata

        Returns:
            True if suspicious pattern detected
        """
        # High access frequency might indicate compromise
        if metadata.last_accessed:
            time_since_access = datetime.now() - metadata.last_accessed
            if time_since_access.seconds < 60 and metadata.access_count > 10:
                return True

        # Sudden spike in access count
        if metadata.access_count > 1000:
            return True

        # Access from unexpected environment
        if metadata.environment == "production" and metadata.access_count < 10:
            # New production secret with low access might be suspicious
            secret_age = (datetime.now() - metadata.created_at).days
            if secret_age > 30:
                return True

        return False
