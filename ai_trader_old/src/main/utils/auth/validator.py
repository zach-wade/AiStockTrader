"""
Main Credential Validator

Main credential validation orchestrator.
"""

# Standard library imports
import logging
from typing import Any

from .security_checks import SecurityChecker
from .types import CredentialType, CredentialValidation, ValidationResult
from .validators import CredentialValidators

logger = logging.getLogger(__name__)


class CredentialValidator:
    """
    Comprehensive credential validation system.

    Validates various types of credentials including API keys, tokens,
    and certificates. Provides security analysis and recommendations.
    """

    def __init__(self):
        """Initialize credential validator."""
        self.validators = CredentialValidators()
        self.security_checker = SecurityChecker()
        logger.info("Credential validator initialized")

    def validate_credential(
        self,
        credential: str,
        credential_type: CredentialType,
        additional_checks: dict[str, Any] | None = None,
    ) -> CredentialValidation:
        """
        Validate a credential of the specified type.

        Args:
            credential: The credential to validate
            credential_type: Type of credential
            additional_checks: Additional validation parameters

        Returns:
            CredentialValidation result
        """
        issues = []
        recommendations = []
        strength_score = 0
        status = ValidationResult.UNKNOWN
        expires_at = None
        metadata = {}

        try:
            # Type-specific validation
            if credential_type == CredentialType.API_KEY:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_api_key(credential)
                )
            elif credential_type == CredentialType.JWT_TOKEN:
                strength_score, type_issues, type_recommendations, expires_at, metadata = (
                    self.validators.validate_jwt_token(credential)
                )
            elif credential_type == CredentialType.OAUTH_TOKEN:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_oauth_token(credential)
                )
            elif credential_type == CredentialType.BASIC_AUTH:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_basic_auth(credential)
                )
            elif credential_type == CredentialType.BEARER_TOKEN:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_bearer_token(credential)
                )
            elif credential_type == CredentialType.WEBHOOK_SECRET:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_webhook_secret(credential)
                )
            else:
                strength_score, type_issues, type_recommendations = (
                    self.validators.validate_generic(credential)
                )

            issues.extend(type_issues)
            recommendations.extend(type_recommendations)

            # Additional security checks
            security_issues, security_recommendations = (
                self.security_checker.perform_security_checks(credential)
            )
            issues.extend(security_issues)
            recommendations.extend(security_recommendations)

            # Determine overall status
            if issues:
                if any("expired" in issue.lower() for issue in issues):
                    status = ValidationResult.EXPIRED
                elif any(
                    "malformed" in issue.lower() or "invalid" in issue.lower() for issue in issues
                ):
                    status = ValidationResult.MALFORMED
                elif strength_score < 50:
                    status = ValidationResult.WEAK
                else:
                    status = ValidationResult.INVALID
            else:
                status = ValidationResult.VALID

            is_valid = status == ValidationResult.VALID

        except Exception as e:
            logger.error(f"Error validating credential: {e}")
            issues.append(f"Validation error: {e!s}")
            status = ValidationResult.INVALID
            is_valid = False

        return CredentialValidation(
            credential_type=credential_type,
            status=status,
            is_valid=is_valid,
            strength_score=strength_score,
            issues=issues,
            recommendations=recommendations,
            expires_at=expires_at,
            metadata=metadata,
        )

    def batch_validate(
        self, credentials: list[tuple[str, CredentialType]]
    ) -> list[CredentialValidation]:
        """
        Validate multiple credentials in batch.

        Args:
            credentials: List of (credential, type) tuples

        Returns:
            List of validation results
        """
        results = []

        for credential, credential_type in credentials:
            try:
                result = self.validate_credential(credential, credential_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch validation error: {e}")
                results.append(
                    CredentialValidation(
                        credential_type=credential_type,
                        status=ValidationResult.INVALID,
                        is_valid=False,
                        strength_score=0,
                        issues=[f"Validation error: {e!s}"],
                        recommendations=["Fix credential format and try again"],
                    )
                )

        return results


# Global validator instance
_global_validator = CredentialValidator()


def validate_credential(
    credential: str,
    credential_type: CredentialType,
    additional_checks: dict[str, Any] | None = None,
) -> CredentialValidation:
    """Validate credential using global validator."""
    return _global_validator.validate_credential(credential, credential_type, additional_checks)


def get_global_validator() -> CredentialValidator:
    """Get the global credential validator instance."""
    return _global_validator
