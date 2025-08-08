"""
Credential Generators

Secure credential generation utilities.
"""

import secrets
from typing import Optional

from .types import CredentialType


class CredentialGenerator:
    """Generate secure credentials of various types."""
    
    def generate_secure_credential(self, 
                                 credential_type: CredentialType,
                                 length: Optional[int] = None) -> str:
        """
        Generate a secure credential of the specified type.
        
        Args:
            credential_type: Type of credential to generate
            length: Optional length override
            
        Returns:
            Generated secure credential
        """
        if credential_type == CredentialType.API_KEY:
            length = length or 32
            return self._generate_api_key(length)
        elif credential_type == CredentialType.WEBHOOK_SECRET:
            length = length or 32
            return self._generate_webhook_secret(length)
        elif credential_type == CredentialType.BEARER_TOKEN:
            length = length or 48
            return self._generate_bearer_token(length)
        else:
            # Generic secure string
            length = length or 24
            return secrets.token_urlsafe(length)
    
    def _generate_api_key(self, length: int) -> str:
        """Generate a secure API key."""
        # Use URL-safe base64 encoding for API keys
        return secrets.token_urlsafe(length)
    
    def _generate_webhook_secret(self, length: int) -> str:
        """Generate a secure webhook secret."""
        # Use hex encoding for webhook secrets
        return secrets.token_hex(length // 2)
    
    def _generate_bearer_token(self, length: int) -> str:
        """Generate a secure bearer token."""
        # Use URL-safe base64 encoding
        return secrets.token_urlsafe(length)


# Global generator instance
_generator = CredentialGenerator()


def generate_secure_credential(credential_type: CredentialType,
                             length: Optional[int] = None) -> str:
    """Generate secure credential using global generator."""
    return _generator.generate_secure_credential(credential_type, length)