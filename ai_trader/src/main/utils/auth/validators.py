"""
Credential Validators

Specific validators for different credential types.
"""

import re
import base64
import json
import logging
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime

from .types import CredentialType, ValidationResult

logger = logging.getLogger(__name__)


class CredentialValidators:
    """Collection of credential-specific validators."""
    
    def __init__(self):
        """Initialize validators with rules."""
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[CredentialType, Dict[str, Any]]:
        """Initialize validation rules for different credential types."""
        return {
            CredentialType.API_KEY: {
                'min_length': 16,
                'max_length': 128,
                'required_chars': ['upper', 'lower', 'digit'],
                'forbidden_patterns': ['123456', 'password', 'secret', 'key'],
                'entropy_threshold': 3.0
            },
            CredentialType.JWT_TOKEN: {
                'min_length': 100,
                'max_length': 4096,
                'required_parts': 3,
                'header_required': ['alg', 'typ'],
                'payload_required': ['exp', 'iat']
            },
            CredentialType.OAUTH_TOKEN: {
                'min_length': 20,
                'max_length': 512,
                'required_chars': ['upper', 'lower', 'digit'],
                'entropy_threshold': 3.5
            },
            CredentialType.BEARER_TOKEN: {
                'min_length': 20,
                'max_length': 1024,
                'entropy_threshold': 3.0
            },
            CredentialType.WEBHOOK_SECRET: {
                'min_length': 16,
                'max_length': 256,
                'required_chars': ['upper', 'lower', 'digit'],
                'entropy_threshold': 3.5
            }
        }
    
    def validate_api_key(self, api_key: str) -> Tuple[int, List[str], List[str]]:
        """Validate API key format and strength."""
        issues = []
        recommendations = []
        score = 0
        
        rules = self.validation_rules[CredentialType.API_KEY]
        
        # Length check
        if len(api_key) < rules['min_length']:
            issues.append(f"API key too short (minimum {rules['min_length']} characters)")
            recommendations.append("Use a longer API key")
        elif len(api_key) > rules['max_length']:
            issues.append(f"API key too long (maximum {rules['max_length']} characters)")
        else:
            score += 20
        
        # Character variety
        has_upper = any(c.isupper() for c in api_key)
        has_lower = any(c.islower() for c in api_key)
        has_digit = any(c.isdigit() for c in api_key)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in api_key)
        
        char_variety = sum([has_upper, has_lower, has_digit, has_special])
        score += char_variety * 10
        
        if not has_upper:
            recommendations.append("Include uppercase letters")
        if not has_lower:
            recommendations.append("Include lowercase letters")
        if not has_digit:
            recommendations.append("Include numbers")
        if not has_special:
            recommendations.append("Consider including special characters")
        
        # Forbidden patterns
        api_key_lower = api_key.lower()
        for pattern in rules['forbidden_patterns']:
            if pattern in api_key_lower:
                issues.append(f"Contains forbidden pattern: {pattern}")
                recommendations.append("Avoid common patterns and dictionary words")
        
        # Entropy check
        entropy = self._calculate_entropy(api_key)
        if entropy < rules['entropy_threshold']:
            issues.append(f"Low entropy: {entropy:.2f}")
            recommendations.append("Use more random characters")
        else:
            score += 30
        
        return min(score, 100), issues, recommendations
    
    def validate_jwt_token(self, token: str) -> Tuple[int, List[str], List[str], Optional[datetime], Dict[str, Any]]:
        """Validate JWT token format and content."""
        issues = []
        recommendations = []
        score = 0
        expires_at = None
        metadata = {}
        
        try:
            # Split token into parts
            parts = token.split('.')
            if len(parts) != 3:
                issues.append("JWT token must have exactly 3 parts")
                return 0, issues, recommendations, expires_at, metadata
            
            score += 20
            
            # Decode header
            try:
                header_data = base64.urlsafe_b64decode(parts[0] + '==')
                header = json.loads(header_data)
                metadata['header'] = header
                
                # Check required header fields
                if 'alg' not in header:
                    issues.append("Missing 'alg' field in header")
                elif header['alg'] == 'none':
                    issues.append("Insecure algorithm: 'none'")
                    recommendations.append("Use a secure signing algorithm")
                else:
                    score += 15
                
                if 'typ' not in header:
                    issues.append("Missing 'typ' field in header")
                else:
                    score += 10
                
            except Exception as e:
                issues.append(f"Invalid JWT header: {e}")
            
            # Decode payload
            try:
                payload_data = base64.urlsafe_b64decode(parts[1] + '==')
                payload = json.loads(payload_data)
                metadata['payload'] = payload
                
                # Check expiration
                if 'exp' in payload:
                    expires_at = datetime.fromtimestamp(payload['exp'])
                    if expires_at < datetime.now():
                        issues.append("JWT token has expired")
                    else:
                        score += 20
                else:
                    issues.append("Missing expiration time")
                    recommendations.append("Include expiration time (exp) in payload")
                
                # Check issued at
                if 'iat' in payload:
                    issued_at = datetime.fromtimestamp(payload['iat'])
                    if issued_at > datetime.now():
                        issues.append("JWT token issued in the future")
                    else:
                        score += 10
                
                # Check not before
                if 'nbf' in payload:
                    not_before = datetime.fromtimestamp(payload['nbf'])
                    if not_before > datetime.now():
                        issues.append("JWT token not valid yet")
                
            except Exception as e:
                issues.append(f"Invalid JWT payload: {e}")
            
            # Signature check (basic validation)
            if len(parts[2]) == 0:
                issues.append("Missing signature")
                recommendations.append("Ensure token is properly signed")
            else:
                score += 15
                
        except Exception as e:
            issues.append(f"JWT validation error: {e}")
        
        return min(score, 100), issues, recommendations, expires_at, metadata
    
    def validate_oauth_token(self, token: str) -> Tuple[int, List[str], List[str]]:
        """Validate OAuth token format and strength."""
        issues = []
        recommendations = []
        score = 0
        
        rules = self.validation_rules[CredentialType.OAUTH_TOKEN]
        
        # Length check
        if len(token) < rules['min_length']:
            issues.append(f"OAuth token too short (minimum {rules['min_length']} characters)")
        elif len(token) > rules['max_length']:
            issues.append(f"OAuth token too long (maximum {rules['max_length']} characters)")
        else:
            score += 30
        
        # Character variety and entropy
        entropy = self._calculate_entropy(token)
        if entropy < rules['entropy_threshold']:
            issues.append(f"Low entropy: {entropy:.2f}")
            recommendations.append("Token should be more random")
        else:
            score += 40
        
        # Format checks
        if token.startswith('Bearer '):
            issues.append("Token should not include 'Bearer ' prefix")
            recommendations.append("Remove 'Bearer ' prefix from token")
        
        # Base64 URL-safe check
        if re.match(r'^[A-Za-z0-9_-]+$', token):
            score += 30
        else:
            recommendations.append("Use URL-safe base64 encoding")
        
        return min(score, 100), issues, recommendations
    
    def validate_basic_auth(self, auth: str) -> Tuple[int, List[str], List[str]]:
        """Validate Basic Auth credentials."""
        issues = []
        recommendations = []
        score = 0
        
        try:
            # Check if it's base64 encoded
            if auth.startswith('Basic '):
                auth = auth[6:]  # Remove 'Basic ' prefix
            
            decoded = base64.b64decode(auth).decode('utf-8')
            if ':' not in decoded:
                issues.append("Basic auth must contain username:password")
                return 0, issues, recommendations
            
            username, password = decoded.split(':', 1)
            
            # Validate username
            if len(username) < 3:
                issues.append("Username too short")
                recommendations.append("Use a longer username")
            else:
                score += 20
            
            # Validate password
            password_score, password_issues, password_recommendations = self.validate_password(password)
            score += password_score // 2  # Reduce impact of password score
            issues.extend(password_issues)
            recommendations.extend(password_recommendations)
            
        except Exception as e:
            issues.append(f"Invalid Basic Auth format: {e}")
        
        return min(score, 100), issues, recommendations
    
    def validate_bearer_token(self, token: str) -> Tuple[int, List[str], List[str]]:
        """Validate Bearer token."""
        issues = []
        recommendations = []
        score = 0
        
        rules = self.validation_rules[CredentialType.BEARER_TOKEN]
        
        # Remove Bearer prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        # Length check
        if len(token) < rules['min_length']:
            issues.append(f"Bearer token too short (minimum {rules['min_length']} characters)")
        elif len(token) > rules['max_length']:
            issues.append(f"Bearer token too long (maximum {rules['max_length']} characters)")
        else:
            score += 30
        
        # Entropy check
        entropy = self._calculate_entropy(token)
        if entropy < rules['entropy_threshold']:
            issues.append(f"Low entropy: {entropy:.2f}")
            recommendations.append("Use a more random token")
        else:
            score += 40
        
        # Format validation
        if re.match(r'^[A-Za-z0-9_.-]+$', token):
            score += 30
        else:
            recommendations.append("Use only alphanumeric characters and safe symbols")
        
        return min(score, 100), issues, recommendations
    
    def validate_webhook_secret(self, secret: str) -> Tuple[int, List[str], List[str]]:
        """Validate webhook secret."""
        issues = []
        recommendations = []
        score = 0
        
        rules = self.validation_rules[CredentialType.WEBHOOK_SECRET]
        
        # Length check
        if len(secret) < rules['min_length']:
            issues.append(f"Webhook secret too short (minimum {rules['min_length']} characters)")
        elif len(secret) > rules['max_length']:
            issues.append(f"Webhook secret too long (maximum {rules['max_length']} characters)")
        else:
            score += 25
        
        # Entropy check
        entropy = self._calculate_entropy(secret)
        if entropy < rules['entropy_threshold']:
            issues.append(f"Low entropy: {entropy:.2f}")
            recommendations.append("Use a more random secret")
        else:
            score += 50
        
        # Character variety
        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        
        char_variety = sum([has_upper, has_lower, has_digit])
        score += char_variety * 8
        
        if char_variety < 3:
            recommendations.append("Include uppercase, lowercase, and digits")
        
        return min(score, 100), issues, recommendations
    
    def validate_generic(self, credential: str) -> Tuple[int, List[str], List[str]]:
        """Generic credential validation."""
        issues = []
        recommendations = []
        score = 0
        
        # Basic length check
        if len(credential) < 8:
            issues.append("Credential too short")
            recommendations.append("Use at least 8 characters")
        else:
            score += 30
        
        # Entropy check
        entropy = self._calculate_entropy(credential)
        if entropy < 2.0:
            issues.append("Low complexity")
            recommendations.append("Use more varied characters")
        else:
            score += 40
        
        # No obvious patterns
        if re.match(r'^(.)\1+$', credential):
            issues.append("Repetitive pattern detected")
            recommendations.append("Avoid repetitive patterns")
        else:
            score += 30
        
        return min(score, 100), issues, recommendations
    
    def validate_password(self, password: str) -> Tuple[int, List[str], List[str]]:
        """Validate password strength."""
        issues = []
        recommendations = []
        score = 0
        
        # Length check
        if len(password) < 8:
            issues.append("Password too short")
            recommendations.append("Use at least 8 characters")
        elif len(password) >= 12:
            score += 25
        else:
            score += 15
        
        # Character variety
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        char_variety = sum([has_upper, has_lower, has_digit, has_special])
        score += char_variety * 15
        
        missing_types = []
        if not has_upper:
            missing_types.append("uppercase letters")
        if not has_lower:
            missing_types.append("lowercase letters")
        if not has_digit:
            missing_types.append("numbers")
        if not has_special:
            missing_types.append("special characters")
        
        if missing_types:
            recommendations.append(f"Add {', '.join(missing_types)}")
        
        # Common patterns
        if password.lower() in ['password', '123456', 'qwerty', 'admin']:
            issues.append("Common password detected")
            recommendations.append("Avoid common passwords")
        
        return min(score, 100), issues, recommendations
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * (probability.bit_length() - 1)
        
        return entropy