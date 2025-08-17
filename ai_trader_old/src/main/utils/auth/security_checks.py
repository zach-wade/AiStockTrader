"""
Security Checks

Additional security validation utilities for credentials.
"""


class SecurityChecker:
    """Perform additional security checks on credentials."""

    def perform_security_checks(self, credential: str) -> tuple[list[str], list[str]]:
        """Perform additional security checks."""
        issues = []
        recommendations = []

        # Check for common leaked credentials (simplified)
        if len(credential) > 0 and credential[0] == credential[-1] and len(set(credential)) < 3:
            issues.append("Credential appears to be a simple pattern")
            recommendations.append("Use a more complex credential")

        # Check for obvious test values
        test_patterns = ["test", "demo", "sample", "example", "fake", "mock"]
        credential_lower = credential.lower()
        for pattern in test_patterns:
            if pattern in credential_lower:
                issues.append(f"Credential contains test pattern: {pattern}")
                recommendations.append("Use production-grade credentials")

        # Check for sequential patterns
        if self._has_sequential_pattern(credential):
            issues.append("Contains sequential pattern")
            recommendations.append("Avoid sequential characters")

        # Check for keyboard patterns
        if self._has_keyboard_pattern(credential):
            issues.append("Contains keyboard pattern")
            recommendations.append("Avoid keyboard patterns like 'qwerty'")

        return issues, recommendations

    def _has_sequential_pattern(self, text: str) -> bool:
        """Check for sequential patterns like 'abc' or '123'."""
        if len(text) < 3:
            return False

        for i in range(len(text) - 2):
            # Check for ascending sequence
            if ord(text[i]) + 1 == ord(text[i + 1]) and ord(text[i + 1]) + 1 == ord(text[i + 2]):
                return True

            # Check for descending sequence
            if ord(text[i]) - 1 == ord(text[i + 1]) and ord(text[i + 1]) - 1 == ord(text[i + 2]):
                return True

        return False

    def _has_keyboard_pattern(self, text: str) -> bool:
        """Check for common keyboard patterns."""
        keyboard_patterns = [
            "qwerty",
            "qwertyuiop",
            "asdfgh",
            "asdfghjkl",
            "zxcvbn",
            "zxcvbnm",
            "123456",
            "1234567890",
            "abcdef",
            "abcdefg",
        ]

        text_lower = text.lower()
        for pattern in keyboard_patterns:
            if pattern in text_lower:
                return True

        return False


# Global security checker instance
_security_checker = SecurityChecker()


def perform_security_checks(credential: str) -> tuple[list[str], list[str]]:
    """Perform security checks using global checker."""
    return _security_checker.perform_security_checks(credential)
