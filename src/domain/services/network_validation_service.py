"""
Network Validation Service - Domain service for IP and network validation business rules.

This service handles business logic for validating IP addresses, network headers,
and network-related request attributes, implementing the Single Responsibility Principle.
"""

import re


class NetworkValidationError(Exception):
    """Exception raised when network validation fails."""

    pass


class NetworkValidationService:
    """
    Domain service for network validation business logic.

    This service contains business rules for validating IP addresses,
    network headers, and related network attributes.
    """

    # Allowed forwarding headers (business decision)
    ALLOWED_FORWARDING_HEADERS = ["X-Forwarded-For", "X-Real-IP", "X-Originating-IP"]

    def __init__(self) -> None:
        """Initialize the service with default settings."""
        # IP whitelist (instance attribute for testing)
        self.whitelist_ips: list[str] = []
        # IP blacklist (instance attribute for testing)
        self.blacklist_ips: list[str] = []
        # Allowed CORS origins (instance attribute for testing)
        self.allowed_origins: list[str] = []

    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """
        Validate IP address format according to business rules.

        Args:
            ip: IP address string

        Returns:
            True if IP format is valid

        Raises:
            NetworkValidationError: If IP address is invalid
        """
        if not ip:
            raise NetworkValidationError("IP address cannot be empty")

        # Basic IPv4 validation (business rule: only support IPv4 for now)
        ipv4_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        if not re.match(ipv4_pattern, ip):
            raise NetworkValidationError(f"Invalid IP address format: {ip}")

        # Validate each octet
        parts = ip.split(".")
        for part in parts:
            try:
                num = int(part)
                if num < 0 or num > 255:
                    raise NetworkValidationError(f"IP address octet out of range: {num}")
            except ValueError:
                raise NetworkValidationError(f"Invalid IP address: {ip}")

        return True

    @classmethod
    def validate_ip_list(cls, ip_string: str) -> bool:
        """
        Validate comma-separated list of IP addresses.

        Args:
            ip_string: Comma-separated IP addresses

        Returns:
            True if all IPs in the list are valid
        """
        if not ip_string:
            return False

        ips = [ip.strip() for ip in ip_string.split(",")]
        for ip in ips:
            try:
                cls.validate_ip_address(ip)
            except NetworkValidationError:
                return False

        return True

    @classmethod
    def validate_ip_format(cls, ip: str) -> bool:
        """
        Validate IP address format.

        Args:
            ip: IP address string

        Returns:
            True if IP format is valid
        """
        # Simple IPv4 validation
        ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
        if re.match(ipv4_pattern, ip):
            # Check each octet is within valid range
            octets = ip.split(".")
            return all(0 <= int(octet) <= 255 for octet in octets)

        # Simple IPv6 validation (basic check)
        ipv6_pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
        if re.match(ipv6_pattern, ip):
            return True

        return False

    def is_whitelisted_ip(self, ip: str) -> bool:
        """
        Check if an IP address is whitelisted.

        Args:
            ip: IP address to check

        Returns:
            True if IP is in whitelist
        """
        return ip in self.whitelist_ips

    def is_blacklisted_ip(self, ip: str) -> bool:
        """
        Check if an IP address is blacklisted.

        Args:
            ip: IP address to check

        Returns:
            True if IP is in blacklist
        """
        return ip in self.blacklist_ips

    @classmethod
    def is_allowed_forwarding_header(cls, header_name: str) -> bool:
        """
        Check if a forwarding header is allowed according to business rules.

        Args:
            header_name: The header name to check

        Returns:
            True if the header is allowed
        """
        return header_name in cls.ALLOWED_FORWARDING_HEADERS

    @classmethod
    def validate_forwarding_headers(cls, headers: dict[str, str]) -> list[str]:
        """
        Validate forwarding headers according to business rules.

        Args:
            headers: Dictionary of request headers

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for suspicious forwarding headers
        for header in cls.ALLOWED_FORWARDING_HEADERS:
            if header in headers:
                value = headers[header]
                if not cls.validate_ip_list(value):
                    errors.append(f"Invalid {header} header format: {value}")

        # Business rule: Check for suspicious header combinations
        if "X-Forwarded-For" in headers and "X-Real-IP" in headers:
            # Multiple forwarding headers might indicate spoofing attempt
            errors.append("Multiple forwarding headers detected")

        return errors

    def validate_cors_origin(self, origin: str) -> bool:
        """
        Validate CORS origin according to business rules.

        Args:
            origin: Origin header value

        Returns:
            True if origin is allowed

        Raises:
            NetworkValidationError: If origin is not allowed
        """
        if not self.allowed_origins:
            # If no origins configured, allow all (for development)
            return True

        if origin not in self.allowed_origins:
            raise NetworkValidationError(f"Origin not allowed: {origin}")

        return True
