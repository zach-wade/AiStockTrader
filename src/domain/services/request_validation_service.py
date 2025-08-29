"""
Refactored Request Validation Service - Facade for focused validation services.

This service provides backward compatibility by delegating to the focused
validation services that implement Single Responsibility Principle.

The original 899-line, 32-method god object has been split into:
- RateLimitingService: Rate limiting business rules
- NetworkValidationService: IP/network validation
- ContentValidationService: Request content validation
- ApiVersioningService: API versioning and endpoint policies
- TradingValidationService: Trading-specific validation
- MarketDataValidationService: Market data validation
- WebhookValidationService: Webhook validation
- CoreRequestValidationService: Basic request validation
"""

from datetime import datetime
from typing import Any

from .api_versioning_service import ApiVersioningError, ApiVersioningService
from .content_validation_service import ContentValidationError, ContentValidationService
from .core_request_validation_service import CoreRequestValidationService, RequestValidationError
from .market_data_validation_service import MarketDataValidationError, MarketDataValidationService
from .network_validation_service import NetworkValidationError, NetworkValidationService
from .rate_limiting_service import RateLimitConfig, RateLimitingService
from .trading_validation_service import TradingValidationService
from .webhook_validation_service import WebhookValidationError, WebhookValidationService


class RequestValidationService:
    """
    Facade service that delegates to focused validation services.

    This maintains backward compatibility while providing clean separation
    of concerns through the focused services.
    """

    def __init__(self) -> None:
        """Initialize the service with all focused validation services."""
        self.core_validation = CoreRequestValidationService()
        self.content_validation = ContentValidationService()
        self.network_validation = NetworkValidationService()
        self.api_versioning = ApiVersioningService()
        self.trading_validation = TradingValidationService()
        self.market_data_validation = MarketDataValidationService()
        self.webhook_validation = WebhookValidationService()

    # Core request validation methods
    def validate_basic_request_format(self, request_data: dict[str, Any]) -> bool:
        """Delegate to CoreRequestValidationService."""
        return self.core_validation.validate_basic_request_format(request_data)

    def validate_request_structure(
        self, request_data: dict[str, Any], required_fields: list[str]
    ) -> bool:
        """Delegate to CoreRequestValidationService."""
        return self.core_validation.validate_request_structure(request_data, required_fields)

    def validate_basic_field_types(
        self, request_data: dict[str, Any], field_types: dict[str, type]
    ) -> bool:
        """Delegate to CoreRequestValidationService."""
        return self.core_validation.validate_basic_field_types(request_data, field_types)

    # Rate limiting methods (delegate to RateLimitingService)
    @classmethod
    def get_rate_limit_for_endpoint(cls, endpoint: str) -> dict[str, int]:
        """Delegate to RateLimitingService."""
        return RateLimitingService.get_rate_limit_for_endpoint(endpoint)

    @classmethod
    def get_rate_limit_config(cls, endpoint: str) -> RateLimitConfig:
        """Delegate to RateLimitingService."""
        return RateLimitingService.get_rate_limit_config(endpoint)

    @classmethod
    def get_request_priority(cls, endpoint: str, method: str) -> str:
        """Delegate to RateLimitingService."""
        return RateLimitingService.get_request_priority(endpoint, method)

    @classmethod
    def get_cooldown_period(cls, endpoint: str) -> int:
        """Delegate to RateLimitingService."""
        return RateLimitingService.get_cooldown_period(endpoint)

    @classmethod
    def get_request_identifier(cls, headers: dict[str, str], fallback: str = "unknown") -> str:
        """Delegate to RateLimitingService."""
        return RateLimitingService.get_request_identifier(headers, fallback)

    # Network validation methods (delegate to NetworkValidationService)
    @classmethod
    def validate_ip_address(cls, ip: str) -> bool:
        """Delegate to NetworkValidationService."""
        try:
            return NetworkValidationService.validate_ip_address(ip)
        except NetworkValidationError as e:
            raise RequestValidationError(str(e))

    @classmethod
    def validate_ip_list(cls, ip_string: str) -> bool:
        """Delegate to NetworkValidationService."""
        return NetworkValidationService.validate_ip_list(ip_string)

    @classmethod
    def validate_ip_format(cls, ip: str) -> bool:
        """Delegate to NetworkValidationService."""
        return NetworkValidationService.validate_ip_format(ip)

    def is_whitelisted_ip(self, ip: str) -> bool:
        """Delegate to NetworkValidationService."""
        return self.network_validation.is_whitelisted_ip(ip)

    def is_blacklisted_ip(self, ip: str) -> bool:
        """Delegate to NetworkValidationService."""
        return self.network_validation.is_blacklisted_ip(ip)

    @classmethod
    def is_allowed_forwarding_header(cls, header_name: str) -> bool:
        """Delegate to NetworkValidationService."""
        return NetworkValidationService.is_allowed_forwarding_header(header_name)

    def validate_cors_origin(self, origin: str) -> bool:
        """Delegate to NetworkValidationService."""
        try:
            return self.network_validation.validate_cors_origin(origin)
        except NetworkValidationError as e:
            raise RequestValidationError(str(e))

    # Content validation methods (delegate to ContentValidationService)
    @classmethod
    def is_suspicious_user_agent(cls, user_agent: str) -> bool:
        """Delegate to ContentValidationService."""
        return ContentValidationService.is_suspicious_user_agent(user_agent)

    def validate_request_id(self, request_id: str) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_request_id(request_id)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    def validate_authorization_header(self, auth_header: str) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_authorization_header(auth_header)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    def validate_content_type(self, content_type: str) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_content_type(content_type)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_date_range(start_date, end_date)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    def validate_pagination_params(self, params: dict[str, Any]) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_pagination_params(params)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    def validate_request_size(self, size_bytes: int, max_size: int | None = None) -> bool:
        """Delegate to ContentValidationService."""
        try:
            return self.content_validation.validate_request_size(size_bytes, max_size)
        except ContentValidationError as e:
            raise RequestValidationError(str(e))

    @classmethod
    def validate_request_headers(cls, headers: dict[str, str]) -> list[str]:
        """Delegate to ContentValidationService."""
        return ContentValidationService.validate_request_headers(headers)

    # API versioning methods (delegate to ApiVersioningService)
    def validate_api_version(self, version: str) -> bool:
        """Delegate to ApiVersioningService."""
        try:
            return self.api_versioning.validate_api_version(version)
        except ApiVersioningError as e:
            raise RequestValidationError(str(e))

    def get_endpoint_timeout(self, endpoint: str) -> int:
        """Delegate to ApiVersioningService."""
        return self.api_versioning.get_endpoint_timeout(endpoint)

    def requires_signature(self, endpoint: str) -> bool:
        """Delegate to ApiVersioningService."""
        return self.api_versioning.requires_signature(endpoint)

    @classmethod
    def should_enforce_signature(cls, endpoint: str, method: str) -> bool:
        """Delegate to ApiVersioningService."""
        return ApiVersioningService.should_enforce_signature(endpoint, method)

    @classmethod
    def get_max_request_size(cls, endpoint: str) -> int:
        """Delegate to ApiVersioningService."""
        return ApiVersioningService.get_max_request_size(endpoint)

    @classmethod
    def is_private_endpoint(cls, endpoint: str) -> bool:
        """Delegate to ApiVersioningService."""
        return ApiVersioningService.is_private_endpoint(endpoint)

    # Trading validation methods (delegate to TradingValidationService)
    @classmethod
    def validate_trading_request(cls, request_data: dict[str, Any]) -> bool:
        """Delegate to TradingValidationService."""
        try:
            return TradingValidationService.validate_trading_request(request_data)
        except ValueError as e:
            raise RequestValidationError(str(e))

    # Market data validation methods (delegate to MarketDataValidationService)
    def validate_market_data_request(self, request_data: dict[str, Any]) -> bool:
        """Delegate to MarketDataValidationService."""
        try:
            return self.market_data_validation.validate_market_data_request(request_data)
        except MarketDataValidationError as e:
            raise RequestValidationError(str(e))

    # Webhook validation methods (delegate to WebhookValidationService)
    def validate_webhook_payload(self, payload: dict[str, Any]) -> bool:
        """Delegate to WebhookValidationService."""
        try:
            return self.webhook_validation.validate_webhook_payload(payload)
        except WebhookValidationError as e:
            raise RequestValidationError(str(e))

    # Properties for compatibility
    @property
    def whitelist_ips(self) -> list[str]:
        """Get whitelist IPs from NetworkValidationService."""
        return self.network_validation.whitelist_ips

    @whitelist_ips.setter
    def whitelist_ips(self, value: list[str]) -> None:
        """Set whitelist IPs on NetworkValidationService."""
        self.network_validation.whitelist_ips = value

    @property
    def blacklist_ips(self) -> list[str]:
        """Get blacklist IPs from NetworkValidationService."""
        return self.network_validation.blacklist_ips

    @blacklist_ips.setter
    def blacklist_ips(self, value: list[str]) -> None:
        """Set blacklist IPs on NetworkValidationService."""
        self.network_validation.blacklist_ips = value

    @property
    def allowed_origins(self) -> list[str]:
        """Get allowed origins from NetworkValidationService."""
        return self.network_validation.allowed_origins

    @allowed_origins.setter
    def allowed_origins(self, value: list[str]) -> None:
        """Set allowed origins on NetworkValidationService."""
        self.network_validation.allowed_origins = value
