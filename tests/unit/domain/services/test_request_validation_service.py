"""
Comprehensive tests for RequestValidationService
"""

from datetime import datetime, timedelta

import pytest

from src.domain.services.request_validation_service import (
    RequestValidationError,
    RequestValidationService,
)


class TestRequestValidationService:
    """Test suite for RequestValidationService"""

    @pytest.fixture
    def service(self):
        """Create a RequestValidationService instance"""
        return RequestValidationService()

    def test_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert hasattr(service, "get_rate_limit_config")
        assert hasattr(service, "validate_request_headers")

    def test_get_rate_limit_config_trading_endpoints(self, service):
        """Test rate limit configuration for trading endpoints"""
        config = service.get_rate_limit_config("/api/orders")
        assert config.requests_per_minute == 60
        assert config.burst_size == 10

        config = service.get_rate_limit_config("/api/positions")
        assert config.requests_per_minute == 60
        assert config.burst_size == 10

    def test_get_rate_limit_config_market_data_endpoints(self, service):
        """Test rate limit configuration for market data endpoints"""
        config = service.get_rate_limit_config("/api/market/quotes")
        assert config.requests_per_minute == 300
        assert config.burst_size == 50

        config = service.get_rate_limit_config("/api/market/bars")
        assert config.requests_per_minute == 300
        assert config.burst_size == 50

    def test_get_rate_limit_config_admin_endpoints(self, service):
        """Test rate limit configuration for admin endpoints"""
        config = service.get_rate_limit_config("/api/admin/users")
        assert config.requests_per_minute == 30
        assert config.burst_size == 5

        config = service.get_rate_limit_config("/api/admin/settings")
        assert config.requests_per_minute == 30
        assert config.burst_size == 5

    def test_get_rate_limit_config_default(self, service):
        """Test default rate limit configuration"""
        config = service.get_rate_limit_config("/api/unknown/endpoint")
        assert config.requests_per_minute == 120
        assert config.burst_size == 20

    def test_validate_request_headers_valid(self, service):
        """Test validation of valid request headers"""
        headers = {
            "User-Agent": "TradingSystem/1.0",
            "Content-Type": "application/json",
            "X-Request-ID": "abc-123-def",
            "Authorization": "Bearer token123",
        }
        assert service.validate_request_headers(headers) is True

    def test_validate_request_headers_missing_user_agent(self, service):
        """Test validation with missing User-Agent"""
        headers = {"Content-Type": "application/json", "X-Request-ID": "abc-123-def"}
        with pytest.raises(RequestValidationError) as exc:
            service.validate_request_headers(headers)
        assert "User-Agent header is required" in str(exc)

    def test_validate_request_headers_invalid_user_agent(self, service):
        """Test validation with invalid User-Agent"""
        headers = {"User-Agent": "bot", "Content-Type": "application/json"}  # Too short
        with pytest.raises(RequestValidationError) as exc:
            service.validate_request_headers(headers)
        assert "Invalid User-Agent" in str(exc)

    def test_validate_trading_request(self, service):
        """Test validation of trading requests"""
        request_data = {"symbol": "AAPL", "quantity": 100, "order_type": "market", "side": "buy"}
        assert service.validate_trading_request(request_data) is True

    def test_validate_trading_request_missing_symbol(self, service):
        """Test validation of trading request without symbol"""
        request_data = {"quantity": 100, "order_type": "market", "side": "buy"}
        with pytest.raises(RequestValidationError) as exc:
            service.validate_trading_request(request_data)
        assert "symbol is required" in str(exc)

    def test_validate_trading_request_invalid_quantity(self, service):
        """Test validation of trading request with invalid quantity"""
        request_data = {
            "symbol": "AAPL",
            "quantity": -100,  # Negative quantity
            "order_type": "market",
            "side": "buy",
        }
        with pytest.raises(RequestValidationError) as exc:
            service.validate_trading_request(request_data)
        assert "Invalid quantity" in str(exc)

    def test_validate_trading_request_invalid_order_type(self, service):
        """Test validation of trading request with invalid order type"""
        request_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "order_type": "invalid_type",
            "side": "buy",
        }
        with pytest.raises(RequestValidationError) as exc:
            service.validate_trading_request(request_data)
        assert "Invalid order type" in str(exc)

    def test_validate_trading_request_invalid_side(self, service):
        """Test validation of trading request with invalid side"""
        request_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "order_type": "market",
            "side": "invalid_side",
        }
        with pytest.raises(RequestValidationError) as exc:
            service.validate_trading_request(request_data)
        assert "Invalid side" in str(exc)

    def test_validate_ip_address_valid(self, service):
        """Test validation of valid IP addresses"""
        valid_ips = ["192.168.1.1", "10.0.0.1", "172.16.0.1", "8.8.8.8"]
        for ip in valid_ips:
            assert service.validate_ip_address(ip) is True

    def test_validate_ip_address_invalid(self, service):
        """Test validation of invalid IP addresses"""
        invalid_ips = [
            "256.256.256.256",  # Out of range
            "192.168.1",  # Incomplete
            "not.an.ip.address",  # Not numeric
            "",  # Empty
        ]
        for ip in invalid_ips:
            with pytest.raises(RequestValidationError):
                service.validate_ip_address(ip)

    def test_is_whitelisted_ip(self, service):
        """Test IP whitelist checking"""
        # Add whitelist IPs
        service.whitelist_ips = ["192.168.1.100", "10.0.0.50"]

        assert service.is_whitelisted_ip("192.168.1.100") is True
        assert service.is_whitelisted_ip("10.0.0.50") is True
        assert service.is_whitelisted_ip("192.168.1.101") is False

    def test_is_blacklisted_ip(self, service):
        """Test IP blacklist checking"""
        # Add blacklist IPs
        service.blacklist_ips = ["192.168.1.200", "10.0.0.100"]

        assert service.is_blacklisted_ip("192.168.1.200") is True
        assert service.is_blacklisted_ip("10.0.0.100") is True
        assert service.is_blacklisted_ip("192.168.1.201") is False

    def test_validate_request_size(self, service):
        """Test request size validation"""
        # Valid size (under 10MB default)
        assert service.validate_request_size(1024 * 1024) is True  # 1MB

        # Too large
        with pytest.raises(RequestValidationError) as exc:
            service.validate_request_size(11 * 1024 * 1024)  # 11MB
        assert "Request size exceeds maximum" in str(exc)

    def test_validate_request_size_custom_limit(self, service):
        """Test request size validation with custom limit"""
        # Set custom limit
        assert service.validate_request_size(500, max_size=1024) is True

        with pytest.raises(RequestValidationError):
            service.validate_request_size(2048, max_size=1024)

    def test_requires_signature(self, service):
        """Test signature requirement checking"""
        # Endpoints requiring signature
        assert service.requires_signature("/api/orders") is True
        assert service.requires_signature("/api/positions") is True
        assert service.requires_signature("/api/admin/settings") is True

        # Endpoints not requiring signature
        assert service.requires_signature("/api/market/quotes") is False
        assert service.requires_signature("/api/health") is False

    def test_validate_api_version(self, service):
        """Test API version validation"""
        # Valid versions
        assert service.validate_api_version("v1") is True
        assert service.validate_api_version("v2") is True

        # Invalid version
        with pytest.raises(RequestValidationError) as exc:
            service.validate_api_version("v0")
        assert "Unsupported API version" in str(exc)

    def test_validate_content_type(self, service):
        """Test content type validation"""
        # Valid content types
        assert service.validate_content_type("application/json") is True
        assert service.validate_content_type("application/x-www-form-urlencoded") is True

        # Invalid content type
        with pytest.raises(RequestValidationError) as exc:
            service.validate_content_type("text/plain")
        assert "Unsupported content type" in str(exc)

    def test_validate_authorization_header(self, service):
        """Test authorization header validation"""
        # Valid Bearer token
        assert service.validate_authorization_header("Bearer abc123xyz") is True

        # Valid API key
        assert service.validate_authorization_header("ApiKey key123secret") is True

        # Invalid format
        with pytest.raises(RequestValidationError) as exc:
            service.validate_authorization_header("InvalidAuth")
        assert "Invalid authorization format" in str(exc)

    def test_validate_request_id(self, service):
        """Test request ID validation"""
        # Valid UUIDs
        valid_ids = ["550e8400-e29b-41d4-a716-446655440000", "abc-123-def-456", "request_12345"]
        for req_id in valid_ids:
            assert service.validate_request_id(req_id) is True

        # Invalid (too short)
        with pytest.raises(RequestValidationError):
            service.validate_request_id("ab")

    def test_get_endpoint_timeout(self, service):
        """Test getting endpoint-specific timeouts"""
        # Trading endpoints - shorter timeout
        assert service.get_endpoint_timeout("/api/orders") == 5000  # 5 seconds

        # Market data - medium timeout
        assert service.get_endpoint_timeout("/api/market/bars") == 10000  # 10 seconds

        # Admin - longer timeout
        assert service.get_endpoint_timeout("/api/admin/reports") == 30000  # 30 seconds

        # Default
        assert service.get_endpoint_timeout("/api/unknown") == 15000  # 15 seconds

    def test_validate_pagination_params(self, service):
        """Test pagination parameter validation"""
        # Valid pagination
        params = {"page": 1, "limit": 50}
        assert service.validate_pagination_params(params) is True

        # Negative page
        params = {"page": -1, "limit": 50}
        with pytest.raises(RequestValidationError) as exc:
            service.validate_pagination_params(params)
        assert "Invalid page number" in str(exc)

        # Limit too high
        params = {"page": 1, "limit": 1001}
        with pytest.raises(RequestValidationError) as exc:
            service.validate_pagination_params(params)
        assert "Limit exceeds maximum" in str(exc)

    def test_validate_date_range(self, service):
        """Test date range validation"""
        # Valid range
        start = datetime.now() - timedelta(days=7)
        end = datetime.now()
        assert service.validate_date_range(start, end) is True

        # End before start
        with pytest.raises(RequestValidationError) as exc:
            service.validate_date_range(end, start)
        assert "End date must be after start date" in str(exc)

        # Range too large (>365 days)
        start = datetime.now() - timedelta(days=400)
        with pytest.raises(RequestValidationError) as exc:
            service.validate_date_range(start, end)
        assert "Date range exceeds maximum" in str(exc)

    def test_validate_market_data_request(self, service):
        """Test market data request validation"""
        request = {
            "symbols": ["AAPL", "GOOGL"],
            "interval": "1min",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }
        assert service.validate_market_data_request(request) is True

        # Too many symbols
        request["symbols"] = ["SYM" + str(i) for i in range(101)]
        with pytest.raises(RequestValidationError) as exc:
            service.validate_market_data_request(request)
        assert "Too many symbols" in str(exc)

        # Invalid interval
        request = {
            "symbols": ["AAPL"],
            "interval": "invalid",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }
        with pytest.raises(RequestValidationError) as exc:
            service.validate_market_data_request(request)
        assert "Invalid interval" in str(exc)

    def test_validate_cors_origin(self, service):
        """Test CORS origin validation"""
        # Allowed origins
        service.allowed_origins = ["https://app.example.com", "http://localhost:3000"]

        assert service.validate_cors_origin("https://app.example.com") is True
        assert service.validate_cors_origin("http://localhost:3000") is True

        # Not allowed
        with pytest.raises(RequestValidationError) as exc:
            service.validate_cors_origin("https://evil.com")
        assert "Origin not allowed" in str(exc)

    def test_validate_webhook_payload(self, service):
        """Test webhook payload validation"""
        payload = {
            "event": "order.filled",
            "timestamp": datetime.now().isoformat(),
            "data": {"order_id": "123", "status": "filled"},
        }
        assert service.validate_webhook_payload(payload) is True

        # Missing event
        payload = {"timestamp": datetime.now().isoformat(), "data": {"order_id": "123"}}
        with pytest.raises(RequestValidationError) as exc:
            service.validate_webhook_payload(payload)
        assert "Missing required field: event" in str(exc)

    def test_get_request_priority(self, service):
        """Test request priority determination"""
        # High priority - trading operations
        assert service.get_request_priority("/api/orders", "POST") == "high"
        assert service.get_request_priority("/api/positions", "DELETE") == "high"

        # Medium priority - market data
        assert service.get_request_priority("/api/market/quotes", "GET") == "medium"

        # Low priority - admin/reporting
        assert service.get_request_priority("/api/admin/reports", "GET") == "low"
        assert service.get_request_priority("/api/health", "GET") == "low"
