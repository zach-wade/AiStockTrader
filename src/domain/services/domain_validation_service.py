"""
Consolidated Domain Validation Service

This service consolidates all validation logic from the previously scattered
validation services into a single, cohesive service following the Single
Responsibility Principle for validation concerns.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from ..entities.order import Order, OrderType
from ..entities.portfolio import Portfolio
from ..entities.position import Position


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, errors=[], warnings=[])

    @classmethod
    def failure(cls, errors: list[str], warnings: list[str] | None = None) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])


class DomainValidationService:
    """
    Consolidated validation service for all domain objects and operations.

    This service replaces the following scattered validation services:
    - validation_service.py
    - trading_validation_service.py
    - portfolio_validation_service.py
    - market_data_validation_service.py
    - content_validation_service.py
    - network_validation_service.py
    - webhook_validation_service.py
    - secrets_validation_service.py
    - core_request_validation_service.py
    - request_validation_service.py
    - request_validation_service_original.py
    """

    # Order Validation
    @staticmethod
    def validate_order(order: Order) -> ValidationResult:
        """Validate an order for correctness and compliance."""
        errors = []
        warnings = []

        # Basic order validation
        if order.quantity.value <= 0:
            errors.append("Order quantity must be positive")

        if order.order_type == OrderType.LIMIT and order.limit_price is None:
            errors.append("Limit orders must have a limit price")

        if order.order_type == OrderType.STOP and order.stop_price is None:
            errors.append("Stop orders must have a stop price")

        if order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                errors.append("Stop-limit orders must have both stop and limit prices")

        # Price validation for limit orders
        if order.limit_price is not None and order.limit_price.value <= 0:
            errors.append("Limit price must be positive")

        if order.stop_price is not None and order.stop_price.value <= 0:
            errors.append("Stop price must be positive")

        # Commission validation
        if order.commission is not None and order.commission.amount < 0:
            errors.append("Commission cannot be negative")

        # Add warnings for large orders
        if order.quantity.value > 10000:
            warnings.append("Large order size may impact market price")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Content and Request Validation (from ContentValidationService & CoreRequestValidationService)
    @staticmethod
    def validate_request_content(
        content_type: str, size_bytes: int, headers: dict[str, str]
    ) -> ValidationResult:
        """Validate request content according to business rules."""
        errors = []
        warnings = []

        # Content type validation
        allowed_types = [
            "application/json",
            "application/xml",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
        ]
        main_type = content_type.split(";")[0].strip().lower()
        if main_type not in [t.lower() for t in allowed_types]:
            errors.append(f"Unsupported content type: {content_type}")

        # Size validation
        max_size = 10 * 1024 * 1024  # 10MB
        if size_bytes > max_size:
            errors.append(f"Request size exceeds maximum allowed size of {max_size} bytes")

        # User-Agent validation
        user_agent = headers.get("User-Agent", "")
        if not user_agent:
            errors.append("User-Agent header is required")
        elif len(user_agent) < 5:
            errors.append(f"Invalid User-Agent: '{user_agent}' is too short")
        else:
            # Check for suspicious user agents
            suspicious = [
                "bot",
                "crawler",
                "scanner",
                "sqlmap",
                "nikto",
                "curl",
                "wget",
                "python-requests",
            ]
            if any(pattern in user_agent.lower() for pattern in suspicious):
                warnings.append(f"Suspicious User-Agent detected: '{user_agent}'")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    @staticmethod
    def validate_request_structure(
        request_data: Any, required_fields: list[str]
    ) -> ValidationResult:
        """Validate basic request structure and required fields."""
        errors = []

        if not isinstance(request_data, dict):
            return ValidationResult.failure(["Request data must be a dictionary"])

        if not request_data:
            return ValidationResult.failure(["Request data cannot be empty"])

        # Check required fields
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")

        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()

    @staticmethod
    def validate_date_range(start_date: datetime, end_date: datetime) -> ValidationResult:
        """Validate date range according to business rules."""
        errors = []

        if end_date < start_date:
            errors.append("End date must be after start date")

        # Check if range is not too large (max 1 year)
        max_days = 365
        date_diff = (end_date - start_date).days
        if date_diff > max_days:
            errors.append(f"Date range exceeds maximum of {max_days} days")

        # Check if dates are not in the future
        now = datetime.now(UTC)
        if start_date > now:
            errors.append("Start date cannot be in the future")

        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()

    @staticmethod
    def validate_pagination_params(params: dict[str, Any]) -> ValidationResult:
        """Validate pagination parameters according to business rules."""
        errors = []

        # Check page number
        page = params.get("page", 1)
        if not isinstance(page, int) or page < 1:
            errors.append(f"Invalid page number: {page}")

        # Check limit
        max_page_size = 100
        default_page_size = 50
        limit = params.get("limit", default_page_size)
        if not isinstance(limit, int) or limit < 1:
            errors.append(f"Invalid page limit: {limit}")
        elif limit > max_page_size:
            errors.append(f"Limit exceeds maximum of {max_page_size}")

        if errors:
            return ValidationResult.failure(errors)
        return ValidationResult.success()

    # Portfolio Validation
    @staticmethod
    def validate_portfolio(portfolio: Portfolio) -> ValidationResult:
        """Validate portfolio state and constraints."""
        errors = []
        warnings = []

        # Check cash balance
        if portfolio.cash_balance.amount < 0:
            errors.append("Portfolio cash balance cannot be negative")

        # Check position limits
        open_positions = portfolio.get_open_positions()
        if portfolio.max_positions and len(open_positions) > portfolio.max_positions:
            errors.append(f"Portfolio exceeds maximum position limit of {portfolio.max_positions}")

        # Validate each position
        for position in open_positions:
            position_result = DomainValidationService.validate_position(position)
            if not position_result.is_valid:
                errors.extend(position_result.errors)
            warnings.extend(position_result.warnings)

        # Check leverage
        if portfolio.max_leverage:
            total_exposure = Decimal("0")
            for pos in open_positions:
                pos_value = pos.get_position_value()
                if pos_value is not None:
                    total_exposure += pos_value.amount

            if portfolio.initial_capital.amount > 0:
                leverage = total_exposure / portfolio.initial_capital.amount
                if leverage > portfolio.max_leverage:
                    errors.append(
                        f"Portfolio leverage {leverage:.2f} exceeds maximum {portfolio.max_leverage}"
                    )

        # Add warnings
        if (
            portfolio.initial_capital is not None
            and portfolio.cash_balance.amount < portfolio.initial_capital.amount * Decimal("0.1")
        ):
            warnings.append("Low cash balance - less than 10% of initial capital")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Position Validation
    @staticmethod
    def validate_position(position: Position) -> ValidationResult:
        """Validate a position for correctness."""
        errors = []
        warnings = []

        # Basic position validation
        if position.quantity.value <= 0:
            errors.append(f"Position {position.symbol} has invalid quantity")

        if position.average_entry_price.value <= 0:
            errors.append(f"Position {position.symbol} has invalid entry price")

        # Check for extreme P&L
        if position.current_price:
            pnl = position.get_unrealized_pnl()
            pos_value = position.get_position_value()
            if pnl and pos_value and abs(pnl.amount) > pos_value.amount * Decimal("0.5"):
                warnings.append(f"Position {position.symbol} has extreme P&L of {pnl.amount}")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Trading Request Validation
    @staticmethod
    def validate_trading_request(
        request: dict[str, Any], portfolio: Portfolio | None = None
    ) -> ValidationResult:
        """Validate a trading request."""
        errors = []
        warnings = []

        # Required fields
        required_fields = ["symbol", "quantity", "side", "order_type"]
        for field in required_fields:
            if field not in request or request[field] is None:
                errors.append(f"Missing required field: {field}")

        if errors:
            return ValidationResult.failure(errors)

        # Validate symbol
        symbol = request.get("symbol", "")
        if not symbol or len(symbol) > 10:
            errors.append("Invalid symbol")

        # Validate quantity
        try:
            quantity = Decimal(str(request.get("quantity", 0)))
            if quantity <= 0:
                errors.append("Quantity must be positive")
            elif quantity > 100000:
                warnings.append("Very large order size")
        except (ValueError, TypeError):
            errors.append("Invalid quantity format")

        # Validate side
        side = request.get("side", "").upper()
        if side not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")

        # Validate order type
        order_type = request.get("order_type", "").upper()
        if order_type not in ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]:
            errors.append("Invalid order type")

        # Validate prices for limit/stop orders
        if order_type in ["LIMIT", "STOP_LIMIT"]:
            limit_price = request.get("limit_price")
            if limit_price is None:
                errors.append(f"{order_type} orders require a limit price")
            else:
                try:
                    price = Decimal(str(limit_price))
                    if price <= 0:
                        errors.append("Limit price must be positive")
                except (ValueError, TypeError):
                    errors.append("Invalid limit price format")

        if order_type in ["STOP", "STOP_LIMIT"]:
            stop_price = request.get("stop_price")
            if stop_price is None:
                errors.append(f"{order_type} orders require a stop price")
            else:
                try:
                    price = Decimal(str(stop_price))
                    if price <= 0:
                        errors.append("Stop price must be positive")
                except (ValueError, TypeError):
                    errors.append("Invalid stop price format")

        # Portfolio-specific validation
        if portfolio and not errors:
            # Check if portfolio has sufficient cash for buy orders
            if side == "BUY":
                estimated_cost = quantity * Decimal(str(request.get("limit_price", 100)))
                if estimated_cost > portfolio.cash_balance.amount:
                    errors.append("Insufficient cash for order")

            # Check if portfolio has the position for sell orders
            elif side == "SELL":
                position = portfolio.get_position(symbol)
                if not position or position.quantity.value < quantity:
                    errors.append("Insufficient position size for sell order")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Market Data Validation
    @staticmethod
    def validate_market_data(data: dict[str, Any]) -> ValidationResult:
        """Validate market data for correctness."""
        errors = []
        warnings = []

        # Check required fields
        required_fields = ["symbol", "price", "timestamp"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required market data field: {field}")

        if errors:
            return ValidationResult.failure(errors)

        # Validate price
        try:
            price = Decimal(str(data["price"]))
            if price <= 0:
                errors.append("Price must be positive")
            elif price > 1000000:
                warnings.append("Unusually high price")
        except (ValueError, TypeError):
            errors.append("Invalid price format")

        # Validate timestamp
        try:
            timestamp = data["timestamp"]
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, (int, float)):
                datetime.fromtimestamp(timestamp, tz=UTC)
            else:
                errors.append("Invalid timestamp format")
        except (ValueError, TypeError):
            errors.append("Invalid timestamp")

        # Validate volume if present
        if "volume" in data:
            try:
                volume = int(data["volume"])
                if volume < 0:
                    errors.append("Volume cannot be negative")
                elif volume == 0:
                    warnings.append("Zero volume")
            except (ValueError, TypeError):
                errors.append("Invalid volume format")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Request Validation (Generic)
    @staticmethod
    def validate_request(
        request: dict[str, Any],
        required_fields: list[str],
        optional_fields: list[str] | None = None,
    ) -> ValidationResult:
        """Generic request validation."""
        errors = []
        warnings = []

        # Check required fields
        for field in required_fields:
            if field not in request or request[field] is None:
                errors.append(f"Missing required field: {field}")

        # Check for unknown fields
        all_fields = set(required_fields)
        if optional_fields:
            all_fields.update(optional_fields)

        unknown_fields = set(request.keys()) - all_fields
        if unknown_fields:
            warnings.append(f"Unknown fields will be ignored: {', '.join(unknown_fields)}")

        # Validate common field types
        if "portfolio_id" in request:
            try:
                UUID(str(request["portfolio_id"]))
            except (ValueError, TypeError):
                errors.append("Invalid portfolio_id format")

        if "order_id" in request:
            try:
                UUID(str(request["order_id"]))
            except (ValueError, TypeError):
                errors.append("Invalid order_id format")

        if "position_id" in request:
            try:
                UUID(str(request["position_id"]))
            except (ValueError, TypeError):
                errors.append("Invalid position_id format")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Content Validation
    @staticmethod
    def validate_content_length(content: str, max_length: int = 10000) -> ValidationResult:
        """Validate content length constraints."""
        errors = []
        warnings = []

        if not content:
            errors.append("Content cannot be empty")
        elif len(content) > max_length:
            errors.append(f"Content exceeds maximum length of {max_length} characters")
        elif len(content) > max_length * 0.8:
            warnings.append("Content approaching maximum length")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)

    # Network Validation
    @staticmethod
    def validate_url(url: str) -> ValidationResult:
        """Validate URL format."""
        errors = []
        warnings = []

        if not url:
            errors.append("URL cannot be empty")
        elif not url.startswith(("http://", "https://")):
            errors.append("URL must start with http:// or https://")
        elif len(url) > 2048:
            errors.append("URL exceeds maximum length")

        # Check for localhost in production
        if "localhost" in url or "127.0.0.1" in url:
            warnings.append("URL points to localhost")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult(is_valid=True, errors=[], warnings=warnings)


class ValidationError(Exception):
    """Exception raised when validation fails."""

    pass


class DatabaseIdentifierValidator:
    """Validator for database identifiers like schema names, table names, etc."""

    @staticmethod
    def validate_schema_name(schema_name: str) -> str:
        """
        Validate and return a schema name.

        Args:
            schema_name: The schema name to validate

        Returns:
            The validated schema name

        Raises:
            ValidationError: If the schema name is invalid
        """
        if not schema_name:
            raise ValidationError("Schema name cannot be empty")

        if not schema_name.isidentifier():
            raise ValidationError(f"Invalid schema name: {schema_name}")

        if len(schema_name) > 63:  # PostgreSQL limit
            raise ValidationError(f"Schema name too long: {schema_name}")

        return schema_name

    @staticmethod
    def validate_table_name(table_name: str) -> str:
        """
        Validate and return a table name.

        Args:
            table_name: The table name to validate

        Returns:
            The validated table name

        Raises:
            ValidationError: If the table name is invalid
        """
        if not table_name:
            raise ValidationError("Table name cannot be empty")

        if not table_name.isidentifier():
            raise ValidationError(f"Invalid table name: {table_name}")

        if len(table_name) > 63:  # PostgreSQL limit
            raise ValidationError(f"Table name too long: {table_name}")

        return table_name
