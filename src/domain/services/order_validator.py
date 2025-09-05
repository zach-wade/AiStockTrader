"""
Order Validator - Domain service for validating trading orders.

This module provides the OrderValidator service which performs comprehensive
validation of trading orders before execution. It ensures orders comply with
business rules, risk limits, and portfolio constraints.

The OrderValidator acts as a gatekeeper, preventing invalid or risky orders from
being submitted to the market. It encapsulates all order validation logic including
parameter validation, capital requirements, position limits, and margin checks.

Key Responsibilities:
    - Validating order parameters (prices, quantities, types)
    - Checking capital availability and margin requirements
    - Enforcing position size and concentration limits
    - Validating short selling constraints
    - Calculating required capital and estimated commissions

Design Patterns:
    - Domain Service: Centralizes validation logic outside of entities
    - Value Object: Uses ValidationResult to encapsulate validation outcomes
    - Dependency Injection: Commission calculator injected for flexibility
    - Configuration Object: OrderConstraints for customizable limits

Architectural Decisions:
    - Validation separated from execution for clarity and testability
    - Returns detailed validation results rather than throwing exceptions
    - All limits configurable through OrderConstraints
    - Commission calculation delegated to separate service

Example:
    >>> from decimal import Decimal
    >>> from domain.services import OrderValidator
    >>> from domain.entities import Order, Portfolio, OrderSide
    >>> from domain.value_objects import Price
    >>>
    >>> validator = OrderValidator(commission_calculator)
    >>> portfolio = Portfolio(cash_balance=Decimal("10000"))
    >>> order = Order(symbol="TSLA", quantity=10, side=OrderSide.BUY)
    >>> current_price = Price(Decimal("700"))
    >>>
    >>> result = validator.validate_order(order, portfolio, current_price)
    >>> if result.is_valid:
    ...     print(f"Order valid. Required capital: ${result.required_capital.amount}")
    ... else:
    ...     print(f"Order rejected: {result.error_message}")

Note:
    This service focuses on pre-trade validation. Post-trade validations and
    real-time risk monitoring would be handled by separate services.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..entities.order import Order, OrderSide, OrderStatus, OrderType
from ..entities.portfolio import Portfolio
from ..value_objects.money import Money
from ..value_objects.price import Price
from ..value_objects.quantity import Quantity
from .commission_calculator import ICommissionCalculator


@dataclass
class ValidationResult:
    """Result of order validation.

    Encapsulates the outcome of order validation including success/failure status,
    error details, and calculated requirements. This value object provides a rich
    result that helps callers understand validation outcomes and requirements.

    Attributes:
        is_valid: True if order passes all validation checks.
        error_message: Description of validation failure if not valid.
        required_capital: Total capital needed to execute the order.
        estimated_commission: Estimated commission for the order.

    Note:
        Using a result object rather than exceptions allows for more nuanced
        validation results and better API design.
    """

    is_valid: bool
    error_message: str | None = None
    required_capital: Money | None = None
    estimated_commission: Money | None = None

    @classmethod
    def success(
        cls, required_capital: Money | None = None, estimated_commission: Money | None = None
    ) -> "ValidationResult":
        """Create successful validation result.

        Factory method for creating a successful validation result with optional
        capital and commission information.

        Args:
            required_capital: Total capital needed for the order.
            estimated_commission: Estimated commission charges.

        Returns:
            ValidationResult with is_valid=True and provided details.

        Example:
            >>> result = ValidationResult.success(
            ...     required_capital=Money(Decimal("1000")),
            ...     estimated_commission=Money(Decimal("1.00"))
            ... )
        """
        return cls(
            is_valid=True,
            required_capital=required_capital,
            estimated_commission=estimated_commission,
        )

    @classmethod
    def failure(cls, error_message: str) -> "ValidationResult":
        """Create failed validation result.

        Factory method for creating a failed validation result with error details.

        Args:
            error_message: Description of why validation failed.

        Returns:
            ValidationResult with is_valid=False and error message.

        Example:
            >>> result = ValidationResult.failure("Insufficient funds")
        """
        return cls(is_valid=False, error_message=error_message)


@dataclass
class OrderConstraints:
    """Constraints for order validation.

    Configurable limits and rules for order validation. These constraints allow
    customization of validation behavior for different account types, risk profiles,
    or regulatory requirements.

    Attributes:
        max_position_size: Maximum shares allowed in a single position.
        max_order_value: Maximum dollar value for a single order.
        min_order_value: Minimum dollar value for a single order (default $1).
        max_portfolio_concentration: Maximum percentage of portfolio in one position
            (default 20%).
        require_margin_for_shorts: Whether to enforce margin requirements for short
            sales (default True).
        short_margin_requirement: Margin requirement for short positions as a
            multiplier (default 1.5 for 150%).

    Example:
        >>> constraints = OrderConstraints(
        ...     max_position_size=Quantity(1000),
        ...     max_order_value=Money(Decimal("50000")),
        ...     max_portfolio_concentration=Decimal("0.15")  # 15% max
        ... )
    """

    max_position_size: Quantity | None = None
    max_order_value: Money | None = None
    min_order_value: Money | None = Money(Decimal("1"))
    max_portfolio_concentration: Decimal = Decimal("0.20")  # 20% max per position
    require_margin_for_shorts: bool = True
    short_margin_requirement: Decimal = Decimal("1.5")  # 150% margin for shorts

    def __post_init__(self) -> None:
        """Validate constraints.

        Ensures constraint values are within valid ranges. Called automatically
        after dataclass initialization.

        Raises:
            ValueError: If portfolio concentration is not between 0 and 1.
            ValueError: If short margin requirement is less than 100%.
        """
        if self.max_portfolio_concentration <= 0 or self.max_portfolio_concentration > 1:
            raise ValueError("Portfolio concentration must be between 0 and 1")
        if self.short_margin_requirement < 1:
            raise ValueError("Short margin requirement must be at least 100%")


class OrderValidator:
    """Service for validating orders against portfolio and market constraints.

    The OrderValidator performs comprehensive pre-trade validation to ensure orders
    meet all business rules, risk limits, and regulatory requirements. It provides
    a centralized validation service that can be used by multiple order entry points.

    Attributes:
        commission_calculator: Service for calculating trading commissions.
        constraints: Configurable limits for validation rules.

    Concurrency Safety:
        This service is concurrent-safe as it maintains no mutable state beyond
        configuration.
    """

    def __init__(
        self,
        commission_calculator: ICommissionCalculator,
        constraints: OrderConstraints | None = None,
    ):
        """Initialize the OrderValidator.

        Args:
            commission_calculator: Service for calculating commissions. Required for
                determining total capital requirements.
            constraints: Optional custom constraints. Uses defaults if not provided.
        """
        self.commission_calculator = commission_calculator
        self.constraints = constraints or OrderConstraints()

    async def validate_order(
        self, order: Order, portfolio: Portfolio, current_price: Price
    ) -> ValidationResult:
        """
        Comprehensive order validation against portfolio and constraints.

        Performs multi-step validation including parameter checks, capital requirements,
        position limits, and margin requirements. Returns detailed results including
        required capital and estimated commissions for valid orders.

        Args:
            order: Order to validate. Must be in PENDING status.
            portfolio: Portfolio to validate against for capital and position checks.
            current_price: Current market price for the symbol, used for value calculations.

        Returns:
            ValidationResult: Contains validation outcome with:
                - is_valid: Whether order passes all checks
                - error_message: Details if validation fails
                - required_capital: Total capital needed if valid
                - estimated_commission: Expected commission if valid

        Validation Steps:
            1. Order parameter validation (status, quantity, prices)
            2. Capital requirement calculation
            3. Portfolio constraint validation (funds, min/max values)
            4. Position limit validation (size, concentration)
            5. Short selling validation (margin requirements)

        Example:
            >>> order = Order(symbol="AAPL", quantity=100, side=OrderSide.BUY)
            >>> result = validator.validate_order(order, portfolio, Price(Decimal("150")))
            >>> if not result.is_valid:
            ...     print(f"Validation failed: {result.error_message}")
            ... else:
            ...     print(f"Order valid, requires ${result.required_capital.amount}")

        Note:
            Validation is performed in a specific order for efficiency. Early failures
            short-circuit the validation process.
        """
        # Step 1: Validate order parameters
        param_result = self._validate_order_parameters(order)
        if not param_result.is_valid:
            return param_result

        # Step 2: Calculate required capital
        required_capital, commission = self._calculate_required_capital(order, current_price)

        # Step 3: Validate against portfolio constraints
        portfolio_result = self._validate_portfolio_constraints(
            order, portfolio, required_capital, current_price
        )
        if not portfolio_result.is_valid:
            return portfolio_result

        # Step 4: Validate position limits
        position_result = await self._validate_position_limits(order, portfolio, current_price)
        if not position_result.is_valid:
            return position_result

        # Step 5: Validate short selling requirements
        if order.side == OrderSide.SELL:
            short_result = self._validate_short_selling(order, portfolio, current_price)
            if not short_result.is_valid:
                return short_result

        return ValidationResult.success(
            required_capital=required_capital, estimated_commission=commission
        )

    def _validate_order_parameters(self, order: Order) -> ValidationResult:
        """Validate basic order parameters.

        Checks that the order has valid basic parameters including status,
        quantity, and required prices for specific order types.

        Args:
            order: Order to validate.

        Returns:
            ValidationResult: Success if parameters are valid, failure with
                specific error message otherwise.

        Validation Rules:
            - Order must be in PENDING status
            - Quantity must be positive
            - Limit orders must have positive limit price
            - Stop orders must have positive stop price
        """
        # Check order status
        if order.status != OrderStatus.PENDING:
            return ValidationResult.failure(f"Cannot submit order with status {order.status}")

        # Check quantity
        if order.quantity.value <= 0:
            return ValidationResult.failure("Order quantity must be positive")

        # Check limit price for limit orders
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None or order.limit_price.value <= 0:
                return ValidationResult.failure("Limit orders must have a positive limit price")

        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price.value <= 0:
                return ValidationResult.failure("Stop orders must have a positive stop price")

        return ValidationResult.success()

    def _calculate_required_capital(
        self, order: Order, current_price: Price
    ) -> tuple[Money, Money]:
        """Calculate capital required for order execution.

        Determines the total capital needed including order value and commission.
        For buy orders, this is the full purchase amount plus commission. For sell
        orders, only commission is required (assuming shares are available).

        Args:
            order: Order to calculate requirements for.
            current_price: Current market price for estimates.

        Returns:
            tuple[Money, Money]: A tuple of (required_capital, commission).
                - required_capital: Total capital needed
                - commission: Estimated commission amount

        Note:
            Uses limit price for limit orders, current price for market orders.
        """
        # Determine execution price
        if order.order_type == OrderType.LIMIT and order.limit_price:
            execution_price = order.limit_price
        else:
            execution_price = current_price

        # Calculate order value
        order_value = Money(order.quantity.value * execution_price.value)

        # Calculate commission
        commission = self.commission_calculator.calculate(order.quantity, order_value)

        # Total required capital
        if order.side == OrderSide.BUY:
            required_capital = Money(order_value.amount + commission.amount)
        else:
            # For sells, we need to ensure we have the shares
            # Capital requirement is just the commission
            required_capital = commission

        return required_capital, commission

    def _validate_portfolio_constraints(
        self, order: Order, portfolio: Portfolio, required_capital: Money, current_price: Price
    ) -> ValidationResult:
        """Validate order against portfolio constraints.

        Checks that the order meets portfolio-level constraints including
        available funds and order value limits.

        Args:
            order: Order to validate.
            portfolio: Portfolio to check against.
            required_capital: Previously calculated capital requirement.
            current_price: Current market price.

        Returns:
            ValidationResult: Success if constraints are met, failure with
                specific error otherwise.

        Checks:
            - Sufficient funds for buy orders
            - Order value within min/max limits
        """
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            if required_capital.amount > portfolio.cash_balance.amount:
                return ValidationResult.failure(
                    f"Insufficient funds. Required: ${required_capital.amount:.2f}, "
                    f"Available: ${portfolio.cash_balance.amount:.2f}"
                )

        # Check min/max order value
        order_value = Money(order.quantity.value * current_price.value)

        if self.constraints.min_order_value:
            if order_value.amount < self.constraints.min_order_value.amount:
                return ValidationResult.failure(
                    f"Order value ${order_value.amount:.2f} below minimum "
                    f"${self.constraints.min_order_value.amount:.2f}"
                )

        if self.constraints.max_order_value:
            if order_value.amount > self.constraints.max_order_value.amount:
                return ValidationResult.failure(
                    f"Order value ${order_value.amount:.2f} exceeds maximum "
                    f"${self.constraints.max_order_value.amount:.2f}"
                )

        return ValidationResult.success()

    async def _validate_position_limits(
        self, order: Order, portfolio: Portfolio, current_price: Price
    ) -> ValidationResult:
        """Validate position concentration limits.

        Ensures the order won't result in excessive position size or portfolio
        concentration, maintaining diversification requirements.

        Args:
            order: Order to validate.
            portfolio: Portfolio to check against.
            current_price: Current market price for value calculations.

        Returns:
            ValidationResult: Success if within limits, failure with specific
                error otherwise.

        Checks:
            - Maximum position size (if configured)
            - Portfolio concentration limit (default 20%)
        """
        # Get current position if exists
        current_position = None
        for position in portfolio.positions.values():
            if position.symbol == order.symbol and not position.is_closed():
                current_position = position
                break

        # Calculate new position size after order
        if current_position:
            if order.side == OrderSide.BUY:
                new_quantity = current_position.quantity + order.quantity
            else:
                new_quantity = current_position.quantity - order.quantity
        else:
            # For SELL orders without position, treat as a short position
            new_quantity = order.quantity

        # Check max position size
        if self.constraints.max_position_size:
            # Extract numeric value from new_quantity - it's always a Quantity object
            new_quantity_value = abs(new_quantity.value)

            if new_quantity_value > self.constraints.max_position_size:
                return ValidationResult.failure(
                    f"Position size {new_quantity_value} exceeds maximum "
                    f"{self.constraints.max_position_size}"
                )

        # Check portfolio concentration - only for BUY orders or when increasing position
        # SELL orders reduce concentration so they shouldn't be limited by this check
        if order.side == OrderSide.BUY:
            # Extract numeric value from new_quantity - it's always a Quantity object
            new_quantity_value = abs(new_quantity.value)
            new_position_value = new_quantity_value * current_price.value
            # Use PortfolioCalculator to get total value
            from .portfolio_calculator import PortfolioCalculator

            total_portfolio_value = PortfolioCalculator.get_total_value(portfolio)

            if total_portfolio_value.amount > 0:
                concentration = new_position_value / total_portfolio_value
                if concentration > self.constraints.max_portfolio_concentration:
                    return ValidationResult.failure(
                        f"Position would represent {concentration*100:.1f}% of portfolio, "
                        f"exceeds maximum {self.constraints.max_portfolio_concentration*100:.1f}%"
                    )

        return ValidationResult.success()

    def _validate_short_selling(
        self, order: Order, portfolio: Portfolio, current_price: Price
    ) -> ValidationResult:
        """Validate short selling requirements.

        Checks margin requirements for short sales. Ensures sufficient margin
        is available when selling shares not owned (short selling).

        Args:
            order: Sell order to validate.
            portfolio: Portfolio to check for existing positions and margin.
            current_price: Current market price for margin calculations.

        Returns:
            ValidationResult: Success if requirements met, failure with specific
                error otherwise.

        Margin Requirements:
            - Short sales require margin (default 150% of short value)
            - Calculated as: shares_to_short * price * margin_requirement
        """
        # Check if we have shares to sell
        current_position = None
        for position in portfolio.positions.values():
            if position.symbol == order.symbol and not position.is_closed():
                current_position = position
                break

        # If no position or short position, this is a short sale
        if not current_position or current_position.quantity.value < order.quantity.value:
            shares_to_short = order.quantity.value
            if current_position:
                shares_to_short = order.quantity.value - current_position.quantity.value

            if shares_to_short > 0 and self.constraints.require_margin_for_shorts:
                # Calculate margin requirement
                short_value = shares_to_short * current_price.value
                margin_required = short_value * self.constraints.short_margin_requirement

                if margin_required > portfolio.cash_balance.amount:
                    return ValidationResult.failure(
                        f"Insufficient margin for short sale. Required: ${margin_required:.2f}, "
                        f"Available: ${portfolio.cash_balance.amount:.2f}"
                    )

        return ValidationResult.success()

    def validate_modification(
        self,
        original_order: Order,
        new_quantity: Quantity | None = None,
        new_limit_price: Price | None = None,
        new_stop_price: Price | None = None,
    ) -> ValidationResult:
        """Validate order modification.

        Ensures proposed modifications to an existing order are valid. Modifications
        are only allowed for pending or partially filled orders.

        Args:
            original_order: Existing order to modify.
            new_quantity: New quantity if changing (must exceed filled amount).
            new_limit_price: New limit price if changing.
            new_stop_price: New stop price if changing.

        Returns:
            ValidationResult: Success if modification is valid, failure with
                specific error otherwise.

        Validation Rules:
            - Can only modify PENDING or PARTIALLY_FILLED orders
            - New quantity must be positive and >= filled quantity
            - New prices must be positive

        Example:
            >>> # Modify quantity on partially filled order
            >>> original = Order(symbol="TSLA", quantity=100, filled_quantity=30)
            >>> result = validator.validate_modification(
            ...     original, new_quantity=Quantity(50)
            ... )
            >>> assert result.is_valid  # 50 > 30 filled, so valid

        Note:
            This validates the modification parameters but doesn't apply them.
            The actual modification would be handled by the order entity or service.
        """
        # Can only modify pending or partially filled orders
        if original_order.status not in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]:
            return ValidationResult.failure(
                f"Cannot modify order with status {original_order.status}"
            )

        # Validate new quantity if provided
        if new_quantity:
            if new_quantity.value <= 0:
                return ValidationResult.failure("Modified quantity must be positive")

            # Cannot reduce below filled quantity
            if new_quantity.value < original_order.filled_quantity.value:
                return ValidationResult.failure(
                    f"Cannot reduce quantity below filled amount "
                    f"({original_order.filled_quantity})"
                )

        # Validate new limit price
        if new_limit_price and new_limit_price.value <= 0:
            return ValidationResult.failure("Modified limit price must be positive")

        # Validate new stop price
        if new_stop_price and new_stop_price.value <= 0:
            return ValidationResult.failure("Modified stop price must be positive")

        return ValidationResult.success()

    # Trading Validation Methods (consolidated from TradingValidationService)
    @staticmethod
    def validate_symbol_format(symbol: str) -> bool:
        """Validate trading symbol format according to business rules."""
        if not symbol or len(symbol) > 10:
            return False
        pattern = r"^[A-Z]{1,10}$"
        return re.match(pattern, symbol.upper()) is not None

    @staticmethod
    def validate_currency_code(currency: str) -> bool:
        """Validate ISO currency code format."""
        if not currency or len(currency) != 3:
            return False
        pattern = r"^[A-Z]{3}$"
        return re.match(pattern, currency.upper()) is not None

    @staticmethod
    def validate_price_range(price: Price) -> bool:
        """Validate price is within acceptable trading range."""
        min_price = Decimal("0.01")
        max_price = Decimal("1000000")
        return min_price <= price.value <= max_price

    @staticmethod
    def validate_quantity_range(quantity: Quantity) -> bool:
        """Validate quantity is within acceptable trading range."""
        min_qty = Decimal("0.01")
        max_qty = Decimal("1000000")
        return min_qty <= abs(quantity.value) <= max_qty

    def validate_trading_order_data(self, order_data: dict[str, Any]) -> list[str]:
        """
        Validate complete trading order data according to business rules.

        Args:
            order_data: Dictionary containing order data

        Returns:
            List of validation error messages, empty if valid
        """
        errors = []

        # Validate required fields
        required_fields = ["symbol", "quantity", "order_type", "side"]
        for field in required_fields:
            if field not in order_data:
                errors.append(f"Required field '{field}' is missing")

        # Validate symbol format
        if "symbol" in order_data:
            if not self.validate_symbol_format(order_data["symbol"]):
                errors.append("Invalid trading symbol format")

        # Validate quantity
        if "quantity" in order_data:
            try:
                qty = Quantity(Decimal(str(order_data["quantity"])))
                if not self.validate_quantity_range(qty):
                    errors.append("Invalid quantity: must be between 0.01 and 1,000,000")
            except (ValueError, TypeError):
                errors.append("Invalid quantity format")

        # Validate order type and side
        valid_types = {"market", "limit", "stop", "stop_limit"}
        valid_sides = {"buy", "sell"}

        if "order_type" in order_data:
            if order_data["order_type"].lower() not in valid_types:
                errors.append(f"Invalid order type: must be one of {valid_types}")

            # Validate price requirements for order types
            if order_data["order_type"].lower() in ["limit", "stop_limit"]:
                if "price" not in order_data:
                    errors.append("Price is required for limit and stop limit orders")
                else:
                    try:
                        price = Price(Decimal(str(order_data["price"])))
                        if not self.validate_price_range(price):
                            errors.append("Invalid price: must be between $0.01 and $1,000,000")
                    except (ValueError, TypeError):
                        errors.append("Invalid price format")

        if "side" in order_data:
            if order_data["side"].lower() not in valid_sides:
                errors.append(f"Invalid order side: must be one of {valid_sides}")

        # Validate optional time in force
        if "time_in_force" in order_data:
            valid_tif = {"day", "gtc", "ioc", "fok"}
            if order_data["time_in_force"].lower() not in valid_tif:
                errors.append(f"Invalid time in force: must be one of {valid_tif}")

        return errors
