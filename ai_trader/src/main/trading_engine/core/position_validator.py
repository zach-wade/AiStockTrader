"""
Position validation logic.

This module provides validation for trading positions to ensure they meet
regulatory, risk, and account-specific constraints.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from main.models.common import Position, Order, OrderSide, PositionSide, AccountInfo
from main.utils.core import AITraderException, ValidationResult, create_event_tracker
from main.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class PositionValidationError(AITraderException):
    """Raised when position validation fails."""
    pass


@dataclass(frozen=True)
class PositionLimits:
    """Position limit configuration."""
    max_position_size: Decimal  # Maximum size per position
    max_position_value: Decimal  # Maximum value per position
    max_positions_per_symbol: int  # Maximum number of positions per symbol
    max_total_positions: int  # Maximum total positions
    max_concentration_pct: Decimal  # Maximum % of portfolio in single position
    min_position_size: Decimal  # Minimum position size
    max_leverage: Decimal  # Maximum leverage allowed
    restricted_symbols: Set[str]  # Symbols not allowed to trade
    
    
@dataclass(frozen=True)
class ValidationContext:
    """Context for position validation."""
    account_info: AccountInfo
    existing_positions: List[Position]
    pending_orders: List[Order]
    position_limits: PositionLimits
    
    
class PositionValidator:
    """
    Validates positions against various constraints.
    
    Ensures positions meet size limits, concentration limits,
    regulatory requirements, and account-specific rules.
    """
    
    def __init__(
        self,
        position_limits: PositionLimits,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize position validator.
        
        Args:
            position_limits: Position limit configuration
            metrics_collector: Optional metrics collector
        """
        self.limits = position_limits
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("position_validator")
        
    def validate_new_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        price: Decimal,
        context: ValidationContext
    ) -> ValidationResult:
        """
        Validate a new position.
        
        Args:
            symbol: Symbol for new position
            side: Position side (LONG/SHORT)
            quantity: Position quantity
            price: Entry price
            context: Validation context
            
        Returns:
            Validation result with any errors/warnings
        """
        errors = []
        warnings = []
        
        # Check restricted symbols
        if symbol in self.limits.restricted_symbols:
            errors.append(f"Symbol {symbol} is restricted")
            
        # Check minimum position size
        if quantity < self.limits.min_position_size:
            errors.append(f"Position size {quantity} below minimum {self.limits.min_position_size}")
            
        # Check maximum position size
        if quantity > self.limits.max_position_size:
            errors.append(f"Position size {quantity} exceeds maximum {self.limits.max_position_size}")
            
        # Check position value
        position_value = quantity * price
        if position_value > self.limits.max_position_value:
            errors.append(f"Position value {position_value} exceeds maximum {self.limits.max_position_value}")
            
        # Check concentration limits
        portfolio_value = context.account_info.portfolio_value
        if portfolio_value > 0:
            concentration_pct = (position_value / portfolio_value) * 100
            if concentration_pct > self.limits.max_concentration_pct:
                errors.append(
                    f"Position concentration {concentration_pct:.1f}% exceeds maximum "
                    f"{self.limits.max_concentration_pct}%"
                )
                
        # Check total position count
        current_positions = len(context.existing_positions)
        if current_positions >= self.limits.max_total_positions:
            errors.append(f"Maximum position count {self.limits.max_total_positions} reached")
            
        # Check positions per symbol
        symbol_positions = [p for p in context.existing_positions if p.symbol == symbol]
        if len(symbol_positions) >= self.limits.max_positions_per_symbol:
            errors.append(
                f"Maximum positions per symbol {self.limits.max_positions_per_symbol} "
                f"reached for {symbol}"
            )
            
        # Check leverage
        leverage = self._calculate_leverage(context, position_value)
        if leverage > self.limits.max_leverage:
            errors.append(f"Leverage {leverage:.2f}x exceeds maximum {self.limits.max_leverage}x")
            
        # Check buying power
        if position_value > context.account_info.buying_power:
            errors.append(f"Insufficient buying power: need {position_value}, have {context.account_info.buying_power}")
            
        # Add warnings for high concentration
        if portfolio_value > 0:
            if concentration_pct > self.limits.max_concentration_pct * Decimal("0.8"):
                warnings.append(f"Position concentration {concentration_pct:.1f}% approaching limit")
                
        # Track validation
        self._track_validation("new_position", symbol, len(errors) == 0)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_position_modification(
        self,
        position: Position,
        new_quantity: Decimal,
        context: ValidationContext
    ) -> ValidationResult:
        """
        Validate position modification.
        
        Args:
            position: Existing position
            new_quantity: New position quantity
            context: Validation context
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Calculate position change
        quantity_change = new_quantity - position.quantity
        
        # If increasing position, validate like new position
        if quantity_change > 0:
            result = self.validate_new_position(
                position.symbol,
                position.side,
                quantity_change,
                position.current_price or position.avg_entry_price,
                context
            )
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            
        # Check minimum position size after modification
        if new_quantity > 0 and new_quantity < self.limits.min_position_size:
            errors.append(f"Modified position size {new_quantity} below minimum {self.limits.min_position_size}")
            
        # Track validation
        self._track_validation("modify_position", position.symbol, len(errors) == 0)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_order(
        self,
        order: Order,
        context: ValidationContext
    ) -> ValidationResult:
        """
        Validate an order before submission.
        
        Args:
            order: Order to validate
            context: Validation context
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Get existing position if any
        existing_position = next(
            (p for p in context.existing_positions if p.symbol == order.symbol),
            None
        )
        
        # Determine position side from order
        position_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        
        # Calculate expected position after order
        if existing_position:
            # Check if order would flip position
            if (existing_position.side == PositionSide.LONG and order.side == OrderSide.SELL) or \
               (existing_position.side == PositionSide.SHORT and order.side == OrderSide.BUY):
                # Closing or flipping position
                remaining_qty = existing_position.quantity - order.quantity
                if remaining_qty < 0:
                    # Position would flip
                    new_position_qty = abs(remaining_qty)
                    new_position_side = PositionSide.SHORT if existing_position.side == PositionSide.LONG else PositionSide.LONG
                    
                    result = self.validate_new_position(
                        order.symbol,
                        new_position_side,
                        new_position_qty,
                        order.limit_price or existing_position.current_price or existing_position.avg_entry_price,
                        context
                    )
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
            else:
                # Adding to position
                new_quantity = existing_position.quantity + order.quantity
                result = self.validate_position_modification(
                    existing_position,
                    new_quantity,
                    context
                )
                errors.extend(result.errors)
                warnings.extend(result.warnings)
        else:
            # New position
            result = self.validate_new_position(
                order.symbol,
                position_side,
                order.quantity,
                order.limit_price or Decimal("0"),  # Price will be validated separately
                context
            )
            errors.extend(result.errors)
            warnings.extend(result.warnings)
            
        # Check for duplicate orders
        duplicate_orders = [
            o for o in context.pending_orders
            if o.symbol == order.symbol and o.side == order.side
        ]
        if duplicate_orders:
            warnings.append(f"Found {len(duplicate_orders)} pending orders for {order.symbol} {order.side.value}")
            
        # Track validation
        self._track_validation("order", order.symbol, len(errors) == 0)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_portfolio(self, context: ValidationContext) -> ValidationResult:
        """
        Validate entire portfolio.
        
        Args:
            context: Validation context
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check total position count
        if len(context.existing_positions) > self.limits.max_total_positions:
            errors.append(f"Total positions {len(context.existing_positions)} exceeds limit {self.limits.max_total_positions}")
            
        # Check concentration by symbol
        portfolio_value = context.account_info.portfolio_value
        if portfolio_value > 0:
            for position in context.existing_positions:
                concentration_pct = (position.market_value / portfolio_value) * 100
                if concentration_pct > self.limits.max_concentration_pct:
                    errors.append(
                        f"{position.symbol} concentration {concentration_pct:.1f}% "
                        f"exceeds limit {self.limits.max_concentration_pct}%"
                    )
                elif concentration_pct > self.limits.max_concentration_pct * Decimal("0.8"):
                    warnings.append(
                        f"{position.symbol} concentration {concentration_pct:.1f}% approaching limit"
                    )
                    
        # Check leverage
        leverage = self._calculate_leverage(context, Decimal("0"))
        if leverage > self.limits.max_leverage:
            errors.append(f"Portfolio leverage {leverage:.2f}x exceeds maximum {self.limits.max_leverage}x")
        elif leverage > self.limits.max_leverage * Decimal("0.8"):
            warnings.append(f"Portfolio leverage {leverage:.2f}x approaching limit")
            
        # Track validation
        self._track_validation("portfolio", "portfolio", len(errors) == 0)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def _calculate_leverage(self, context: ValidationContext, additional_value: Decimal) -> Decimal:
        """Calculate portfolio leverage including additional position."""
        total_position_value = sum(p.market_value for p in context.existing_positions) + additional_value
        equity = context.account_info.portfolio_value
        
        if equity > 0:
            return total_position_value / equity
        else:
            return Decimal("0")
            
    def _track_validation(self, validation_type: str, symbol: str, is_valid: bool) -> None:
        """Track validation event."""
        self.event_tracker.track("validation", {
            "type": validation_type,
            "symbol": symbol,
            "is_valid": is_valid
        })
        
        if self.metrics:
            self.metrics.increment(
                "position_validator.validation",
                tags={
                    "type": validation_type,
                    "result": "valid" if is_valid else "invalid"
                }
            )