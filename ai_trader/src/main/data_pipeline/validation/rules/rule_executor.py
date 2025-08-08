"""
Validation Rule Executor

Executes validation rules against data using safe expression evaluation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from main.utils.core import get_logger
from .rule_definitions import ValidationRule, RuleExecutionResult, FailureAction

logger = get_logger(__name__)


class RuleExecutor:
    """Executes validation rules against data."""
    
    def __init__(self):
        """Initialize rule executor."""
        self.execution_context = {}
        self.execution_stats = {
            "total_executed": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
    
    def execute_rule(
        self,
        rule: ValidationRule,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> RuleExecutionResult:
        """
        Execute a single validation rule.
        
        Args:
            rule: Validation rule to execute
            data: Data to validate
            context: Additional context for rule execution
            
        Returns:
            RuleExecutionResult with execution details
        """
        self.execution_stats["total_executed"] += 1
        context = context or {}
        
        try:
            # Skip if rule is disabled
            if not rule.enabled:
                return RuleExecutionResult(
                    rule_name=rule.name,
                    passed=True,
                    metadata={"skipped": True, "reason": "Rule disabled"}
                )
            
            # Execute based on data type
            if isinstance(data, pd.DataFrame):
                passed = self._evaluate_dataframe_rule(rule.expression, data, context)
            elif isinstance(data, dict):
                passed = self._evaluate_dict_rule(rule.expression, data, context)
            elif isinstance(data, list):
                passed = self._evaluate_list_rule(rule.expression, data, context)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            if passed:
                self.execution_stats["passed"] += 1
            else:
                self.execution_stats["failed"] += 1
            
            return RuleExecutionResult(
                rule_name=rule.name,
                passed=passed,
                error_message=None if passed else rule.error_message,
                metadata={
                    "severity": rule.severity.value,
                    "failure_action": rule.failure_action.value
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing rule '{rule.name}': {e}")
            self.execution_stats["errors"] += 1
            
            return RuleExecutionResult(
                rule_name=rule.name,
                passed=False,
                error_message=f"Rule execution error: {str(e)}",
                metadata={"execution_error": True}
            )
    
    def execute_rules(
        self,
        rules: List[ValidationRule],
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        context: Optional[Dict[str, Any]] = None,
        fail_fast: bool = True
    ) -> List[RuleExecutionResult]:
        """
        Execute multiple validation rules.
        
        Args:
            rules: List of validation rules
            data: Data to validate
            context: Additional context
            fail_fast: Stop on first failure
            
        Returns:
            List of execution results
        """
        results = []
        
        for rule in rules:
            result = self.execute_rule(rule, data, context)
            results.append(result)
            
            if fail_fast and not result.passed:
                if rule.failure_action == FailureAction.STOP_PROCESSING:
                    logger.warning(f"Stopping validation due to rule '{rule.name}' failure")
                    break
        
        return results
    
    def _evaluate_dataframe_rule(
        self,
        expression: str,
        df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule expression for DataFrame."""
        try:
            # Create safe evaluation context
            safe_context = {
                "df": df,
                "pd": pd,
                "np": np,
                "len": len,
                "all": all,
                "any": any,
                "min": min,
                "max": max,
                "sum": sum,
                **context
            }
            
            # Add helper functions
            safe_context.update(self._get_dataframe_helpers(df))
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"DataFrame rule evaluation error: {e}")
            return False
    
    def _evaluate_dict_rule(
        self,
        expression: str,
        data: dict,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule expression for dictionary."""
        try:
            # Create safe evaluation context
            safe_context = {
                "data": data,
                "len": len,
                "all": all,
                "any": any,
                "min": min,
                "max": max,
                **context
            }
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Dict rule evaluation error: {e}")
            return False
    
    def _evaluate_list_rule(
        self,
        expression: str,
        data: list,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate rule expression for list."""
        try:
            # Create safe evaluation context
            safe_context = {
                "data": data,
                "len": len,
                "all": all,
                "any": any,
                "min": min,
                "max": max,
                "sum": sum,
                **context
            }
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"List rule evaluation error: {e}")
            return False
    
    def _get_dataframe_helpers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get helper functions for DataFrame validation."""
        return {
            "check_positive_prices": lambda: self._check_positive_prices(df),
            "check_ohlc_relationships": lambda: self._check_ohlc_relationships(df),
            "check_non_negative_volume": lambda: self._check_non_negative_volume(df),
            "check_reasonable_price_range": lambda: self._check_reasonable_price_range(df),
            "has_column": lambda col: col in df.columns,
            "column_not_null": lambda col: df[col].notna().all() if col in df.columns else False,
            "column_unique": lambda col: df[col].nunique() == len(df) if col in df.columns else False
        }
    
    def _check_positive_prices(self, df: pd.DataFrame) -> bool:
        """Check if all price columns are positive."""
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] <= 0).any():
                    return False
        return True
    
    def _check_ohlc_relationships(self, df: pd.DataFrame) -> bool:
        """Check OHLC relationship validity."""
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            return True  # Skip check if columns missing
        
        # High >= Low
        if (df['high'] < df['low']).any():
            return False
        
        # High >= Open and High >= Close
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            return False
        
        # Low <= Open and Low <= Close  
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            return False
        
        return True
    
    def _check_non_negative_volume(self, df: pd.DataFrame) -> bool:
        """Check if volume is non-negative."""
        if 'volume' in df.columns:
            return (df['volume'] >= 0).all()
        return True
    
    def _check_reasonable_price_range(self, df: pd.DataFrame) -> bool:
        """Check if prices are within reasonable range."""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in df.columns:
                # Check for unreasonable values (e.g., > $1M per share)
                if (df[col] > 1_000_000).any():
                    return False
                # Check for tiny values that might be errors
                if ((df[col] > 0) & (df[col] < 0.0001)).any():
                    return False
        
        return True
    
    def get_execution_stats(self) -> Dict[str, int]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            "total_executed": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }