"""
Validation Rules Engine

Main orchestrator that combines parser, executor, and registry for rule-based validation.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime

from main.interfaces.validation.rules import IRuleEngine
from main.interfaces.data_pipeline.validation import ValidationStage
from main.utils.core import get_logger

from .rule_definitions import ValidationRule, RuleProfile, RuleExecutionResult
from .rule_parser import RuleParser
from .rule_executor import RuleExecutor
from .rule_registry import RuleRegistry

logger = get_logger(__name__)


class ValidationRulesEngine:
    """
    Main validation rules engine that orchestrates rule-based validation.
    
    Implements IRuleEngine interface and coordinates parser, executor, and registry.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        custom_rules: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize validation rules engine.
        
        Args:
            config_path: Path to YAML configuration file
            custom_rules: Custom rule definitions
        """
        self.parser = RuleParser(config_path)
        self.executor = RuleExecutor()
        self.registry = RuleRegistry()
        self.custom_rules = custom_rules or []
        
        # Initialize engine
        self._initialize()
        
        logger.info("ValidationRulesEngine initialized")
    
    def _initialize(self) -> None:
        """Initialize the engine with rules and profiles."""
        # Load configuration
        config = self.parser.load_config()
        
        # Parse and register profiles
        profiles = self.parser.parse_profiles()
        self.registry.register_profiles(profiles)
        
        # Parse and register rules
        for rule_type in ["market_data_rules", "news_rules", "fundamentals_rules"]:
            rules = self.parser.parse_rules(rule_type)
            self.registry.register_rules(rules)
        
        # Register custom rules
        for rule_config in self.custom_rules:
            try:
                rule = self.parser._create_rule_from_config(
                    rule_config.get("name", "custom_rule"),
                    rule_config
                )
                self.registry.register_rule(rule)
            except Exception as e:
                logger.error(f"Failed to register custom rule: {e}")
        
        rule_counts = self.registry.get_rule_count()
        logger.info(f"Loaded {rule_counts['total']} validation rules")
    
    # IRuleEngine interface methods
    async def load_rules(self, config_path: str) -> List[ValidationRule]:
        """Load validation rules from configuration."""
        parser = RuleParser(config_path)
        config = parser.load_config()
        
        all_rules = []
        for rule_type in ["market_data_rules", "news_rules", "fundamentals_rules"]:
            rules = parser.parse_rules(rule_type)
            all_rules.extend(rules)
        
        return all_rules
    
    async def evaluate_rule(
        self,
        rule: ValidationRule,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Evaluate a single rule against data."""
        result = self.executor.execute_rule(rule, data, context)
        return result.passed
    
    async def evaluate_rules(
        self,
        rules: List[ValidationRule],
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Evaluate multiple rules against data."""
        results = self.executor.execute_rules(rules, data, context)
        
        passed = all(r.passed for r in results)
        errors = [r.error_message for r in results if r.error_message]
        
        return passed, errors
    
    async def get_rules_for_stage(
        self,
        stage: ValidationStage,
        data_type: Optional[str] = None
    ) -> List[ValidationRule]:
        """Get rules applicable to a validation stage."""
        if data_type:
            return self.registry.get_rules_for_data_type(data_type, stage=stage)
        return self.registry.get_rules_for_stage(stage)
    
    async def get_rule_by_name(self, name: str) -> Optional[ValidationRule]:
        """Get a specific rule by name."""
        return self.registry.get_rule(name)
    
    async def register_rule(self, rule: ValidationRule) -> None:
        """Register a new validation rule."""
        self.registry.register_rule(rule)
    
    async def unregister_rule(self, name: str) -> None:
        """Unregister a validation rule."""
        if name in self.registry.rules:
            del self.registry.rules[name]
            logger.info(f"Unregistered rule: {name}")
    
    # Additional public methods
    def validate_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        data_type: str,
        stage: ValidationStage,
        profile: str = "standard",
        fail_fast: bool = True
    ) -> Tuple[bool, List[RuleExecutionResult]]:
        """
        Validate data using rules for specific type and stage.
        
        Args:
            data: Data to validate
            data_type: Type of data (market_data, news, etc.)
            stage: Validation stage
            profile: Validation profile to use
            fail_fast: Stop on first failure
            
        Returns:
            Tuple of (passed, results)
        """
        # Get applicable rules
        rules = self.registry.get_rules_for_profile(profile, stage, data_type)
        
        if not rules:
            logger.debug(f"No rules found for {data_type} at {stage.value} with profile {profile}")
            return True, []
        
        # Execute rules
        context = {
            "data_type": data_type,
            "stage": stage.value,
            "profile": profile,
            "timestamp": datetime.utcnow()
        }
        
        results = self.executor.execute_rules(rules, data, context, fail_fast)
        passed = all(r.passed for r in results)
        
        # Log summary
        failed_count = sum(1 for r in results if not r.passed)
        if failed_count > 0:
            logger.warning(
                f"Validation failed for {data_type}: {failed_count}/{len(results)} rules failed"
            )
        
        return passed, results
    
    def get_profile(self, profile_name: str) -> Optional[RuleProfile]:
        """Get validation profile by name."""
        return self.registry.get_profile(profile_name)
    
    def get_required_fields(self, data_type: str) -> List[str]:
        """Get required fields for a data type."""
        return list(self.registry.get_required_fields(data_type))
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration for validation."""
        return {
            "rule_counts": self.registry.get_rule_count(),
            "execution_stats": self.executor.get_execution_stats(),
            "profiles": list(self.registry.profiles.keys())
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.executor.reset_stats()