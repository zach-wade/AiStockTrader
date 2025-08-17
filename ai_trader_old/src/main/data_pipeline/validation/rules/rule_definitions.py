"""
Validation Rule Definitions and Models

Contains rule dataclasses, enums, and core rule structures.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Local imports
from main.interfaces.data_pipeline.validation import ValidationSeverity, ValidationStage


class FailureAction(Enum):
    """Actions to take on validation failure."""

    DROP_ROW = "DROP_ROW"
    SKIP_SYMBOL = "SKIP_SYMBOL"
    FLAG_ROW = "FLAG_ROW"
    FLAG_AND_CONTINUE = "FLAG_AND_CONTINUE"
    USE_LAST_GOOD = "USE_LAST_GOOD"
    CONTINUE_WITH_WARNING = "CONTINUE_WITH_WARNING"
    STOP_PROCESSING = "STOP_PROCESSING"


@dataclass
class ValidationRule:
    """Individual validation rule definition."""

    name: str
    expression: str
    error_message: str
    severity: ValidationSeverity
    applies_to_profiles: list[str]
    data_types: list[str]
    stages: list[ValidationStage]
    failure_action: FailureAction = FailureAction.FLAG_AND_CONTINUE
    enabled: bool = True
    priority: int = 0  # Lower numbers = higher priority


@dataclass
class RuleProfile:
    """Validation rule profile configuration."""

    name: str
    description: str
    enabled_checks: list[str]
    thresholds: dict[str, Any]
    failure_actions: dict[str, FailureAction]


@dataclass
class RuleExecutionResult:
    """Result of rule execution."""

    rule_name: str
    passed: bool
    error_message: str | None = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


# Default rule configurations for common validation scenarios
DEFAULT_MARKET_DATA_RULES = {
    "positive_prices": {
        "expression": "all(df[col] > 0 for col in ['open', 'high', 'low', 'close'] if col in df.columns)",
        "error_message": "Negative or zero prices detected",
        "severity": ValidationSeverity.ERROR,
        "data_types": ["market_data"],
        "stages": [ValidationStage.INGEST],
    },
    "ohlc_relationships": {
        "expression": "all(df['high'] >= df['low']) and all(df['high'] >= df[['open', 'close']].max(axis=1))",
        "error_message": "Invalid OHLC relationships detected",
        "severity": ValidationSeverity.ERROR,
        "data_types": ["market_data"],
        "stages": [ValidationStage.INGEST, ValidationStage.POST_ETL],
    },
    "volume_non_negative": {
        "expression": "all(df['volume'] >= 0)",
        "error_message": "Negative volume detected",
        "severity": ValidationSeverity.ERROR,
        "data_types": ["market_data"],
        "stages": [ValidationStage.INGEST],
    },
}


DEFAULT_NEWS_RULES = {
    "title_not_empty": {
        "expression": "data.get('title', '').strip() != ''",
        "error_message": "News title is empty",
        "severity": ValidationSeverity.WARNING,
        "data_types": ["news"],
        "stages": [ValidationStage.INGEST],
    },
    "valid_sentiment": {
        "expression": "-1 <= data.get('sentiment_score', 0) <= 1",
        "error_message": "Invalid sentiment score",
        "severity": ValidationSeverity.WARNING,
        "data_types": ["news"],
        "stages": [ValidationStage.POST_ETL],
    },
}


DEFAULT_FUNDAMENTALS_RULES = {
    "valid_market_cap": {
        "expression": "data.get('market_cap', 0) >= 0",
        "error_message": "Invalid market cap",
        "severity": ValidationSeverity.WARNING,
        "data_types": ["fundamentals"],
        "stages": [ValidationStage.INGEST],
    },
    "valid_pe_ratio": {
        "expression": "data.get('pe_ratio') is None or data.get('pe_ratio') > -1000",
        "error_message": "Invalid P/E ratio",
        "severity": ValidationSeverity.WARNING,
        "data_types": ["fundamentals"],
        "stages": [ValidationStage.POST_ETL],
    },
}
