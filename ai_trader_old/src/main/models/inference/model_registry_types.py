# File: src/ai_trader/models/inference/model_registry_types.py

"""
Common data models and enumerations for the Model Registry.

Defines the structure for ModelVersion metadata, which is central
to tracking and managing ML models.
"""

# Standard library imports
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json  # For serialization in to_dict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelVersion:
    """
    Represents a specific version of a machine learning model.
    Contains metadata, performance metrics, and deployment status.
    """

    model_id: str
    version: str  # Unique version string (e.g., 'v1', '1.0.1')
    model_type: str  # e.g., 'xgboost', 'lightgbm', 'ensemble'
    created_at: datetime
    trained_on: str  # Date range or description of training data (e.g., '2023-01-01_to_2023-12-31')
    features: List[str]  # List of feature names used by this model version
    hyperparameters: Dict[str, Any]  # Model hyperparameters
    metrics: Dict[str, float]  # Performance metrics (e.g., accuracy, precision, f1, Sharpe)
    status: str  # 'candidate', 'production', 'archived', 'failed'
    deployment_pct: float  # Percentage of live traffic routed to this version (0.0 - 100.0)
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional metadata (e.g., preprocessing steps, data sources)

    # Path to the actual serialized model file (will be set by ModelFileManager)
    # This field is mutable even if dataclass is frozen for special cases, or set explicitly via __post_init__
    # For simplicity, if not frozen, it's just a normal field. If frozen, this would need careful handling.
    # Given ModelRegistry will manage this, it's typically derived, not part of initial dataclass.
    # Let's add it explicitly for clarity, assuming ModelRegistry manages its persistence.
    model_file_path: Optional[Path] = None  # Path to the .pkl file on disk

    def to_dict(self) -> Dict[str, Any]:
        """Converts ModelVersion object to a dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if data["model_file_path"]:
            data["model_file_path"] = str(
                data["model_file_path"]
            )  # Convert Path to string for JSON
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Creates a ModelVersion object from a dictionary."""
        # Handle optional and special types during deserialization
        created_at = datetime.fromisoformat(data["created_at"])
        model_file_path = Path(data["model_file_path"]) if data.get("model_file_path") else None

        return cls(
            model_id=data["model_id"],
            version=data["version"],
            model_type=data["model_type"],
            created_at=created_at,
            trained_on=data["trained_on"],
            features=data["features"],
            hyperparameters=data["hyperparameters"],
            metrics=data["metrics"],
            status=data["status"],
            deployment_pct=data["deployment_pct"],
            metadata=data.get("metadata", {}),
            model_file_path=model_file_path,
        )


@dataclass
class ModelDeployment:
    """
    Represents a deployed model configuration.
    """

    model_id: str
    version: str
    deployment_name: str
    endpoint: str  # API endpoint or service URL
    deployment_type: str  # 'rest_api', 'grpc', 'embedded', 'batch'
    environment: str  # 'dev', 'staging', 'production'
    created_at: datetime
    updated_at: datetime
    status: str  # 'active', 'inactive', 'deploying', 'failed'
    health_check_url: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDeployment":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class DeploymentStatus:
    """Current deployment status."""

    deployment_id: str
    model_name: str
    version: str
    environment: str
    status: str
    progress_percentage: float
    healthy_instances: int
    total_instances: int
    start_time: datetime
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Basic model information."""

    model_id: str
    name: str
    description: str
    model_type: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    model_id: str
    version: str
    metrics: Dict[str, float]
    timestamp: datetime
    environment: str = "production"
    dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
