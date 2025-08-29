"""
Base Request DTO for Use Cases

Provides a base class for all request DTOs with common fields and behavior.
Eliminates repeated boilerplate across request objects.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from .base import UseCaseRequest


@dataclass(kw_only=True)
class BaseRequestDTO(UseCaseRequest):
    """
    Base class for all request DTOs with common fields.

    Provides standard fields that all requests need:
    - request_id: Unique identifier for the request
    - correlation_id: For tracing related requests
    - metadata: Arbitrary metadata dictionary

    Subclasses should define their specific fields and inherit from this base.

    Uses kw_only=True to allow derived classes to have required fields
    before optional ones from the base class.
    """

    # Common fields with defaults using field factory
    request_id: UUID = field(default_factory=uuid4)
    correlation_id: UUID | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    # No __post_init__ needed - field defaults handle initialization

    def with_correlation_id(self, correlation_id: UUID) -> "BaseRequestDTO":
        """Set the correlation ID and return self for chaining."""
        self.correlation_id = correlation_id
        return self

    def with_metadata(self, key: str, value: Any) -> "BaseRequestDTO":
        """Add metadata and return self for chaining."""
        self.metadata[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert request to dictionary representation."""
        return {
            "request_id": str(self.request_id),
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "metadata": self.metadata,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ["request_id", "correlation_id", "metadata"]
            },
        }
