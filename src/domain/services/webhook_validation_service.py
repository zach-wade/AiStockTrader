"""
Webhook Validation Service - Domain service for webhook validation business rules.

This service handles business logic for validating webhook payloads and data,
implementing the Single Responsibility Principle.
"""

from datetime import datetime
from typing import Any


class WebhookValidationError(Exception):
    """Exception raised when webhook validation fails."""

    pass


class WebhookValidationService:
    """
    Domain service for webhook validation business logic.

    This service contains business rules for validating webhook payloads,
    event types, and webhook-related data structures.
    """

    def validate_webhook_payload(self, payload: dict[str, Any]) -> bool:
        """
        Validate webhook payload according to business rules.

        Args:
            payload: Webhook payload dictionary

        Returns:
            True if payload is valid

        Raises:
            WebhookValidationError: If payload is invalid
        """
        # Check required fields
        required_fields = ["event", "timestamp", "data"]
        for field in required_fields:
            if field not in payload:
                raise WebhookValidationError(f"Missing required field: {field}")

        # Validate event type
        valid_events = [
            "order.filled",
            "order.cancelled",
            "order.rejected",
            "position.opened",
            "position.closed",
            "alert.triggered",
        ]
        event = payload.get("event", "")
        if event not in valid_events:
            raise WebhookValidationError(f"Invalid webhook event: {event}")

        # Validate timestamp format (ISO format)
        timestamp = payload.get("timestamp", "")
        try:
            # Try to parse the timestamp
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            raise WebhookValidationError(f"Invalid timestamp format: {timestamp}")

        # Validate data is a dictionary
        data = payload.get("data")
        if not isinstance(data, dict):
            raise WebhookValidationError("Webhook data must be a dictionary")

        return True

    @classmethod
    def validate_webhook_event_type(cls, event_type: str) -> bool:
        """
        Validate webhook event type according to business rules.

        Args:
            event_type: Event type string

        Returns:
            True if event type is valid

        Raises:
            WebhookValidationError: If event type is invalid
        """
        valid_events = [
            "order.filled",
            "order.cancelled",
            "order.rejected",
            "position.opened",
            "position.closed",
            "alert.triggered",
        ]

        if event_type not in valid_events:
            raise WebhookValidationError(f"Invalid webhook event type: {event_type}")

        return True
