"""
Base Use Case

Provides the foundation for all use cases in the application layer.
Implements common patterns like logging, validation, and error handling.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Type variables for request and response
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")


@dataclass
class UseCaseRequest:
    """Base class for use case requests."""

    request_id: UUID | None = None
    correlation_id: UUID | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize request with defaults."""
        if self.request_id is None:
            self.request_id = uuid4()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UseCaseResponse:
    """Base class for use case responses."""

    success: bool
    data: Any | None = None
    error: str | None = None
    request_id: UUID | None = None

    @classmethod
    def success_response(cls, data: Any, request_id: UUID) -> "UseCaseResponse":
        """Create a successful response."""
        return cls(success=True, data=data, request_id=request_id)

    @classmethod
    def error_response(cls, error: str, request_id: UUID) -> "UseCaseResponse":
        """Create an error response."""
        return cls(success=False, error=error, request_id=request_id)


class UseCase(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all use cases.

    Provides a consistent interface and common functionality for
    business logic orchestration.
    """

    def __init__(self, name: str | None = None) -> None:
        """
        Initialize use case.

        Args:
            name: Optional name for the use case (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    async def execute(self, request: TRequest) -> TResponse:
        """
        Execute the use case.

        This method provides the template for use case execution with
        logging, validation, and error handling.

        Args:
            request: The use case request

        Returns:
            The use case response
        """
        request_id = getattr(request, "request_id", uuid4())

        self.logger.info(
            f"Executing {self.name}",
            extra={
                "request_id": str(request_id),
                "use_case": self.name,
            },
        )

        try:
            # Validate the request
            validation_error = await self.validate(request)
            if validation_error:
                self.logger.warning(
                    f"Validation failed for {self.name}: {validation_error}",
                    extra={"request_id": str(request_id)},
                )
                return self._create_error_response(validation_error, request_id)

            # Execute the business logic
            response = await self.process(request)

            self.logger.info(
                f"Successfully executed {self.name}",
                extra={
                    "request_id": str(request_id),
                    "success": getattr(response, "success", True),
                },
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Error executing {self.name}: {e}",
                extra={"request_id": str(request_id)},
                exc_info=True,
            )
            return self._create_error_response(str(e), request_id)

    @abstractmethod
    async def validate(self, request: TRequest) -> str | None:
        """
        Validate the request.

        Args:
            request: The request to validate

        Returns:
            Error message if validation fails, None otherwise
        """
        pass

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """
        Process the request and execute business logic.

        Args:
            request: The validated request

        Returns:
            The response
        """
        pass

    def _create_error_response(self, error: str, request_id: UUID) -> TResponse:
        """
        Create an error response.

        Args:
            error: Error message
            request_id: Request ID

        Returns:
            Error response
        """
        # This is a default implementation that assumes TResponse has these fields
        # Subclasses can override if needed
        return UseCaseResponse.error_response(error, request_id)  # type: ignore


class TransactionalUseCase(UseCase[TRequest, TResponse]):
    """
    Base class for use cases that require database transactions.

    Automatically manages transaction lifecycle.
    """

    def __init__(self, unit_of_work: Any, name: str | None = None) -> None:
        """
        Initialize transactional use case.

        Args:
            unit_of_work: Unit of work for transaction management
            name: Optional name for the use case
        """
        super().__init__(name)
        self.unit_of_work = unit_of_work

    async def execute(self, request: TRequest) -> TResponse:
        """
        Execute the use case within a transaction.

        Args:
            request: The use case request

        Returns:
            The use case response
        """
        request_id = getattr(request, "request_id", uuid4())

        self.logger.info(
            f"Starting transaction for {self.name}", extra={"request_id": str(request_id)}
        )

        async with self.unit_of_work as uow:
            try:
                # Call parent execute which handles validation and processing
                response = await super().execute(request)

                # Commit if successful
                if getattr(response, "success", True):
                    await uow.commit()
                    self.logger.info(
                        f"Transaction committed for {self.name}",
                        extra={"request_id": str(request_id)},
                    )
                else:
                    await uow.rollback()
                    self.logger.info(
                        f"Transaction rolled back for {self.name}",
                        extra={"request_id": str(request_id)},
                    )

                return response

            except Exception as e:
                await uow.rollback()
                self.logger.error(
                    f"Transaction rolled back due to error in {self.name}: {e}",
                    extra={"request_id": str(request_id)},
                    exc_info=True,
                )
                raise
