"""
Application Interfaces - Repository Contracts

This module defines the interface contracts that the infrastructure layer
must implement. Following the dependency inversion principle, the application
layer defines what it needs, and the infrastructure layer provides it.
"""

from .exceptions import (
    ConcurrencyError,
    ConnectionError,
    DuplicateEntityError,
    EntityNotFoundError,
    FactoryError,
    IntegrityError,
    OrderNotFoundError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    RepositoryError,
    TimeoutError,
    TransactionAlreadyActiveError,
    TransactionCommitError,
    TransactionError,
    TransactionNotActiveError,
    TransactionRollbackError,
    ValidationError,
)
from .repositories import IOrderRepository, IPortfolioRepository, IPositionRepository
from .unit_of_work import ITransactionManager, IUnitOfWork, IUnitOfWorkFactory

__all__ = [
    # Repository interfaces
    "IOrderRepository",
    "IPositionRepository",
    "IPortfolioRepository",
    # Unit of Work interfaces
    "IUnitOfWork",
    "IUnitOfWorkFactory",
    "ITransactionManager",
    # Exceptions
    "RepositoryError",
    "EntityNotFoundError",
    "OrderNotFoundError",
    "PositionNotFoundError",
    "PortfolioNotFoundError",
    "DuplicateEntityError",
    "ConcurrencyError",
    "ValidationError",
    "TransactionError",
    "TransactionNotActiveError",
    "TransactionAlreadyActiveError",
    "TransactionCommitError",
    "TransactionRollbackError",
    "ConnectionError",
    "TimeoutError",
    "IntegrityError",
    "FactoryError",
]
