"""
Application Layer - Use Cases and Orchestration

This layer contains:
- Commands: Actions that change state
- Queries: Actions that retrieve information
- Handlers: Orchestration of business logic
- Services: Application-level services
- Interfaces: Repository contracts and abstractions

Depends on domain layer, orchestrates business logic.
Defines interfaces that infrastructure layer must implement.
"""

from .interfaces import (  # Repository interfaces; Unit of Work interfaces; Exceptions
    ConcurrencyError,
    ConnectionError,
    DuplicateEntityError,
    EntityNotFoundError,
    FactoryError,
    IntegrityError,
    IOrderRepository,
    IPortfolioRepository,
    IPositionRepository,
    ITransactionManager,
    IUnitOfWork,
    IUnitOfWorkFactory,
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
