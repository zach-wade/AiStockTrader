"""
Repository Provider

Provides repository instances and manages dependencies to break circular imports.
"""

# Standard library imports
from typing import Any, Protocol, runtime_checkable

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import RepositoryConfig
from main.utils.core import get_logger

from .repository_factory import RepositoryFactory, get_repository_factory

logger = get_logger(__name__)


@runtime_checkable
class IRepositoryProvider(Protocol):
    """Interface for repository providers."""

    def get_market_data_repository(self) -> Any:
        """Get market data repository."""
        ...

    def get_company_repository(self) -> Any:
        """Get company repository."""
        ...

    def get_feature_repository(self) -> Any:
        """Get feature repository."""
        ...

    def get_scanner_repository(self) -> Any:
        """Get scanner data repository."""
        ...

    def get_news_repository(self) -> Any:
        """Get news repository."""
        ...

    def get_financials_repository(self) -> Any:
        """Get financials repository."""
        ...


class RepositoryProvider(IRepositoryProvider):
    """
    Provides repository instances with lazy loading and dependency management.

    Breaks circular dependencies by providing repositories on-demand
    and managing their lifecycle.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: RepositoryConfig | None = None,
        factory: RepositoryFactory | None = None,
    ):
        """
        Initialize the repository provider.

        Args:
            db_adapter: Database adapter for repositories
            config: Default repository configuration
            factory: Optional repository factory (uses global if None)
        """
        self._db_adapter = db_adapter
        self._config = config or RepositoryConfig()
        self._factory = factory or get_repository_factory()

        # Lazy-loaded repository cache
        self._repositories: dict[str, Any] = {}

        logger.debug("RepositoryProvider initialized")

    def get_market_data_repository(self):
        """Get market data repository."""
        return self._get_repository("market_data")

    def get_company_repository(self):
        """Get company repository."""
        return self._get_repository("company")

    def get_feature_repository(self):
        """Get feature repository."""
        return self._get_repository("feature")

    def get_scanner_repository(self):
        """Get scanner data repository."""
        return self._get_repository("scanner")

    def get_news_repository(self):
        """Get news repository."""
        return self._get_repository("news")

    def get_financials_repository(self):
        """Get financials repository."""
        return self._get_repository("financials")

    def get_sentiment_repository(self):
        """Get sentiment repository."""
        return self._get_repository("sentiment")

    def get_ratings_repository(self):
        """Get ratings repository."""
        return self._get_repository("ratings")

    def get_dividends_repository(self):
        """Get dividends repository."""
        return self._get_repository("dividends")

    def get_social_sentiment_repository(self):
        """Get social sentiment repository."""
        return self._get_repository("social_sentiment")

    def get_guidance_repository(self):
        """Get guidance repository."""
        return self._get_repository("guidance")

    def get_repository(self, name: str) -> Any | None:
        """Get repository by name."""
        return self._get_repository(name)

    def get_all_repositories(self) -> dict[str, Any]:
        """Get all available repositories."""
        all_repos = {}

        # Get all registered repository names from factory
        for name in self._factory.registry.list_repositories():
            repo = self._get_repository(name)
            if repo:
                all_repos[name] = repo

        return all_repos

    def refresh_repository(self, name: str) -> Any | None:
        """Refresh a repository instance (recreate)."""
        if name in self._repositories:
            del self._repositories[name]

        return self._get_repository(name)

    def configure_repository(self, name: str, config: RepositoryConfig):
        """Configure a specific repository."""
        self._factory.configure_repository(name, config)

        # Clear cached instance to pick up new config
        if name in self._repositories:
            del self._repositories[name]

    def clear_cache(self):
        """Clear all cached repository instances."""
        self._repositories.clear()
        self._factory.clear_cache()
        logger.debug("Repository provider cache cleared")

    def health_check(self) -> dict[str, bool]:
        """Check health of all repositories."""
        health_status = {}

        for name in self._factory.registry.list_repositories():
            try:
                repo = self._get_repository(name)
                health_status[name] = repo is not None
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_status[name] = False

        return health_status

    def _get_repository(self, name: str) -> Any | None:
        """Get repository with lazy loading."""
        if name not in self._repositories:
            try:
                repo = self._factory.create_repository(
                    name,
                    self._db_adapter,
                    self._config,
                    use_singleton=False,  # Provider manages instances
                )

                if repo:
                    self._repositories[name] = repo
                    logger.debug(f"Loaded repository: {name}")
                else:
                    logger.warning(f"Failed to create repository: {name}")
                    return None

            except Exception as e:
                logger.error(f"Error creating repository '{name}': {e}")
                return None

        return self._repositories.get(name)


class RepositoryServiceLocator:
    """
    Service locator pattern for repository access.

    Provides a static interface to access repositories without
    direct dependency injection.
    """

    _provider: RepositoryProvider | None = None

    @classmethod
    def initialize(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        """Initialize the service locator with a repository provider."""
        self._provider = RepositoryProvider(db_adapter, config)
        logger.info("Repository service locator initialized")

    @classmethod
    def get_provider(cls) -> RepositoryProvider | None:
        """Get the repository provider."""
        if cls._provider is None:
            logger.warning("Repository service locator not initialized")
        return cls._provider

    @classmethod
    def get_market_data_repository(cls):
        """Get market data repository."""
        provider = cls.get_provider()
        return provider.get_market_data_repository() if provider else None

    @classmethod
    def get_company_repository(cls):
        """Get company repository."""
        provider = cls.get_provider()
        return provider.get_company_repository() if provider else None

    @classmethod
    def get_feature_repository(cls):
        """Get feature repository."""
        provider = cls.get_provider()
        return provider.get_feature_repository() if provider else None

    @classmethod
    def get_scanner_repository(cls):
        """Get scanner repository."""
        provider = cls.get_provider()
        return provider.get_scanner_repository() if provider else None

    @classmethod
    def get_news_repository(cls):
        """Get news repository."""
        provider = cls.get_provider()
        return provider.get_news_repository() if provider else None

    @classmethod
    def get_financials_repository(cls):
        """Get financials repository."""
        provider = cls.get_provider()
        return provider.get_financials_repository() if provider else None

    @classmethod
    def get_repository(cls, name: str):
        """Get repository by name."""
        provider = cls.get_provider()
        return provider.get_repository(name) if provider else None

    @classmethod
    def clear_cache(cls):
        """Clear repository cache."""
        if cls._provider:
            cls._provider.clear_cache()

    @classmethod
    def reset(cls):
        """Reset the service locator."""
        cls._provider = None
        logger.info("Repository service locator reset")


# Convenience functions for common patterns
def create_repository_provider(
    db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None
) -> RepositoryProvider:
    """Create a repository provider instance."""
    return RepositoryProvider(db_adapter, config)


def initialize_repository_service_locator(
    db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None
):
    """Initialize the global repository service locator."""
    RepositoryServiceLocator.initialize(db_adapter, config)
