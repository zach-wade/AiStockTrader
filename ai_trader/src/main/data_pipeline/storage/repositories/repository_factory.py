"""
Repository Factory

Factory for creating and managing repository instances with dependency injection.
"""

from typing import Dict, Type, Optional, Any, TypeVar, Generic, Union, cast
import inspect

from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import RepositoryConfig

from .market_data_repository import MarketDataRepository
from .company_repository import CompanyRepository
from .feature_repository import FeatureRepository
from .scanner_data_repository import ScannerDataRepository
from .news_repository import NewsRepository
from .financials_repository import FinancialsRepository
from .specialized_repositories import (
    SentimentRepository,
    RatingsRepository,
    DividendsRepository,
    SocialSentimentRepository,
    GuidanceRepository
)

from main.utils.core import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
RepositoryT = TypeVar('RepositoryT')


class RepositoryRegistry:
    """Registry for repository types and their configurations."""
    
    def __init__(self):
        self._repository_types: Dict[str, Type] = {}
        self._repository_configs: Dict[str, RepositoryConfig] = {}
        self._instances: Dict[str, Any] = {}
        
        # Register built-in repositories
        self._register_built_in_repositories()
    
    def _register_built_in_repositories(self):
        """Register all built-in repository types."""
        self.register('market_data', MarketDataRepository)
        self.register('company', CompanyRepository)
        self.register('feature', FeatureRepository)
        self.register('scanner', ScannerDataRepository)
        self.register('news', NewsRepository)
        self.register('financials', FinancialsRepository)
        self.register('sentiment', SentimentRepository)
        self.register('ratings', RatingsRepository)
        self.register('dividends', DividendsRepository)
        self.register('social_sentiment', SocialSentimentRepository)
        self.register('guidance', GuidanceRepository)
    
    def register(
        self,
        name: str,
        repository_type: Type,
        config: Optional[RepositoryConfig] = None
    ):
        """Register a repository type."""
        self._repository_types[name] = repository_type
        if config:
            self._repository_configs[name] = config
        
        logger.debug(f"Registered repository: {name} -> {repository_type.__name__}")
    
    def get_repository_type(self, name: str) -> Optional[Type]:
        """Get repository type by name."""
        return self._repository_types.get(name)
    
    def get_repository_config(self, name: str) -> Optional[RepositoryConfig]:
        """Get repository configuration by name."""
        return self._repository_configs.get(name)
    
    def list_repositories(self) -> Dict[str, Type]:
        """List all registered repositories."""
        return self._repository_types.copy()
    
    def clear_instances(self):
        """Clear cached repository instances."""
        self._instances.clear()
    
    def get_instance(self, name: str) -> Optional[RepositoryT]:
        """Get cached repository instance."""
        return self._instances.get(name)
    
    def set_instance(self, name: str, instance: RepositoryT):
        """Cache repository instance."""
        self._instances[name] = instance


class RepositoryFactory:
    """
    Factory for creating repository instances with dependency injection.
    
    Handles creation, configuration, and lifecycle management of repositories.
    """
    
    def __init__(self):
        self.registry = RepositoryRegistry()
        self._default_config = RepositoryConfig()
    
    def create_repository(
        self,
        name: str,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None,
        use_singleton: bool = True
    ) -> RepositoryT:
        """
        Create a repository instance.
        
        Args:
            name: Repository name
            db_adapter: Database adapter
            config: Optional repository configuration
            use_singleton: Whether to use singleton pattern
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository type is not registered
            RuntimeError: If repository creation fails
        """
        # Check for cached instance if singleton
        if use_singleton:
            cached = self.registry.get_instance(name)
            if cached:
                return cached
        
        # Get repository type
        repo_type = self.registry.get_repository_type(name)
        if not repo_type:
            raise ValueError(f"Repository type '{name}' not registered. Available types: {list(self.registry.list_repositories().keys())}")
        
        # Get configuration
        repo_config = config or self.registry.get_repository_config(name) or self._default_config
        
        # Create instance - let exceptions bubble up
        instance = self._create_instance(repo_type, db_adapter, repo_config)
        
        # Cache if singleton
        if use_singleton and instance:
            self.registry.set_instance(name, instance)
        
        logger.debug(f"Created repository: {name} ({repo_type.__name__})")
        return instance
    
    def create_market_data_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> MarketDataRepository:
        """Create MarketDataRepository instance."""
        return cast(MarketDataRepository, self.create_repository('market_data', db_adapter, config))
    
    def create_company_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> CompanyRepository:
        """Create CompanyRepository instance."""
        return cast(CompanyRepository, self.create_repository('company', db_adapter, config))
    
    def create_feature_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> FeatureRepository:
        """Create FeatureRepository instance."""
        return cast(FeatureRepository, self.create_repository('feature', db_adapter, config))
    
    def create_scanner_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> ScannerDataRepository:
        """Create ScannerDataRepository instance."""
        return cast(ScannerDataRepository, self.create_repository('scanner', db_adapter, config))
    
    def create_news_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> NewsRepository:
        """Create NewsRepository instance."""
        return cast(NewsRepository, self.create_repository('news', db_adapter, config))
    
    def create_financials_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> FinancialsRepository:
        """Create FinancialsRepository instance."""
        return cast(FinancialsRepository, self.create_repository('financials', db_adapter, config))
    
    def create_social_sentiment_repository(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> SocialSentimentRepository:
        """Create SocialSentimentRepository instance."""
        return cast(SocialSentimentRepository, self.create_repository('social_sentiment', db_adapter, config))
    
    def create_all_repositories(
        self,
        db_adapter: IAsyncDatabase,
        config: Optional[RepositoryConfig] = None
    ) -> Dict[str, RepositoryT]:
        """Create all registered repositories."""
        repositories = {}
        
        for name in self.registry.list_repositories():
            try:
                repo = self.create_repository(name, db_adapter, config)
                repositories[name] = repo
            except Exception as e:
                logger.warning(f"Failed to create repository '{name}': {e}")
        
        logger.info(f"Created {len(repositories)} repositories")
        return repositories
    
    def register_custom_repository(
        self,
        name: str,
        repository_type: Type,
        config: Optional[RepositoryConfig] = None
    ):
        """Register a custom repository type."""
        self.registry.register(name, repository_type, config)
    
    def configure_repository(
        self,
        name: str,
        config: RepositoryConfig
    ):
        """Configure a repository type."""
        if name in self.registry.list_repositories():
            self.registry._repository_configs[name] = config
        else:
            logger.warning(f"Repository '{name}' not registered")
    
    def set_default_config(self, config: RepositoryConfig):
        """Set default repository configuration."""
        self._default_config = config
    
    def clear_cache(self):
        """Clear all cached repository instances."""
        self.registry.clear_instances()
        logger.debug("Repository cache cleared")
    
    def _create_instance(
        self,
        repo_type: Type[RepositoryT],
        db_adapter: IAsyncDatabase,
        config: RepositoryConfig
    ) -> RepositoryT:
        """Create repository instance with proper dependency injection."""
        
        # Get constructor signature
        sig = inspect.signature(repo_type.__init__)
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Build constructor arguments
        kwargs = {}
        
        if 'db_adapter' in params:
            kwargs['db_adapter'] = db_adapter
        
        if 'config' in params:
            kwargs['config'] = config
        
        # Create instance
        return repo_type(**kwargs)


# Global factory instance
_repository_factory = None


def get_repository_factory() -> RepositoryFactory:
    """Get global repository factory instance."""
    global _repository_factory
    if _repository_factory is None:
        _repository_factory = RepositoryFactory()
    return _repository_factory


def create_repository(
    name: str,
    db_adapter: IAsyncDatabase,
    config: Optional[RepositoryConfig] = None
) -> RepositoryT:
    """Convenience function to create a repository."""
    factory = get_repository_factory()
    return factory.create_repository(name, db_adapter, config)