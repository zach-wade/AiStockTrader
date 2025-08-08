"""
Company Repository Interface

Interface for company data and layer qualification operations.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from .base import IRepository, OperationResult


class ICompanyRepository(IRepository):
    """Interface for company data repositories."""
    
    @abstractmethod
    async def get_company(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get company information by symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company data or None
        """
        pass
    
    @abstractmethod
    async def get_companies(
        self,
        symbols: Optional[List[str]] = None,
        layer: Optional[int] = None,
        is_active: bool = True
    ) -> pd.DataFrame:
        """
        Get multiple companies with optional filtering.
        
        Args:
            symbols: Optional list of symbols
            layer: Optional layer filter (1, 2, or 3)
            is_active: Filter for active companies
            
        Returns:
            DataFrame with company data
        """
        pass
    
    @abstractmethod
    async def update_layer_qualification(
        self,
        symbol: str,
        layer: int,
        qualified: bool,
        reason: Optional[str] = None
    ) -> OperationResult:
        """
        Update layer qualification status.
        
        Args:
            symbol: Stock symbol
            layer: Layer number (1, 2, or 3)
            qualified: Qualification status
            reason: Optional reason for change
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_layer_qualified_symbols(
        self,
        layer: int
    ) -> List[str]:
        """
        Get symbols qualified for a specific layer.
        
        Args:
            layer: Layer number (1, 2, or 3)
            
        Returns:
            List of qualified symbols
        """
        pass
    
    @abstractmethod
    async def batch_update_layer_qualifications(
        self,
        updates: List[Dict[str, Any]]
    ) -> OperationResult:
        """
        Batch update layer qualifications.
        
        Args:
            updates: List of update dictionaries with symbol, layer, qualified
            
        Returns:
            Operation result with update count
        """
        pass
    
    @abstractmethod
    async def get_sector_companies(
        self,
        sector: str,
        layer: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get companies in a specific sector.
        
        Args:
            sector: Sector name
            layer: Optional layer filter
            
        Returns:
            DataFrame with company data
        """
        pass
    
    @abstractmethod
    async def get_industry_companies(
        self,
        industry: str,
        layer: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get companies in a specific industry.
        
        Args:
            industry: Industry name
            layer: Optional layer filter
            
        Returns:
            DataFrame with company data
        """
        pass
    
    @abstractmethod
    async def update_company_metadata(
        self,
        symbol: str,
        metadata: Dict[str, Any]
    ) -> OperationResult:
        """
        Update company metadata.
        
        Args:
            symbol: Stock symbol
            metadata: Metadata to update
            
        Returns:
            Operation result
        """
        pass
    
    @abstractmethod
    async def get_company_statistics(
        self
    ) -> Dict[str, Any]:
        """
        Get statistics about companies in the database.
        
        Returns:
            Dictionary with statistics (counts by layer, sector, etc.)
        """
        pass
    
    @abstractmethod
    async def search_companies(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search companies by text query.
        
        Args:
            query: Search query
            fields: Optional fields to search in
            limit: Maximum results
            
        Returns:
            DataFrame with matching companies
        """
        pass