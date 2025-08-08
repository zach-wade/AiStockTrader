"""
Ingestion Clients

Standardized clients for fetching data from various sources.
"""

from .base_client import BaseIngestionClient, ClientConfig, FetchResult
from .polygon_market_client import PolygonMarketClient
from .polygon_news_client import PolygonNewsClient
from .polygon_fundamentals_client import PolygonFundamentalsClient
from .polygon_corporate_actions_client import PolygonCorporateActionsClient

__all__ = [
    'BaseIngestionClient',
    'ClientConfig',
    'FetchResult',
    'PolygonMarketClient',
    'PolygonNewsClient',
    'PolygonFundamentalsClient',
    'PolygonCorporateActionsClient',
]