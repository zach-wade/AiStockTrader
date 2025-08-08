#!/usr/bin/env python3
"""
Test archive integration with bulk loaders.

This script tests that the archive save functionality works correctly
for news, corporate actions, and fundamentals loaders.
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
sys.path.insert(0, 'src')

from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.ingestion.loaders.news import NewsBulkLoader
from main.data_pipeline.ingestion.loaders.corporate_actions import CorporateActionsBulkLoader
from main.data_pipeline.ingestion.loaders.fundamentals import FundamentalsBulkLoader
from main.data_pipeline.services.ingestion import TextProcessingService, DeduplicationService
from main.interfaces.ingestion import BulkLoadConfig
from unittest.mock import AsyncMock, MagicMock


async def test_news_archive_integration():
    """Test that news loader correctly archives data."""
    print("\n=== Testing News Archive Integration ===")
    
    # Create temporary archive directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup archive
        archive_config = {
            'storage_type': 'local',
            'local_path': temp_dir
        }
        archive = DataArchive(archive_config)
        
        # Mock database adapter
        db_adapter = AsyncMock()
        db_adapter.acquire = MagicMock()
        db_adapter.acquire.return_value.__aenter__ = AsyncMock()
        db_adapter.acquire.return_value.__aexit__ = AsyncMock()
        
        # Mock services
        text_processor = MagicMock(spec=TextProcessingService)
        text_processor.process_article = lambda x: {
            'title': x.get('title', ''),
            'content': x.get('content', ''),
            'symbols': ['AAPL', 'GOOGL'],
            'keywords': ['tech', 'news'],
            'sentiment_positive': 0.7,
            'sentiment_negative': 0.1,
            'sentiment_overall': 'positive'
        }
        
        deduplicator = AsyncMock(spec=DeduplicationService)
        deduplicator.deduplicate_batch = AsyncMock(return_value=([], 0))
        
        # Create loader with archive
        config = BulkLoadConfig(
            buffer_size=10,
            max_memory_mb=100,
            batch_timeout_seconds=10.0
        )
        
        loader = NewsBulkLoader(
            db_adapter=db_adapter,
            text_processor=text_processor,
            deduplicator=deduplicator,
            archive=archive,
            config=config
        )
        
        # Test data
        test_articles = [
            {
                'id': 'test_1',
                'title': 'Apple Reports Record Quarter',
                'content': 'Apple Inc. reported record revenue...',
                'published_utc': datetime.now(timezone.utc).isoformat(),
                'publisher': {'name': 'Reuters'},
                'article_url': 'https://example.com/article1'
            },
            {
                'id': 'test_2',
                'title': 'Google Announces AI Update',
                'content': 'Google unveiled new AI features...',
                'published_utc': datetime.now(timezone.utc).isoformat(),
                'publisher': {'name': 'Bloomberg'},
                'article_url': 'https://example.com/article2'
            }
        ]
        
        # Deduplicate will return the articles as-is
        deduplicator.deduplicate_batch = AsyncMock(return_value=(test_articles, 0))
        
        # Load data (should trigger archive)
        result = await loader.load(
            data=test_articles,
            symbols=['AAPL', 'GOOGL'],
            source='polygon'
        )
        
        # Force flush to trigger archive
        await loader.flush_all()
        
        # Check archive was populated
        archive_path = Path(temp_dir) / 'raw' / 'polygon' / 'news' / 'symbol=_aggregated'
        
        # List files in archive
        if archive_path.exists():
            files = list(archive_path.rglob('*.parquet'))
            print(f"✓ Archive created {len(files)} parquet files")
            
            if files:
                # Read first file to verify content
                import pandas as pd
                df = pd.read_parquet(files[0])
                print(f"✓ Archived {len(df)} news records")
                
                # Check metadata file
                meta_file = files[0].with_suffix('.parquet.meta')
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"✓ Metadata includes {metadata.get('record_count', 0)} records")
                    print(f"✓ Symbols: {metadata.get('symbols', [])}")
        else:
            print("✗ Archive directory not created")
            
    print("=== News Archive Test Complete ===")


async def test_corporate_actions_archive():
    """Test that corporate actions loader correctly archives data."""
    print("\n=== Testing Corporate Actions Archive Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup archive
        archive_config = {
            'storage_type': 'local',
            'local_path': temp_dir
        }
        archive = DataArchive(archive_config)
        
        # Mock database adapter
        db_adapter = AsyncMock()
        
        # Create loader with archive
        config = BulkLoadConfig(
            buffer_size=10,
            max_memory_mb=100,
            batch_timeout_seconds=10.0
        )
        
        loader = CorporateActionsBulkLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=config
        )
        
        # Test data
        test_actions = [
            {
                'symbol': 'AAPL',
                'type': 'dividend',
                'ex_date': datetime.now(timezone.utc),
                'amount': 0.25,
                'currency': 'USD'
            },
            {
                'symbol': 'GOOGL',
                'type': 'split',
                'ex_date': datetime.now(timezone.utc),
                'split_ratio': 2.0
            }
        ]
        
        # Load data (should trigger archive)
        result = await loader.load(
            data=test_actions,
            symbols=['AAPL', 'GOOGL'],
            source='polygon'
        )
        
        # Force flush to trigger archive
        await loader.flush_all()
        
        # Check archive
        archive_path = Path(temp_dir) / 'raw' / 'polygon' / 'corporate_actions'
        
        if archive_path.exists():
            files = list(archive_path.rglob('*.parquet'))
            print(f"✓ Archive created {len(files)} parquet files")
            
            # Check for different action types
            dividend_files = [f for f in files if 'dividend' in str(f)]
            split_files = [f for f in files if 'split' in str(f)]
            
            if dividend_files:
                print(f"✓ Archived dividend actions")
            if split_files:
                print(f"✓ Archived split actions")
        else:
            print("✗ Archive directory not created")
            
    print("=== Corporate Actions Archive Test Complete ===")


async def test_fundamentals_archive():
    """Test that fundamentals loader correctly archives data."""
    print("\n=== Testing Fundamentals Archive Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup archive
        archive_config = {
            'storage_type': 'local',
            'local_path': temp_dir
        }
        archive = DataArchive(archive_config)
        
        # Mock database adapter
        db_adapter = AsyncMock()
        
        # Create loader with archive
        config = BulkLoadConfig(
            buffer_size=10,
            max_memory_mb=100,
            batch_timeout_seconds=10.0
        )
        
        loader = FundamentalsBulkLoader(
            db_adapter=db_adapter,
            archive=archive,
            config=config
        )
        
        # Test data
        test_fundamentals = [
            {
                'symbol': 'AAPL',
                'year': 2024,
                'period': 'Q1',
                'revenue': 100000000,
                'net_income': 25000000,
                'eps_basic': 1.50
            },
            {
                'symbol': 'GOOGL',
                'year': 2024,
                'period': 'Q1',
                'revenue': 80000000,
                'net_income': 20000000,
                'eps_basic': 1.25
            }
        ]
        
        # Load data (should trigger archive)
        result = await loader.load(
            data=test_fundamentals,
            symbols=['AAPL', 'GOOGL'],
            source='polygon'
        )
        
        # Force flush to trigger archive
        await loader.flush_all()
        
        # Check archive
        archive_path = Path(temp_dir) / 'raw' / 'polygon' / 'fundamentals'
        
        if archive_path.exists():
            files = list(archive_path.rglob('*.parquet'))
            print(f"✓ Archive created {len(files)} parquet files")
            
            # Check for period-based files
            q1_files = [f for f in files if '2024_Q1' in str(f)]
            
            if q1_files:
                print(f"✓ Archived Q1 2024 fundamentals")
                
                # Read and check content
                import pandas as pd
                df = pd.read_parquet(q1_files[0])
                print(f"✓ Archived {len(df)} financial records")
                
                # Check metadata
                meta_file = q1_files[0].with_suffix('.parquet.meta')
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"✓ Metadata shows {metadata.get('metrics', {}).get('has_revenue', 0)} records with revenue")
        else:
            print("✗ Archive directory not created")
            
    print("=== Fundamentals Archive Test Complete ===")


async def main():
    """Run all archive integration tests."""
    print("=" * 60)
    print("Archive Integration Tests")
    print("=" * 60)
    
    try:
        await test_news_archive_integration()
        await test_corporate_actions_archive()
        await test_fundamentals_archive()
        
        print("\n" + "=" * 60)
        print("✅ All archive integration tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)