"""
Integration tests for multi-layer scanner system.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from typing import Dict, List

from main.scanners.orchestrator_parallel import ScannerOrchestrator
from main.universe.universe_manager import UniverseManager
from main.repositories.company_repository import CompanyRepository


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiLayerScanning:
    """Test multi-layer scanner coordination and universe management."""
    
    @pytest_asyncio.fixture
    async def scanner_orchestrator(self, test_config, test_db_pool):
        """Create scanner orchestrator instance."""
        orchestrator = ScannerOrchestrator(test_config)
        orchestrator.db_pool = test_db_pool
        
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest_asyncio.fixture
    async def universe_manager(self, test_config, test_db_pool):
        """Create universe manager instance."""
        company_repo = CompanyRepository(test_db_pool)
        
        # Insert test companies
        test_companies = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'market_cap': 3000000000000},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'market_cap': 2500000000000},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'market_cap': 2000000000000},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'market_cap': 1500000000000},
            {'symbol': 'META', 'name': 'Meta Platforms', 'market_cap': 1000000000000},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corp.', 'market_cap': 1200000000000},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'market_cap': 800000000000},
        ]
        
        for company in test_companies:
            await company_repo.create(company)
        
        manager = UniverseManager(test_config, company_repo)
        await manager.initialize()
        
        yield manager
        
        # Cleanup
        for company in test_companies:
            await company_repo.delete(company['symbol'])
    
    async def test_layer_progression(
        self,
        scanner_orchestrator,
        universe_manager
    ):
        """Test symbols progress through scanner layers."""
        # Run Layer 0 scan
        layer0_results = await scanner_orchestrator.run_layer_scan(
            layer=0,
            scan_type='daily'
        )
        
        # Verify Layer 0 results
        assert layer0_results['success']
        assert layer0_results['layer'] == 0
        assert len(layer0_results['qualified_symbols']) > 0
        
        # Update universe with Layer 0 results
        await universe_manager.update_layer_qualification(
            layer=0,
            qualified_symbols=layer0_results['qualified_symbols']
        )
        
        # Run Layer 1 scan
        layer1_results = await scanner_orchestrator.run_layer_scan(
            layer=1,
            scan_type='daily'
        )
        
        # Verify Layer 1 results
        assert layer1_results['success']
        assert layer1_results['layer'] == 1
        # Layer 1 should have fewer symbols than Layer 0
        assert len(layer1_results['qualified_symbols']) <= len(layer0_results['qualified_symbols'])
        
        # Verify universe state
        universe_stats = await universe_manager.get_universe_stats()
        assert universe_stats['layer0_count'] == len(layer0_results['qualified_symbols'])
        assert universe_stats['layer1_count'] == len(layer1_results['qualified_symbols'])
    
    async def test_concurrent_layer_scanning(
        self,
        scanner_orchestrator,
        performance_threshold
    ):
        """Test concurrent scanning across multiple layers."""
        import time
        import asyncio
        
        start_time = time.time()
        
        # Run multiple layer scans concurrently
        scan_tasks = [
            scanner_orchestrator.run_layer_scan(layer=0, scan_type='realtime'),
            scanner_orchestrator.run_layer_scan(layer=1, scan_type='realtime'),
            scanner_orchestrator.run_layer_scan(layer=2, scan_type='realtime'),
        ]
        
        results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        end_time = time.time()
        scan_time_ms = (end_time - start_time) * 1000
        
        # Verify results
        successful_scans = [r for r in results if isinstance(r, dict) and r.get('success')]
        assert len(successful_scans) >= 2  # At least 2 layers should succeed
        
        # Check performance
        assert scan_time_ms < performance_threshold['scanner']['scan_time_ms']
        
        # Verify no race conditions
        symbols_by_layer = {}
        for result in successful_scans:
            if isinstance(result, dict):
                layer = result['layer']
                symbols_by_layer[layer] = set(result['qualified_symbols'])
        
        # Higher layers should be subsets of lower layers
        if 0 in symbols_by_layer and 1 in symbols_by_layer:
            assert symbols_by_layer[1].issubset(symbols_by_layer[0])
    
    async def test_scan_failure_recovery(
        self,
        scanner_orchestrator,
        monkeypatch
    ):
        """Test scanner recovery from failures."""
        # Track scan attempts
        scan_attempts = []
        
        # Mock scanner to fail first attempt
        original_scan = scanner_orchestrator._execute_layer_scan
        
        async def mock_failing_scan(layer, scan_type):
            scan_attempts.append((layer, scan_type))
            if len(scan_attempts) == 1:
                raise Exception("Simulated scan failure")
            return await original_scan(layer, scan_type)
        
        monkeypatch.setattr(
            scanner_orchestrator,
            '_execute_layer_scan',
            mock_failing_scan
        )
        
        # Run scan with retry
        result = await scanner_orchestrator.run_layer_scan(
            layer=0,
            scan_type='daily',
            retry_on_failure=True
        )
        
        # Verify recovery
        assert result['success']
        assert len(scan_attempts) == 2  # Initial attempt + retry
        assert 'retry_count' in result
        assert result['retry_count'] == 1
    
    async def test_universe_consistency(
        self,
        scanner_orchestrator,
        universe_manager,
        test_db_pool
    ):
        """Test universe consistency across scanner operations."""
        # Get initial universe state
        initial_stats = await universe_manager.get_universe_stats()
        
        # Run full scan cycle
        scan_results = await scanner_orchestrator.run_full_scan_cycle()
        
        # Verify scan results
        assert scan_results['success']
        assert 'layers_scanned' in scan_results
        assert scan_results['layers_scanned'] >= 3
        
        # Check universe updates
        final_stats = await universe_manager.get_universe_stats()
        
        # Verify consistency
        assert final_stats['total_companies'] == initial_stats['total_companies']
        assert final_stats['layer0_count'] >= 0
        assert final_stats['layer1_count'] <= final_stats['layer0_count']
        assert final_stats['layer2_count'] <= final_stats['layer1_count']
        assert final_stats['layer3_count'] <= final_stats['layer2_count']
        
        # Verify database consistency
        async with test_db_pool.acquire() as conn:
            # Check no orphaned qualifications
            orphan_check = await conn.fetchval("""
                SELECT COUNT(*) FROM layer_qualifications lq
                LEFT JOIN companies c ON lq.symbol = c.symbol
                WHERE c.symbol IS NULL
            """)
            assert orphan_check == 0
    
    async def test_scan_metadata_tracking(
        self,
        scanner_orchestrator,
        test_db_pool
    ):
        """Test scan metadata and history tracking."""
        # Run scan
        result = await scanner_orchestrator.run_layer_scan(
            layer=1,
            scan_type='daily'
        )
        
        # Verify metadata
        assert 'scan_id' in result
        assert 'start_time' in result
        assert 'end_time' in result
        assert 'duration_ms' in result
        assert 'symbols_scanned' in result
        assert 'scan_metadata' in result
        
        metadata = result['scan_metadata']
        assert 'scanner_version' in metadata
        assert 'config_hash' in metadata
        assert 'filters_applied' in metadata
        
        # Verify scan history in database
        async with test_db_pool.acquire() as conn:
            scan_record = await conn.fetchrow(
                "SELECT * FROM scan_history WHERE scan_id = $1",
                result['scan_id']
            )
            
            assert scan_record is not None
            assert scan_record['layer'] == 1
            assert scan_record['scan_type'] == 'daily'
            assert scan_record['status'] == 'completed'
            assert scan_record['qualified_count'] == len(result['qualified_symbols'])