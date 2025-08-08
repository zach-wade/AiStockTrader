"""
Integration tests for Performance.

Tests performance characteristics, scalability, and resource efficiency
across all events components under various load conditions.
"""

import asyncio
import pytest
import time
import resource
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any, Set
import json
import gc
import psutil
import os

from main.interfaces.events import IEventBus, Event, EventType
from main.events.core import EventBusFactory, EventBusConfig
from main.events.handlers.scanner_feature_bridge import ScannerFeatureBridge
from main.events.handlers.feature_pipeline_handler import FeaturePipelineHandler
from main.events.types import (
    ScannerAlertEvent, FeatureRequestEvent, 
    FeatureComputedEvent, ScanAlert, AlertType
)
from tests.fixtures.events.mock_database import create_mock_db_pool


@pytest.fixture
async def performance_optimized_system():
    """Create event system optimized for performance testing."""
    # Performance-optimized configuration
    mock_config = {
        'events': {
            'batch_size': 25,
            'batch_interval_seconds': 0.01,
            'max_queue_size': 10000,
            'worker_pool_size': 4
        },
        'scanner_bridge': {
            'batch_size': 20,
            'batch_timeout': 0.01,
            'max_batches': 1000,
            'priority_boost_threshold': 0.8
        },
        'feature_pipeline': {
            'batch_size': 30,
            'queue_timeout': 60,
            'max_retries': 1,  # Reduced for performance
            'computation_timeout': 30
        }
    }
    
    # High-performance mock feature service
    mock_feature_service = AsyncMock()
    
    async def fast_compute_features(symbols, features, **kwargs):
        # Minimal processing delay
        await asyncio.sleep(0.001)
        return {
            symbol: {
                feature: {'computed_at': time.time(), 'value': hash(symbol) % 1000}
                for feature in features
            }
            for symbol in symbols
        }
    
    mock_feature_service.compute_features.side_effect = fast_compute_features
    
    # Initialize system components with performance config
    config = EventBusConfig(
        max_queue_size=10000,
        max_workers=4,
        enable_history=False,
        enable_dlq=False
    )
    event_bus = EventBusFactory.create(config)
    await event_bus.start()
    
    bridge = ScannerFeatureBridge(event_bus=event_bus, config=mock_config)
    await bridge.start()
    
    pipeline_handler = FeaturePipelineHandler(
        event_bus=event_bus,
        feature_service=mock_feature_service,
        config=mock_config
    )
    await pipeline_handler.start()
    
    # Subscribe pipeline to requests
    event_bus.subscribe(EventType.FEATURE_REQUEST, pipeline_handler.handle_feature_request)
    
    yield {
        'event_bus': event_bus,
        'bridge': bridge,
        'pipeline_handler': pipeline_handler,
        'feature_service': mock_feature_service,
        'config': mock_config
    }
    
    # Cleanup
    await pipeline_handler.stop()
    await bridge.stop()
    await event_bus.stop()


@pytest.fixture
def performance_test_data():
    """Create test data for performance testing."""
    symbols_pool = [f'PERF_STOCK_{i:04d}' for i in range(1000)]
    alert_types = ['high_volume', 'breakout', 'catalyst_detected', 'momentum_shift', 'gap_up', 'gap_down']
    scanner_names = [f'scanner_{i}' for i in range(10)]
    
    return {
        'symbols_pool': symbols_pool,
        'alert_types': alert_types,
        'scanner_names': scanner_names
    }


class TestPerformanceIntegration:
    """Test performance characteristics across all components."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_alert_processing(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test system throughput under high alert volume."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        test_data = performance_test_data
        
        # Performance tracking
        start_time = time.time()
        num_alerts = 1000
        
        # Generate high-volume alerts
        alerts = []
        for i in range(num_alerts):
            alert = ScannerAlertEvent(
                symbol=test_data['symbols_pool'][i % len(test_data['symbols_pool'])],
                alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                score=0.3 + (i % 70) / 100.0,
                scanner_name=test_data['scanner_names'][i % len(test_data['scanner_names'])],
                metadata={'batch_id': i // 50, 'test_index': i}
            )
            alerts.append(alert)
        
        # Measure publishing throughput
        publish_start = time.time()
        
        # Publish in optimized batches
        batch_size = 50
        for i in range(0, num_alerts, batch_size):
            batch = alerts[i:i + batch_size]
            await asyncio.gather(*[event_bus.publish(alert) for alert in batch])
        
        publish_end = time.time()
        publish_duration = publish_end - publish_start
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Verify throughput performance
        publish_throughput = num_alerts / publish_duration
        total_throughput = num_alerts / total_duration
        
        # Performance assertions
        assert publish_throughput > 500, f"Publish throughput {publish_throughput:.2f} alerts/sec too low"
        assert total_throughput > 200, f"Total throughput {total_throughput:.2f} alerts/sec too low"
        
        # Verify processing completeness
        bridge_stats = bridge.get_stats()
        assert bridge_stats['alerts_received_total'] >= num_alerts * 0.95  # 95% processed
        assert bridge_stats['feature_requests_sent_total'] >= 10  # Multiple batches sent
        
        print(f"High throughput test: {publish_throughput:.2f} pub/sec, {total_throughput:.2f} total/sec")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_scalability(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test scalability under concurrent processing load."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        test_data = performance_test_data
        
        # Concurrent processing scenarios
        scenarios = [
            {'name': 'scanner_alerts', 'count': 300, 'delay': 0.001},
            {'name': 'feature_requests', 'count': 200, 'delay': 0.001},
            {'name': 'mixed_events', 'count': 400, 'delay': 0.001}
        ]
        
        async def alert_generator(scenario):
            """Generate alerts for a scenario."""
            scenario_start = time.time()
            events_published = 0
            
            for i in range(scenario['count']):
                if scenario['name'] == 'scanner_alerts':
                    event = ScannerAlertEvent(
                        symbol=test_data['symbols_pool'][i % 100],
                        alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                        score=0.5 + (i % 50) / 100.0,
                        scanner_name=f"{scenario['name']}_scanner_{i % 3}"
                    )
                elif scenario['name'] == 'feature_requests':
                    event = FeatureRequestEvent(
                        symbols=[test_data['symbols_pool'][i % 100]],
                        features=['price_features', 'volume_features'],
                        requester=f"{scenario['name']}_requester_{i % 3}",
                        priority=5 + (i % 5)
                    )
                else:  # mixed_events
                    if i % 2 == 0:
                        event = ScannerAlertEvent(
                            symbol=test_data['symbols_pool'][i % 100],
                            alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                            score=0.6 + (i % 40) / 100.0,
                            scanner_name=f"mixed_scanner_{i % 2}"
                        )
                    else:
                        event = FeatureRequestEvent(
                            symbols=[test_data['symbols_pool'][i % 100]],
                            features=['trend_features'],
                            requester=f"mixed_requester_{i % 2}",
                            priority=6
                        )
                
                await event_bus.publish(event)
                events_published += 1
                
                if scenario['delay'] > 0:
                    await asyncio.sleep(scenario['delay'])
            
            scenario_end = time.time()
            return {
                'scenario': scenario['name'],
                'events_published': events_published,
                'duration': scenario_end - scenario_start,
                'throughput': events_published / (scenario_end - scenario_start)
            }
        
        # Run scenarios concurrently
        concurrent_start = time.time()
        results = await asyncio.gather(*[alert_generator(scenario) for scenario in scenarios])
        concurrent_end = time.time()
        
        # Wait for processing completion
        await asyncio.sleep(3.0)
        
        total_concurrent_duration = concurrent_end - concurrent_start
        total_events = sum(result['events_published'] for result in results)
        concurrent_throughput = total_events / total_concurrent_duration
        
        # Verify concurrent scalability
        assert concurrent_throughput > 800, f"Concurrent throughput {concurrent_throughput:.2f} events/sec too low"
        
        # Verify each scenario maintained good performance
        for result in results:
            assert result['throughput'] > 200, f"Scenario {result['scenario']} throughput {result['throughput']:.2f} too low"
        
        # Verify system handled concurrent load
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        # Should have processed significant portion of events
        alert_count = sum(r['events_published'] for r in results if 'alert' in r['scenario'])
        request_count = sum(r['events_published'] for r in results if 'request' in r['scenario'])
        
        assert bridge_stats['alerts_received_total'] >= alert_count * 0.9
        assert pipeline_stats['requests_received'] >= request_count * 0.9
        
        print(f"Concurrent scalability: {concurrent_throughput:.2f} events/sec across {len(scenarios)} scenarios")
    
    @pytest.mark.asyncio 
    async def test_memory_efficiency_under_load(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test memory efficiency during sustained high load."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        test_data = performance_test_data
        
        # Memory tracking
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection before test
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Sustained load configuration
        waves = 10
        events_per_wave = 500
        memory_measurements = []
        
        for wave in range(waves):
            wave_start_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate wave of events
            wave_events = []
            for i in range(events_per_wave):
                event = ScannerAlertEvent(
                    symbol=test_data['symbols_pool'][i % 200],  # Cycling symbols
                    alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                    score=0.5 + (i % 50) / 100.0,
                    scanner_name=f"memory_test_scanner_{wave}_{i % 3}",
                    metadata={'wave': wave, 'index': i, 'test_data': 'x' * 100}  # Add some data
                )
                wave_events.append(event)
            
            # Process wave
            wave_start_time = time.time()
            await asyncio.gather(*[event_bus.publish(event) for event in wave_events])
            wave_end_time = time.time()
            
            # Allow processing
            await asyncio.sleep(0.5)
            
            # Measure memory after wave
            wave_end_memory = process.memory_info().rss / 1024 / 1024
            
            memory_measurements.append({
                'wave': wave,
                'start_memory_mb': wave_start_memory,
                'end_memory_mb': wave_end_memory,
                'memory_delta_mb': wave_end_memory - wave_start_memory,
                'events_processed': events_per_wave,
                'processing_time': wave_end_time - wave_start_time
            })
            
            # Force garbage collection periodically
            if wave % 3 == 2:
                gc.collect()
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - baseline_memory
        
        # Verify memory efficiency
        avg_memory_per_wave = sum(m['memory_delta_mb'] for m in memory_measurements) / waves
        max_memory_growth = max(m['memory_delta_mb'] for m in memory_measurements)
        
        # Memory efficiency assertions
        assert total_memory_growth < 100, f"Total memory growth {total_memory_growth:.2f}MB too high"
        assert avg_memory_per_wave < 15, f"Average memory per wave {avg_memory_per_wave:.2f}MB too high"
        assert max_memory_growth < 30, f"Max wave memory growth {max_memory_growth:.2f}MB too high"
        
        # Verify processing completed
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        total_expected_alerts = waves * events_per_wave
        assert bridge_stats['alerts_received_total'] >= total_expected_alerts * 0.95
        
        print(f"Memory efficiency: {total_memory_growth:.2f}MB growth, {avg_memory_per_wave:.2f}MB avg/wave")
    
    @pytest.mark.asyncio
    async def test_latency_characteristics(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test end-to-end latency characteristics."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        test_data = performance_test_data
        
        # Latency tracking
        latency_measurements = []
        processed_events = []
        
        # Track feature computations
        original_compute = feature_service.compute_features
        
        async def latency_tracking_compute(symbols, features, **kwargs):
            computation_start = time.time()
            result = await original_compute(symbols, features, **kwargs)
            computation_end = time.time()
            
            processed_events.append({
                'symbols': symbols,
                'features': features,
                'computation_latency': computation_end - computation_start,
                'timestamp': computation_end
            })
            
            return result
        
        feature_service.compute_features.side_effect = latency_tracking_compute
        
        # Test various latency scenarios
        latency_test_cases = [
            {'name': 'single_symbol', 'symbols': 1, 'count': 50},
            {'name': 'multi_symbol', 'symbols': 5, 'count': 30},
            {'name': 'high_priority', 'symbols': 1, 'count': 20, 'priority': 9},
            {'name': 'batch_processing', 'symbols': 1, 'count': 100, 'batch': True}
        ]
        
        for test_case in latency_test_cases:
            case_latencies = []
            
            for i in range(test_case['count']):
                publish_start = time.time()
                
                # Create alert based on test case
                symbols = [test_data['symbols_pool'][j] for j in range(i, i + test_case['symbols'])]
                
                alert = ScannerAlertEvent(
                    symbol=symbols[0] if symbols else f"LATENCY_TEST_{i}",
                    alert_type='high_volume',
                    score=0.8,
                    scanner_name=f"latency_test_{test_case['name']}",
                    metadata={'latency_test': test_case['name'], 'test_index': i}
                )
                
                await event_bus.publish(alert)
                publish_end = time.time()
                
                publish_latency = publish_end - publish_start
                case_latencies.append({
                    'test_case': test_case['name'],
                    'test_index': i,
                    'publish_latency': publish_latency,
                    'publish_timestamp': publish_end
                })
                
                # Small delay for non-batch scenarios
                if not test_case.get('batch', False):
                    await asyncio.sleep(0.01)
            
            latency_measurements.extend(case_latencies)
        
        # Wait for all processing to complete
        await asyncio.sleep(2.0)
        
        # Analyze latency characteristics
        publish_latencies = [m['publish_latency'] for m in latency_measurements]
        computation_latencies = [e['computation_latency'] for e in processed_events]
        
        # Latency statistics
        avg_publish_latency = sum(publish_latencies) / len(publish_latencies)
        max_publish_latency = max(publish_latencies)
        p95_publish_latency = sorted(publish_latencies)[int(len(publish_latencies) * 0.95)]
        
        if computation_latencies:
            avg_computation_latency = sum(computation_latencies) / len(computation_latencies)
            max_computation_latency = max(computation_latencies)
            p95_computation_latency = sorted(computation_latencies)[int(len(computation_latencies) * 0.95)]
        else:
            avg_computation_latency = max_computation_latency = p95_computation_latency = 0
        
        # Latency performance assertions
        assert avg_publish_latency < 0.01, f"Average publish latency {avg_publish_latency:.4f}s too high"
        assert p95_publish_latency < 0.05, f"P95 publish latency {p95_publish_latency:.4f}s too high"
        assert max_publish_latency < 0.1, f"Max publish latency {max_publish_latency:.4f}s too high"
        
        if computation_latencies:
            assert avg_computation_latency < 0.01, f"Average computation latency {avg_computation_latency:.4f}s too high"
            assert p95_computation_latency < 0.05, f"P95 computation latency {p95_computation_latency:.4f}s too high"
        
        # Verify processing completeness
        bridge_stats = bridge.get_stats()
        total_alerts = sum(case['count'] for case in latency_test_cases)
        assert bridge_stats['alerts_received_total'] >= total_alerts * 0.95
        
        print(f"Latency: pub avg={avg_publish_latency:.4f}s p95={p95_publish_latency:.4f}s, "
              f"comp avg={avg_computation_latency:.4f}s p95={p95_computation_latency:.4f}s")
    
    @pytest.mark.asyncio
    async def test_batching_efficiency(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test batching efficiency across components."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        feature_service = system['feature_service']
        test_data = performance_test_data
        
        # Batching tracking
        batch_tracking = {
            'feature_requests': [],
            'computations': [],
            'batching_efficiency': []
        }
        
        # Track batching in feature service
        original_compute = feature_service.compute_features
        
        async def batch_tracking_compute(symbols, features, **kwargs):
            batch_info = {
                'timestamp': time.time(),
                'symbol_count': len(symbols),
                'feature_count': len(features),
                'symbols': symbols,
                'features': features
            }
            batch_tracking['computations'].append(batch_info)
            return await original_compute(symbols, features, **kwargs)
        
        feature_service.compute_features.side_effect = batch_tracking_compute
        
        # Track feature request events
        original_publish = event_bus.publish
        
        async def request_tracking_publish(event):
            if event.event_type == EventType.FEATURE_REQUEST:
                batch_tracking['feature_requests'].append({
                    'timestamp': time.time(),
                    'symbol_count': len(event.symbols),
                    'feature_count': len(event.features),
                    'priority': getattr(event, 'priority', 5)
                })
            return await original_publish(event)
        
        event_bus.publish = request_tracking_publish
        
        # Generate batching test scenarios
        batching_scenarios = [
            {
                'name': 'small_bursts',
                'pattern': [(3, 0.01), (3, 0.01), (3, 0.01)],  # Small bursts
                'expected_batches': 1
            },
            {
                'name': 'large_burst',
                'pattern': [(50, 0.001)],  # Large burst
                'expected_batches': 3  # Should create multiple batches
            },
            {
                'name': 'sustained_flow',
                'pattern': [(1, 0.002)] * 30,  # Sustained flow
                'expected_batches': 2
            },
            {
                'name': 'mixed_patterns',
                'pattern': [(5, 0.01), (1, 0.001), (10, 0.002), (2, 0.01)],
                'expected_batches': 2
            }
        ]
        
        for scenario in batching_scenarios:
            scenario_start = time.time()
            scenario_alerts = 0
            
            for count, delay in scenario['pattern']:
                # Generate alerts for this pattern
                alerts = []
                for i in range(count):
                    alert = ScannerAlertEvent(
                        symbol=test_data['symbols_pool'][(scenario_alerts + i) % 100],
                        alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                        score=0.7,
                        scanner_name=f"batch_test_{scenario['name']}",
                        metadata={'scenario': scenario['name']}
                    )
                    alerts.append(alert)
                
                # Publish pattern
                await asyncio.gather(*[event_bus.publish(alert) for alert in alerts])
                scenario_alerts += count
                
                if delay > 0:
                    await asyncio.sleep(delay)
            
            # Wait for scenario processing
            await asyncio.sleep(0.2)
            
            scenario_end = time.time()
            scenario_duration = scenario_end - scenario_start
            
            # Calculate batching efficiency for this scenario
            scenario_requests = [req for req in batch_tracking['feature_requests'] 
                               if req['timestamp'] >= scenario_start and req['timestamp'] <= scenario_end]
            scenario_computations = [comp for comp in batch_tracking['computations']
                                   if comp['timestamp'] >= scenario_start and comp['timestamp'] <= scenario_end]
            
            batching_efficiency = {
                'scenario': scenario['name'],
                'alerts_generated': scenario_alerts,
                'feature_requests': len(scenario_requests),
                'computations': len(scenario_computations),
                'avg_symbols_per_request': sum(req['symbol_count'] for req in scenario_requests) / max(len(scenario_requests), 1),
                'avg_symbols_per_computation': sum(comp['symbol_count'] for comp in scenario_computations) / max(len(scenario_computations), 1),
                'batching_ratio': len(scenario_computations) / max(scenario_alerts, 1),
                'duration': scenario_duration
            }
            
            batch_tracking['batching_efficiency'].append(batching_efficiency)
        
        # Analyze overall batching efficiency
        total_alerts = sum(eff['alerts_generated'] for eff in batch_tracking['batching_efficiency'])
        total_computations = len(batch_tracking['computations'])
        overall_batching_ratio = total_computations / max(total_alerts, 1)
        
        # Batching efficiency assertions
        assert overall_batching_ratio < 0.8, f"Overall batching ratio {overall_batching_ratio:.3f} indicates poor batching"
        
        # Verify individual scenario efficiency
        for efficiency in batch_tracking['batching_efficiency']:
            # Should batch alerts into fewer computations
            assert efficiency['batching_ratio'] <= 1.0, f"Scenario {efficiency['scenario']} batching ratio too high"
            
            # Should achieve reasonable symbol aggregation
            if efficiency['computations'] > 0:
                assert efficiency['avg_symbols_per_computation'] >= 1.0
        
        # Verify system performance
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        assert bridge_stats['alerts_received_total'] >= total_alerts * 0.95
        assert bridge_stats['feature_requests_sent_total'] >= 1
        
        print(f"Batching efficiency: {overall_batching_ratio:.3f} computation/alert ratio, "
              f"{total_computations} computations for {total_alerts} alerts")
    
    @pytest.mark.asyncio
    async def test_resource_utilization_efficiency(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test efficient resource utilization under load."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        test_data = performance_test_data
        
        # Resource monitoring
        process = psutil.Process(os.getpid())
        resource_measurements = []
        
        # Baseline measurements
        baseline_cpu = process.cpu_percent()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        time.sleep(0.1)  # Let CPU measurement stabilize
        
        # Resource-intensive load test
        load_phases = [
            {'name': 'ramp_up', 'duration': 1.0, 'rate': 100},      # 100 events/sec
            {'name': 'sustained', 'duration': 3.0, 'rate': 200},    # 200 events/sec
            {'name': 'peak', 'duration': 1.0, 'rate': 400},         # 400 events/sec
            {'name': 'ramp_down', 'duration': 1.0, 'rate': 100}     # 100 events/sec
        ]
        
        total_events_generated = 0
        
        for phase in load_phases:
            phase_start = time.time()
            phase_events = 0
            
            events_per_interval = phase['rate'] * 0.1  # Events per 100ms interval
            intervals = int(phase['duration'] / 0.1)
            
            for interval in range(intervals):
                interval_start = time.time()
                
                # Generate events for this interval
                interval_events = []
                for i in range(int(events_per_interval)):
                    event = ScannerAlertEvent(
                        symbol=test_data['symbols_pool'][(total_events_generated + i) % 500],
                        alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                        score=0.6 + (i % 40) / 100.0,
                        scanner_name=f"resource_test_{phase['name']}_{i % 3}",
                        metadata={'phase': phase['name'], 'interval': interval}
                    )
                    interval_events.append(event)
                
                # Publish interval events
                await asyncio.gather(*[event_bus.publish(event) for event in interval_events])
                phase_events += len(interval_events)
                total_events_generated += len(interval_events)
                
                # Resource measurement
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                resource_measurements.append({
                    'timestamp': time.time(),
                    'phase': phase['name'],
                    'interval': interval,
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'events_generated': len(interval_events),
                    'cumulative_events': total_events_generated
                })
                
                # Maintain interval timing
                interval_end = time.time()
                interval_duration = interval_end - interval_start
                if interval_duration < 0.1:
                    await asyncio.sleep(0.1 - interval_duration)
            
            phase_end = time.time()
            phase_duration = phase_end - phase_start
            
            print(f"Phase {phase['name']}: {phase_events} events in {phase_duration:.2f}s "
                  f"({phase_events/phase_duration:.1f} events/sec)")
        
        # Wait for processing completion
        await asyncio.sleep(2.0)
        
        # Analyze resource utilization
        avg_cpu = sum(m['cpu_percent'] for m in resource_measurements) / len(resource_measurements)
        max_cpu = max(m['cpu_percent'] for m in resource_measurements)
        peak_memory = max(m['memory_mb'] for m in resource_measurements)
        memory_growth = peak_memory - baseline_memory
        
        # Resource efficiency assertions
        assert avg_cpu < 80, f"Average CPU utilization {avg_cpu:.1f}% too high"
        assert max_cpu < 95, f"Peak CPU utilization {max_cpu:.1f}% too high"
        assert memory_growth < 150, f"Memory growth {memory_growth:.1f}MB too high"
        
        # Verify processing efficiency
        bridge_stats = bridge.get_stats()
        pipeline_stats = pipeline_handler.get_stats()
        
        processing_efficiency = bridge_stats['alerts_received_total'] / total_events_generated
        assert processing_efficiency > 0.9, f"Processing efficiency {processing_efficiency:.3f} too low"
        
        # Resource per event efficiency
        avg_cpu_per_event = avg_cpu / (total_events_generated / sum(phase['duration'] for phase in load_phases))
        assert avg_cpu_per_event < 0.5, f"CPU per event {avg_cpu_per_event:.3f}% too high"
        
        print(f"Resource efficiency: {avg_cpu:.1f}% avg CPU, {peak_memory:.1f}MB peak memory, "
              f"{processing_efficiency:.3f} processing efficiency")
    
    @pytest.mark.asyncio
    async def test_scalability_limits(
        self,
        performance_optimized_system,
        performance_test_data
    ):
        """Test system behavior at scalability limits."""
        system = performance_optimized_system
        event_bus = system['event_bus']
        bridge = system['bridge']
        pipeline_handler = system['pipeline_handler']
        test_data = performance_test_data
        
        # Scalability test configuration
        scalability_tests = [
            {'name': 'symbol_count', 'symbols': 10000, 'events': 1000, 'expected_degradation': 0.2},
            {'name': 'event_rate', 'symbols': 100, 'events': 5000, 'expected_degradation': 0.3},
            {'name': 'batch_size', 'symbols': 200, 'events': 2000, 'expected_degradation': 0.15}
        ]
        
        scalability_results = []
        
        for test in scalability_tests:
            print(f"Running scalability test: {test['name']}")
            
            test_start = time.time()
            performance_metrics = {
                'events_published': 0,
                'publish_errors': 0,
                'processing_time': 0
            }
            
            # Generate large symbol pool for symbol_count test
            if test['name'] == 'symbol_count':
                test_symbols = [f'SCALE_SYMBOL_{i:05d}' for i in range(test['symbols'])]
            else:
                test_symbols = test_data['symbols_pool'][:test['symbols']]
            
            # Generate events based on test type
            events = []
            for i in range(test['events']):
                if test['name'] == 'batch_size':
                    # Create larger batches by using many symbols per event
                    symbol_batch = test_symbols[i:(i+10) % len(test_symbols)]
                    symbol = symbol_batch[0] if symbol_batch else f'BATCH_SYMBOL_{i}'
                else:
                    symbol = test_symbols[i % len(test_symbols)]
                
                event = ScannerAlertEvent(
                    symbol=symbol,
                    alert_type=test_data['alert_types'][i % len(test_data['alert_types'])],
                    score=0.5 + (i % 50) / 100.0,
                    scanner_name=f"scale_test_{test['name']}_{i % 5}",
                    metadata={'test_type': test['name'], 'test_index': i}
                )
                events.append(event)
            
            # Publish events with performance tracking
            publish_start = time.time()
            
            if test['name'] == 'event_rate':
                # High-rate publishing
                batch_size = 100
                for i in range(0, len(events), batch_size):
                    batch = events[i:i + batch_size]
                    try:
                        await asyncio.gather(*[event_bus.publish(event) for event in batch])
                        performance_metrics['events_published'] += len(batch)
                    except Exception:
                        performance_metrics['publish_errors'] += len(batch)
            else:
                # Standard publishing
                for event in events:
                    try:
                        await event_bus.publish(event)
                        performance_metrics['events_published'] += 1
                    except Exception:
                        performance_metrics['publish_errors'] += 1
                    
                    # Small delay for non-rate tests
                    if test['name'] != 'event_rate' and performance_metrics['events_published'] % 100 == 0:
                        await asyncio.sleep(0.01)
            
            publish_end = time.time()
            publish_duration = publish_end - publish_start
            
            # Wait for processing
            processing_wait_start = time.time()
            await asyncio.sleep(3.0)
            processing_wait_end = time.time()
            
            test_end = time.time()
            total_duration = test_end - test_start
            performance_metrics['processing_time'] = total_duration
            
            # Collect final statistics
            bridge_stats = bridge.get_stats()
            pipeline_stats = pipeline_handler.get_stats()
            
            # Calculate scalability metrics
            publish_throughput = performance_metrics['events_published'] / publish_duration
            overall_throughput = performance_metrics['events_published'] / total_duration
            processing_efficiency = bridge_stats['alerts_received_total'] / performance_metrics['events_published']
            error_rate = performance_metrics['publish_errors'] / test['events']
            
            scalability_result = {
                'test_name': test['name'],
                'symbols_tested': len(test_symbols),
                'events_generated': test['events'],
                'events_published': performance_metrics['events_published'],
                'publish_errors': performance_metrics['publish_errors'],
                'publish_throughput': publish_throughput,
                'overall_throughput': overall_throughput,
                'processing_efficiency': processing_efficiency,
                'error_rate': error_rate,
                'total_duration': total_duration,
                'bridge_stats': bridge_stats,
                'pipeline_stats': pipeline_stats
            }
            
            scalability_results.append(scalability_result)
            
            # Scalability assertions
            assert processing_efficiency > (1.0 - test['expected_degradation']), \
                f"Processing efficiency {processing_efficiency:.3f} below threshold for {test['name']}"
            assert error_rate < 0.05, f"Error rate {error_rate:.3f} too high for {test['name']}"
            assert publish_throughput > 50, f"Publish throughput {publish_throughput:.2f} too low for {test['name']}"
            
            print(f"  {test['name']}: {publish_throughput:.1f} pub/sec, {processing_efficiency:.3f} efficiency, "
                  f"{error_rate:.3f} error rate")
        
        # Cross-test analysis
        throughputs = [result['publish_throughput'] for result in scalability_results]
        efficiencies = [result['processing_efficiency'] for result in scalability_results]
        
        # System should maintain reasonable performance across all scalability tests
        min_throughput = min(throughputs)
        min_efficiency = min(efficiencies)
        
        assert min_throughput > 50, f"Minimum throughput {min_throughput:.2f} across tests too low"
        assert min_efficiency > 0.8, f"Minimum efficiency {min_efficiency:.3f} across tests too low"
        
        print(f"Scalability limits: min throughput {min_throughput:.1f} events/sec, "
              f"min efficiency {min_efficiency:.3f}")
        
        return scalability_results