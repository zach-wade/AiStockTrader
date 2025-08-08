#!/usr/bin/env python3
"""
Data Pipeline Integration Example

Demonstrates how to use the data pipeline for various data operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from main.orchestration.managers.data_pipeline_manager import DataPipelineManager
from main.data_pipeline.orchestrator import DataPipelineOrchestrator, PipelineMode, DataFlowConfig
from main.trading_engine.core.trading_system import TradingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_data_pipeline_example():
    """Basic example of running the data pipeline"""
    
    # Mock orchestrator for this example
    class MockOrchestrator:
        def __init__(self):
            self.config = {
                'data_pipeline': {
                    'health_check_interval': 30,
                    'metrics_interval': 60,
                    'enabled_stages': ['ingestion', 'processing', 'historical']
                }
            }
    
    orchestrator = MockOrchestrator()
    
    try:
        # Create data pipeline manager
        pipeline_manager = DataPipelineManager(orchestrator)
        
        # Initialize the pipeline
        logger.info("Initializing data pipeline...")
        await pipeline_manager.initialize()
        
        # Start the pipeline
        logger.info("Starting data pipeline...")
        await pipeline_manager.start()
        
        # Run a full pipeline execution
        logger.info("Running full pipeline...")
        result = await pipeline_manager.run_pipeline()
        
        logger.info(f"Pipeline run completed: {result}")
        
        # Get pipeline status
        status = await pipeline_manager.get_pipeline_status()
        logger.info(f"Pipeline status: {status['status']}")
        logger.info(f"Active symbols: {status['metrics']['active_symbols']}")
        
        # Stop the pipeline
        logger.info("Stopping data pipeline...")
        await pipeline_manager.stop()
        
    except Exception as e:
        logger.error(f"Error in basic pipeline example: {e}")


async def targeted_data_ingestion_example():
    """Example of targeted data ingestion for specific symbols"""
    
    class MockOrchestrator:
        def __init__(self):
            self.config = {'data_pipeline': {}}
    
    orchestrator = MockOrchestrator()
    pipeline_manager = DataPipelineManager(orchestrator)
    
    try:
        await pipeline_manager.initialize()
        await pipeline_manager.start()
        
        # Define target symbols and date range
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"Ingesting market data for {symbols}")
        
        # Run market data ingestion
        result = await pipeline_manager.run_ingestion(
            data_type='market_data',
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe='1day'
        )
        
        logger.info(f"Ingestion completed: {result.records_processed} records processed")
        
        # Run news data ingestion
        logger.info("Ingesting news data...")
        news_result = await pipeline_manager.run_ingestion(
            data_type='news',
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"News ingestion completed: {news_result.records_processed} articles")
        
        # Get updated metrics
        metrics = await pipeline_manager.get_pipeline_metrics()
        logger.info(f"Total records ingested: {metrics.ingestion_records}")
        logger.info(f"Active symbols: {len(metrics.active_symbols)}")
        
        await pipeline_manager.stop()
        
    except Exception as e:
        logger.error(f"Error in targeted ingestion example: {e}")


async def real_time_streaming_example():
    """Example of real-time data streaming"""
    
    # This example demonstrates how to set up real-time data streaming
    # In a real implementation, you would have actual data sources configured
    
    logger.info("=== Real-Time Streaming Example ===")
    
    # Configure for real-time mode
    flow_config = DataFlowConfig(
        mode=PipelineMode.REAL_TIME,
        batch_size=10,
        interval_seconds=1.0,
        max_workers=5
    )
    
    # Mock components for demonstration
    class MockComponents:
        async def run_full_ingestion(self):
            return {'status': 'success', 'records': 100}
        
        async def run_processing(self):
            return {'status': 'success', 'processed': 95}
        
        async def run_backfill_and_analysis(self):
            return {'status': 'success', 'gaps_filled': 5}
    
    mock = MockComponents()
    
    # In real usage, you would create actual orchestrator with real components
    # orchestrator = DataPipelineOrchestrator(
    #     config=config,
    #     ingestion_orchestrator=ingestion_orchestrator,
    #     historical_manager=historical_manager,
    #     processing_manager=processing_manager,
    #     flow_config=flow_config
    # )
    
    logger.info("Real-time streaming configured")
    logger.info(f"Mode: {flow_config.mode.value}")
    logger.info(f"Batch size: {flow_config.batch_size}")
    logger.info(f"Interval: {flow_config.interval_seconds}s")


async def data_quality_monitoring_example():
    """Example of monitoring data quality"""
    
    class MockOrchestrator:
        def __init__(self):
            self.config = {'data_pipeline': {}}
    
    orchestrator = MockOrchestrator()
    pipeline_manager = DataPipelineManager(orchestrator)
    
    try:
        await pipeline_manager.initialize()
        await pipeline_manager.start()
        
        # Trigger health check
        logger.info("Running pipeline health check...")
        health_status = await pipeline_manager.trigger_health_check()
        
        logger.info(f"Health status: {'Healthy' if health_status['healthy'] else 'Issues detected'}")
        
        if health_status['issues']:
            logger.warning(f"Issues found: {health_status['issues']}")
        
        # Monitor data quality metrics
        for _ in range(3):
            # Simulate pipeline runs
            await pipeline_manager.run_pipeline(['ingestion'])
            
            # Check metrics
            metrics = await pipeline_manager.get_pipeline_metrics()
            
            quality_score = metrics.quality_score
            error_rate = metrics.failed_records / max(metrics.ingestion_records, 1)
            
            logger.info(f"Quality score: {quality_score:.2%}")
            logger.info(f"Error rate: {error_rate:.2%}")
            logger.info(f"Throughput: {metrics.throughput_records_per_second:.2f} records/sec")
            
            await asyncio.sleep(2)
        
        await pipeline_manager.stop()
        
    except Exception as e:
        logger.error(f"Error in quality monitoring example: {e}")


async def selective_pipeline_stages_example():
    """Example of running specific pipeline stages"""
    
    class MockOrchestrator:
        def __init__(self):
            self.config = {
                'data_pipeline': {
                    'enabled_stages': ['ingestion', 'processing', 'historical']
                }
            }
    
    orchestrator = MockOrchestrator()
    pipeline_manager = DataPipelineManager(orchestrator)
    
    try:
        await pipeline_manager.initialize()
        await pipeline_manager.start()
        
        # Run only ingestion stage
        logger.info("Running ingestion stage only...")
        result = await pipeline_manager.run_pipeline(stages=['ingestion'])
        logger.info(f"Ingestion completed: {result}")
        
        # Run only processing stage
        logger.info("Running processing stage only...")
        result = await pipeline_manager.run_pipeline(stages=['processing'])
        logger.info(f"Processing completed: {result}")
        
        # Run ingestion and processing, skip historical
        logger.info("Running ingestion and processing stages...")
        result = await pipeline_manager.run_pipeline(stages=['ingestion', 'processing'])
        logger.info(f"Stages completed: {result}")
        
        await pipeline_manager.stop()
        
    except Exception as e:
        logger.error(f"Error in selective stages example: {e}")


async def integration_with_trading_system_example():
    """Example of data pipeline integration with trading system"""
    
    logger.info("=== Data Pipeline + Trading System Integration ===")
    
    # This demonstrates how the data pipeline feeds into the trading system
    
    # 1. Data Pipeline collects and processes market data
    # 2. Processed data is stored in appropriate storage tiers
    # 3. Trading algorithms access this data for decision making
    # 4. Real-time updates flow through the streaming pipeline
    
    workflow = """
    Data Pipeline Workflow:
    
    1. Ingestion Stage:
       - Collect market data from multiple sources (Alpaca, Polygon, Yahoo)
       - Collect news and sentiment data
       - Collect corporate actions and fundamentals
       
    2. Processing Stage:
       - Standardize data formats
       - Validate data quality
       - Calculate derived metrics
       - Apply transformations
       
    3. Storage Stage:
       - Hot storage (PostgreSQL) for recent data (< 30 days)
       - Cold storage (Data Lake) for historical data
       - Feature store for ML features
       
    4. Distribution:
       - Real-time feeds to trading algorithms
       - Batch data for backtesting
       - Historical data for analysis
    """
    
    logger.info(workflow)
    
    # Example configuration for integrated system
    integrated_config = {
        'data_pipeline': {
            'collection': {
                'sources': ['alpaca', 'polygon', 'yahoo'],
                'data_types': ['market_data', 'news', 'fundamentals']
            },
            'streaming': {
                'enabled': True,
                'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                'update_frequency': '1s'
            },
            'storage': {
                'hot_storage_days': 30,
                'cold_storage_path': 'data_lake/',
                'feature_store_enabled': True
            }
        },
        'trading_system': {
            'data_source': 'integrated_pipeline',
            'real_time_data': True,
            'historical_lookback': 365
        }
    }
    
    logger.info("Integration configuration created")
    logger.info(f"Hot storage window: {integrated_config['data_pipeline']['storage']['hot_storage_days']} days")
    logger.info(f"Streaming symbols: {integrated_config['data_pipeline']['streaming']['symbols']}")


def main():
    """Run all examples"""
    logger.info("=== Data Pipeline Integration Examples ===")
    
    # Run basic example
    logger.info("\n--- Basic Data Pipeline Example ---")
    asyncio.run(basic_data_pipeline_example())
    
    # Run targeted ingestion example
    logger.info("\n--- Targeted Data Ingestion Example ---")
    asyncio.run(targeted_data_ingestion_example())
    
    # Run real-time streaming example
    logger.info("\n--- Real-Time Streaming Example ---")
    asyncio.run(real_time_streaming_example())
    
    # Run data quality monitoring example
    logger.info("\n--- Data Quality Monitoring Example ---")
    asyncio.run(data_quality_monitoring_example())
    
    # Run selective stages example
    logger.info("\n--- Selective Pipeline Stages Example ---")
    asyncio.run(selective_pipeline_stages_example())
    
    # Run integration example
    logger.info("\n--- Trading System Integration Example ---")
    asyncio.run(integration_with_trading_system_example())


if __name__ == "__main__":
    main()