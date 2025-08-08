# File: src/ai_trader/models/outcome_classifier.py

"""
Outcome Classifier for V3 Training Data Labeling.

This component classifies what actually happened after a catalyst was identified,
providing nuanced outcome classifications for training catalyst specialist models.
It orchestrates data fetching, metric calculation, labeling, and reporting.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, date, timezone
import pandas as pd # Used by helper components, not directly here as orchestrator

# Corrected absolute imports
from main.data_pipeline.ingestion.clients.polygon_market_client import PolygonMarketClient
from main.config.config_manager import get_config

# Import the new outcome classifier types and helper classes
from main.models.outcome_classifier_types import OutcomeLabel, OutcomeMetrics
from main.models.outcome_classifier_helpers.outcome_data_fetcher import OutcomeDataFetcher
from main.models.outcome_classifier_helpers.entry_price_determiner import EntryPriceDeterminer
from main.models.outcome_classifier_helpers.outcome_metrics_calculator import OutcomeMetricsCalculator
from main.models.outcome_classifier_helpers.outcome_labeler import OutcomeLabeler
from main.models.outcome_classifier_helpers.outcome_reporter import OutcomeReporter

logger = logging.getLogger(__name__)


class OutcomeClassifier:
    """
    Advanced outcome classifier for V3 catalyst training data.
    
    This class orchestrates the process of analyzing post-catalyst price action
    and classifying outcomes, delegating specific tasks to specialized helper components.
    It provides high-quality labels and detailed metrics for ML training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the OutcomeClassifier and its composing helper components.
        
        Args:
            config: Optional. Configuration dictionary. If None, `get_config()` is used.
        """
        self.config = config or get_config()
        
        # Initialize PolygonMarketClient (shared dependency for data fetching)
        import os
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            raise ValueError("POLYGON_API_KEY environment variable is not set")
        
        from main.data_pipeline.core.enums import DataLayer
        self.market_client = PolygonMarketClient(
            api_key=polygon_api_key,
            layer=DataLayer.LIQUID  # Use appropriate layer based on requirements
        )

        # Classification thresholds (can be tuned via config if desired)
        self.thresholds = {
            'successful_breakout': {
                'min_return_3d': 0.05, 'min_max_favorable': 0.08, 'min_follow_through': 0.3
            },
            'failed_breakdown': {
                'max_return_3d': -0.03, 'max_max_favorable': 0.02, 'min_adverse_move': -0.05
            },
            'sideways_fizzle': {
                'max_abs_return_3d': 0.02, 'max_volatility': 0.03, 'max_max_move': 0.04
            },
            'modest_gain': {
                'min_return_3d': 0.02, 'max_return_3d': 0.05, 'min_consistency': 0.2
            },
            'reversal_pattern': {
                'min_initial_move': 0.03, 'min_reversal_strength': 0.4, 'max_final_return': 0.01
            }
        }
        
        # Evaluation periods for returns calculation
        self.evaluation_periods = self.config.get('outcome_classifier', {}).get('evaluation_periods', [1, 3, 5, 7, 14])
        self.primary_period = self.config.get('outcome_classifier', {}).get('primary_classification_period', 3)
        
        # Initialize helper components
        self._data_fetcher = OutcomeDataFetcher(market_client=self.market_client)
        self._entry_price_determiner = EntryPriceDeterminer()
        self._metrics_calculator = OutcomeMetricsCalculator(evaluation_periods=self.evaluation_periods)
        self._outcome_labeler = OutcomeLabeler(thresholds=self.thresholds)
        self._outcome_reporter = OutcomeReporter()
        
        logger.info("🔧 OutcomeClassifier initialized with helper components.")
    
    async def classify_outcome(
        self, 
        symbol: str, 
        entry_date: date,
        entry_time: Optional[datetime] = None,
        evaluation_days: int = 7,
        include_intraday: bool = True
    ) -> OutcomeMetrics:
        """
        Classifies the outcome of a catalyst event for a single symbol.
        Orchestrates data fetching, entry price determination, metric calculation, and labeling.
        
        Args:
            symbol: Stock symbol.
            entry_date: Date the catalyst was identified (e.g., market open).
            entry_time: Specific entry time (defaults to market open if not provided).
            evaluation_days: How many days to evaluate post-catalyst.
            include_intraday: Whether to include intraday extreme analysis (requires intraday data).
            
        Returns:
            OutcomeMetrics object with comprehensive classification and metrics.
            Returns OutcomeMetrics with NO_DATA or CALCULATION_ERROR label on failure.
        """
        logger.debug(f"Classifying outcome for {symbol} on {entry_date} (Eval days: {evaluation_days})")
        
        try:
            # 1. Fetch price data
            # Add a buffer of 5 days before for context/entry price lookup and 5 days after for safety.
            total_fetch_days = evaluation_days + 10 # Buffer for entry_date and future
            price_data = await self._data_fetcher.get_price_data_for_outcome_evaluation(
                symbol, entry_date, total_fetch_days 
            )
            
            if price_data is None or price_data.empty:
                logger.warning(f"No sufficient price data found for {symbol} around {entry_date}. Skipping classification.")
                return OutcomeMetrics(
                    entry_price=0.0,
                    outcome_label=OutcomeLabel.NO_DATA,
                    price_data_quality="no_price_data_fetched",
                    evaluation_period_days=evaluation_days # Retain requested eval days
                )
            
            # 2. Determine entry price
            entry_price, data_quality_note = self._entry_price_determiner.determine_entry_price(
                price_data, entry_date, entry_time # entry_time is currently for intraday consideration
            )
            
            if entry_price <= 0:
                logger.warning(f"Invalid entry price ({entry_price}) determined for {symbol} on {entry_date}.")
                return OutcomeMetrics(
                    entry_price=0.0,
                    outcome_label=OutcomeLabel.NO_DATA, # Or specific 'INVALID_ENTRY_PRICE'
                    price_data_quality=data_quality_note,
                    evaluation_period_days=evaluation_days
                )
            
            # 3. Calculate comprehensive metrics
            calculated_metrics = self._metrics_calculator.calculate_comprehensive_metrics(
                price_data, entry_price, entry_date, evaluation_days, include_intraday
            )

            # If metrics calculation itself failed, it will return an empty dict or indicate error
            if not calculated_metrics or 'calculation_error' in calculated_metrics:
                logger.warning(f"Error calculating metrics for {symbol} on {entry_date}. Details: {calculated_metrics.get('calculation_error', 'unknown error')}")
                return OutcomeMetrics(
                    entry_price=entry_price,
                    outcome_label=OutcomeLabel.CALCULATION_ERROR,
                    price_data_quality=data_quality_note,
                    evaluation_period_days=evaluation_days
                )
            
            # 4. Classify outcome based on metrics
            outcome_label, confidence = self._outcome_labeler.classify_based_on_metrics(calculated_metrics)
            
            # 5. Populate final OutcomeMetrics object
            final_metrics = OutcomeMetrics(
                entry_price=entry_price,
                outcome_label=outcome_label,
                confidence_score=confidence,
                evaluation_period_days=evaluation_days, # Use requested evaluation days
                price_data_quality=data_quality_note,
                **calculated_metrics # Unpack all calculated metrics
            )
            
            logger.debug(f"{symbol} {entry_date}: Classified as {outcome_label.value} (Confidence: {confidence:.2f}, 3d Return: {final_metrics.return_3d:.2%})")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error classifying outcome for {symbol} on {entry_date}: {e}", exc_info=True)
            return OutcomeMetrics(
                entry_price=0.0,
                outcome_label=OutcomeLabel.CALCULATION_ERROR,
                price_data_quality=f"classification_system_error: {str(e)}",
                evaluation_period_days=evaluation_days
            )
    
    async def classify_batch(
        self, 
        symbols_and_dates: List[Tuple[str, date]], 
        evaluation_days: int = 7,
        batch_size: int = 20
    ) -> Dict[Tuple[str, date], OutcomeMetrics]:
        """
        Classifies outcomes for a batch of symbol-date pairs efficiently in parallel.
        
        Args:
            symbols_and_dates: List of (symbol, date) tuples.
            evaluation_days: Evaluation period in days.
            batch_size: Number of symbol-date pairs to process concurrently in each batch.
            
        Returns:
            Dictionary mapping (symbol, date) to OutcomeMetrics object.
        """
        logger.info(f"Classifying outcomes for {len(symbols_and_dates)} symbol-date pairs in batches.")
        
        results: Dict[Tuple[str, date], OutcomeMetrics] = {}
        
        for i in range(0, len(symbols_and_dates), batch_size):
            batch = symbols_and_dates[i:i + batch_size]
            
            tasks = []
            for symbol, date_val in batch:
                # Assuming entry_time and include_intraday are defaults or handled by caller
                tasks.append(self.classify_outcome(symbol, date_val, evaluation_days=evaluation_days))
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (symbol, date_val), outcome_metric in zip(batch, batch_results):
                if isinstance(outcome_metric, Exception):
                    logger.error(f"Error processing {symbol} {date_val} in batch: {outcome_metric}", exc_info=True)
                    # Return a specific error OutcomeMetrics for individual failures
                    results[(symbol, date_val)] = OutcomeMetrics(
                        entry_price=0.0, # Indicate no valid entry price
                        outcome_label=OutcomeLabel.CALCULATION_ERROR,
                        price_data_quality=f"batch_processing_error: {outcome_metric}"
                    )
                else:
                    results[(symbol, date_val)] = outcome_metric
            
            if i + batch_size < len(symbols_and_dates): # Don't sleep after last batch
                await asyncio.sleep(self.config.get('outcome_classifier', {}).get('batch_rate_limit_seconds', 0.5)) # Configurable rate limit
        
        success_count = sum(1 for m in results.values() 
                          if m.outcome_label not in [OutcomeLabel.NO_DATA, OutcomeLabel.CALCULATION_ERROR])
        
        logger.info(f"Successfully classified {success_count}/{len(symbols_and_dates)} outcomes in total.")
        return results
    
    def generate_classification_report(self, outcomes: List[OutcomeMetrics]) -> Dict[str, Any]:
        """
        Generates a comprehensive report summarizing the classification results.
        Delegates to the OutcomeReporter.
        
        Args:
            outcomes: A list of OutcomeMetrics objects.
            
        Returns:
            A dictionary containing various statistics and summaries of the classifications.
        """
        return self._outcome_reporter.generate_classification_report(outcomes)


# Example usage and testing (removed from main class for cleaner separation)
# This part would typically be in a separate script or test module.