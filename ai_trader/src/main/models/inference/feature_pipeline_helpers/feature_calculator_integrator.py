"""
Feature calculator integration for model inference.

This module integrates feature calculators with the inference pipeline,
ensuring efficient feature computation for real-time predictions.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer,
    process_in_batches
)
from main.utils.monitoring import record_metric

from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.feature_pipeline.calculators.base_calculator import BaseFeatureCalculator

logger = get_logger(__name__)


class FeatureCalculatorIntegrator(ErrorHandlingMixin):
    """
    Integrates feature calculators with model inference.
    
    Features:
    - Dynamic feature calculator loading
    - Parallel feature computation
    - Feature dependency resolution
    - Calculation optimization
    - Error handling and fallbacks
    """
    
    def __init__(
        self,
        feature_engine: UnifiedFeatureEngine,
        max_parallel_calculators: int = 10
    ):
        """
        Initialize feature calculator integrator.
        
        Args:
            feature_engine: Unified feature engine instance
            max_parallel_calculators: Max calculators to run in parallel
        """
        super().__init__()
        self.feature_engine = feature_engine
        self.max_parallel_calculators = max_parallel_calculators
        
        # Calculator registry
        self._calculators: Dict[str, BaseFeatureCalculator] = {}
        self._calculator_dependencies: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self._calculation_times: Dict[str, List[float]] = {}
        
        # Initialize calculators
        self._initialize_calculators()
        
    def _initialize_calculators(self):
        """Initialize and register available calculators."""
        # Get all available calculators from feature engine
        calculator_types = [
            'technical', 'statistical', 'risk', 'sentiment',
            'microstructure', 'correlation', 'regime'
        ]
        
        for calc_type in calculator_types:
            try:
                calculator = self.feature_engine.get_calculator(calc_type)
                if calculator:
                    self._calculators[calc_type] = calculator
                    # Analyze dependencies (simplified for now)
                    self._calculator_dependencies[calc_type] = set()
                    logger.debug(f"Registered {calc_type} calculator")
            except Exception as e:
                logger.warning(f"Failed to register {calc_type} calculator: {e}")
    
    @timer
    async def calculate_features(
        self,
        symbols: List[str],
        feature_groups: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate features for inference.
        
        Args:
            symbols: List of symbols
            feature_groups: Feature groups to calculate
            start_date: Start date for calculations
            end_date: End date for calculations
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with calculated features
        """
        with self._handle_error("calculating features"):
            # Resolve dependencies
            required_groups = self._resolve_dependencies(feature_groups)
            
            # Order by dependencies
            ordered_groups = self._topological_sort(required_groups)
            
            # Calculate features in batches
            all_features = {}
            
            # Process in parallel batches
            batch_size = min(self.max_parallel_calculators, len(ordered_groups))
            
            for i in range(0, len(ordered_groups), batch_size):
                batch = ordered_groups[i:i + batch_size]
                
                # Calculate batch in parallel
                batch_results = await asyncio.gather(
                    *[
                        self._calculate_single_group(
                            group, symbols, start_date, end_date, **kwargs
                        )
                        for group in batch
                    ],
                    return_exceptions=True
                )
                
                # Process results
                for group, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to calculate {group}: {result}")
                        # Use fallback or skip
                        continue
                    
                    all_features[group] = result
                    
                    # Record timing
                    if hasattr(result, '_calculation_time'):
                        self._record_calculation_time(
                            group, 
                            result._calculation_time
                        )
            
            # Combine all features
            combined = self._combine_features(all_features)
            
            # Record metrics
            record_metric(
                'feature_calculator_integrator.features_calculated',
                len(combined.columns) if not combined.empty else 0,
                tags={
                    'symbols': len(symbols),
                    'groups': len(feature_groups)
                }
            )
            
            return combined
    
    async def _calculate_single_group(
        self,
        group: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate features for a single group."""
        start_time = datetime.utcnow()
        
        calculator = self._calculators.get(group)
        if not calculator:
            raise ValueError(f"No calculator found for group: {group}")
        
        # Prepare parameters
        params = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            **kwargs
        }
        
        # Calculate features
        if hasattr(calculator, 'calculate_async'):
            features = await calculator.calculate_async(**params)
        else:
            # Run sync calculator in executor
            features = await asyncio.get_event_loop().run_in_executor(
                None,
                calculator.calculate,
                **params
            )
        
        # Add timing info
        calculation_time = (datetime.utcnow() - start_time).total_seconds()
        features._calculation_time = calculation_time
        
        return features
    
    def _resolve_dependencies(
        self,
        feature_groups: List[str]
    ) -> Set[str]:
        """Resolve all dependencies for requested feature groups."""
        required = set(feature_groups)
        
        # Add dependencies recursively
        def add_deps(group: str):
            if group in self._calculator_dependencies:
                for dep in self._calculator_dependencies[group]:
                    if dep not in required:
                        required.add(dep)
                        add_deps(dep)
        
        for group in list(required):
            add_deps(group)
        
        return required
    
    def _topological_sort(
        self,
        groups: Set[str]
    ) -> List[str]:
        """Sort feature groups by dependencies."""
        # Build adjacency list
        graph = {g: list(self._calculator_dependencies.get(g, [])) for g in groups}
        
        # Calculate in-degrees
        in_degree = {g: 0 for g in groups}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find nodes with no dependencies
        queue = [g for g in groups if in_degree[g] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Update in-degrees
            for dep in graph[node]:
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        
        # If not all nodes processed, there's a cycle
        if len(result) != len(groups):
            logger.warning("Dependency cycle detected, using partial order")
            # Add remaining nodes
            result.extend([g for g in groups if g not in result])
        
        return result
    
    def _combine_features(
        self,
        feature_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Combine features from multiple groups."""
        if not feature_dict:
            return pd.DataFrame()
        
        # Get all dataframes
        dfs = list(feature_dict.values())
        
        if len(dfs) == 1:
            return dfs[0]
        
        # Align indices
        aligned_dfs = []
        common_index = None
        
        for df in dfs:
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # Reindex all dataframes to common index
        for df in dfs:
            aligned_dfs.append(df.reindex(common_index))
        
        # Concatenate along columns
        combined = pd.concat(aligned_dfs, axis=1)
        
        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        return combined
    
    def _record_calculation_time(self, group: str, time_seconds: float):
        """Record calculation time for performance monitoring."""
        if group not in self._calculation_times:
            self._calculation_times[group] = []
        
        self._calculation_times[group].append(time_seconds)
        
        # Keep only recent times
        if len(self._calculation_times[group]) > 100:
            self._calculation_times[group] = self._calculation_times[group][-100:]
    
    def get_calculator_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each calculator."""
        stats = {}
        
        for group, times in self._calculation_times.items():
            if times:
                stats[group] = {
                    'mean_time': np.mean(times),
                    'median_time': np.median(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times),
                    'std_time': np.std(times),
                    'sample_count': len(times)
                }
        
        return stats
    
    def optimize_calculation_order(
        self,
        feature_groups: List[str],
        time_budget_seconds: Optional[float] = None
    ) -> List[str]:
        """
        Optimize calculation order based on historical performance.
        
        Args:
            feature_groups: Groups to calculate
            time_budget_seconds: Optional time budget
            
        Returns:
            Optimized order of calculation
        """
        # Get average times
        avg_times = {}
        for group in feature_groups:
            if group in self._calculation_times:
                avg_times[group] = np.mean(self._calculation_times[group])
            else:
                avg_times[group] = 1.0  # Default estimate
        
        # Sort by dependencies first, then by time
        required_groups = self._resolve_dependencies(feature_groups)
        ordered = self._topological_sort(required_groups)
        
        # Within each dependency level, sort by time
        # (This is simplified - real implementation would be more sophisticated)
        if time_budget_seconds:
            # Filter to fit in budget
            cumulative_time = 0
            filtered = []
            
            for group in ordered:
                group_time = avg_times.get(group, 1.0)
                if cumulative_time + group_time <= time_budget_seconds:
                    filtered.append(group)
                    cumulative_time += group_time
                else:
                    logger.warning(
                        f"Skipping {group} due to time budget "
                        f"({cumulative_time + group_time:.1f}s > {time_budget_seconds}s)"
                    )
            
            return filtered
        
        return ordered
    
    async def validate_features(
        self,
        features: pd.DataFrame,
        expected_columns: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate calculated features.
        
        Args:
            features: Calculated features
            expected_columns: Expected feature columns
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing = [col for col in expected_columns if col not in features.columns]
        
        if missing:
            logger.warning(f"Missing features: {missing}")
        
        # Additional validation
        if not features.empty:
            # Check for NaN values
            nan_columns = features.columns[features.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"Features with NaN values: {nan_columns}")
            
            # Check for infinite values
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            inf_columns = []
            for col in numeric_cols:
                if np.isinf(features[col]).any():
                    inf_columns.append(col)
            
            if inf_columns:
                logger.warning(f"Features with infinite values: {inf_columns}")
        
        return len(missing) == 0, missing