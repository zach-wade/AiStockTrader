"""
Model registry enhancements for advanced model management.

This module provides enhanced functionality for the model registry including:
- Model versioning and lifecycle management
- A/B testing and traffic routing
- Performance tracking and comparison
- Automated deployment workflows
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

from main.models.inference.model_registry_types import (
    ModelInfo,
    ModelVersion,
    ModelDeployment,
    ModelMetrics,
    DeploymentStatus
)

logger = get_logger(__name__)


@dataclass
class ModelComparison:
    """Results of model comparison."""
    model_a: str
    model_b: str
    metric_comparisons: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    confidence: float = 0.0
    sample_size: int = 0
    comparison_date: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_name: str
    model_a: str  # Control model
    model_b: str  # Test model
    traffic_split: float = 0.5  # Fraction to model_b
    min_samples: int = 1000
    max_duration_hours: int = 168  # 1 week
    success_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'profit'])
    confidence_threshold: float = 0.95


class ModelVersionManager(ErrorHandlingMixin):
    """
    Manages model versions and lifecycle.
    
    Features:
    - Semantic versioning
    - Version promotion/demotion
    - Version comparison
    - Rollback capabilities
    """
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize version manager."""
        super().__init__()
        self.db_pool = db_pool
        
    async def create_version(
        self,
        model_name: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> ModelVersion:
        """Create a new model version."""
        with self._handle_error("creating model version"):
            async with self.db_pool.acquire() as conn:
                # Insert version record
                query = """
                    INSERT INTO model_versions (
                        model_name, version, metadata, created_at, status
                    ) VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """
                
                version_id = await conn.fetchval(
                    query,
                    model_name,
                    version,
                    json.dumps(metadata),
                    datetime.utcnow(),
                    'created'
                )
                
                model_version = ModelVersion(
                    id=version_id,
                    model_name=model_name,
                    version=version,
                    created_at=datetime.utcnow(),
                    status='created',
                    metadata=metadata
                )
                
                logger.info(f"Created version {version} for model {model_name}")
                
                return model_version
    
    async def promote_version(
        self,
        model_name: str,
        version: str,
        environment: str = 'production'
    ) -> bool:
        """Promote a model version to an environment."""
        with self._handle_error("promoting model version"):
            async with self.db_pool.acquire() as conn:
                # Update deployment status
                query = """
                    UPDATE model_deployments
                    SET status = 'active', 
                        promoted_at = $1,
                        environment = $2
                    WHERE model_name = $3 AND version = $4
                """
                
                await conn.execute(
                    query,
                    datetime.utcnow(),
                    environment,
                    model_name,
                    version
                )
                
                # Deactivate other versions in same environment
                deactivate_query = """
                    UPDATE model_deployments
                    SET status = 'inactive'
                    WHERE model_name = $1 
                    AND environment = $2 
                    AND version != $3
                    AND status = 'active'
                """
                
                await conn.execute(
                    deactivate_query,
                    model_name,
                    environment,
                    version
                )
                
                record_metric(
                    'model_registry.version_promoted',
                    1,
                    tags={
                        'model': model_name,
                        'version': version,
                        'environment': environment
                    }
                )
                
                logger.info(
                    f"Promoted {model_name} v{version} to {environment}"
                )
                
                return True
    
    async def rollback_version(
        self,
        model_name: str,
        target_version: str,
        environment: str = 'production'
    ) -> bool:
        """Rollback to a previous version."""
        with self._handle_error("rolling back model version"):
            # Get current version
            current = await self.get_active_version(model_name, environment)
            
            if not current:
                logger.error(f"No active version found for {model_name}")
                return False
            
            # Promote target version
            success = await self.promote_version(
                model_name,
                target_version,
                environment
            )
            
            if success:
                # Log rollback
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO model_rollbacks (
                            model_name, from_version, to_version,
                            environment, timestamp, reason
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        model_name,
                        current.version,
                        target_version,
                        environment,
                        datetime.utcnow(),
                        'manual_rollback'
                    )
                
                logger.info(
                    f"Rolled back {model_name} from v{current.version} "
                    f"to v{target_version}"
                )
            
            return success
    
    async def get_active_version(
        self,
        model_name: str,
        environment: str = 'production'
    ) -> Optional[ModelVersion]:
        """Get currently active version."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT v.*, d.environment, d.promoted_at
                FROM model_versions v
                JOIN model_deployments d ON v.id = d.version_id
                WHERE v.model_name = $1 
                AND d.environment = $2
                AND d.status = 'active'
                ORDER BY d.promoted_at DESC
                LIMIT 1
            """
            
            row = await conn.fetchrow(query, model_name, environment)
            
            if row:
                return ModelVersion(
                    id=row['id'],
                    model_name=row['model_name'],
                    version=row['version'],
                    created_at=row['created_at'],
                    status=row['status'],
                    metadata=json.loads(row['metadata'])
                )
            
            return None
    
    async def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        metric_names: List[str]
    ) -> ModelComparison:
        """Compare two model versions."""
        with self._handle_error("comparing model versions"):
            # Get metrics for both versions
            metrics_a = await self._get_version_metrics(
                model_name, version_a, metric_names
            )
            metrics_b = await self._get_version_metrics(
                model_name, version_b, metric_names
            )
            
            # Calculate comparisons
            comparisons = {}
            for metric in metric_names:
                if metric in metrics_a and metric in metrics_b:
                    comparisons[metric] = {
                        'version_a': metrics_a[metric],
                        'version_b': metrics_b[metric],
                        'difference': metrics_b[metric] - metrics_a[metric],
                        'improvement_pct': (
                            (metrics_b[metric] - metrics_a[metric]) / 
                            metrics_a[metric] * 100
                            if metrics_a[metric] != 0 else 0
                        )
                    }
            
            # Determine winner
            improvements = sum(
                1 for m in comparisons.values()
                if m['difference'] > 0
            )
            
            winner = None
            confidence = 0.0
            
            if improvements > len(comparisons) / 2:
                winner = version_b
                confidence = improvements / len(comparisons)
            elif improvements < len(comparisons) / 2:
                winner = version_a
                confidence = (len(comparisons) - improvements) / len(comparisons)
            
            return ModelComparison(
                model_a=f"{model_name}_v{version_a}",
                model_b=f"{model_name}_v{version_b}",
                metric_comparisons=comparisons,
                winner=winner,
                confidence=confidence,
                sample_size=1000  # Placeholder
            )
    
    async def _get_version_metrics(
        self,
        model_name: str,
        version: str,
        metric_names: List[str]
    ) -> Dict[str, float]:
        """Get metrics for a specific version."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT metric_name, metric_value
                FROM model_metrics
                WHERE model_name = $1 AND version = $2
                AND metric_name = ANY($3)
                ORDER BY timestamp DESC
            """
            
            rows = await conn.fetch(query, model_name, version, metric_names)
            
            # Get latest value for each metric
            metrics = {}
            seen = set()
            for row in rows:
                if row['metric_name'] not in seen:
                    metrics[row['metric_name']] = row['metric_value']
                    seen.add(row['metric_name'])
            
            return metrics


class ModelMetricsTracker(ErrorHandlingMixin):
    """
    Tracks and analyzes model performance metrics.
    
    Features:
    - Real-time metric collection
    - Historical analysis
    - Anomaly detection
    - Metric aggregation
    """
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize metrics tracker."""
        super().__init__()
        self.db_pool = db_pool
        
        # Metric buffers for batch insertion
        self._metric_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100
        self._last_flush = datetime.utcnow()
        
    async def track_metric(
        self,
        model_name: str,
        version: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a model metric."""
        metric_data = {
            'model_name': model_name,
            'version': version,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        self._metric_buffer.append(metric_data)
        
        # Flush if buffer is full or time elapsed
        if (len(self._metric_buffer) >= self._buffer_size or
            (datetime.utcnow() - self._last_flush).seconds > 60):
            await self._flush_metrics()
    
    async def _flush_metrics(self) -> None:
        """Flush metric buffer to database."""
        if not self._metric_buffer:
            return
        
        async with self.db_pool.acquire() as conn:
            # Batch insert
            await conn.executemany(
                """
                INSERT INTO model_metrics (
                    model_name, version, metric_name,
                    metric_value, timestamp, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (
                        m['model_name'], m['version'], m['metric_name'],
                        m['metric_value'], m['timestamp'], 
                        json.dumps(m['metadata'])
                    )
                    for m in self._metric_buffer
                ]
            )
        
        logger.debug(f"Flushed {len(self._metric_buffer)} metrics")
        self._metric_buffer.clear()
        self._last_flush = datetime.utcnow()
    
    async def get_metric_history(
        self,
        model_name: str,
        version: str,
        metric_name: str,
        hours: int = 24
    ) -> List[Tuple[datetime, float]]:
        """Get metric history for a model."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT timestamp, metric_value
                FROM model_metrics
                WHERE model_name = $1 
                AND version = $2
                AND metric_name = $3
                AND timestamp > $4
                ORDER BY timestamp
            """
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            rows = await conn.fetch(
                query, model_name, version, metric_name, cutoff
            )
            
            return [(row['timestamp'], row['metric_value']) for row in rows]
    
    async def get_metric_summary(
        self,
        model_name: str,
        version: str,
        hours: int = 24
    ) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value,
                    STDDEV(metric_value) as std_value,
                    COUNT(*) as sample_count
                FROM model_metrics
                WHERE model_name = $1 
                AND version = $2
                AND timestamp > $3
                GROUP BY metric_name
            """
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            rows = await conn.fetch(query, model_name, version, cutoff)
            
            summary = {}
            for row in rows:
                summary[row['metric_name']] = {
                    'mean': row['avg_value'],
                    'min': row['min_value'],
                    'max': row['max_value'],
                    'std': row['std_value'] or 0,
                    'count': row['sample_count']
                }
            
            return summary


class ModelRegistryEnhancements(ErrorHandlingMixin):
    """
    Enhanced model registry with advanced features.
    
    Combines version management, metrics tracking, and A/B testing
    capabilities for comprehensive model lifecycle management.
    """
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize enhanced registry."""
        super().__init__()
        self.db_pool = db_pool
        self.version_manager = ModelVersionManager(db_pool)
        self.metrics_tracker = ModelMetricsTracker(db_pool)
        
        # A/B test tracking
        self._active_ab_tests: Dict[str, ABTestConfig] = {}
        
    async def create_ab_test(
        self,
        config: ABTestConfig
    ) -> bool:
        """Create a new A/B test."""
        with self._handle_error("creating A/B test"):
            # Validate models exist
            model_a_exists = await self._model_exists(config.model_a)
            model_b_exists = await self._model_exists(config.model_b)
            
            if not (model_a_exists and model_b_exists):
                logger.error("One or both models don't exist")
                return False
            
            # Store test config
            self._active_ab_tests[config.test_name] = config
            
            # Initialize test in database
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO ab_tests (
                        test_name, model_a, model_b,
                        traffic_split, start_time, status
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    config.test_name,
                    config.model_a,
                    config.model_b,
                    config.traffic_split,
                    datetime.utcnow(),
                    'active'
                )
            
            logger.info(
                f"Created A/B test '{config.test_name}' between "
                f"{config.model_a} and {config.model_b}"
            )
            
            return True
    
    async def route_request(
        self,
        test_name: str,
        request_id: str
    ) -> str:
        """Route a request in an A/B test."""
        config = self._active_ab_tests.get(test_name)
        
        if not config:
            raise ValueError(f"No active A/B test named '{test_name}'")
        
        # Simple hash-based routing for consistency
        import hashlib
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        
        if (hash_value % 100) < (config.traffic_split * 100):
            selected_model = config.model_b
        else:
            selected_model = config.model_a
        
        # Log routing decision
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ab_test_assignments (
                    test_name, request_id, assigned_model, timestamp
                ) VALUES ($1, $2, $3, $4)
                """,
                test_name,
                request_id,
                selected_model,
                datetime.utcnow()
            )
        
        return selected_model
    
    async def analyze_ab_test(
        self,
        test_name: str
    ) -> Dict[str, Any]:
        """Analyze results of an A/B test."""
        config = self._active_ab_tests.get(test_name)
        
        if not config:
            raise ValueError(f"No active A/B test named '{test_name}'")
        
        # Get test metrics
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    assigned_model,
                    COUNT(*) as request_count,
                    AVG(metric_value) as avg_metric,
                    STDDEV(metric_value) as std_metric
                FROM ab_test_assignments a
                JOIN model_metrics m ON a.request_id = m.request_id
                WHERE a.test_name = $1
                AND m.metric_name = ANY($2)
                GROUP BY assigned_model
            """
            
            rows = await conn.fetch(
                query,
                test_name,
                config.success_metrics
            )
            
            results = {row['assigned_model']: dict(row) for row in rows}
        
        # Perform statistical test
        if len(results) == 2:
            # Simple t-test approximation
            model_a_stats = results.get(config.model_a, {})
            model_b_stats = results.get(config.model_b, {})
            
            if (model_a_stats.get('request_count', 0) > 30 and
                model_b_stats.get('request_count', 0) > 30):
                
                # Calculate test statistic
                mean_diff = (
                    model_b_stats.get('avg_metric', 0) -
                    model_a_stats.get('avg_metric', 0)
                )
                
                pooled_std = (
                    (model_a_stats.get('std_metric', 1) ** 2 / 
                     model_a_stats.get('request_count', 1) +
                     model_b_stats.get('std_metric', 1) ** 2 / 
                     model_b_stats.get('request_count', 1)) ** 0.5
                )
                
                if pooled_std > 0:
                    z_score = mean_diff / pooled_std
                    # Approximate p-value
                    p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
                    
                    significant = p_value < (1 - config.confidence_threshold)
                    winner = config.model_b if mean_diff > 0 else config.model_a
                else:
                    significant = False
                    winner = None
                    p_value = 1.0
            else:
                significant = False
                winner = None
                p_value = None
        else:
            significant = False
            winner = None
            p_value = None
        
        return {
            'test_name': test_name,
            'model_a': config.model_a,
            'model_b': config.model_b,
            'results': results,
            'significant': significant,
            'winner': winner,
            'p_value': p_value,
            'duration_hours': (
                datetime.utcnow() - 
                self._get_test_start_time(test_name)
            ).total_seconds() / 3600
        }
    
    async def _model_exists(self, model_identifier: str) -> bool:
        """Check if a model exists."""
        async with self.db_pool.acquire() as conn:
            # Parse model_name and version from identifier
            parts = model_identifier.rsplit('_v', 1)
            if len(parts) == 2:
                model_name, version = parts
                query = """
                    SELECT EXISTS(
                        SELECT 1 FROM model_versions
                        WHERE model_name = $1 AND version = $2
                    )
                """
                return await conn.fetchval(query, model_name, version)
            
            return False
    
    def _normal_cdf(self, z: float) -> float:
        """Approximate normal CDF for z-score."""
        # Simple approximation
        import math
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def _get_test_start_time(self, test_name: str) -> datetime:
        """Get test start time (placeholder)."""
        # In real implementation, would query from database
        return datetime.utcnow() - timedelta(hours=24)