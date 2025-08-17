"""
Traffic routing for model inference requests.

This module handles intelligent routing of prediction requests including:
- Load balancing across model instances
- A/B testing traffic splits
- Gradual rollout strategies
- Sticky sessions and affinity
"""

# Standard library imports
import asyncio
import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger, timer
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


class RoutingStrategy(Enum):
    """Traffic routing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    STICKY = "sticky"
    CANARY = "canary"
    AB_TEST = "ab_test"


@dataclass
class ModelEndpoint:
    """Model endpoint information."""

    endpoint_id: str
    model_name: str
    version: str
    host: str
    port: int
    weight: float = 1.0
    healthy: bool = True
    active_connections: int = 0
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingConfig:
    """Traffic routing configuration."""

    strategy: RoutingStrategy
    sticky_session_duration_minutes: int = 60
    health_check_interval_seconds: int = 30
    max_retries: int = 3
    timeout_seconds: int = 10

    # A/B test configuration
    ab_test_enabled: bool = False
    ab_test_traffic_split: float = 0.5  # Fraction to variant B

    # Canary configuration
    canary_enabled: bool = False
    canary_traffic_percentage: float = 10.0

    # Load balancing weights
    endpoint_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    endpoint: ModelEndpoint
    request_id: str
    routing_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrafficRouter(ErrorHandlingMixin):
    """
    Routes traffic to model endpoints based on configured strategy.

    Features:
    - Multiple routing strategies
    - Health-aware routing
    - Session affinity
    - A/B testing support
    - Performance tracking
    """

    def __init__(self, db_pool: DatabasePool):
        """Initialize traffic router."""
        super().__init__()
        self.db_pool = db_pool

        # Endpoint registry
        self._endpoints: Dict[str, List[ModelEndpoint]] = {}
        self._endpoint_index: Dict[str, int] = {}  # For round-robin

        # Session affinity
        self._session_affinity: Dict[str, Tuple[str, datetime]] = {}

        # Performance tracking
        self._routing_metrics: Dict[str, Dict[str, int]] = {}

        # Health check tasks
        self._health_check_tasks: Dict[str, asyncio.Task] = {}

    async def register_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Register a model endpoint."""
        with self._handle_error("registering endpoint"):
            model_key = f"{endpoint.model_name}_v{endpoint.version}"

            if model_key not in self._endpoints:
                self._endpoints[model_key] = []
                self._endpoint_index[model_key] = 0

            # Check if endpoint already exists
            existing_idx = None
            for idx, ep in enumerate(self._endpoints[model_key]):
                if ep.endpoint_id == endpoint.endpoint_id:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                # Update existing
                self._endpoints[model_key][existing_idx] = endpoint
            else:
                # Add new
                self._endpoints[model_key].append(endpoint)

            # Start health checking
            await self._start_health_check(endpoint)

            logger.info(
                f"Registered endpoint {endpoint.endpoint_id} for "
                f"{model_key} at {endpoint.host}:{endpoint.port}"
            )

            record_metric(
                "traffic_router.endpoint_registered",
                1,
                tags={"model": endpoint.model_name, "version": endpoint.version},
            )

    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a model endpoint."""
        with self._handle_error("unregistering endpoint"):
            # Find and remove endpoint
            for model_key, endpoints in self._endpoints.items():
                for idx, ep in enumerate(endpoints):
                    if ep.endpoint_id == endpoint_id:
                        # Stop health check
                        if endpoint_id in self._health_check_tasks:
                            self._health_check_tasks[endpoint_id].cancel()
                            del self._health_check_tasks[endpoint_id]

                        # Remove endpoint
                        del endpoints[idx]

                        logger.info(f"Unregistered endpoint {endpoint_id}")

                        return True

            return False

    @timer
    async def route_request(
        self,
        model_name: str,
        version: str,
        request_id: str,
        routing_config: RoutingConfig,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route a request to an endpoint.

        Args:
            model_name: Target model name
            version: Target model version
            request_id: Unique request ID
            routing_config: Routing configuration
            session_id: Optional session ID for affinity
            metadata: Optional request metadata

        Returns:
            Routing decision with selected endpoint
        """
        with self._handle_error("routing request"):
            model_key = f"{model_name}_v{version}"

            # Get available endpoints
            endpoints = self._get_healthy_endpoints(model_key)

            if not endpoints:
                raise ValueError(f"No healthy endpoints for {model_key}")

            # Check session affinity
            if session_id and routing_config.strategy == RoutingStrategy.STICKY:
                endpoint = self._check_session_affinity(session_id, endpoints, routing_config)
                if endpoint:
                    return RoutingDecision(
                        endpoint=endpoint,
                        request_id=request_id,
                        routing_reason="session_affinity",
                        metadata={"session_id": session_id},
                    )

            # Route based on strategy
            if routing_config.strategy == RoutingStrategy.ROUND_ROBIN:
                endpoint = self._route_round_robin(model_key, endpoints)
            elif routing_config.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
                endpoint = self._route_weighted_round_robin(endpoints, routing_config)
            elif routing_config.strategy == RoutingStrategy.LEAST_CONNECTIONS:
                endpoint = self._route_least_connections(endpoints)
            elif routing_config.strategy == RoutingStrategy.RANDOM:
                endpoint = self._route_random(endpoints)
            elif routing_config.strategy == RoutingStrategy.HASH_BASED:
                endpoint = self._route_hash_based(request_id, endpoints)
            elif routing_config.strategy == RoutingStrategy.CANARY:
                endpoint = self._route_canary(endpoints, routing_config, request_id)
            elif routing_config.strategy == RoutingStrategy.AB_TEST:
                endpoint = self._route_ab_test(endpoints, routing_config, request_id)
            else:
                # Default to random
                endpoint = self._route_random(endpoints)

            # Update metrics
            self._update_routing_metrics(endpoint, routing_config.strategy)

            # Update session affinity if needed
            if session_id and routing_config.strategy == RoutingStrategy.STICKY:
                self._update_session_affinity(session_id, endpoint, routing_config)

            # Record routing decision
            await self._record_routing_decision(request_id, endpoint, routing_config.strategy)

            return RoutingDecision(
                endpoint=endpoint,
                request_id=request_id,
                routing_reason=routing_config.strategy.value,
                metadata=metadata or {},
            )

    def _get_healthy_endpoints(self, model_key: str) -> List[ModelEndpoint]:
        """Get healthy endpoints for a model."""
        endpoints = self._endpoints.get(model_key, [])
        return [ep for ep in endpoints if ep.healthy]

    def _route_round_robin(self, model_key: str, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Round-robin routing."""
        if not endpoints:
            raise ValueError("No endpoints available")

        # Get and update index
        idx = self._endpoint_index.get(model_key, 0)
        endpoint = endpoints[idx % len(endpoints)]
        self._endpoint_index[model_key] = (idx + 1) % len(endpoints)

        return endpoint

    def _route_weighted_round_robin(
        self, endpoints: List[ModelEndpoint], config: RoutingConfig
    ) -> ModelEndpoint:
        """Weighted round-robin routing."""
        # Build cumulative weights
        weights = []
        cumulative = 0.0

        for ep in endpoints:
            weight = config.endpoint_weights.get(ep.endpoint_id, ep.weight)
            cumulative += weight
            weights.append(cumulative)

        # Select based on weight
        if cumulative > 0:
            rand_val = secure_uniform(0, cumulative)
            idx = bisect.bisect_left(weights, rand_val)
            return endpoints[min(idx, len(endpoints) - 1)]

        # Fallback to random
        return secure_choice(endpoints)

    def _route_least_connections(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Least connections routing."""
        return min(endpoints, key=lambda ep: ep.active_connections)

    def _route_random(self, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Random routing."""
        return secure_choice(endpoints)

    def _route_hash_based(self, request_id: str, endpoints: List[ModelEndpoint]) -> ModelEndpoint:
        """Hash-based routing for consistency."""
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return endpoints[hash_val % len(endpoints)]

    def _route_canary(
        self, endpoints: List[ModelEndpoint], config: RoutingConfig, request_id: str
    ) -> ModelEndpoint:
        """Canary routing."""
        # Separate canary and stable endpoints
        canary_endpoints = [
            ep for ep in endpoints if ep.metadata.get("deployment_type") == "canary"
        ]
        stable_endpoints = [
            ep for ep in endpoints if ep.metadata.get("deployment_type") != "canary"
        ]

        if not canary_endpoints:
            # No canary available
            return self._route_random(stable_endpoints)

        if not stable_endpoints:
            # Only canary available
            return self._route_random(canary_endpoints)

        # Route based on percentage
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        if (hash_val % 100) < config.canary_traffic_percentage:
            return self._route_random(canary_endpoints)
        else:
            return self._route_random(stable_endpoints)

    def _route_ab_test(
        self, endpoints: List[ModelEndpoint], config: RoutingConfig, request_id: str
    ) -> ModelEndpoint:
        """A/B test routing."""
        # Separate A and B endpoints
        a_endpoints = [ep for ep in endpoints if ep.metadata.get("ab_variant") == "A"]
        b_endpoints = [ep for ep in endpoints if ep.metadata.get("ab_variant") == "B"]

        if not a_endpoints or not b_endpoints:
            # Not properly configured for A/B
            return self._route_random(endpoints)

        # Route based on split
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        if (hash_val % 1000) < (config.ab_test_traffic_split * 1000):
            return self._route_random(b_endpoints)
        else:
            return self._route_random(a_endpoints)

    def _check_session_affinity(
        self, session_id: str, available_endpoints: List[ModelEndpoint], config: RoutingConfig
    ) -> Optional[ModelEndpoint]:
        """Check session affinity."""
        if session_id in self._session_affinity:
            endpoint_id, timestamp = self._session_affinity[session_id]

            # Check if affinity is still valid
            if (datetime.utcnow() - timestamp).total_seconds() < (
                config.sticky_session_duration_minutes * 60
            ):

                # Find endpoint
                for ep in available_endpoints:
                    if ep.endpoint_id == endpoint_id:
                        return ep

        return None

    def _update_session_affinity(
        self, session_id: str, endpoint: ModelEndpoint, config: RoutingConfig
    ) -> None:
        """Update session affinity."""
        self._session_affinity[session_id] = (endpoint.endpoint_id, datetime.utcnow())

        # Clean old sessions periodically
        if len(self._session_affinity) > 10000:
            self._cleanup_old_sessions(config)

    def _cleanup_old_sessions(self, config: RoutingConfig) -> None:
        """Clean up old session affinity entries."""
        cutoff = datetime.utcnow() - timedelta(minutes=config.sticky_session_duration_minutes)

        to_remove = [
            session_id
            for session_id, (_, timestamp) in self._session_affinity.items()
            if timestamp < cutoff
        ]

        for session_id in to_remove:
            del self._session_affinity[session_id]

    def _update_routing_metrics(self, endpoint: ModelEndpoint, strategy: RoutingStrategy) -> None:
        """Update routing metrics."""
        if endpoint.endpoint_id not in self._routing_metrics:
            self._routing_metrics[endpoint.endpoint_id] = {"total_requests": 0, "by_strategy": {}}

        metrics = self._routing_metrics[endpoint.endpoint_id]
        metrics["total_requests"] += 1

        strategy_key = strategy.value
        if strategy_key not in metrics["by_strategy"]:
            metrics["by_strategy"][strategy_key] = 0
        metrics["by_strategy"][strategy_key] += 1

        # Update active connections
        endpoint.active_connections += 1

    async def _record_routing_decision(
        self, request_id: str, endpoint: ModelEndpoint, strategy: RoutingStrategy
    ) -> None:
        """Record routing decision in database."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO routing_decisions (
                    request_id, endpoint_id, model_name,
                    version, strategy, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                request_id,
                endpoint.endpoint_id,
                endpoint.model_name,
                endpoint.version,
                strategy.value,
                datetime.utcnow(),
            )

        record_metric(
            "traffic_router.request_routed",
            1,
            tags={
                "model": endpoint.model_name,
                "version": endpoint.version,
                "strategy": strategy.value,
                "endpoint": endpoint.endpoint_id,
            },
        )

    async def _start_health_check(self, endpoint: ModelEndpoint) -> None:
        """Start health checking for an endpoint."""
        if endpoint.endpoint_id in self._health_check_tasks:
            # Cancel existing task
            self._health_check_tasks[endpoint.endpoint_id].cancel()

        # Create new health check task
        task = asyncio.create_task(self._health_check_loop(endpoint))
        self._health_check_tasks[endpoint.endpoint_id] = task

    async def _health_check_loop(self, endpoint: ModelEndpoint) -> None:
        """Health check loop for an endpoint."""
        while True:
            try:
                # Perform health check
                healthy = await self._check_endpoint_health(endpoint)

                # Update status
                endpoint.healthy = healthy
                endpoint.last_health_check = datetime.utcnow()

                # Log status changes
                if not healthy:
                    logger.warning(f"Endpoint {endpoint.endpoint_id} marked unhealthy")

                # Wait before next check
                await asyncio.sleep(30)  # Default 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {endpoint.endpoint_id}: {e}")
                await asyncio.sleep(30)

    async def _check_endpoint_health(self, endpoint: ModelEndpoint) -> bool:
        """
        Check if an endpoint is healthy.

        In real implementation, would make HTTP health check request.
        """
        # Simulate health check
        # Local imports
        from main.utils.core import secure_uniform

        return secure_uniform(0, 1) > 0.05  # 95% healthy

    def get_routing_stats(
        self, model_name: Optional[str] = None, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            "total_endpoints": 0,
            "healthy_endpoints": 0,
            "total_requests": 0,
            "by_strategy": {},
            "by_endpoint": {},
        }

        # Filter endpoints
        if model_name and version:
            model_key = f"{model_name}_v{version}"
            endpoints = self._endpoints.get(model_key, [])
        else:
            endpoints = []
            for eps in self._endpoints.values():
                endpoints.extend(eps)

        # Calculate stats
        stats["total_endpoints"] = len(endpoints)
        stats["healthy_endpoints"] = sum(1 for ep in endpoints if ep.healthy)

        # Aggregate metrics
        for endpoint_id, metrics in self._routing_metrics.items():
            stats["total_requests"] += metrics["total_requests"]

            for strategy, count in metrics["by_strategy"].items():
                if strategy not in stats["by_strategy"]:
                    stats["by_strategy"][strategy] = 0
                stats["by_strategy"][strategy] += count

            stats["by_endpoint"][endpoint_id] = metrics["total_requests"]

        return stats

    async def update_endpoint_weight(self, endpoint_id: str, weight: float) -> bool:
        """Update endpoint weight for weighted routing."""
        for endpoints in self._endpoints.values():
            for ep in endpoints:
                if ep.endpoint_id == endpoint_id:
                    ep.weight = weight
                    logger.info(f"Updated weight for {endpoint_id} to {weight}")
                    return True

        return False
