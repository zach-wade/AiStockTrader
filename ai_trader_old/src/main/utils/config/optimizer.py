"""
Configuration Optimizer

Intelligent configuration optimization with parameter tuning and performance-based adjustments.
"""

# Standard library imports
import asyncio
from collections import deque
from collections.abc import Callable
from datetime import datetime
import logging
import statistics
from typing import Any

from .types import (
    ConfigParameter,
    OptimizationResult,
    OptimizationStrategy,
    OptimizationTarget,
    ParameterType,
)

logger = logging.getLogger(__name__)


class ConfigOptimizer:
    """
    Intelligent configuration optimizer.

    Uses various optimization strategies to automatically tune configuration
    parameters based on performance metrics and system behavior.
    """

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        max_iterations: int = 100,
        convergence_threshold: float = 0.01,
    ):
        """
        Initialize configuration optimizer.

        Args:
            strategy: Optimization strategy to use
            max_iterations: Maximum optimization iterations
            convergence_threshold: Threshold for convergence detection
        """
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Parameter registry
        self.parameters: dict[str, ConfigParameter] = {}
        self.optimization_targets: list[OptimizationTarget] = []

        # Metrics collection
        self.metrics_history: deque = deque(maxlen=1000)
        self.performance_baseline: dict[str, float] | None = None

        # Optimization history
        self.optimization_history: list[OptimizationResult] = []

        # Callbacks
        self.metric_collectors: list[Callable[[], dict[str, float]]] = []
        self.config_validators: list[Callable[[dict[str, Any]], bool]] = []
        self.config_appliers: list[Callable[[dict[str, Any]], None]] = []

        logger.info(f"Config optimizer initialized with {strategy.value} strategy")

    def register_parameter(self, parameter: ConfigParameter):
        """Register a configuration parameter for optimization."""
        self.parameters[parameter.name] = parameter
        logger.debug(f"Registered parameter: {parameter.name}")

    def register_parameters(self, parameters: list[ConfigParameter]):
        """Register multiple configuration parameters."""
        for param in parameters:
            self.register_parameter(param)

    def add_optimization_target(self, target: OptimizationTarget):
        """Add an optimization target metric."""
        self.optimization_targets.append(target)
        logger.debug(f"Added optimization target: {target.metric_name}")

    def add_metric_collector(self, collector: Callable[[], dict[str, float]]):
        """Add a metric collection function."""
        self.metric_collectors.append(collector)
        logger.debug("Added metric collector")

    def add_config_validator(self, validator: Callable[[dict[str, Any]], bool]):
        """Add a configuration validation function."""
        self.config_validators.append(validator)
        logger.debug("Added config validator")

    def add_config_applier(self, applier: Callable[[dict[str, Any]], None]):
        """Add a configuration application function."""
        self.config_appliers.append(applier)
        logger.debug("Added config applier")

    async def collect_metrics(self) -> dict[str, float]:
        """Collect current performance metrics."""
        metrics = {}

        for collector in self.metric_collectors:
            try:
                if asyncio.iscoroutinefunction(collector):
                    collector_metrics = await collector()
                else:
                    collector_metrics = collector()
                metrics.update(collector_metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

        # Store in history
        self.metrics_history.append({"timestamp": datetime.now(), "metrics": metrics.copy()})

        return metrics

    def validate_configuration(self, config: dict[str, Any]) -> bool:
        """Validate a configuration against constraints and validators."""

        # Check parameter constraints
        for param_name, value in config.items():
            if param_name in self.parameters:
                param = self.parameters[param_name]
                if not param.validate_value(value):
                    logger.warning(f"Parameter {param_name} failed validation: {value}")
                    return False

        # Check dependencies
        for param_name, param in self.parameters.items():
            if param.constraint and param.constraint.dependencies:
                for dep in param.constraint.dependencies:
                    if dep not in config:
                        logger.warning(f"Missing dependency {dep} for parameter {param_name}")
                        return False

        # Run custom validators
        for validator in self.config_validators:
            try:
                if not validator(config):
                    logger.warning("Custom validator failed")
                    return False
            except Exception as e:
                logger.error(f"Error in config validator: {e}")
                return False

        return True

    async def apply_configuration(self, config: dict[str, Any]) -> bool:
        """Apply configuration using registered appliers."""
        if not self.validate_configuration(config):
            return False

        for applier in self.config_appliers:
            try:
                if asyncio.iscoroutinefunction(applier):
                    await applier(config)
                else:
                    applier(config)
            except Exception as e:
                logger.error(f"Error applying config: {e}")
                return False

        return True

    def calculate_fitness(self, metrics: dict[str, float]) -> float:
        """Calculate fitness score for given metrics."""
        if not self.optimization_targets:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for target in self.optimization_targets:
            if target.metric_name not in metrics:
                continue

            metric_value = metrics[target.metric_name]

            if target.target_value is not None:
                # Calculate distance from target
                distance = abs(metric_value - target.target_value)
                score = max(0, 1 - (distance / target.target_value))
            else:
                # Use raw metric value (higher is better unless minimize=True)
                score = 1.0 / (1.0 + metric_value) if target.minimize else metric_value

            total_score += score * target.weight
            total_weight += target.weight

        return total_score / total_weight if total_weight > 0 else 0.0

    async def optimize_configuration(
        self, current_config: dict[str, Any], test_duration: float = 30.0
    ) -> OptimizationResult:
        """
        Optimize configuration parameters.

        Args:
            current_config: Current configuration
            test_duration: Time to test each configuration

        Returns:
            OptimizationResult with optimization details
        """
        start_time = datetime.now()

        # Collect baseline metrics
        logger.info("Collecting baseline metrics...")
        baseline_metrics = await self.collect_metrics()
        baseline_fitness = self.calculate_fitness(baseline_metrics)

        best_config = current_config.copy()
        best_metrics = baseline_metrics.copy()
        best_fitness = baseline_fitness

        iteration = 0
        convergence_count = 0

        logger.info(f"Starting optimization with baseline fitness: {baseline_fitness:.4f}")

        while iteration < self.max_iterations:
            iteration += 1

            # Generate parameter variations
            variations = self.generate_parameter_variations(best_config)

            if not variations:
                logger.info("No more variations to test")
                break

            improved = False

            # Test each variation
            for variation in variations:
                logger.debug(f"Testing variation {iteration}: {variation}")

                # Apply configuration
                if not await self.apply_configuration(variation):
                    logger.warning("Failed to apply configuration, skipping")
                    continue

                # Wait for system to stabilize
                await asyncio.sleep(test_duration)

                # Collect metrics
                metrics = await self.collect_metrics()
                fitness = self.calculate_fitness(metrics)

                # Check if this is better
                if fitness > best_fitness:
                    best_config = variation.copy()
                    best_metrics = metrics.copy()
                    best_fitness = fitness
                    improved = True

                    logger.info(f"Improved fitness: {fitness:.4f} (iteration {iteration})")
                    convergence_count = 0
                else:
                    convergence_count += 1

            # Check convergence
            if not improved:
                convergence_count += 1
                if convergence_count >= 5:  # No improvement for 5 iterations
                    logger.info("Convergence detected, stopping optimization")
                    break

        # Apply final best configuration
        await self.apply_configuration(best_config)

        # Calculate results
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()

        performance_improvement = (
            (best_fitness - baseline_fitness) / baseline_fitness * 100
            if baseline_fitness > 0
            else 0
        )

        changed_parameters = [
            param for param in current_config if current_config.get(param) != best_config.get(param)
        ]

        result = OptimizationResult(
            original_config=current_config,
            optimized_config=best_config,
            performance_improvement=performance_improvement,
            optimization_time=optimization_time,
            iterations=iteration,
            metrics_before=baseline_metrics,
            metrics_after=best_metrics,
            changed_parameters=changed_parameters,
            timestamp=end_time,
        )

        self.optimization_history.append(result)

        logger.info(
            f"Optimization completed: {performance_improvement:.2f}% improvement "
            f"in {optimization_time:.1f}s over {iteration} iterations"
        )

        return result

    def generate_parameter_variations(
        self, current_config: dict[str, Any], variation_factor: float = 0.1
    ) -> list[dict[str, Any]]:
        """Generate parameter variations for optimization."""
        variations = []

        for param_name, param in self.parameters.items():
            if param_name not in current_config:
                continue

            current_value = current_config[param_name]

            # Generate variations based on parameter type
            if param.param_type == ParameterType.INTEGER:
                variations.extend(
                    self._generate_integer_variations(
                        current_config, param_name, current_value, variation_factor
                    )
                )
            elif param.param_type == ParameterType.FLOAT:
                variations.extend(
                    self._generate_float_variations(
                        current_config, param_name, current_value, variation_factor
                    )
                )
            elif param.param_type == ParameterType.BOOLEAN:
                variations.extend(
                    self._generate_boolean_variations(current_config, param_name, current_value)
                )
            elif param.param_type == ParameterType.ENUM:
                variations.extend(
                    self._generate_enum_variations(current_config, param_name, current_value, param)
                )

        return variations

    def _generate_integer_variations(
        self, config: dict[str, Any], param_name: str, current_value: int, variation_factor: float
    ) -> list[dict[str, Any]]:
        """Generate integer parameter variations."""
        variations = []
        param = self.parameters[param_name]

        # Calculate variation range
        variation_range = max(1, int(current_value * variation_factor))

        # Generate up/down variations
        for delta in [-variation_range, variation_range]:
            new_value = current_value + delta

            # Apply constraints
            if param.constraint:
                if param.constraint.min_value is not None:
                    new_value = max(new_value, param.constraint.min_value)
                if param.constraint.max_value is not None:
                    new_value = min(new_value, param.constraint.max_value)
                if param.constraint.step_size is not None:
                    new_value = (
                        round(new_value / param.constraint.step_size) * param.constraint.step_size
                    )

            if new_value != current_value:
                new_config = config.copy()
                new_config[param_name] = int(new_value)
                variations.append(new_config)

        return variations

    def _generate_float_variations(
        self, config: dict[str, Any], param_name: str, current_value: float, variation_factor: float
    ) -> list[dict[str, Any]]:
        """Generate float parameter variations."""
        variations = []
        param = self.parameters[param_name]

        # Generate up/down variations
        for multiplier in [1 - variation_factor, 1 + variation_factor]:
            new_value = current_value * multiplier

            # Apply constraints
            if param.constraint:
                if param.constraint.min_value is not None:
                    new_value = max(new_value, param.constraint.min_value)
                if param.constraint.max_value is not None:
                    new_value = min(new_value, param.constraint.max_value)
                if param.constraint.step_size is not None:
                    new_value = (
                        round(new_value / param.constraint.step_size) * param.constraint.step_size
                    )

            if abs(new_value - current_value) > 1e-6:
                new_config = config.copy()
                new_config[param_name] = new_value
                variations.append(new_config)

        return variations

    def _generate_boolean_variations(
        self, config: dict[str, Any], param_name: str, current_value: bool
    ) -> list[dict[str, Any]]:
        """Generate boolean parameter variations."""
        new_config = config.copy()
        new_config[param_name] = not current_value
        return [new_config]

    def _generate_enum_variations(
        self, config: dict[str, Any], param_name: str, current_value: Any, param: ConfigParameter
    ) -> list[dict[str, Any]]:
        """Generate enum parameter variations."""
        variations = []

        if param.constraint and param.constraint.allowed_values:
            for value in param.constraint.allowed_values:
                if value != current_value:
                    new_config = config.copy()
                    new_config[param_name] = value
                    variations.append(new_config)

        return variations

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get optimization history."""
        return [result.to_dict() for result in self.optimization_history]

    def get_parameter_impact_analysis(self) -> dict[str, dict[str, float]]:
        """Analyze parameter impact on performance."""
        if not self.optimization_history:
            return {}

        impact_analysis = {}

        for param_name, param in self.parameters.items():
            impact_data = {
                "correlation_score": 0.0,
                "change_frequency": 0,
                "avg_improvement": 0.0,
                "impact_score": param.impact_score,
            }

            # Analyze historical changes
            improvements = []
            changes = 0

            for result in self.optimization_history:
                if param_name in result.changed_parameters:
                    changes += 1
                    improvements.append(result.performance_improvement)

            if changes > 0:
                impact_data["change_frequency"] = changes
                impact_data["avg_improvement"] = statistics.mean(improvements)
                impact_data["correlation_score"] = abs(statistics.mean(improvements)) / 100

            impact_analysis[param_name] = impact_data

        return impact_analysis

    def reset_optimization_history(self):
        """Reset optimization history."""
        self.optimization_history.clear()
        self.metrics_history.clear()
        logger.info("Optimization history reset")


# Global optimizer instance
_global_optimizer = ConfigOptimizer()


def get_global_optimizer() -> ConfigOptimizer:
    """Get the global configuration optimizer instance."""
    return _global_optimizer
