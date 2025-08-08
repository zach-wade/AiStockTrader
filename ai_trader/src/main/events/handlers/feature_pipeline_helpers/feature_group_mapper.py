"""
Feature group mapping for scanner alerts.

This module contains the main FeatureGroupMapper class that maps scanner 
alerts to the appropriate feature groups for computation.
"""

from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timezone

from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import record_metric, timer
from main.events.types import ScanAlert, AlertType
from main.events.handlers.feature_pipeline_helpers.feature_types import (
    FeatureGroup,
    FeatureGroupConfig,
    FeatureRequest
)
from main.events.handlers.feature_pipeline_helpers.feature_config import (
    initialize_group_configs,
    initialize_alert_mappings,
    get_conditional_group_rules,
    get_priority_calculation_rules
)

logger = get_logger(__name__)


class FeatureGroupMapper(ErrorHandlingMixin):
    """
    Maps scanner alerts to feature computation requests.
    
    This mapper determines which features need to be computed based on
    the alert type, score, and other contextual information.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature group mapper.
        
        Args:
            config: Optional configuration overrides
        """
        super().__init__()
        self.config = config or {}
        
        # Initialize configurations
        self.group_configs = self._initialize_group_configs()
        self.alert_mappings = self._initialize_alert_mappings()
        self.conditional_rules = get_conditional_group_rules()
        self.priority_rules = get_priority_calculation_rules()
        
        logger.info(
            f"FeatureGroupMapper initialized with {len(self.group_configs)} "
            f"feature groups and {len(self.alert_mappings)} alert mappings"
        )
    
    @timer
    def map_alert_to_features(self, alert: ScanAlert) -> FeatureRequest:
        """
        Map a scanner alert to required feature groups.
        
        Args:
            alert: Scanner alert to map
        
        Returns:
            Feature request with required groups
        """
        try:
            # Get base feature groups for alert type
            base_groups = self.alert_mappings.get(
                alert.alert_type,
                self.alert_mappings[AlertType.UNKNOWN]
            )
            
            # Record alert mapping metrics
            record_metric("feature_group_mapper.alert_mapped", 1, tags={
                "alert_type": alert.alert_type.value,
                "is_default": alert.alert_type not in self.alert_mappings
            })
            
            # Add conditional groups based on alert data
            additional_groups = self._get_conditional_groups(alert)
            
            # Combine groups
            all_groups = list(set(base_groups + additional_groups))
            
            # Record groups expansion
            record_metric("feature_group_mapper.groups_selected", 
                         len(all_groups), 
                         metric_type="histogram",
                         tags={
                             "alert_type": alert.alert_type.value,
                             "base_groups": len(base_groups),
                             "additional_groups": len(additional_groups)
                         })
            
            # Calculate priority
            priority = self._calculate_priority(alert)
            
            # Create feature request
            request = FeatureRequest(
                symbol=alert.symbol,
                feature_groups=all_groups,
                alert_type=alert.alert_type,
                priority=priority,
                metadata={
                    'alert_id': id(alert),
                    'alert_timestamp': alert.timestamp,
                    'alert_data': alert.data
                }
            )
            
            logger.info(
                f"Mapped {alert.alert_type.value} to {len(all_groups)} "
                f"feature groups with priority {priority}"
            )
            
            return request
        except Exception as e:
            self.handle_error(e, f"mapping alert {alert.alert_type}")
            # Return a basic request with default groups
            return FeatureRequest(
                symbol=alert.symbol,
                feature_groups=self.alert_mappings[AlertType.UNKNOWN],
                alert_type=alert.alert_type,
                priority=5,  # Default priority
                metadata={'error': str(e)}
            )
    
    @timer
    def get_required_data_types(self, feature_groups: List[FeatureGroup]) -> Set[str]:
        """
        Get all data types required for the given feature groups.
        
        Args:
            feature_groups: List of feature groups
            
        Returns:
            Set of required data type names
        """
        data_types = set()
        for group in feature_groups:
            if group in self.group_configs:
                data_types.update(self.group_configs[group].required_data)
        
        # Record data types query
        record_metric("feature_group_mapper.data_types_query", 1, tags={
            "group_count": len(feature_groups),
            "data_type_count": len(data_types)
        })
        
        logger.debug(f"Required data types for {len(feature_groups)} groups: {data_types}")
        return data_types
    
    def prioritize_requests(self, requests: List[FeatureRequest]) -> List[FeatureRequest]:
        """
        Sort feature requests by priority.
        
        Args:
            requests: List of feature requests
            
        Returns:
            Sorted list with highest priority first
        """
        return sorted(
            requests,
            key=lambda r: (r.priority, -r.created_at.timestamp()),
            reverse=True
        )
    
    def _initialize_group_configs(self) -> Dict[FeatureGroup, FeatureGroupConfig]:
        """Initialize or override feature group configurations."""
        configs = initialize_group_configs()
        
        # Apply any config overrides
        if 'group_configs' in self.config:
            for group_name, overrides in self.config['group_configs'].items():
                if hasattr(FeatureGroup, group_name):
                    group = FeatureGroup[group_name]
                    if group in configs:
                        # Update existing config
                        for key, value in overrides.items():
                            setattr(configs[group], key, value)
        
        return configs
    
    def _initialize_alert_mappings(self) -> Dict[AlertType, List[FeatureGroup]]:
        """Initialize or override alert to feature group mappings."""
        mappings = initialize_alert_mappings()
        
        # Apply any config overrides
        if 'alert_mappings' in self.config:
            for alert_name, groups in self.config['alert_mappings'].items():
                if hasattr(AlertType, alert_name):
                    alert_type = AlertType[alert_name]
                    # Convert string group names to FeatureGroup enums
                    feature_groups = []
                    for group_name in groups:
                        if hasattr(FeatureGroup, group_name):
                            feature_groups.append(FeatureGroup[group_name])
                    if feature_groups:
                        mappings[alert_type] = feature_groups
        
        return mappings
    
    def _get_conditional_groups(self, alert: ScanAlert) -> List[FeatureGroup]:
        """
        Get additional feature groups based on alert conditions.
        
        Args:
            alert: Scanner alert
            
        Returns:
            List of additional feature groups
        """
        additional_groups = []
        
        # High score alerts get advanced features
        if alert.score > self.conditional_rules['high_score_threshold']:
            additional_groups.extend(self.conditional_rules['high_score_additions'])
        
        # Volume spike alerts
        if alert.data.get('volume_multiplier', 1.0) > self.conditional_rules['volume_spike_multiplier']:
            additional_groups.extend(self.conditional_rules['volume_spike_additions'])
        
        # News keyword-based additions
        if 'keywords' in alert.data:
            for keyword, groups in self.conditional_rules['news_keyword_additions'].items():
                if keyword in alert.data['keywords']:
                    additional_groups.extend(groups)
        
        # Time-based additions
        current_hour = datetime.now(timezone.utc).hour
        if 4 <= current_hour < 9:  # Pre-market
            additional_groups.extend(self.conditional_rules['time_based_additions']['pre_market'])
        elif 20 <= current_hour < 24:  # Post-market
            additional_groups.extend(self.conditional_rules['time_based_additions']['post_market'])
        
        return list(set(additional_groups))  # Remove duplicates
    
    def _calculate_priority(self, alert: ScanAlert) -> int:
        """
        Calculate request priority based on alert type and score.
        
        Args:
            alert: Scanner alert
            
        Returns:
            Priority value (0-10)
        """
        # Base priority from alert type
        base_priority = self.priority_rules['base_priority_map'].get(
            alert.alert_type,
            0
        )
        
        # Add score-based boost
        score_boost = int(alert.score * self.priority_rules['score_multiplier'])
        
        # Add market phase boost
        current_hour = datetime.now(timezone.utc).hour
        phase_boost = 0
        if 4 <= current_hour < 9:
            phase_boost = self.priority_rules['market_phase_boost']['pre_market']
        elif 9 <= current_hour < 10:
            phase_boost = self.priority_rules['market_phase_boost']['market_open']
        elif 15 <= current_hour < 16:
            phase_boost = self.priority_rules['market_phase_boost']['market_close']
        elif 16 <= current_hour < 20:
            phase_boost = self.priority_rules['market_phase_boost']['post_market']
        
        # Add volatility boost if present
        volatility_level = alert.data.get('volatility_level', 'normal')
        volatility_boost = self.priority_rules['volatility_boost'].get(volatility_level, 0)
        
        # Calculate final priority
        priority = base_priority + score_boost + phase_boost + volatility_boost
        
        # Apply min/max bounds
        priority = max(
            self.priority_rules['min_priority'],
            min(priority, self.priority_rules['max_priority'])
        )
        
        logger.debug(
            f"Calculated priority {priority} for {alert.alert_type.value}: "
            f"base={base_priority}, score_boost={score_boost}, "
            f"phase_boost={phase_boost}, volatility_boost={volatility_boost}"
        )
        
        return priority
    
    def get_computation_params(
        self,
        feature_groups: List[FeatureGroup]
    ) -> Dict[str, Any]:
        """
        Get merged computation parameters for the given feature groups.
        
        Args:
            feature_groups: List of feature groups
            
        Returns:
            Merged computation parameters
        """
        params = {}
        for group in feature_groups:
            if group in self.group_configs:
                group_params = self.group_configs[group].computation_params
                for key, value in group_params.items():
                    if key not in params:
                        params[key] = value
                    elif isinstance(value, list) and isinstance(params[key], list):
                        # Merge lists
                        params[key] = list(set(params[key] + value))
                    elif isinstance(value, dict) and isinstance(params[key], dict):
                        # Merge dicts
                        params[key].update(value)
        
        return params
    
    def _get_all_dependencies(self, groups: List[FeatureGroup]) -> List[FeatureGroup]:
        """
        Expand feature groups to include all dependencies.
        
        Args:
            groups: Initial feature groups
            
        Returns:
            Expanded list including all dependencies
        """
        all_groups = set()
        to_process = list(groups)
        
        while to_process:
            group = to_process.pop()
            if group not in all_groups:
                all_groups.add(group)
                if group in self.group_configs:
                    dependencies = self.group_configs[group].dependencies
                    to_process.extend(dependencies)
        
        return list(all_groups)