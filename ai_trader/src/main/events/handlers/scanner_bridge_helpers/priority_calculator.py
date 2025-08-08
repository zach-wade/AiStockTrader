# File: src/main/events/scanner_bridge_helpers/priority_calculator.py

from typing import Dict, Any, Optional
import yaml
import os
from main.utils.core import get_logger, ErrorHandlingMixin
from main.utils.monitoring import record_metric, timer

logger = get_logger(__name__)

class PriorityCalculator(ErrorHandlingMixin):
    """
    Calculates the priority of a feature computation request based on
    alert score and alert type.
    
    Now loads priority boosts from configuration file.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initializes the PriorityCalculator from configuration file."""
        super().__init__()
        
        # Load configuration
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            config_path = os.path.join(base_dir, 'config', 'events', 'priority_boosts.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.priority_boost_map = config['priority_boosts']
            self.priority_config = config.get('priority_config', {
                'min_priority': 1,
                'max_priority': 10,
                'base_multiplier': 10
            })
        except Exception as e:
            self.handle_error(e, "loading priority boosts configuration")
            # Provide defaults if config loading fails
            self.priority_boost_map = {}
            self.priority_config = {
                'min_priority': 1,
                'max_priority': 10,
                'base_multiplier': 10
            }
            
            logger.info(f"PriorityCalculator initialized from {config_path}")

    @timer
    def calculate_priority(self, score: float, alert_type: str) -> int:
        """
        Calculates the final priority for a feature request.

        Args:
            score: The numerical score associated with the alert (e.g., 0.0 to 1.0).
            alert_type: The type of the alert (string).

        Returns:
            An integer representing the priority (e.g., 1 to 10, higher is more urgent).
            Priority is capped between 1 and 10.
        """
        # Base priority derived from score using configured multiplier
        base_priority = int(score * self.priority_config['base_multiplier'])
        
        # Apply boost for specific alert types
        boost = self.priority_boost_map.get(alert_type, 0)
        
        # Ensure priority is within configured range
        final_priority = max(
            self.priority_config['min_priority'], 
            min(base_priority + boost, self.priority_config['max_priority'])
        )
        
        # Record priority calculation metrics
        record_metric("priority_calculator.priority_calculated", 
                     final_priority, 
                     metric_type="histogram",
                     tags={
                         "alert_type": alert_type,
                         "has_boost": boost > 0,
                         "boost_value": boost
                     })
        
        logger.debug(f"Calculated priority for alert '{alert_type}' (score {score:.2f}): {final_priority}.")
        return final_priority