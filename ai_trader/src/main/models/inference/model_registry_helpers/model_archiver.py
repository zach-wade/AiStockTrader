# File: src/ai_trader/models/inference/model_registry_helpers/model_archiver.py

import logging
from datetime import datetime, timedelta, timezone
from typing import List

# Corrected absolute import for ModelVersion
from main.models.inference.model_registry_types import ModelVersion

logger = logging.getLogger(__name__)

class ModelArchiver:
    """
    Manages the archiving process for old or unused model versions.
    Changes their status to 'archived' and removes their deployment traffic.
    """

    def __init__(self):
        logger.debug("ModelArchiver initialized.")

    def archive_old_model_versions(self, all_versions: List[ModelVersion], days_to_keep: int) -> int:
        """
        Identifies and updates the status of model versions that are older than a specified
        number of days and are not currently active ('production' or 'candidate').
        These models are marked as 'archived' with 0% deployment traffic.

        Args:
            all_versions: A list of all ModelVersion objects across all model_ids.
                          (Assumed to be mutable list of objects from registry's state).
            days_to_keep: The age in days beyond which non-active models should be archived.

        Returns:
            The number of model versions that were newly marked as archived.
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        archived_count = 0
        
        for version in all_versions:
            # Criteria for archiving:
            # 1. Created before the cutoff date
            # 2. Not currently 'production' or 'candidate' (i.e., not actively serving traffic)
            # 3. Not already 'archived'
            if (version.created_at < cutoff_date and 
                version.status not in ['production', 'candidate'] and
                version.status != 'archived'):
                
                version.status = 'archived'
                version.deployment_pct = 0.0 # Ensure no traffic
                archived_count += 1
                logger.info(f"Archived model '{version.model_id}' version '{version.version}' (created: {version.created_at.isoformat()}).")
        
        if archived_count > 0:
            logger.info(f"Completed archiving of {archived_count} old model versions.")
        
        return archived_count