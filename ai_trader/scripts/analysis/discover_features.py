# File: scripts/analysis/discover_features.py

import importlib.util
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, List

# --- Configuration ---
# Set the root path of your project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define the paths relative to the project root
CALCULATORS_PATH = PROJECT_ROOT / "feature_pipeline/calculators"
FEATURE_REGISTRY_PATH = PROJECT_ROOT / "data_pipeline/processing/features/metadata/feature_sets.json"
REPORTS_PATH = PROJECT_ROOT / "reports/feature_discovery"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureDiscoveryEngine:
    """
    Scans the codebase to find all implemented feature calculators and compares
    them against the registered features in feature_sets.json to find any
    new, undocumented features.
    """

    def __init__(self):
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        # Assuming a common base class for all feature calculators
        # This might need to be imported if it's in a different location
        # from feature_pipeline.calculators.base_calculator import FeatureCalculator
        # self.base_class = FeatureCalculator
        self.base_class_name = "FeatureCalculator" # Using name for simplicity

    def _load_registered_feature_sets(self) -> Dict[str, List[str]]:
        """Loads the current feature registry from the JSON file."""
        if not FEATURE_REGISTRY_PATH.exists():
            logger.warning("Feature registry file not found. Returning empty registry.")
            return {}
        with open(FEATURE_REGISTRY_PATH, 'r') as f:
            return json.load(f)

    def _discover_code_features(self) -> Dict[str, Set[str]]:
        """
        Dynamically imports all modules in the calculators directory and inspects
        them to find all methods that appear to be feature calculations.
        """
        discovered_features = {}
        logger.info(f"Scanning for calculators in: {CALCULATORS_PATH}")

        for py_file in CALCULATORS_PATH.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                # Dynamically load the module from its file path
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Inspect the module for classes that are feature calculators
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a feature calculator (but not the base class itself)
                    if name != self.base_class_name and hasattr(obj, 'get_feature_names'):
                        calculator_instance = obj()
                        features = calculator_instance.get_feature_names()
                        if features:
                            discovered_features[py_file.name] = set(features)
                            logger.info(f"-> Found {len(features)} features in {py_file.name}")

            except Exception as e:
                logger.error(f"Could not inspect file {py_file.name}: {e}")
        
        return discovered_features

    def run(self):
        """
        Executes the discovery process and generates a report.
        """
        logger.info("--- Starting Feature Discovery Engine ---")
        
        # 1. Load registered features
        registered_sets = self._load_registered_feature_sets()
        all_registered_features = {feature for f_set in registered_sets.values() for feature in f_set['features']}
        logger.info(f"Found {len(all_registered_features)} features registered in feature_sets.json.")

        # 2. Discover features implemented in the code
        features_in_code = self._discover_code_features()
        all_code_features = {feature for f_set in features_in_code.values() for feature in f_set}
        logger.info(f"Discovered {len(all_code_features)} unique features in the codebase.")

        # 3. Find the difference
        newly_discovered_features = all_code_features - all_registered_features
        
        # 4. Generate the report
        self._generate_report(newly_discovered_features, features_in_code)

        logger.info("--- Feature Discovery Complete ---")

    def _generate_report(self, new_features: Set[str], features_by_file: Dict[str, Set[str]]):
        """Generates and saves a human-readable Markdown report."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_path = REPORTS_PATH / f"feature_discovery_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write(f"# Feature Discovery Report - {timestamp}\n\n")
            if not new_features:
                f.write("✅ **System is in sync.** No new, unregistered features were found.\n")
                logger.info("✅ System in sync. No new features to report.")
                return

            f.write(f"⚠️ **Action Required:** Found **{len(new_features)}** new, unregistered features in the codebase.\n\n")
            f.write("Please review these features and add them to `data_pipeline/processing/features/metadata/feature_sets.json` if they are ready for production use.\n\n")
            
            # Group new features by the file they were found in
            report_by_file = defaultdict(list)
            for feature in sorted(list(new_features)):
                for file, feature_set in features_by_file.items():
                    if feature in feature_set:
                        report_by_file[file].append(feature)
                        break
            
            for file, features in report_by_file.items():
                f.write(f"### 📄 From: `{file}`\n")
                for feature in features:
                    f.write(f"- `{feature}`\n")
                f.write("\n")
        
        logger.warning(f"ACTION REQUIRED: Found {len(new_features)} new features. Report saved to: {report_path}")


if __name__ == "__main__":
    engine = FeatureDiscoveryEngine()
    engine.run()