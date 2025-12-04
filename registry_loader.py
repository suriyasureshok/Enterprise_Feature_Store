"""
Registry Loader
Loads and validates the feature registry YAML file.
Provides helper methods for accessing feature group metadata.
"""

import yaml
from typing import Dict, Any

class FeatureRegistry:
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    # -------------------------------------------------------------
    # Load YAML Registry
    # -------------------------------------------------------------
    def _load_registry(self) -> Dict[str, Any]:
        try:
            with open(self.registry_path, "r") as f:
                registry = yaml.safe_load(f)
            print(f"Feature Registry Loaded Successfully from: {self.registry_path}")
            return registry
        except Exception as e:
            raise RuntimeError(f"Failed to load registry YAML: {e}")

    # -------------------------------------------------------------
    # Get ALL Feature Groups
    # -------------------------------------------------------------
    def get_feature_groups(self) -> Dict[str, Any]:
        return self.registry.get("feature_groups", {})

    # -------------------------------------------------------------
    # Get Specific Feature Group Definition
    # -------------------------------------------------------------
    def get_group(self, group_name: str) -> Dict[str, Any]:
        groups = self.get_feature_groups()
        if group_name not in groups:
            raise KeyError(f"Feature group '{group_name}' not found in registry.")
        return groups[group_name]

    # -------------------------------------------------------------
    # Validate Registry Structure (Optional)
    # -------------------------------------------------------------
    def validate(self):
        if "feature_groups" not in self.registry:
            raise ValueError("Registry missing 'feature_groups' section.")
        for name, group in self.registry["feature_groups"].items():
            if "keys" not in group:
                raise ValueError(f"Feature group '{name}' missing 'keys'")
            if "features" not in group:
                raise ValueError(f"Feature group '{name}' missing 'features'")
        print("Feature Registry Validation Successful!")