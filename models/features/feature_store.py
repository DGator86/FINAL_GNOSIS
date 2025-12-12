"""
Feature Store for caching and versioning features
Supports both in-memory (Redis) and persistent (Parquet) storage
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Persistent feature storage with versioning

    Features:
    - Save/load features to/from Parquet files
    - Feature schema versioning
    - Metadata tracking (timestamps, data sources, transformations)
    - Fast retrieval for backtesting
    """

    def __init__(self, base_path: str = "./data/features"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.base_path / "metadata.json"
        self.metadata = self._load_metadata()

    def save_features(
        self,
        features: pd.DataFrame,
        version: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save features to storage

        Args:
            features: Feature DataFrame
            version: Version string (e.g., "v1.0.0", "20240101")
            metadata: Additional metadata dict
        """
        version_path = self.base_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Save features
        features_file = version_path / "features.parquet"
        features.to_parquet(features_file, compression="gzip")

        # Save feature names
        feature_names_file = version_path / "feature_names.json"
        with open(feature_names_file, "w") as f:
            json.dump(features.columns.tolist(), f)

        # Save metadata
        meta = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "shape": features.shape,
            "feature_count": len(features.columns),
            "row_count": len(features),
            "checksum": self._compute_checksum(features),
            **(metadata or {}),
        }

        meta_file = version_path / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # Update global metadata
        self.metadata[version] = meta
        self._save_metadata()

        logger.info(f"Saved features version {version}: {features.shape}")

    def load_features(
        self,
        version: str,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load features from storage

        Args:
            version: Version string
            columns: Optional list of specific columns to load (for efficiency)

        Returns:
            Feature DataFrame
        """
        version_path = self.base_path / version
        if not version_path.exists():
            raise ValueError(f"Version {version} not found")

        features_file = version_path / "features.parquet"

        if columns:
            features = pd.read_parquet(features_file, columns=columns)
        else:
            features = pd.read_parquet(features_file)

        logger.info(f"Loaded features version {version}: {features.shape}")
        return features

    def get_feature_names(self, version: str) -> List[str]:
        """Get list of feature names for a version"""
        version_path = self.base_path / version
        feature_names_file = version_path / "feature_names.json"

        with open(feature_names_file, "r") as f:
            return json.load(f)

    def list_versions(self) -> List[str]:
        """List all available feature versions"""
        return list(self.metadata.keys())

    def get_latest_version(self) -> str:
        """Get most recent feature version"""
        versions = self.list_versions()
        if not versions:
            raise ValueError("No feature versions found")

        # Sort by timestamp
        sorted_versions = sorted(
            versions, key=lambda v: self.metadata[v]["timestamp"], reverse=True
        )
        return sorted_versions[0]

    def delete_version(self, version: str):
        """Delete a feature version"""
        import shutil

        version_path = self.base_path / version
        if version_path.exists():
            shutil.rmtree(version_path)

        if version in self.metadata:
            del self.metadata[version]
            self._save_metadata()

        logger.info(f"Deleted features version {version}")

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute checksum for data integrity verification"""
        data_str = df.to_json()
        return hashlib.md5(data_str.encode()).hexdigest()

    def _load_metadata(self) -> Dict:
        """Load global metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save global metadata"""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)


class RedisFeatureCache:
    """
    In-memory feature cache using Redis
    For real-time serving (last 1000 rows)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        try:
            import redis

            self.redis_client = redis.from_url(redis_url)
            self.enabled = True
            logger.info("Redis feature cache enabled")
        except (ImportError, Exception) as e:
            logger.warning(f"Redis not available: {e}. Cache disabled.")
            self.enabled = False

    def set(self, key: str, features: pd.DataFrame, expiry: int = 3600):
        """Cache features with expiry"""
        if not self.enabled:
            return

        # Serialize to JSON
        data = features.to_json()
        self.redis_client.setex(key, expiry, data)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached features"""
        if not self.enabled:
            return None

        data = self.redis_client.get(key)
        if data:
            return pd.read_json(data)
        return None

    def delete(self, key: str):
        """Delete cached features"""
        if self.enabled:
            self.redis_client.delete(key)


# Example usage
if __name__ == "__main__":
    # Initialize feature store
    store = FeatureStore("./data/features")

    # Mock features
    features = pd.DataFrame(np.random.randn(1000, 50))
    features.columns = [f"feature_{i}" for i in range(50)]

    # Save
    store.save_features(
        features,
        version="v1.0.0",
        metadata={
            "description": "Initial feature set",
            "data_source": "alpaca",
            "date_range": "2024-01-01 to 2024-12-31",
        },
    )

    # Load
    loaded = store.load_features("v1.0.0")
    print(f"Loaded features: {loaded.shape}")

    # List versions
    print(f"Available versions: {store.list_versions()}")
