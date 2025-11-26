"""Utility functions for S3 storage integration.

Provides helper functions to integrate S3 storage with existing storage classes
like FeatureStore, LedgerStore, etc.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from storage.massive_s3_adapter import MassiveS3StorageAdapter


def get_s3_adapter(
    enabled: Optional[bool] = None,
    endpoint: Optional[str] = None,
    bucket: Optional[str] = None,
) -> MassiveS3StorageAdapter:
    """Get a configured S3 storage adapter instance.

    Args:
        enabled: Override the enabled status
        endpoint: Override the endpoint URL
        bucket: Override the bucket name

    Returns:
        Configured MassiveS3StorageAdapter instance
    """
    return MassiveS3StorageAdapter(
        enabled=enabled,
        endpoint=endpoint,
        bucket_name=bucket,
    )


def backup_features_to_s3(
    local_feature_dir: str | Path,
    version: str,
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> bool:
    """Backup feature store version to S3.

    Args:
        local_feature_dir: Path to local features directory
        version: Feature version to backup
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        True if backup succeeded, False otherwise
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, skipping feature backup")
        return False

    local_feature_dir = Path(local_feature_dir)
    version_dir = local_feature_dir / version

    if not version_dir.exists():
        logger.error(f"Feature version directory does not exist: {version_dir}")
        return False

    # Upload all files in the version directory
    s3_prefix = f"features/{version}"
    count = adapter.sync_directory(version_dir, s3_prefix)

    if count > 0:
        logger.info(f"✓ Backed up {count} feature files for version {version} to S3")
        return True
    else:
        logger.warning(f"No files backed up for feature version {version}")
        return False


def restore_features_from_s3(
    version: str,
    local_feature_dir: str | Path,
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> bool:
    """Restore feature store version from S3.

    Args:
        version: Feature version to restore
        local_feature_dir: Path to local features directory
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        True if restore succeeded, False otherwise
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, skipping feature restore")
        return False

    local_feature_dir = Path(local_feature_dir)
    version_dir = local_feature_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # List and download all files for this version
    s3_prefix = f"features/{version}"
    objects = adapter.list_objects(prefix=s3_prefix)

    if not objects:
        logger.warning(f"No feature files found in S3 for version {version}")
        return False

    downloaded_count = 0
    for obj in objects:
        s3_key = obj["key"]
        # Extract filename from S3 key
        filename = s3_key.split("/")[-1]
        local_path = version_dir / filename

        if adapter.download_file(s3_key, local_path):
            downloaded_count += 1

    if downloaded_count > 0:
        logger.info(f"✓ Restored {downloaded_count} feature files for version {version} from S3")
        return True
    else:
        logger.warning(f"Failed to restore feature version {version}")
        return False


def backup_ledger_to_s3(
    ledger_path: str | Path,
    archive_name: Optional[str] = None,
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> bool:
    """Backup ledger file to S3.

    Args:
        ledger_path: Path to local ledger file
        archive_name: Optional custom archive name (default: ledger_YYYYMMDD_HHMMSS.jsonl)
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        True if backup succeeded, False otherwise
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, skipping ledger backup")
        return False

    ledger_path = Path(ledger_path)

    if not ledger_path.exists():
        logger.error(f"Ledger file does not exist: {ledger_path}")
        return False

    # Generate archive name with timestamp
    if not archive_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"ledger_{timestamp}.jsonl"

    s3_key = f"ledger/archive/{archive_name}"

    # Add metadata with file stats
    metadata = {
        "original_path": str(ledger_path),
        "backup_timestamp": datetime.now().isoformat(),
        "file_size": str(ledger_path.stat().st_size),
    }

    if adapter.upload_file(ledger_path, s3_key, metadata=metadata):
        logger.info(f"✓ Backed up ledger to S3: {archive_name}")
        return True
    else:
        logger.error("Failed to backup ledger to S3")
        return False


def backup_directory_to_s3(
    local_dir: str | Path,
    s3_prefix: str,
    pattern: str = "*",
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> int:
    """Backup a local directory to S3.

    Args:
        local_dir: Path to local directory
        s3_prefix: S3 prefix (folder) to backup to
        pattern: Glob pattern for files to include (default: all files)
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        Number of files backed up
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, skipping directory backup")
        return 0

    count = adapter.sync_directory(local_dir, s3_prefix, pattern=pattern)

    if count > 0:
        logger.info(f"✓ Backed up {count} files from {local_dir} to s3://{s3_prefix}")

    return count


def list_s3_backups(
    prefix: str = "",
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> list[dict]:
    """List backups in S3 with optional prefix filter.

    Args:
        prefix: Optional prefix to filter backups (e.g., "ledger/", "features/")
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        List of backup metadata dictionaries
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, cannot list backups")
        return []

    objects = adapter.list_objects(prefix=prefix)

    return objects


def create_backup_manifest(
    backup_type: str,
    local_paths: list[str | Path],
    s3_keys: list[str],
    s3_adapter: Optional[MassiveS3StorageAdapter] = None,
) -> bool:
    """Create a backup manifest file in S3 to track what was backed up.

    Args:
        backup_type: Type of backup (e.g., "features", "ledger", "models")
        local_paths: List of local file paths that were backed up
        s3_keys: List of corresponding S3 keys
        s3_adapter: Optional S3 adapter instance (creates new one if not provided)

    Returns:
        True if manifest was created successfully, False otherwise
    """
    adapter = s3_adapter or get_s3_adapter()

    if not adapter.enabled:
        logger.debug("S3 storage disabled, skipping manifest creation")
        return False

    manifest = {
        "backup_type": backup_type,
        "timestamp": datetime.now().isoformat(),
        "files": [
            {
                "local_path": str(local_path),
                "s3_key": s3_key,
            }
            for local_path, s3_key in zip(local_paths, s3_keys)
        ],
        "total_files": len(local_paths),
    }

    manifest_json = json.dumps(manifest, indent=2)
    manifest_key = f"manifests/{backup_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if adapter.upload_bytes(manifest_json.encode("utf-8"), manifest_key):
        logger.info(f"✓ Created backup manifest: {manifest_key}")
        return True
    else:
        logger.error("Failed to create backup manifest")
        return False
