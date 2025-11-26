#!/usr/bin/env python
"""
Example script demonstrating MASSIVE.COM S3 storage integration.

This script shows how to use the S3 adapter for various operations:
- Uploading and downloading files
- Working with bytes directly
- Listing and managing objects
- Syncing directories
- Generating presigned URLs

Run this script to test your S3 integration:
    python examples/s3_storage_example.py
"""

import os
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from storage import (
    MassiveS3StorageAdapter,
    backup_features_to_s3,
    backup_ledger_to_s3,
    list_s3_backups,
)


def example_basic_operations():
    """Example: Basic S3 operations."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic S3 Operations")
    logger.info("=" * 60)

    # Initialize S3 adapter (reads credentials from .env)
    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled. Enable it in .env to run this example.")
        return

    # Create a test file
    test_file = Path("test_data.txt")
    test_file.write_text(f"Test data created at {datetime.now()}")
    logger.info(f"Created test file: {test_file}")

    # Upload file to S3
    s3_key = f"examples/test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if s3.upload_file(test_file, s3_key):
        logger.info(f"✓ Uploaded file to S3: {s3_key}")

        # Check if file exists
        if s3.object_exists(s3_key):
            logger.info(f"✓ Confirmed file exists in S3")

        # Download file
        downloaded_file = Path("downloaded_test_data.txt")
        if s3.download_file(s3_key, downloaded_file):
            logger.info(f"✓ Downloaded file from S3: {downloaded_file}")
            logger.info(f"   Content: {downloaded_file.read_text()}")

        # Clean up
        downloaded_file.unlink(missing_ok=True)
        s3.delete_object(s3_key)
        logger.info(f"✓ Cleaned up test files")

    # Clean up local file
    test_file.unlink(missing_ok=True)


def example_bytes_operations():
    """Example: Working with bytes directly."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Bytes Operations (No Local Files)")
    logger.info("=" * 60)

    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled.")
        return

    # Upload bytes directly
    test_data = f"Binary data created at {datetime.now()}".encode("utf-8")
    s3_key = f"examples/bytes_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"

    if s3.upload_bytes(test_data, s3_key):
        logger.info(f"✓ Uploaded {len(test_data)} bytes to S3: {s3_key}")

        # Download bytes directly
        downloaded_data = s3.download_bytes(s3_key)
        if downloaded_data:
            logger.info(f"✓ Downloaded {len(downloaded_data)} bytes from S3")
            logger.info(f"   Content: {downloaded_data.decode('utf-8')}")

        # Clean up
        s3.delete_object(s3_key)
        logger.info(f"✓ Cleaned up test object")


def example_list_objects():
    """Example: Listing objects in S3."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Listing Objects")
    logger.info("=" * 60)

    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled.")
        return

    # Upload a few test files
    test_keys = []
    for i in range(3):
        s3_key = f"examples/list_test_{i}.txt"
        s3.upload_bytes(f"Test file {i}".encode(), s3_key)
        test_keys.append(s3_key)
        logger.info(f"✓ Uploaded test file: {s3_key}")

    # List objects
    objects = s3.list_objects(prefix="examples/")
    logger.info(f"\nFound {len(objects)} objects in 'examples/' prefix:")
    for obj in objects:
        logger.info(f"  - {obj['key']} ({obj['size']} bytes)")

    # Clean up
    for s3_key in test_keys:
        s3.delete_object(s3_key)
    logger.info(f"\n✓ Cleaned up {len(test_keys)} test files")


def example_directory_sync():
    """Example: Syncing a directory to S3."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Directory Sync")
    logger.info("=" * 60)

    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled.")
        return

    # Create a test directory
    test_dir = Path("test_sync_dir")
    test_dir.mkdir(exist_ok=True)

    # Create some test files
    (test_dir / "file1.txt").write_text("File 1 content")
    (test_dir / "file2.txt").write_text("File 2 content")
    (test_dir / "file3.txt").write_text("File 3 content")

    logger.info(f"Created test directory with 3 files")

    # Sync directory to S3
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"examples/sync_test_{timestamp}"
    count = s3.sync_directory(test_dir, s3_prefix)
    logger.info(f"✓ Synced {count} files to S3 prefix: {s3_prefix}")

    # List uploaded files
    objects = s3.list_objects(prefix=s3_prefix)
    logger.info(f"\nUploaded files:")
    for obj in objects:
        logger.info(f"  - {obj['key']}")

    # Clean up S3
    for obj in objects:
        s3.delete_object(obj["key"])

    # Clean up local directory
    for file in test_dir.iterdir():
        file.unlink()
    test_dir.rmdir()
    logger.info(f"\n✓ Cleaned up test directory and S3 objects")


def example_presigned_urls():
    """Example: Generating presigned URLs."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Presigned URLs")
    logger.info("=" * 60)

    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled.")
        return

    # Upload a test file
    s3_key = f"examples/presigned_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    test_content = "This file can be accessed via presigned URL"
    s3.upload_bytes(test_content.encode(), s3_key)
    logger.info(f"✓ Uploaded test file: {s3_key}")

    # Generate presigned URL (valid for 1 hour)
    url = s3.get_presigned_url(s3_key, expiration=3600)
    if url:
        logger.info(f"\n✓ Generated presigned URL (valid for 1 hour):")
        logger.info(f"  {url}")
        logger.info(f"\nYou can share this URL to give temporary access to the file.")

    # Clean up
    s3.delete_object(s3_key)
    logger.info(f"\n✓ Cleaned up test file")


def example_high_level_utilities():
    """Example: Using high-level utility functions."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: High-Level Utilities")
    logger.info("=" * 60)

    s3 = MassiveS3StorageAdapter()

    if not s3.enabled:
        logger.warning("S3 storage is disabled.")
        return

    # Example: List all backups
    logger.info("\nListing existing backups:")

    # List ledger backups
    ledger_backups = list_s3_backups(prefix="ledger/archive/")
    logger.info(f"  Ledger backups: {len(ledger_backups)}")
    for backup in ledger_backups[:5]:  # Show first 5
        logger.info(f"    - {backup['key']}")

    # List feature backups
    feature_backups = list_s3_backups(prefix="features/")
    logger.info(f"\n  Feature backups: {len(feature_backups)}")
    for backup in feature_backups[:5]:  # Show first 5
        logger.info(f"    - {backup['key']}")

    logger.info(f"\nNote: Use backup_features_to_s3() and backup_ledger_to_s3()")
    logger.info(f"      for production backup operations.")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 60)
    logger.info("MASSIVE.COM S3 Storage Integration Examples")
    logger.info("=" * 60)

    # Check if S3 is enabled
    s3 = MassiveS3StorageAdapter()
    if not s3.enabled:
        logger.error("\n❌ S3 storage is not enabled!")
        logger.info("\nTo enable S3 storage:")
        logger.info("1. Update .env file:")
        logger.info("   MASSIVE_S3_ENABLED=true")
        logger.info("   MASSIVE_S3_ACCESS_KEY_ID=your_access_key_id")
        logger.info("   MASSIVE_S3_SECRET_ACCESS_KEY=your_secret_access_key")
        logger.info("\n2. Run this script again")
        return

    logger.info(f"\n✓ S3 storage is enabled")
    logger.info(f"  Endpoint: {s3.endpoint}")
    logger.info(f"  Bucket: {s3.bucket_name}")
    logger.info(f"  Region: {s3.region}")

    # Run examples
    try:
        example_basic_operations()
        example_bytes_operations()
        example_list_objects()
        example_directory_sync()
        example_presigned_urls()
        example_high_level_utilities()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
