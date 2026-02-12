"""MASSIVE.COM S3-compatible storage adapter.

Provides cloud storage capabilities using MASSIVE.COM's S3-compatible API.
Supports uploading/downloading files, feature versioning, ledger archiving, and more.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger


class MassiveS3StorageAdapter:
    """S3-compatible storage adapter for MASSIVE.COM cloud storage.

    Provides methods for:
    - Uploading/downloading files and objects
    - Listing and deleting objects
    - Syncing local directories to S3
    - Feature versioning and archival
    - Ledger and log backups

    Authentication via environment variables:
    - MASSIVE_S3_ACCESS_KEY_ID
    - MASSIVE_S3_SECRET_ACCESS_KEY
    - MASSIVE_S3_ENDPOINT (default: https://files.massive.com)
    - MASSIVE_S3_BUCKET (default: flatfiles)
    - MASSIVE_S3_REGION (default: us-east-1)
    - MASSIVE_S3_ENABLED (default: false)
    """

    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        """Initialize the MASSIVE S3 storage adapter.

        Args:
            access_key_id: S3 access key ID (reads from MASSIVE_S3_ACCESS_KEY_ID if not provided)
            secret_access_key: S3 secret access key (reads from MASSIVE_S3_SECRET_ACCESS_KEY if not provided)
            endpoint: S3 endpoint URL (reads from MASSIVE_S3_ENDPOINT if not provided)
            bucket_name: S3 bucket name (reads from MASSIVE_S3_BUCKET if not provided)
            region: S3 region (reads from MASSIVE_S3_REGION if not provided)
            enabled: Whether S3 storage is enabled (reads from MASSIVE_S3_ENABLED if not provided)
        """
        # Read configuration from environment variables with fallbacks
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("MASSIVE_S3_ENABLED", "false").lower() == "true"
        )

        self.access_key_id = access_key_id or os.getenv("MASSIVE_S3_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("MASSIVE_S3_SECRET_ACCESS_KEY")
        self.endpoint = endpoint or os.getenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com")
        self.bucket_name = bucket_name or os.getenv("MASSIVE_S3_BUCKET", "flatfiles")
        self.region = region or os.getenv("MASSIVE_S3_REGION", "us-east-1")

        self.client: Optional[boto3.client] = None

        if not self.enabled:
            logger.info("MASSIVE S3 storage is disabled (MASSIVE_S3_ENABLED=false)")
            return

        # Validate credentials
        if not self.access_key_id or not self.secret_access_key:
            logger.warning(
                "⚠️  MASSIVE S3 credentials not configured. "
                "Set MASSIVE_S3_ACCESS_KEY_ID and MASSIVE_S3_SECRET_ACCESS_KEY environment variables."
            )
            self.enabled = False
            return

        try:
            # Initialize boto3 S3 client with custom endpoint
            self.client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region,
                config=Config(signature_version="s3v4"),
            )

            # Test connection by checking if bucket exists
            self._verify_bucket()

            logger.info(
                f"✓ MASSIVE S3 storage initialized: endpoint={self.endpoint}, bucket={self.bucket_name}"
            )

        except NoCredentialsError:
            logger.error("❌ Invalid S3 credentials provided")
            self.enabled = False
            self.client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            self.enabled = False
            self.client = None

    def _verify_bucket(self) -> None:
        """Verify that the configured bucket exists and is accessible."""
        if not self.client:
            return

        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                logger.warning(f"Bucket '{self.bucket_name}' does not exist, attempting to create...")
                try:
                    self.client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"✓ Created bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"❌ Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"❌ Cannot access bucket '{self.bucket_name}': {e}")
                raise

    def upload_file(
        self,
        local_path: str | Path,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Upload a local file to S3.

        Args:
            local_path: Path to the local file
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata to attach to the object

        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping upload")
            return False

        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Local file does not exist: {local_path}")
            return False

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )

            logger.info(f"✓ Uploaded {local_path} → s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to upload {local_path}: {e}")
            return False

    def download_file(
        self,
        s3_key: str,
        local_path: str | Path,
    ) -> bool:
        """Download a file from S3 to local storage.

        Args:
            s3_key: S3 object key (path in bucket)
            local_path: Path where the file should be saved locally

        Returns:
            True if download succeeded, False otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping download")
            return False

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path),
            )

            logger.info(f"✓ Downloaded s3://{self.bucket_name}/{s3_key} → {local_path}")
            return True

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                logger.warning(f"Object not found in S3: {s3_key}")
            else:
                logger.error(f"❌ Failed to download {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download {s3_key}: {e}")
            return False

    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Upload bytes directly to S3 without writing to local file.

        Args:
            data: Bytes to upload
            s3_key: S3 object key (path in bucket)
            metadata: Optional metadata to attach to the object

        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping upload")
            return False

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data,
                **extra_args,
            )

            logger.info(f"✓ Uploaded {len(data)} bytes → s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to upload bytes to {s3_key}: {e}")
            return False

    def download_bytes(self, s3_key: str) -> Optional[bytes]:
        """Download bytes directly from S3.

        Args:
            s3_key: S3 object key (path in bucket)

        Returns:
            Bytes if download succeeded, None otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping download")
            return None

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )
            data = response["Body"].read()

            logger.info(f"✓ Downloaded {len(data)} bytes from s3://{self.bucket_name}/{s3_key}")
            return data

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                logger.warning(f"Object not found in S3: {s3_key}")
            else:
                logger.error(f"❌ Failed to download {s3_key}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to download {s3_key}: {e}")
            return None

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List objects in S3 bucket with optional prefix filter.

        Args:
            prefix: Optional prefix to filter objects
            max_keys: Maximum number of objects to return

        Returns:
            List of object metadata dictionaries
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping list")
            return []

        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys,
            )

            objects = []
            for obj in response.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj["ETag"],
                })

            logger.info(f"✓ Listed {len(objects)} objects with prefix '{prefix}'")
            return objects

        except Exception as e:
            logger.error(f"❌ Failed to list objects: {e}")
            return []

    def delete_object(self, s3_key: str) -> bool:
        """Delete an object from S3.

        Args:
            s3_key: S3 object key (path in bucket)

        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping delete")
            return False

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )

            logger.info(f"✓ Deleted s3://{self.bucket_name}/{s3_key}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete {s3_key}: {e}")
            return False

    def object_exists(self, s3_key: str) -> bool:
        """Check if an object exists in S3.

        Args:
            s3_key: S3 object key (path in bucket)

        Returns:
            True if object exists, False otherwise
        """
        if not self.enabled or not self.client:
            return False

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key,
            )
            return True
        except ClientError:
            return False

    def sync_directory(
        self,
        local_dir: str | Path,
        s3_prefix: str,
        pattern: str = "*",
    ) -> int:
        """Sync a local directory to S3.

        Args:
            local_dir: Path to local directory
            s3_prefix: S3 prefix (folder) to sync to
            pattern: Glob pattern for files to include (default: all files)

        Returns:
            Number of files uploaded
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping sync")
            return 0

        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.error(f"Local directory does not exist: {local_dir}")
            return 0

        uploaded_count = 0

        for file_path in local_dir.rglob(pattern):
            if file_path.is_file():
                # Compute relative path and S3 key
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")

                if self.upload_file(file_path, s3_key):
                    uploaded_count += 1

        logger.info(f"✓ Synced {uploaded_count} files from {local_dir} to s3://{self.bucket_name}/{s3_prefix}")
        return uploaded_count

    def get_presigned_url(
        self,
        s3_key: str,
        expiration: int = 3600,
        http_method: str = "GET",
    ) -> Optional[str]:
        """Generate a presigned URL for temporary access to an S3 object.

        Args:
            s3_key: S3 object key (path in bucket)
            expiration: URL expiration time in seconds (default: 1 hour)
            http_method: HTTP method (GET, PUT, etc.)

        Returns:
            Presigned URL if successful, None otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("S3 storage disabled, skipping presigned URL generation")
            return None

        try:
            client_method = "get_object" if http_method == "GET" else "put_object"
            url = self.client.generate_presigned_url(
                ClientMethod=client_method,
                Params={
                    "Bucket": self.bucket_name,
                    "Key": s3_key,
                },
                ExpiresIn=expiration,
            )

            logger.info(f"✓ Generated presigned URL for {s3_key} (expires in {expiration}s)")
            return url

        except Exception as e:
            logger.error(f"❌ Failed to generate presigned URL: {e}")
            return None
