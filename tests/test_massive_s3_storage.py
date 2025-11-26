"""Tests for MASSIVE.COM S3 storage adapter."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storage.massive_s3_adapter import MassiveS3StorageAdapter


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for S3 configuration."""
    monkeypatch.setenv("MASSIVE_S3_ENABLED", "true")
    monkeypatch.setenv("MASSIVE_S3_ACCESS_KEY_ID", "test_access_key")
    monkeypatch.setenv("MASSIVE_S3_SECRET_ACCESS_KEY", "test_secret_key")
    monkeypatch.setenv("MASSIVE_S3_ENDPOINT", "https://files.massive.com")
    monkeypatch.setenv("MASSIVE_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("MASSIVE_S3_REGION", "us-east-1")


@pytest.fixture
def mock_disabled_env(monkeypatch):
    """Set up environment with S3 disabled."""
    monkeypatch.setenv("MASSIVE_S3_ENABLED", "false")


@pytest.fixture
def mock_boto3_client():
    """Create a mock boto3 S3 client."""
    with patch("storage.massive_s3_adapter.boto3.client") as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


def test_adapter_disabled_when_env_var_false(mock_disabled_env):
    """Test that adapter is disabled when MASSIVE_S3_ENABLED is false."""
    adapter = MassiveS3StorageAdapter()

    assert adapter.enabled is False
    assert adapter.client is None


def test_adapter_disabled_when_missing_credentials(monkeypatch):
    """Test that adapter is disabled when credentials are missing."""
    monkeypatch.setenv("MASSIVE_S3_ENABLED", "true")
    # Don't set credentials

    adapter = MassiveS3StorageAdapter()

    assert adapter.enabled is False
    assert adapter.client is None


def test_adapter_initialization_with_env_vars(mock_env_vars, mock_boto3_client):
    """Test adapter initialization with environment variables."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    assert adapter.enabled is True
    assert adapter.access_key_id == "test_access_key"
    assert adapter.secret_access_key == "test_secret_key"
    assert adapter.endpoint == "https://files.massive.com"
    assert adapter.bucket_name == "test-bucket"
    assert adapter.region == "us-east-1"


def test_adapter_initialization_with_explicit_params(mock_boto3_client):
    """Test adapter initialization with explicit parameters."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter(
        enabled=True,
        access_key_id="explicit_key",
        secret_access_key="explicit_secret",
        endpoint="https://custom.endpoint.com",
        bucket_name="custom-bucket",
        region="us-west-2",
    )

    assert adapter.enabled is True
    assert adapter.access_key_id == "explicit_key"
    assert adapter.secret_access_key == "explicit_secret"
    assert adapter.endpoint == "https://custom.endpoint.com"
    assert adapter.bucket_name == "custom-bucket"
    assert adapter.region == "us-west-2"


def test_upload_file_success(mock_env_vars, mock_boto3_client, tmp_path):
    """Test successful file upload to S3."""
    # Mock successful bucket verification and upload
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Create a test file
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")

    # Upload file
    result = adapter.upload_file(test_file, "test/path/file.txt")

    assert result is True
    mock_boto3_client.upload_file.assert_called_once()


def test_upload_file_nonexistent(mock_env_vars, mock_boto3_client):
    """Test upload of non-existent file."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Try to upload non-existent file
    result = adapter.upload_file("/nonexistent/file.txt", "test/path/file.txt")

    assert result is False
    mock_boto3_client.upload_file.assert_not_called()


def test_download_file_success(mock_env_vars, mock_boto3_client, tmp_path):
    """Test successful file download from S3."""
    # Mock successful bucket verification and download
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Download file
    local_path = tmp_path / "downloaded_file.txt"
    result = adapter.download_file("test/path/file.txt", local_path)

    assert result is True
    mock_boto3_client.download_file.assert_called_once()


def test_upload_bytes_success(mock_env_vars, mock_boto3_client):
    """Test successful bytes upload to S3."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Upload bytes
    test_data = b"test bytes content"
    result = adapter.upload_bytes(test_data, "test/path/data.bin")

    assert result is True
    mock_boto3_client.put_object.assert_called_once()


def test_download_bytes_success(mock_env_vars, mock_boto3_client):
    """Test successful bytes download from S3."""
    # Mock successful bucket verification and download
    mock_boto3_client.head_bucket.return_value = {}
    mock_boto3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"test bytes content")
    }

    adapter = MassiveS3StorageAdapter()

    # Download bytes
    result = adapter.download_bytes("test/path/data.bin")

    assert result == b"test bytes content"
    mock_boto3_client.get_object.assert_called_once()


def test_list_objects_success(mock_env_vars, mock_boto3_client):
    """Test successful object listing."""
    # Mock successful bucket verification and listing
    mock_boto3_client.head_bucket.return_value = {}
    mock_boto3_client.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "test/file1.txt",
                "Size": 100,
                "LastModified": "2024-01-01",
                "ETag": "abc123",
            },
            {
                "Key": "test/file2.txt",
                "Size": 200,
                "LastModified": "2024-01-02",
                "ETag": "def456",
            },
        ]
    }

    adapter = MassiveS3StorageAdapter()

    # List objects
    objects = adapter.list_objects(prefix="test/")

    assert len(objects) == 2
    assert objects[0]["key"] == "test/file1.txt"
    assert objects[0]["size"] == 100
    assert objects[1]["key"] == "test/file2.txt"
    assert objects[1]["size"] == 200


def test_delete_object_success(mock_env_vars, mock_boto3_client):
    """Test successful object deletion."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Delete object
    result = adapter.delete_object("test/path/file.txt")

    assert result is True
    mock_boto3_client.delete_object.assert_called_once()


def test_object_exists_true(mock_env_vars, mock_boto3_client):
    """Test object existence check when object exists."""
    # Mock successful bucket verification and head_object
    mock_boto3_client.head_bucket.return_value = {}
    mock_boto3_client.head_object.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Check if object exists
    result = adapter.object_exists("test/path/file.txt")

    assert result is True


def test_object_exists_false(mock_env_vars, mock_boto3_client):
    """Test object existence check when object doesn't exist."""
    from botocore.exceptions import ClientError

    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}
    # Mock object not found
    mock_boto3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404"}}, "HeadObject"
    )

    adapter = MassiveS3StorageAdapter()

    # Check if object exists
    result = adapter.object_exists("test/path/file.txt")

    assert result is False


def test_operations_when_disabled(mock_disabled_env):
    """Test that operations return False/empty when adapter is disabled."""
    adapter = MassiveS3StorageAdapter()

    assert adapter.upload_file("/some/file.txt", "s3/key") is False
    assert adapter.download_file("s3/key", "/local/path") is False
    assert adapter.upload_bytes(b"data", "s3/key") is False
    assert adapter.download_bytes("s3/key") is None
    assert adapter.list_objects() == []
    assert adapter.delete_object("s3/key") is False
    assert adapter.object_exists("s3/key") is False


def test_presigned_url_generation(mock_env_vars, mock_boto3_client):
    """Test presigned URL generation."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}
    mock_boto3_client.generate_presigned_url.return_value = "https://presigned.url/test"

    adapter = MassiveS3StorageAdapter()

    # Generate presigned URL
    url = adapter.get_presigned_url("test/path/file.txt", expiration=3600)

    assert url == "https://presigned.url/test"
    mock_boto3_client.generate_presigned_url.assert_called_once()


def test_sync_directory_success(mock_env_vars, mock_boto3_client, tmp_path):
    """Test successful directory sync to S3."""
    # Mock successful bucket verification
    mock_boto3_client.head_bucket.return_value = {}

    adapter = MassiveS3StorageAdapter()

    # Create test directory with files
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content1")
    (test_dir / "file2.txt").write_text("content2")

    # Sync directory
    count = adapter.sync_directory(test_dir, "test_prefix")

    assert count == 2
    assert mock_boto3_client.upload_file.call_count == 2
