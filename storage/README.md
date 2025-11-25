# MASSIVE.COM S3 Storage Integration

Cloud storage integration for Super Gnosis using MASSIVE.COM's S3-compatible API.

## Features

- ✅ S3-compatible cloud storage for data backups
- ✅ Feature store versioning and archival
- ✅ Ledger and trading results backup
- ✅ ML model storage and versioning
- ✅ Log archival
- ✅ Presigned URL generation for secure sharing
- ✅ Automatic directory syncing

## Quick Start

### 1. Set up credentials

Copy the `.env.example` file to `.env` and add your MASSIVE.COM credentials:

```bash
# Enable S3 storage
MASSIVE_S3_ENABLED=true

# MASSIVE.COM credentials
MASSIVE_S3_ACCESS_KEY_ID=your_access_key_id_here
MASSIVE_S3_SECRET_ACCESS_KEY=your_secret_access_key_here

# S3 endpoint and bucket
MASSIVE_S3_ENDPOINT=https://files.massive.com
MASSIVE_S3_BUCKET=flatfiles
MASSIVE_S3_REGION=us-east-1
```

### 2. Enable in configuration

Update `config/config.yaml`:

```yaml
storage:
  enabled: true                     # Enable S3 storage
  provider: "massive"               # Provider name
  endpoint: "https://files.massive.com"
  bucket: "flatfiles"
  region: "us-east-1"
  auto_sync: false                  # Enable automatic syncing
  sync_features: true               # Sync feature store
  sync_ledger: true                 # Sync trading ledger
  sync_logs: false                  # Sync log files
  sync_models: false                # Sync ML models
```

### 3. Install dependencies

```bash
pip install boto3>=1.35.0
# or
pip install -r requirements.txt
```

## Usage Examples

### Basic Adapter Usage

```python
from storage import MassiveS3StorageAdapter

# Initialize adapter (reads from environment variables)
s3 = MassiveS3StorageAdapter()

# Upload a file
s3.upload_file("data/ledger.jsonl", "backups/ledger_2024.jsonl")

# Download a file
s3.download_file("backups/ledger_2024.jsonl", "data/restored_ledger.jsonl")

# Upload bytes directly
data = b"some binary data"
s3.upload_bytes(data, "data/binary_file.bin")

# Download bytes directly
downloaded = s3.download_bytes("data/binary_file.bin")

# List objects with prefix
objects = s3.list_objects(prefix="backups/")
for obj in objects:
    print(f"{obj['key']}: {obj['size']} bytes, modified {obj['last_modified']}")

# Check if object exists
if s3.object_exists("backups/ledger_2024.jsonl"):
    print("Backup exists!")

# Delete an object
s3.delete_object("old/unused_file.txt")

# Sync entire directory
s3.sync_directory("data/features", "s3_prefix/features")

# Generate presigned URL for sharing
url = s3.get_presigned_url("reports/monthly_report.pdf", expiration=3600)
print(f"Share this URL: {url}")
```

### Using Utility Functions

```python
from storage import (
    backup_features_to_s3,
    restore_features_from_s3,
    backup_ledger_to_s3,
    backup_directory_to_s3,
    list_s3_backups,
)

# Backup feature store version
backup_features_to_s3("data/features", version="v1.0.0")

# Restore feature store version
restore_features_from_s3(version="v1.0.0", local_feature_dir="data/features")

# Backup ledger with auto-generated name
backup_ledger_to_s3("data/ledger.jsonl")

# Backup entire directory
backup_directory_to_s3("runs/2024-01-15", "s3_prefix/runs/2024-01-15")

# List all backups
ledger_backups = list_s3_backups(prefix="ledger/archive/")
for backup in ledger_backups:
    print(f"Backup: {backup['key']} ({backup['size']} bytes)")
```

### Integration with Feature Store

```python
from models.features.feature_store import FeatureStore
from storage import backup_features_to_s3, restore_features_from_s3
import pandas as pd

# Create and save features locally
store = FeatureStore("./data/features")
features_df = pd.DataFrame({...})  # Your features
store.save_features(features_df, version="v1.2.0", metadata={"source": "production"})

# Backup to S3
backup_features_to_s3("./data/features", version="v1.2.0")

# Later, restore from S3 on another machine
restore_features_from_s3(version="v1.2.0", local_feature_dir="./data/features")
restored_features = store.load_features("v1.2.0")
```

### Automatic Ledger Archival

```python
from pathlib import Path
from datetime import datetime
from storage import backup_ledger_to_s3

def archive_old_ledger():
    """Archive ledger and start fresh."""
    ledger_path = Path("data/ledger.jsonl")

    if ledger_path.exists():
        # Backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_ledger_to_s3(
            ledger_path,
            archive_name=f"ledger_{timestamp}.jsonl"
        )

        # Clear local ledger or create new one
        ledger_path.unlink()
        print(f"Ledger archived to S3 and cleared locally")

# Run monthly
archive_old_ledger()
```

### Scheduled Backups

```python
from storage import backup_directory_to_s3
from datetime import datetime

def scheduled_backup():
    """Backup important directories to S3."""
    timestamp = datetime.now().strftime("%Y%m%d")

    # Backup features
    backup_directory_to_s3(
        "data/features",
        f"backups/{timestamp}/features"
    )

    # Backup agent memory
    backup_directory_to_s3(
        "data/agent_memory",
        f"backups/{timestamp}/agent_memory"
    )

    # Backup models
    backup_directory_to_s3(
        "models",
        f"backups/{timestamp}/models",
        pattern="*.pkl"  # Only backup pickle files
    )

# Run daily via cron or scheduler
scheduled_backup()
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MASSIVE_S3_ENABLED` | `false` | Enable/disable S3 storage |
| `MASSIVE_S3_ACCESS_KEY_ID` | - | S3 access key ID (required) |
| `MASSIVE_S3_SECRET_ACCESS_KEY` | - | S3 secret access key (required) |
| `MASSIVE_S3_ENDPOINT` | `https://files.massive.com` | S3 endpoint URL |
| `MASSIVE_S3_BUCKET` | `flatfiles` | S3 bucket name |
| `MASSIVE_S3_REGION` | `us-east-1` | S3 region |

### Config.yaml Options

| Option | Default | Description |
|--------|---------|-------------|
| `storage.enabled` | `false` | Enable S3 storage |
| `storage.provider` | `massive` | Provider name (massive, aws, minio) |
| `storage.endpoint` | `https://files.massive.com` | S3 endpoint |
| `storage.bucket` | `flatfiles` | Bucket name |
| `storage.region` | `us-east-1` | Region |
| `storage.auto_sync` | `false` | Auto-sync data to S3 |
| `storage.sync_features` | `true` | Sync feature store |
| `storage.sync_ledger` | `true` | Sync ledger |
| `storage.sync_logs` | `false` | Sync logs |
| `storage.sync_models` | `false` | Sync ML models |

## Best Practices

### 1. Version Your Features

Always use semantic versioning for features:

```python
# Good
backup_features_to_s3("data/features", version="v1.2.0")

# Also good
backup_features_to_s3("data/features", version="20240115_production")
```

### 2. Archive Old Ledgers Regularly

Don't let ledger files grow indefinitely:

```python
# Weekly archival
if datetime.now().weekday() == 0:  # Monday
    backup_ledger_to_s3("data/ledger.jsonl")
```

### 3. Use Presigned URLs for Sharing

Don't expose your S3 credentials. Generate temporary URLs instead:

```python
# Generate 1-hour access URL
url = s3.get_presigned_url("reports/analysis.pdf", expiration=3600)
# Share this URL securely
```

### 4. Organize Your S3 Structure

Use a logical folder structure:

```
/
├── features/
│   ├── v1.0.0/
│   ├── v1.1.0/
│   └── v1.2.0/
├── ledger/
│   └── archive/
│       ├── ledger_20240101.jsonl
│       └── ledger_20240115.jsonl
├── models/
│   ├── production/
│   └── experimental/
├── backups/
│   ├── 20240115/
│   └── 20240122/
└── manifests/
```

### 5. Monitor Storage Costs

List objects regularly to monitor storage usage:

```python
all_objects = s3.list_objects()
total_size = sum(obj['size'] for obj in all_objects)
print(f"Total storage: {total_size / (1024**3):.2f} GB")
```

## Error Handling

The adapter handles errors gracefully:

```python
from storage import MassiveS3StorageAdapter

s3 = MassiveS3StorageAdapter()

# All operations return False/None on failure
if not s3.upload_file("data.csv", "backup/data.csv"):
    print("Upload failed - check logs for details")

# Check if adapter is enabled before operations
if s3.enabled:
    s3.upload_file("important.json", "backups/important.json")
else:
    print("S3 storage is disabled")
```

## Testing

Run the test suite:

```bash
# Run all S3 tests
pytest tests/test_massive_s3_storage.py -v

# Run specific test
pytest tests/test_massive_s3_storage.py::test_upload_file_success -v
```

## Troubleshooting

### Adapter shows as disabled

**Cause**: Missing or invalid credentials

**Solution**:
1. Check that `MASSIVE_S3_ENABLED=true` in `.env`
2. Verify `MASSIVE_S3_ACCESS_KEY_ID` and `MASSIVE_S3_SECRET_ACCESS_KEY` are set
3. Check logs for detailed error messages

### Bucket not found error

**Cause**: Bucket doesn't exist or wrong bucket name

**Solution**:
1. Verify bucket name in configuration matches your MASSIVE.COM bucket
2. The adapter will attempt to create the bucket if it doesn't exist (requires permissions)

### Connection timeout

**Cause**: Network issues or incorrect endpoint

**Solution**:
1. Verify `MASSIVE_S3_ENDPOINT=https://files.massive.com`
2. Check network connectivity
3. Verify firewall settings allow HTTPS traffic

## API Reference

See the source code documentation:
- `storage/massive_s3_adapter.py` - Main adapter class
- `storage/utils.py` - Utility functions
- `config/config_models.py` - Configuration models

## Support

For issues or questions:
1. Check the logs (uses `loguru` logger)
2. Review error messages in the console
3. Consult MASSIVE.COM documentation: https://massive.com/docs

## License

Part of the Super Gnosis trading system.
