# Repository Issue Analysis

## Test suite blockers
- **`test_uw_raw.py` exits the test run when credentials are missing.** The module calls `exit(1)` during import if an Unusual Whales token is not set, which triggers a `SystemExit` before any tests run. This breaks `pytest` collection in environments without the secret and prevents other tests from executing.
- **Integration-style tests depend on live services and credentials.** Tests in `test_setup.py` attempt real Alpaca connections and live data fetches and return `False` instead of failing assertions, so they silently succeed or fail based on external network state. These are not marked as integration tests and have no skipping logic when credentials are absent, reducing reliability of automated runs.

## Dependency gaps
- **Runtime imports are missing from `pyproject.toml`.** Tests and adapters rely on packages such as `httpx`, `python-dotenv`, and Alpaca SDK modules, but these are not declared in the core dependencies or optional groups. Environments created from `pyproject.toml` alone will miss required libraries, leading to import errors or skipped functionality.

## Recommendations
- Wrap credentialed tests with `pytest.skip` guards (or mark them `integration`) when required environment variables are absent, and avoid calling `exit` during import. Replace `print`/`return False` patterns with assertions so failures are reported clearly.
- Add missing dependencies (e.g., `httpx`, `python-dotenv`, Alpaca SDK packages) to the appropriate dependency group in `pyproject.toml`, or gate the corresponding code paths behind optional extras.
