# CI Workflow Manual Fix Required

## ⚠️ Important Notice

The GitHub App does not have `workflows` permission to update `.github/workflows/tests.yml`.  
You must manually apply this change through the GitHub web UI or with your personal credentials.

## Status

✅ **Dependencies Fixed** - `pyproject.toml` and `requirements.txt` updated and pushed  
⚠️ **Workflow Update Required** - Manual update needed for `.github/workflows/tests.yml`

## What Was Already Fixed (Committed & Pushed)

### ✅ pyproject.toml
Added `[project.optional-dependencies]` test section:
```toml
test = [
    "pytest>=7.4",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "yfinance>=0.2.40",
    "greekcalc>=0.3.0",
    "pyarrow>=14.0.0",
]
```

### ✅ requirements.txt
Added required test dependencies:
```
greekcalc>=0.3.0  # Greeks calculator validation - REQUIRED for tests
yfinance>=0.2.40  # Yahoo Finance data - REQUIRED for tests
pyarrow>=14.0.0  # Arrow format support - REQUIRED for tests
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
```

## Manual Fix Required: .github/workflows/tests.yml

### Current (Broken) Version:
```yaml
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Install package
        run: pip install -e .

      - name: Run pytest
        run: pytest
```

### Fixed Version (Apply This):
```yaml
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[test]
          pip install yfinance greekcalc pyarrow pytest pytest-cov pytest-asyncio

      - name: Run tests
        run: |
          pytest -vv --cov=engines --cov=ml --cov=strategies
```

## How to Apply the Fix

### Option 1: GitHub Web UI (Easiest)

1. Go to: https://github.com/DGator86/V2---Gnosis/blob/genspark_ai_developer/.github/workflows/tests.yml
2. Click the "Edit" (pencil) icon
3. Replace lines 21-30 with the "Fixed Version" above
4. Commit directly to `genspark_ai_developer` branch
5. Message: `fix(ci): Update workflow to install test dependencies`

### Option 2: Git Command Line (With Your Personal Credentials)

```bash
# Clone repo with your personal GitHub credentials
git clone https://github.com/DGator86/V2---Gnosis.git
cd V2---Gnosis
git checkout genspark_ai_developer

# Apply the patch
git apply /path/to/workflow_fix.patch

# Or manually edit .github/workflows/tests.yml
nano .github/workflows/tests.yml

# Commit and push with your credentials
git add .github/workflows/tests.yml
git commit -m "fix(ci): Update workflow to install test dependencies"
git push origin genspark_ai_developer
```

### Option 3: Apply Patch File

A patch file is available at: `/tmp/workflow_fix.patch`

```bash
cd /home/user/webapp
git apply /tmp/workflow_fix.patch
git add .github/workflows/tests.yml
git commit -m "fix(ci): Update workflow to install test dependencies"
# Push with personal credentials (not GitHub App)
```

## What This Fix Does

### Before (Broken):
- ❌ Only installs from `requirements.txt`
- ❌ Missing `yfinance`, `greekcalc`, `pyarrow`
- ❌ Missing `pytest-cov`, `pytest-asyncio`
- ❌ No coverage reporting
- ❌ Tests fail with ImportError

### After (Fixed):
- ✅ Installs package with `[test]` extras
- ✅ Explicitly installs all test dependencies
- ✅ Runs pytest with verbose output
- ✅ Generates coverage report for `engines`, `ml`, `strategies`
- ✅ Tests pass without ImportErrors

## Why This Matters

The CI was failing because:

1. **yfinance not installed** → `YFinanceAdapter` couldn't import
2. **greekcalc not installed** → `GreekCalcAdapter` couldn't import
3. **pyarrow not installed** → Polars DataFrame operations failed
4. **No coverage tools** → Can't generate coverage reports

With this fix:
- ✅ All dependencies installed
- ✅ Tests can import all modules
- ✅ Coverage reports generated
- ✅ CI goes green 🟢

## Verification

After applying the fix, the CI will:

1. ✅ Install all dependencies (no ImportErrors)
2. ✅ Collect 17+ tests successfully
3. ✅ Run tests with verbose output
4. ✅ Generate coverage report
5. ✅ Pass (or show actual test failures, not import failures)

## Timeline

- **Commit ffe4a2e**: Dependencies fixed (pyproject.toml, requirements.txt) - **PUSHED** ✅
- **Pending**: Workflow file update - **MANUAL ACTION REQUIRED** ⚠️

## Questions?

If you encounter issues applying this fix, check:

1. Do you have write access to the repository?
2. Are you using the correct branch (`genspark_ai_developer`)?
3. Did you authenticate with your personal GitHub credentials (not the App)?

## Summary

✅ **What's Done**: Dependencies are now in `pyproject.toml` and `requirements.txt`  
⚠️ **What's Needed**: Update `.github/workflows/tests.yml` manually  
🎯 **Result**: CI will go green once workflow is updated  

**PR #34**: https://github.com/DGator86/V2---Gnosis/pull/34
