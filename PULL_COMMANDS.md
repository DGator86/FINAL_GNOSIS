# 🔄 Git Pull Commands for Termius

## Quick Pull (if you're already on the branch)

```bash
cd /path/to/FINAL_GNOSIS
git pull origin claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC
```

---

## Full Setup (if starting fresh)

### 1. Navigate to repository
```bash
cd /path/to/FINAL_GNOSIS
```

### 2. Check current branch
```bash
git branch
```

### 3. Fetch all remote changes
```bash
git fetch origin
```

### 4. Switch to the feature branch
```bash
git checkout claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC
```

### 5. Pull latest changes
```bash
git pull origin claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC
```

---

## What You'll Get

After pulling, you'll have these new/updated files:

### **New Files:**
1. `UNUSUAL_WHALES_API_ENDPOINTS.md` - Complete API endpoint catalog
2. `test_new_uw_features.py` - Test script for new features

### **Updated Files:**
1. `.env.example` - Fixed to use UNUSUAL_WHALES_API_TOKEN
2. `engines/inputs/unusual_whales_adapter.py` - Added:
   - `get_greek_exposure()` - Pulls GEX, VEX, Charm
   - `get_dark_pool()` - Pulls dark pool trades
3. `UNUSUAL_WHALES_FIX_SUMMARY.md` - Corrected JWT → UUID documentation
4. `test_uw_comprehensive.py` - Fixed token format detection

---

## Verify Pull Success

```bash
# Check that new files exist
ls -la UNUSUAL_WHALES_API_ENDPOINTS.md
ls -la test_new_uw_features.py

# Check latest commit
git log -1 --oneline
# Should show: 332dc2a feat: add Greek Exposure (Vanna/Charm) and Dark Pool data methods
```

---

## Test New Features

```bash
# Make test script executable
chmod +x test_new_uw_features.py

# Run test (requires UNUSUAL_WHALES_API_TOKEN in .env)
python test_new_uw_features.py
```

---

## Alternative: Clone Fresh

If you want a clean copy:

```bash
# Clone the repository
git clone https://github.com/DGator86/FINAL_GNOSIS.git
cd FINAL_GNOSIS

# Checkout the feature branch
git checkout claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC

# Pull latest
git pull
```

---

## Branch Info

- **Branch Name**: `claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC`
- **Latest Commit**: `332dc2a`
- **Commits on this branch**:
  1. `6c35803` - fix: correct Unusual Whales authentication (UUID not JWT)
  2. `332dc2a` - feat: add Greek Exposure and Dark Pool methods

---

## If You Get Conflicts

If you have local changes that conflict:

```bash
# Stash your changes
git stash

# Pull the latest
git pull origin claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC

# Re-apply your changes
git stash pop
```

---

## Set Up Environment

After pulling, update your `.env` file:

```bash
# Copy example if you don't have .env
cp .env.example .env

# Edit .env and add your token
nano .env
# or
vim .env

# Add this line:
# UNUSUAL_WHALES_API_TOKEN=8932cd23-72b3-4f74-9848-13f9103b9df5
```

---

## Quick Reference

```bash
# One-liner to pull everything
cd /path/to/FINAL_GNOSIS && git fetch origin && git checkout claude/fix-api-auth-token-012Juo1DixdQh59FRzJrDeLC && git pull

# Verify new methods are available
python -c "from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter; a = UnusualWhalesOptionsAdapter(); print('✅ Methods available:', [m for m in dir(a) if not m.startswith('_')])"
```
