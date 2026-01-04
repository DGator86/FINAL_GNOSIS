import sys
import os

print("="*50)
print("DEBUG: PYTHON ENVIRONMENT STARTUP CHECK")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print("="*50)
print("PYTHONPATH:")
for p in sys.path:
    print(f"  - {p}")
print("="*50)

print("Attempting to import sqlalchemy...")
try:
    import sqlalchemy
    print(f"✅ SUCCESS: SQLAlchemy found at {sqlalchemy.__file__}")
    print(f"Version: {sqlalchemy.__version__}")
except ImportError as e:
    print(f"❌ FAILURE: {e}")
    # List site-packages to see what IS there
    import site
    print("Contents of site-packages:")
    for site_pkg in site.getsitepackages():
        if os.path.exists(site_pkg):
            print(f"Listing {site_pkg}:")
            try:
                print(os.listdir(site_pkg)[:20]) # Limit output
            except Exception as ls_err:
                print(f"Error listing {site_pkg}: {ls_err}")

print("="*50)
print("Starting Application...")
import uvicorn
uvicorn.run("web_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
