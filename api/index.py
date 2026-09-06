import sys
from pathlib import Path
import traceback

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

try:
    from main import app
except Exception:
    print("=== VERCEL STARTUP ERROR ===")
    traceback.print_exc()
    raise