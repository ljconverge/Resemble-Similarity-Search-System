import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "Algorithm", "services", "tools"):
    sub_path = str(ROOT / sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)
