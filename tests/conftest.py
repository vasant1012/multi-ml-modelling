"""Ensure the project root is importable when pytest is launched from any folder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
