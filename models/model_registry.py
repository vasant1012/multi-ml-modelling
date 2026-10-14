"""Small model registry with in-process caching."""
from functools import lru_cache
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=4)
def load_model(relative_path: str):
    """Load a joblib model once. Return None when it has not been trained yet."""
    path = ROOT / relative_path
    return joblib.load(path) if path.exists() else None
