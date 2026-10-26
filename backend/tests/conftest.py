import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.main import app
from backend.graph import build_review_graph

legacy_app = types.ModuleType("app")
legacy_app.__path__ = [str(PROJECT_ROOT / "backend")]
legacy_app.__package__ = "app"
legacy_app.__spec__ = None
legacy_app.nodes = __import__("backend.nodes", fromlist=["*"])
legacy_app.graph = __import__("backend.graph", fromlist=["*"])

sys.modules["app"] = legacy_app
sys.modules["app.nodes"] = legacy_app.nodes
sys.modules["app.nodes.emotion"] = legacy_app.nodes.emotion
sys.modules["app.graph"] = legacy_app.graph


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def graph():
    return build_review_graph()