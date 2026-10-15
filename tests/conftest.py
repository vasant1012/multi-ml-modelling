import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")
from app.api.main import app


@pytest.fixture
def client():
    return TestClient(app)