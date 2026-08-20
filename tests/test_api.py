from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_validation_error():
    # Missing required fields like 'credit_score'
    response = client.post("/predict", json={
        "customer_id": 1,
        "geography": "France"
    })
    assert response.status_code == 422 # Unprocessable Entity
