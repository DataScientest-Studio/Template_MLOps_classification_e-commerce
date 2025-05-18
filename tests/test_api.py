from fastapi.testclient import TestClient
from Projet_Formation_MLOps_DataScientest_2025.api import app

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "L'API fonctionne correctement"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Bienvenue" in data["message"]

def test_predict():
    sample_text = {"predicted": "1", "label": "label_1"}
    response = client.post("/predict", json=sample_text)
    assert response.status_code == 200
    data = response.json()
    assert "predicted" in data
    assert "label" in data
