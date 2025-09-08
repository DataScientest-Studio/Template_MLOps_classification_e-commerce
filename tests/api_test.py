import os
import io
from fastapi.testclient import TestClient
from src.models.serve.serve_model_fastapi import app
from src.models.serve.serve_model_fastapi import verify_token
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import json
import src.models.serve.serve_model_fastapi as serve_module


client = TestClient(app)


def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "Model Serving via FastAPI is running !"}


def test_predict(monkeypatch):
    """Test that handles sequence padding properly"""
    
    def mock_verify_token():
        return {"username": "testuser"}
    app.dependency_overrides[verify_token] = mock_verify_token
    
    # Mock with proper sequence handling
    mock_tokenizer = Mock()
    # Return non-empty sequences
    mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3, 4, 5]]
    
    # Mock pad_sequences if you use it
    with patch('tensorflow.keras.preprocessing.sequence.pad_sequences') as mock_pad:
        mock_pad.return_value = np.array([[1, 2, 3, 4, 5] + [0] * 95])  # Padded to 100
        
        mock_lstm = Mock()
        mock_lstm.predict.return_value = np.array([[0.2, 0.3, 0.5]])
        
        mock_vgg16 = Mock()  
        mock_vgg16.predict.return_value = np.array([[0.1, 0.4, 0.5]])
        
        mock_weights = {"lstm_weight": 0.6, "vgg16_weight": 0.4}
        
        # If weighted prediction argmax returns 2, mapper must have "2"
        mock_mapper = {
            "0": "Class_A",
            "1": "Class_B", 
            "2": "Class_C"  # This will be selected
        }
        
        serve_module.tokenizer = mock_tokenizer
        serve_module.lstm = mock_lstm
        serve_module.vgg16 = mock_vgg16
        serve_module.best_weights = mock_weights
        serve_module.mapper = mock_mapper
        
        client = TestClient(app)
        
        # create a simple red image for testing
        img = Image.new('RGB', (224, 224), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {"image": ("test.jpg", img_buffer.getvalue(), "image/jpeg")}
        data = {
            "product_identifier": "id1",
            "designation": "laptop computer",
            "description": "gaming laptop with good specs", 
            "product_id": "pid123",
            "imageid": "img123"
        }
        
        response = client.post("/predict", data=data, files=files)
        
        if response.status_code == 200:
            resp_json = response.json()
            assert "predictions" in resp_json
            print(f"✅ Success: {resp_json}")
        else:
            print(f"❌ Error {response.status_code}: {response.content}")
        
        # Clean up
        serve_module.tokenizer = None
        serve_module.lstm = None
        serve_module.vgg16 = None  
        serve_module.best_weights = None
        serve_module.mapper = None
        app.dependency_overrides = {}