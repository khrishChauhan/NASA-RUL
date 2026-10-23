import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.dataset_service import dataset_service

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["models_loaded"]["ImprovedRULLSTM"] is True
    assert data["scaler_loaded"] is True
    assert data["dataset_loaded"] is True

def test_overview():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert data["total_test_engines"] == 100
    assert data["total_train_engines"] == 100
    assert data["best_test_mae"] == 17.6053
    assert len(data["benchmark_sample"]) > 0

def test_list_engines():
    response = client.get("/api/engines")
    assert response.status_code == 200
    engines = response.json()
    assert len(engines) == 100
    eng1 = engines[0]
    assert eng1["engine_id"] == 1
    assert eng1["observed_cycles"] == 31
    assert eng1["predicted_rul"] > 0
    assert eng1["actual_rul"] == 112.0

def test_get_engine_detail():
    response = client.get("/api/engines/1")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 1
    assert data["observed_cycles"] == 31
    assert len(data["history"]) == 31
    assert len(data["final_sequence_features"]) == 19

def test_predict_test_engine():
    response = client.post("/api/predict", json={"engine_id": 1, "model_name": "ImprovedRULLSTM"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "ImprovedRULLSTM"
    assert data["sequence_length_used"] == 30
    assert data["predicted_rul"] > 0
    assert data["actual_rul"] == 112.0
    assert data["health_status"] in ["Healthy", "Warning", "Critical"]
    assert len(data["recent_telemetry"]) == 30

def test_predict_transformer():
    response = client.post("/api/predict", json={"engine_id": 1, "model_name": "RULTransformer"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "RULTransformer"
    assert data["predicted_rul"] > 0

def test_predict_invalid_engine():
    response = client.post("/api/predict", json={"engine_id": 999})
    assert response.status_code == 422  # Pydantic validation rejects le=100

def test_predict_custom_insufficient_cycles():
    payload = {
        "rows": [
            {
                "cycle": 1,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "sensor_2": 642.0,
                "sensor_3": 1580.0,
                "sensor_4": 1400.0,
                "sensor_5": 14.62,
                "sensor_6": 21.61,
                "sensor_7": 554.0,
                "sensor_8": 2388.0,
                "sensor_9": 9050.0,
                "sensor_11": 47.3,
                "sensor_12": 521.0,
                "sensor_13": 2388.0,
                "sensor_14": 8130.0,
                "sensor_15": 8.4,
                "sensor_16": 0.03,
                "sensor_17": 392.0,
                "sensor_20": 39.0,
                "sensor_21": 23.4
            }
        ],
        "model_name": "ImprovedRULLSTM"
    }
    response = client.post("/api/predict/custom", json=payload)
    assert response.status_code == 400
    assert "at least 30" in response.json()["detail"]

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 3
    lstm_info = next(m for m in data["models"] if m["model_name"] == "ImprovedRULLSTM")
    assert lstm_info["mae"] == 17.6053
    assert lstm_info["rmse"] == 24.0977
    assert lstm_info["r2"] == 0.6637
    assert lstm_info["is_active"] is True

def test_sensors_endpoint():
    response = client.get("/api/sensors/1")
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 1
    assert len(data["available_features"]) == 19
    assert len(data["readings"]) == 31

def test_predict_sample_endpoint():
    response = client.post("/api/predict/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "ImprovedRULLSTM"
    assert data["sequence_length_used"] == 30
    assert data["predicted_rul"] >= 0.0
    assert len(data["recent_telemetry"]) == 30

def test_get_sample_endpoint():
    response = client.get("/api/sample")
    assert response.status_code == 200
    data = response.json()
    assert len(data["rows"]) == 30
    assert len(data["features"]) == 19

def test_production_sample_fallback_without_raw_dataset():
    """
    Simulates production deployment where raw dataset files are excluded from Git.
    Verifies that the bundled sample asset allows real LSTM inference to succeed.
    """
    orig_test = dataset_service.test_df
    orig_train = dataset_service.train_df
    try:
        dataset_service.test_df = None
        dataset_service.train_df = None
        
        assert dataset_service.is_loaded() is True
        
        res = client.post("/api/predict/sample")
        assert res.status_code == 200
        data = res.json()
        assert data["model_name"] == "ImprovedRULLSTM"
        assert data["sequence_length_used"] == 30
        assert isinstance(data["predicted_rul"], (int, float))
        assert data["predicted_rul"] >= 0.0
        assert len(data["recent_telemetry"]) == 30
        
        sample_res = client.get("/api/sample")
        assert sample_res.status_code == 200
        assert len(sample_res.json()["rows"]) == 30
    finally:
        dataset_service.test_df = orig_test
        dataset_service.train_df = orig_train


