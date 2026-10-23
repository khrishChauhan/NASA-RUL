from typing import Optional, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    scaler_loaded: bool
    dataset_loaded: bool
    device: str
    version: str = "1.0.0"


class EngineBenchmarkItem(BaseModel):
    engine_id: int
    observed_cycles: int
    predicted_rul: float
    actual_rul: float
    absolute_error: float
    health_status: str  # "Critical", "Warning", "Healthy"


class OverviewResponse(BaseModel):
    project_title: str
    description: str
    total_test_engines: int
    total_train_engines: int
    total_train_cycles: int
    selected_model: str
    best_test_mae: float
    best_test_rmse: float
    best_test_r2: float
    benchmark_sample: list[EngineBenchmarkItem]
    models_summary: list[dict[str, Any]]


class EngineListItem(BaseModel):
    engine_id: int
    observed_cycles: int
    predicted_rul: float
    actual_rul: float
    absolute_error: float
    health_status: str


class EngineDetailResponse(BaseModel):
    engine_id: int
    observed_cycles: int
    predicted_rul: float
    actual_rul: float
    history: list[dict[str, Any]]
    final_sequence_features: list[str]


class PredictEngineRequest(BaseModel):
    engine_id: int = Field(..., ge=1, le=100, description="Test engine identifier from FD001 (1-100)")
    model_name: Optional[str] = Field("ImprovedRULLSTM", description="Model architecture to use (ImprovedRULLSTM or RULTransformer)")


class CustomSequenceRow(BaseModel):
    cycle: Optional[int] = None
    setting_1: float
    setting_2: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictCustomSequenceRequest(BaseModel):
    rows: list[CustomSequenceRow]
    model_name: Optional[str] = "ImprovedRULLSTM"
    engine_identifier: Optional[str] = "Custom Upload"


class PredictResponse(BaseModel):
    model_name: str
    predicted_rul: float
    sequence_length_used: int
    engine_identifier: str
    health_status: str
    status: str
    note: str
    actual_rul: Optional[float] = None
    recent_telemetry: Optional[list[dict[str, Any]]] = None


class ModelMetricDetail(BaseModel):
    model_name: str
    mae: float
    rmse: float
    r2: float
    status: str
    description: str
    is_active: bool
    weights_available: bool
    parameters: Optional[int] = None


class ModelsResponse(BaseModel):
    models: list[ModelMetricDetail]
    historical_benchmark_note: str
    active_production_model: str


class SensorStat(BaseModel):
    name: str
    description: str
    unit: str
    mean: float
    std: float
    min: float
    max: float
    correlation_with_rul: float


class SensorHistoryResponse(BaseModel):
    engine_id: int
    total_cycles: int
    available_features: list[SensorStat]
    readings: list[dict[str, Any]]
