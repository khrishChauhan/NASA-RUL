import io
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Any
from backend.app.core.config import settings
from backend.app.services.dataset_service import dataset_service
from backend.app.services.model_service import model_service
from backend.app.schemas.rul import (
    HealthResponse,
    OverviewResponse,
    EngineListItem,
    EngineDetailResponse,
    PredictEngineRequest,
    PredictCustomSequenceRequest,
    PredictResponse,
    ModelsResponse,
    ModelMetricDetail,
    SensorHistoryResponse,
    SensorStat,
    EngineBenchmarkItem
)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health():
    models_status = model_service.get_loaded_models_status()
    scaler_loaded = model_service.scaler is not None
    dataset_loaded = dataset_service.is_loaded()
    
    overall_status = "healthy" if (models_status.get("ImprovedRULLSTM") and scaler_loaded and dataset_loaded) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        models_loaded=models_status,
        scaler_loaded=scaler_loaded,
        dataset_loaded=dataset_loaded,
        device=str(model_service.device),
        version="1.0.0"
    )

@router.get("/overview", response_model=OverviewResponse)
def get_overview():
    test_engines = dataset_service.get_test_engine_ids()
    total_test = len(test_engines)
    total_train = 100
    total_train_cycles = 20631
    
    # Generate benchmark sample for first 10 test engines
    sample_items = []
    for eng_id in test_engines[:10]:
        try:
            seq = dataset_service.get_engine_last_sequence(eng_id, 30)
            pred = model_service.predict_sequence(seq, "ImprovedRULLSTM")
            actual = dataset_service.get_engine_actual_rul(eng_id) or 0.0
            cycles = dataset_service.get_engine_cycle_count(eng_id)
            err = round(abs(pred - actual), 2)
            sample_items.append(EngineBenchmarkItem(
                engine_id=eng_id,
                observed_cycles=cycles,
                predicted_rul=pred,
                actual_rul=actual,
                absolute_error=err,
                health_status=model_service.categorize_health(pred)
            ))
        except Exception:
            continue

    models_summary = [
        {"name": "LSTM", "mae": 17.6053, "rmse": 24.0977, "r2": 0.6637, "status": "Production Active"},
        {"name": "Transformer", "mae": 18.8980, "rmse": 25.2003, "r2": 0.6322, "status": "Available"},
        {"name": "XGBoost", "mae": 24.3677, "rmse": 34.1241, "r2": 0.3257, "status": "Benchmark Baseline"}
    ]

    return OverviewResponse(
        project_title="NASA-RUL — Predictive Maintenance Intelligence",
        description="Deep sequence models estimating Remaining Useful Life on NASA C-MAPSS FD001 turbofan degradation telemetry.",
        total_test_engines=total_test,
        total_train_engines=total_train,
        total_train_cycles=total_train_cycles,
        selected_model="ImprovedRULLSTM",
        best_test_mae=17.6053,
        best_test_rmse=24.0977,
        best_test_r2=0.6637,
        benchmark_sample=sample_items,
        models_summary=models_summary
    )

@router.get("/engines", response_model=list[EngineListItem])
def list_engines():
    test_engines = dataset_service.get_test_engine_ids()
    if not test_engines:
        raise HTTPException(status_code=404, detail="No test engines found in dataset.")
    
    items = []
    for eng_id in test_engines:
        try:
            cycles = dataset_service.get_engine_cycle_count(eng_id)
            seq = dataset_service.get_engine_last_sequence(eng_id, 30)
            pred = model_service.predict_sequence(seq, "ImprovedRULLSTM")
            actual = dataset_service.get_engine_actual_rul(eng_id) or 0.0
            err = round(abs(pred - actual), 2)
            items.append(EngineListItem(
                engine_id=eng_id,
                observed_cycles=cycles,
                predicted_rul=pred,
                actual_rul=actual,
                absolute_error=err,
                health_status=model_service.categorize_health(pred)
            ))
        except Exception as e:
            continue
            
    return items

@router.get("/engines/{engine_id}", response_model=EngineDetailResponse)
def get_engine_detail(engine_id: int):
    if engine_id not in dataset_service.get_test_engine_ids():
        raise HTTPException(status_code=404, detail=f"Engine ID {engine_id} not found in FD001 test dataset.")
    
    cycles = dataset_service.get_engine_cycle_count(engine_id)
    seq = dataset_service.get_engine_last_sequence(engine_id, 30)
    pred = model_service.predict_sequence(seq, "ImprovedRULLSTM")
    actual = dataset_service.get_engine_actual_rul(engine_id) or 0.0
    history = dataset_service.get_engine_full_history(engine_id)
    
    return EngineDetailResponse(
        engine_id=engine_id,
        observed_cycles=cycles,
        predicted_rul=pred,
        actual_rul=actual,
        history=history,
        final_sequence_features=dataset_service.selected_features
    )

@router.post("/predict", response_model=PredictResponse)
def predict_rul(payload: PredictEngineRequest):
    eng_id = payload.engine_id
    if eng_id not in dataset_service.get_test_engine_ids():
        raise HTTPException(status_code=404, detail=f"Engine ID {eng_id} not found in FD001 dataset.")
    
    chosen_model = payload.model_name or "ImprovedRULLSTM"
    try:
        seq = dataset_service.get_engine_last_sequence(eng_id, 30)
        pred = model_service.predict_sequence(seq, chosen_model)
        actual = dataset_service.get_engine_actual_rul(eng_id)
        recent_telemetry = dataset_service.get_engine_recent_telemetry_dicts(eng_id, 30)
        
        health = model_service.categorize_health(pred)
        return PredictResponse(
            model_name=chosen_model,
            predicted_rul=pred,
            sequence_length_used=30,
            engine_identifier=f"Test Engine #{eng_id}",
            health_status=health,
            status="success",
            note="Prediction calculated from the final 30 observed cycles preceding maintenance truncation. Actual label is historical benchmark ground truth.",
            actual_rul=actual,
            recent_telemetry=recent_telemetry
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/custom", response_model=PredictResponse)
def predict_custom_sequence(payload: PredictCustomSequenceRequest):
    rows = payload.rows
    if len(rows) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Sequence length validation failed: received {len(rows)} cycles, but model requires at least 30 consecutive cycles."
        )
    
    # Convert list of rows to DataFrame
    df = pd.DataFrame([r.model_dump() for r in rows])
    if "cycle" in df.columns and df["cycle"].notnull().all():
        df = df.sort_values("cycle")
        
    missing_cols = [c for c in dataset_service.selected_features if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required sensor features: {missing_cols}. Exactly 19 features required."
        )
        
    seq = df[dataset_service.selected_features].tail(30).to_numpy(dtype=np.float32)
    chosen_model = payload.model_name or "ImprovedRULLSTM"
    
    try:
        pred = model_service.predict_sequence(seq, chosen_model)
        health = model_service.categorize_health(pred)
        recent_telemetry = df[["cycle"] + dataset_service.selected_features].tail(30).to_dict(orient="records") if "cycle" in df.columns else df[dataset_service.selected_features].tail(30).to_dict(orient="records")
        return PredictResponse(
            model_name=chosen_model,
            predicted_rul=pred,
            sequence_length_used=30,
            engine_identifier=payload.engine_identifier or "Custom Input",
            health_status=health,
            status="success",
            note="Prediction estimated from the final 30 cycles of uploaded sensor sequence. This is a statistical estimation, not a guarantee of safe operation.",
            actual_rul=None,
            recent_telemetry=recent_telemetry
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/upload", response_model=PredictResponse)
async def predict_csv_upload(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("ImprovedRULLSTM")
):
    if not file.filename.endswith(".csv") and not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or space/tab-delimited text file.")
    
    contents = await file.read()
    try:
        # Try comma separated first
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), sep=r"\s+")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded CSV: {e}")

    # Check required columns
    missing_cols = [c for c in dataset_service.selected_features if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded CSV is missing {len(missing_cols)} required features: {missing_cols}. The model requires exactly 19 sensor channels."
        )

    # Check non-numeric / NaN values
    feature_df = df[dataset_service.selected_features]
    if feature_df.isnull().any().any():
        raise HTTPException(status_code=400, detail="Uploaded file contains missing (NaN) or null sensor readings.")

    # Sort by cycle if present
    if "cycle" in df.columns:
        df = df.sort_values("cycle")

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient cycles: uploaded file contains {len(df)} rows, but model requires at least 30 consecutive operational cycles."
        )

    tail_30 = df[dataset_service.selected_features].tail(30).to_numpy(dtype=np.float32)
    chosen_model = model_name or "ImprovedRULLSTM"

    try:
        pred = model_service.predict_sequence(tail_30, chosen_model)
        health = model_service.categorize_health(pred)
        cols_for_snapshot = (["cycle"] if "cycle" in df.columns else []) + dataset_service.selected_features
        recent_telemetry = df[cols_for_snapshot].tail(30).to_dict(orient="records")
        return PredictResponse(
            model_name=chosen_model,
            predicted_rul=pred,
            sequence_length_used=30,
            engine_identifier=f"Upload: {file.filename}",
            health_status=health,
            status="success",
            note="Prediction computed from the final 30 consecutive rows of uploaded telemetry.",
            actual_rul=None,
            recent_telemetry=recent_telemetry
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/sample", response_model=PredictResponse)
def predict_sample():
    """
    Generates a realistic 30-cycle operational sample from the FD001 dataset
    using only the 19 sensor channels, and runs real LSTM inference.
    """
    try:
        seq, telemetry, sample_label = dataset_service.get_random_contiguous_sample(30)
        pred = model_service.predict_sequence(seq, "ImprovedRULLSTM")
        health = model_service.categorize_health(pred)
        return PredictResponse(
            model_name="ImprovedRULLSTM",
            predicted_rul=pred,
            sequence_length_used=30,
            engine_identifier=sample_label,
            health_status=health,
            status="success",
            note="Prediction calculated from a real contiguous 30-cycle operational window sampled from the FD001 benchmark. This is a statistical ML estimate based on sensor degradation patterns, not an operational airworthiness guarantee.",
            actual_rul=None,
            recent_telemetry=telemetry
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sample")
def get_sample_telemetry():
    """
    Returns a sample contiguous 30-cycle telemetry block with the 19 features.
    """
    try:
        _, telemetry, sample_label = dataset_service.get_random_contiguous_sample(30)
        return {
            "label": sample_label,
            "rows": telemetry,
            "features": dataset_service.selected_features,
            "total_cycles": len(telemetry)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models", response_model=ModelsResponse)
def get_models_info():
    models_list = [
        ModelMetricDetail(
            model_name="ImprovedRULLSTM",
            mae=17.6053,
            rmse=24.0977,
            r2=0.6637,
            status="Production Active",
            description="Two-layer stacked LSTM network with 128 hidden units and dropout. Evaluates temporal wear patterns over 30 sliding cycles.",
            is_active=True,
            weights_available=True,
            parameters=200000
        ),
        ModelMetricDetail(
            model_name="RULTransformer",
            mae=18.8980,
            rmse=25.2003,
            r2=0.6322,
            status="Alternative Architecture",
            description="Self-attention Transformer Encoder with 2 layers, 4 heads, and learnable positional encodings over 30 cycles.",
            is_active=False,
            weights_available=True,
            parameters=72385
        ),
        ModelMetricDetail(
            model_name="XGBoost",
            mae=24.3677,
            rmse=34.1241,
            r2=0.3257,
            status="Tabular Baseline",
            description="Gradient-boosted decision trees (500 estimators) trained on isolated cycle snapshots with Tree SHAP explainability.",
            is_active=False,
            weights_available=False,
            parameters=None
        )
    ]

    return ModelsResponse(
        models=models_list,
        historical_benchmark_note="Verified official test results from the current experiment on the 100 test engines of NASA C-MAPSS FD001. Lower MAE and RMSE indicate higher accuracy; higher R² indicates superior variance explanation. Results reflect historical benchmark evaluations and are not claimed as state of the art.",
        active_production_model="ImprovedRULLSTM"
    )

@router.get("/sensors/{engine_id}", response_model=SensorHistoryResponse)
def get_sensor_history(engine_id: int):
    if engine_id not in dataset_service.get_test_engine_ids():
        raise HTTPException(status_code=404, detail=f"Engine ID {engine_id} not found in FD001 test dataset.")
    
    cycles = dataset_service.get_engine_cycle_count(engine_id)
    history = dataset_service.get_engine_full_history(engine_id)
    metadata = dataset_service.get_sensor_metadata_list()
    sensor_stats_models = [SensorStat(**m) for m in metadata]
    
    return SensorHistoryResponse(
        engine_id=engine_id,
        total_cycles=cycles,
        available_features=sensor_stats_models,
        readings=history
    )
