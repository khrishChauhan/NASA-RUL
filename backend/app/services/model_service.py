import torch
import torch.nn as nn
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from backend.app.core.config import settings
from backend.app.models.architectures import ImprovedRULLSTM, RULTransformer

class ModelService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.scaler = None
        self.config: dict[str, Any] = {}
        self.models: dict[str, nn.Module] = {}
        self.feature_columns: list[str] = []
        self.sequence_length: int = settings.SEQUENCE_LENGTH
        
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load config
        if settings.LSTM_CONFIG_PATH.exists():
            with open(settings.LSTM_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.feature_columns = self.config.get("input_features", [])
                self.sequence_length = self.config.get("sequence_length", 30)

        # 2. Load scaler
        if settings.SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(settings.SCALER_PATH)
            except Exception as e:
                print(f"[ModelService] Error loading scaler: {e}")
                self.scaler = None

        # 3. Load ImprovedRULLSTM
        if settings.LSTM_WEIGHTS_PATH.exists():
            try:
                input_size = len(self.feature_columns) if self.feature_columns else 19
                lstm = ImprovedRULLSTM(input_size=input_size, hidden_size=128)
                weights = torch.load(
                    settings.LSTM_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=True
                )
                lstm.load_state_dict(weights)
                lstm.eval()
                self.models["ImprovedRULLSTM"] = lstm
                print("[ModelService] Successfully loaded ImprovedRULLSTM checkpoint.")
            except Exception as e:
                print(f"[ModelService] Error loading ImprovedRULLSTM: {e}")

        # 4. Load RULTransformer
        if settings.TRANSFORMER_WEIGHTS_PATH.exists():
            try:
                input_size = len(self.feature_columns) if self.feature_columns else 19
                transformer = RULTransformer(
                    input_size=input_size,
                    sequence_length=self.sequence_length
                )
                weights = torch.load(
                    settings.TRANSFORMER_WEIGHTS_PATH,
                    map_location=self.device,
                    weights_only=True
                )
                transformer.load_state_dict(weights)
                transformer.eval()
                self.models["RULTransformer"] = transformer
                print("[ModelService] Successfully loaded RULTransformer checkpoint.")
            except Exception as e:
                print(f"[ModelService] Error loading RULTransformer: {e}")

    def is_model_loaded(self, model_name: str = "ImprovedRULLSTM") -> bool:
        return model_name in self.models and self.scaler is not None

    def get_loaded_models_status(self) -> dict[str, bool]:
        return {
            "ImprovedRULLSTM": "ImprovedRULLSTM" in self.models,
            "RULTransformer": "RULTransformer" in self.models,
            "XGBoost": False,  # Weights were not serialized in repo; documented as historical benchmark
        }

    def predict_sequence(
        self,
        raw_sequence: np.ndarray,
        model_name: Optional[str] = "ImprovedRULLSTM"
    ) -> float:
        """
        raw_sequence must be shape (sequence_length, 19) containing unscaled raw features in feature order.
        """
        chosen_model = model_name or "ImprovedRULLSTM"
        if chosen_model not in self.models:
            # Fallback to ImprovedRULLSTM if requested model not available
            if "ImprovedRULLSTM" in self.models:
                chosen_model = "ImprovedRULLSTM"
            else:
                raise RuntimeError(f"No trained model artifact is loaded for {chosen_model}.")

        if self.scaler is None:
            raise RuntimeError("Preprocessing StandardScaler artifact is not loaded.")

        if raw_sequence.ndim != 2:
            raise ValueError(f"Expected 2D array of shape ({self.sequence_length}, {len(self.feature_columns)}), got ndim={raw_sequence.ndim}.")

        if raw_sequence.shape[0] < self.sequence_length:
            raise ValueError(
                f"Insufficient cycles: provided sequence has {raw_sequence.shape[0]} cycles, but model requires at least {self.sequence_length} cycles."
            )

        if raw_sequence.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.feature_columns)} features, got {raw_sequence.shape[1]}."
            )

        # Slice the final sequence_length rows
        tail_sequence = raw_sequence[-self.sequence_length:]

        # Validate non-finite numbers
        if not np.all(np.isfinite(tail_sequence)):
            raise ValueError("Input sequence contains NaN or Infinite values.")

        # Apply training-fitted scaler with feature names preserved
        df_tail = pd.DataFrame(tail_sequence, columns=self.feature_columns)
        scaled_sequence = self.scaler.transform(df_tail)

        # Convert to tensor: shape (1, sequence_length, input_size)
        tensor_input = torch.tensor(
            np.expand_dims(scaled_sequence, axis=0),
            dtype=torch.float32
        ).to(self.device)

        model = self.models[chosen_model]
        model.eval()

        with torch.no_grad():
            raw_pred = model(tensor_input).cpu().numpy().flatten()[0]

        # Enforce physical reality: RUL cannot be negative or non-finite
        if not np.isfinite(raw_pred):
            raise ValueError("Model output returned non-finite value.")

        final_rul = float(np.maximum(raw_pred, 0.0))
        return round(final_rul, 2)

    def categorize_health(self, rul_cycles: float) -> str:
        if rul_cycles < 30.0:
            return "Critical"
        elif rul_cycles <= 80.0:
            return "Warning"
        else:
            return "Healthy"

model_service = ModelService()
