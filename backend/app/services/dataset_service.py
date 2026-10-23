import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Any
from backend.app.core.config import settings

SENSOR_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "setting_1": {"desc": "Altitude & Operating Setting 1", "unit": "Norm"},
    "setting_2": {"desc": "Mach Number & Setting 2", "unit": "Norm"},
    "sensor_2": {"desc": "Total temperature at LPC outlet", "unit": "°R"},
    "sensor_3": {"desc": "Total temperature at HPC outlet", "unit": "°R"},
    "sensor_4": {"desc": "Total temperature at LPT outlet", "unit": "°R"},
    "sensor_5": {"desc": "Pressure at fan inlet", "unit": "psia"},
    "sensor_6": {"desc": "Total pressure in bypass-duct", "unit": "psia"},
    "sensor_7": {"desc": "Total pressure at HPC outlet", "unit": "psia"},
    "sensor_8": {"desc": "Physical fan speed", "unit": "rpm"},
    "sensor_9": {"desc": "Physical core speed", "unit": "rpm"},
    "sensor_11": {"desc": "Static pressure at HPC outlet", "unit": "psia"},
    "sensor_12": {"desc": "Ratio of fuel flow to Ps30", "unit": "pps/psi"},
    "sensor_13": {"desc": "Corrected fan speed", "unit": "rpm"},
    "sensor_14": {"desc": "Corrected core speed", "unit": "rpm"},
    "sensor_15": {"desc": "Bypass Ratio", "unit": "ratio"},
    "sensor_16": {"desc": "Burner fuel-air ratio", "unit": "ratio"},
    "sensor_17": {"desc": "Bleed Enthalpy", "unit": "BTU/lbm"},
    "sensor_20": {"desc": "HPT coolant bleed", "unit": "lbm/s"},
    "sensor_21": {"desc": "LPT coolant bleed", "unit": "lbm/s"},
}

class DatasetService:
    def __init__(self):
        self.columns = (
            ["engine_id", "cycle"]
            + [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )
        self.selected_features = [
            "setting_1", "setting_2",
            "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7",
            "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
            "sensor_15", "sensor_16", "sensor_17", "sensor_20", "sensor_21"
        ]
        
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.rul_df: Optional[pd.DataFrame] = None
        self.sensor_correlations: dict[str, float] = {}
        self.sensor_stats: dict[str, dict[str, float]] = {}
        
        self._load_datasets()

    def _load_datasets(self):
        if settings.TEST_DATA_PATH.exists():
            self.test_df = pd.read_csv(
                settings.TEST_DATA_PATH,
                sep=r"\s+",
                header=None,
                names=self.columns
            )
        
        if settings.RUL_DATA_PATH.exists():
            self.rul_df = pd.read_csv(
                settings.RUL_DATA_PATH,
                sep=r"\s+",
                header=None,
                names=["RUL"]
            )
            
        if settings.TRAIN_DATA_PATH.exists():
            self.train_df = pd.read_csv(
                settings.TRAIN_DATA_PATH,
                sep=r"\s+",
                header=None,
                names=self.columns
            )
            # Compute RUL for training engines to get true correlation
            max_cycles = self.train_df.groupby("engine_id")["cycle"].max()
            train_rul = self.train_df["engine_id"].map(max_cycles) - self.train_df["cycle"]
            
            # Correlation with RUL
            for feat in self.selected_features:
                if feat in self.train_df.columns:
                    corr = float(self.train_df[feat].corr(train_rul))
                    self.sensor_correlations[feat] = round(corr, 4) if not np.isnan(corr) else 0.0
                    self.sensor_stats[feat] = {
                        "mean": float(round(self.train_df[feat].mean(), 3)),
                        "std": float(round(self.train_df[feat].std(), 3)),
                        "min": float(round(self.train_df[feat].min(), 3)),
                        "max": float(round(self.train_df[feat].max(), 3)),
                    }

    def is_loaded(self) -> bool:
        return self.test_df is not None and self.rul_df is not None

    def get_test_engine_ids(self) -> list[int]:
        if self.test_df is None:
            return []
        return sorted([int(x) for x in self.test_df["engine_id"].unique()])

    def get_engine_cycle_count(self, engine_id: int) -> int:
        if self.test_df is None:
            return 0
        engine_rows = self.test_df[self.test_df["engine_id"] == engine_id]
        return len(engine_rows)

    def get_engine_actual_rul(self, engine_id: int) -> Optional[float]:
        if self.rul_df is None or engine_id < 1 or engine_id > len(self.rul_df):
            return None
        # RUL file is 0-indexed corresponding to engine_id = index + 1
        return float(self.rul_df.iloc[engine_id - 1]["RUL"])

    def get_engine_last_sequence(self, engine_id: int, sequence_length: int = 30) -> np.ndarray:
        if self.test_df is None:
            raise ValueError("Test dataset not available.")
        engine_data = self.test_df[self.test_df["engine_id"] == engine_id].sort_values("cycle")
        if len(engine_data) < sequence_length:
            raise ValueError(f"Engine {engine_id} has {len(engine_data)} cycles, requires {sequence_length}.")
        features = engine_data[self.selected_features].iloc[-sequence_length:].to_numpy(dtype=np.float32)
        return features

    def get_engine_recent_telemetry_dicts(self, engine_id: int, count: int = 30) -> list[dict[str, Any]]:
        if self.test_df is None:
            return []
        engine_data = self.test_df[self.test_df["engine_id"] == engine_id].sort_values("cycle").tail(count)
        cols_to_keep = ["cycle"] + self.selected_features
        return engine_data[cols_to_keep].to_dict(orient="records")

    def get_engine_full_history(self, engine_id: int) -> list[dict[str, Any]]:
        if self.test_df is None:
            return []
        engine_data = self.test_df[self.test_df["engine_id"] == engine_id].sort_values("cycle")
        cols_to_keep = ["cycle"] + self.selected_features
        return engine_data[cols_to_keep].to_dict(orient="records")

    def get_sensor_metadata_list(self) -> list[dict[str, Any]]:
        result = []
        for feat in self.selected_features:
            meta = SENSOR_DESCRIPTIONS.get(feat, {"desc": feat, "unit": ""})
            stats = self.sensor_stats.get(feat, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})
            corr = self.sensor_correlations.get(feat, 0.0)
            result.append({
                "name": feat,
                "description": meta["desc"],
                "unit": meta["unit"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "correlation_with_rul": corr,
            })
        return result

    def get_random_contiguous_sample(self, sequence_length: int = 30) -> tuple[np.ndarray, list[dict[str, Any]], str]:
        """
        Samples a real contiguous 30-cycle operational window from the available FD001 dataset
        using the exact 19 sensor features only.
        Returns: (features_array (30, 19), list_of_dicts_with_relative_cycles, sample_label)
        """
        source_df = self.test_df if self.test_df is not None else self.train_df
        if source_df is None:
            raise ValueError("No dataset available to generate sample.")

        engine_ids = sorted(source_df["engine_id"].unique())
        chosen_id = int(np.random.choice(engine_ids))
        engine_data = source_df[source_df["engine_id"] == chosen_id].sort_values("cycle")

        max_start = len(engine_data) - sequence_length
        if max_start < 0:
            raise ValueError(f"Engine {chosen_id} has fewer than {sequence_length} cycles.")

        start_idx = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        window = engine_data.iloc[start_idx : start_idx + sequence_length]

        # Extract features only (never include engine_id or cycle as model features)
        features_array = window[self.selected_features].to_numpy(dtype=np.float32)

        # Prepare telemetry dicts with relative cycles 1..30 for inspection
        telemetry = []
        for i, (_, row) in enumerate(window.iterrows()):
            item = {"cycle": i + 1}
            for col in self.selected_features:
                item[col] = round(float(row[col]), 4)
            telemetry.append(item)

        start_c = int(window["cycle"].iloc[0])
        end_c = int(window["cycle"].iloc[-1])
        sample_label = f"Sample FD001 (Ref #{chosen_id}, Cycles {start_c}–{end_c})"
        return features_array, telemetry, sample_label

dataset_service = DatasetService()
