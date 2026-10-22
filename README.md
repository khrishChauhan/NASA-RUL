# NASA-RUL: Turbofan Engine Remaining Useful Life Prediction

Predictive maintenance machine learning system for estimating the Remaining Useful Life (RUL) of aircraft gas turbine engines using the NASA C-MAPSS dataset.

---

## 1. Project Overview and Motivation

In mission-critical industrial systems—such as commercial aviation engines—unplanned downtime and in-flight component failures incur catastrophic financial costs and compromise passenger safety. Traditional maintenance routines rely on fixed schedules or reactive repairs after an anomaly manifests.

**NASA-RUL** is an end-to-end predictive-maintenance machine learning workflow built on NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark dataset. The project demonstrates how degradation signals hidden within high-dimensional time-series sensor telemetry can be extracted to estimate the Remaining Useful Life (RUL) of turbofan engines, enabling proactive maintenance scheduling and extending equipment lifespans.

---

## 2. Problem Statement

Given multidimensional operational settings and sensor telemetry collected across flight cycles:
- **Input**: Sensor readings and operational settings across operating cycles for each engine unit.
- **Output**: Remaining Useful Life ($RUL \in \mathbb{R}_{\ge 0}$), defined as the number of operational flight cycles remaining before engine failure.

In the test phase, engine telemetry is monitored up to an arbitrary operational point prior to system failure. The model must ingest historical telemetry cycles and accurately predict how many operational cycles remain before reaching the failure threshold.

---

## 3. Dataset Description and FD001 Data Structure

This project utilizes the **FD001** subset of the NASA C-MAPSS dataset:
- **Operating Conditions**: Sea level (1 condition)
- **Fault Modes**: High-Pressure Compressor (HPC) degradation (1 fault mode)
- **Engines in Training Set**: 100 units (run continuously to failure, totaling 20,631 cycles)
- **Engines in Test Set**: 100 units (truncated at arbitrary cycles prior to failure, totaling 13,096 cycles)
- **Ground Truth Test Labels**: `RUL_FD001.txt` containing 100 scalar values representing the true remaining cycles after the final recorded cycle for each test engine.

### Data Schema (26 Columns)
Each record in the raw space-delimited text files contains 26 attributes:

| Column Category | Column Names | Description |
|---|---|---|
| Engine Identifier | `engine_id` | Integer unit ID (1 to 100) |
| Time Tracking | `cycle` | Operational flight cycle count (starts at 1) |
| Operational Settings | `setting_1`, `setting_2`, `setting_3` | Altitude, Mach number, and throttle resolver angle |
| Sensor Telemetry | `sensor_1` to `sensor_21` | Measurements including temperatures, pressures, spool speeds, and coolant bleeds |

---

## 4. Exploratory Data Analysis and Preprocessing

1. **Integrity & Completeness**:
   - Zero missing values (`null` count: 0) and zero duplicate rows across all 20,631 training cycles.
   - Run lengths in training vary naturally between 128 cycles (Engine #39) and 362 cycles (Engine #69).
2. **True RUL Ground Truth Construction**:
   - For training engines, the final observed cycle represents catastrophic failure ($cycle_{max}$).
   - The uncapped RUL at cycle $t$ is computed as:
     $$RUL_t = cycle_{max} - cycle_t$$
3. **Engine-Level Data Splitting**:
   - To strictly prevent data leakage across time-series sequences, train/validation splitting is conducted using `GroupShuffleSplit` on `engine_id` (80% training / 80 engines, 20% validation / 20 engines; 0 overlapping engines).
4. **Feature Normalization**:
   - A `StandardScaler` is fitted **strictly** on the training engines and subsequently applied to validation and test sequences to preserve isolation and avoid lookahead bias.

---

## 5. Feature Selection and Sensor Correlation Analysis

Analysis of sensor variance revealed that several channels remain flat across all engines under sea-level conditions.

### Removed Constant Features (Zero Variance)
Five non-informative features were identified and removed:
- `setting_3` (std = 0.0)
- `sensor_1` (Fan inlet temperature, std = 0.0)
- `sensor_10` (Bleed enthalpy, std = 0.0)
- `sensor_18` (High-pressure turbine demand, std = 0.0)
- `sensor_19` (Bypass ratio setting, std = 0.0)

### Selected Features (19 Inputs)
The remaining 19 informative channels capture thermal, mechanical, and pressure wear:
- Operational settings: `setting_1`, `setting_2`
- Telemetry sensors: `sensor_2`, `sensor_3`, `sensor_4`, `sensor_5`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_16`, `sensor_17`, `sensor_20`, `sensor_21`

Sensors showing the highest correlation with degradation include `sensor_11` (Static pressure at HPC outlet), `sensor_4` (Low-pressure turbine outlet temperature), `sensor_12` (Ratio of fuel flow to static pressure), and `sensor_9` (Core speed).

---

## 6. Explanation of the RUL Target and 30-Cycle Sliding Windows

### Capped vs. Uncapped RUL
Early in an engine's lifecycle, degradation has not yet begun, and sensor readings reflect healthy baseline operations. Predicting very large linear RUL values (e.g., 300+ cycles) during early cycles can destabilize model gradients. A piecewise linear target capped at $RUL_{max} = 125$ cycles was evaluated during baseline screening:
$$RUL_{capped} = \min(RUL_{true}, 125)$$
The deep learning models (LSTM and Transformer) were trained and evaluated against true, uncapped RUL to deliver realistic degradation forecasts throughout the entire engine lifespan.

### 30-Cycle Sliding Sequence Generation
Turbofan degradation is a temporal progression. Rather than predicting solely from single isolated cycles, sequential models ingest a sliding temporal window of the past **30 consecutive cycles** ($T = 30$):
$$\mathbf{X}_{t} \in \mathbb{R}^{30 \times 19}$$
- Each engine is processed independently in chronological cycle order.
- Sequences are generated by sliding the 30-cycle window forward by 1 cycle.
- Engines with fewer than 30 cycles are padded or evaluated at minimum sequence threshold (all 100 test engines in FD001 possess $\ge 31$ cycles).

---

## 7. Models Implemented

Five model architectures spanning classical regression, ensemble learning, and deep sequence modeling were implemented and compared:

1. **Linear Regression**: Baseline ordinary least squares model on tabular features.
2. **Random Forest Regressor**: Non-linear ensemble (100 estimators, max depth 15, min samples leaf 2).
3. **XGBoost Regressor**: Gradient-boosted decision trees (500 estimators, max depth 5, learning rate 0.03, subsample 0.8) equipped with Tree SHAP explainability.
4. **Improved LSTM (Long Short-Term Memory)**: Recurrent neural network designed to capture temporal degradation dynamics across sequential cycles.
5. **Transformer Encoder**: Self-attention architecture mapping temporal inter-cycle relationships across sensor channels.

---

## 8. Model Architecture and Training Approach

### 1. Improved LSTM Architecture
- **Input Layer**: Shape `(batch_size, 30, 19)`
- **Recurrent Backbone**: 2-layer stacked LSTM, `hidden_size = 128`, `batch_first = True`, `dropout = 0.2`
- **Output Head**: Linear projection layer mapping `128 -> 1`
- **Total Parameters**: ~200,000 parameters
- **Optimizer & Loss**: Adam optimizer (`lr = 0.001`), Mean Squared Error (`nn.MSELoss`)
- **Regularization**: Validation loss tracking with checkpoint restoration of best weights.

### 2. Transformer Architecture
- **Input Projection**: Linear layer projecting `19 -> 64` (`d_model = 64`)
- **Positional Encoding**: Learnable embedding matrix `(1, 30, 64)` scaled by 0.02
- **Encoder Stack**: 2 Transformer Encoder layers with 4 attention heads (`nhead = 4`), feedforward dimension 128, and dropout 0.1
- **Pooling & Head**: Mean pooling across the temporal sequence dimension followed by a linear projection `64 -> 1`
- **Total Parameters**: 72,385 parameters
- **Optimizer & Schedule**: AdamW (`lr = 0.0003`, `weight_decay = 1e-4`) with gradient clipping (`max_norm = 1.0`) and `ReduceLROnPlateau` scheduler.

---

## 9. Evaluation Metrics

Model performance is quantified using three standard regression metrics:

1. **Mean Absolute Error (MAE)**: Average absolute deviation in cycle prediction:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
   *(Lower is better; directly interpretable in operating cycles).*

2. **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors more heavily:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
   *(Lower is better; critical for safety when late predictions are costly).*

3. **Coefficient of Determination ($R^2$)**: Proportion of variance in RUL explained by the model:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   *(Higher is better; 1.0 represents a perfect fit).*

---

## 10. Official FD001 Test-Set Comparison

The official FD001 test benchmark consists of **100 independent test engines**. For each test engine, models ingest the final 30 observed cycles preceding maintenance truncation and predict the remaining cycles. Predictions are benchmarked against the ground truth labels in `RUL_FD001.txt`.

### Verified Official Test Results

| Model | MAE (cycles) | RMSE (cycles) | $R^2$ Score |
|---|---:|---:|---:|
| **LSTM** | **17.6053** | **24.0977** | **0.6637** |
| Transformer | 18.8980 | 25.2003 | 0.6322 |
| XGBoost | 24.3677 | 34.1241 | 0.3257 |

*Note: These metrics reflect the exact results from this experimental benchmark. Lower MAE and RMSE indicate superior precision; higher $R^2$ indicates stronger explanatory power. While competitive for standard benchmark configurations, these results do not claim state-of-the-art status.*

### Key Observations
- **LSTM achieved the best performance** across all three evaluation metrics, delivering the lowest MAE (17.61 cycles) and RMSE (24.10 cycles), and highest $R^2$ (0.6637).
- Both temporal deep learning models (LSTM and Transformer) significantly outperformed tabular tree-based models (XGBoost), demonstrating the necessity of sequential feature modeling for cumulative mechanical wear.

---

## 11. Installation and Setup Instructions for Windows

Follow these steps to set up the environment and run the project locally on Windows:

### Prerequisites
- Python 3.10 to 3.12 installed on your system.
- Git installed.

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/NASA-RUL.git
cd NASA-RUL
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate in PowerShell
.\.venv\Scripts\Activate.ps1

# (Optional: If running in Command Prompt / cmd.exe)
# .\.venv\Scripts\activate.bat
```

> **PowerShell Execution Policy Note**: If script execution is restricted on Windows, enable it for the current process:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Instructions to Run the Notebook

### 1. Verify Dataset Placement
Ensure the NASA C-MAPSS FD001 raw data files are located in `data/raw/cmapss/`:
```
data/raw/cmapss/train_FD001.txt
data/raw/cmapss/test_FD001.txt
data/raw/cmapss/RUL_FD001.txt
```

### 2. Launch Jupyter Notebook
With `.venv` activated:
```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```
Alternatively, open the repository in VS Code or Cursor, select `.venv\Scripts\python.exe` as the Jupyter kernel, and execute cells sequentially.

---

## 13. Project Folder Structure

```
NASA-RUL/
│
├── configs/                      # Configuration files for experiments
│   └── .gitkeep
├── data/                         # Dataset root (excluded from Git)
│   ├── raw/                      # Raw C-MAPSS telemetry files
│   │   ├── cmapss/               # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   │   └── .gitkeep
│   └── processed/                # Intermediate processed datasets
│       └── .gitkeep
├── models/                       # Saved model checkpoints and artifacts
│   ├── .gitkeep
│   ├── lstm_fd001.pth            # Trained PyTorch state dict for best LSTM (838 KB)
│   ├── lstm_fd001_config.json    # Architecture & input feature metadata (675 B)
│   ├── scaler_fd001.joblib       # Fitted StandardScaler instance (1.5 KB)
│   └── transformer_fd001.pth     # Trained PyTorch state dict for Transformer (302 KB)
├── notebooks/                    # Interactive research notebooks
│   └── 01_data_exploration.ipynb # End-to-end EDA, training, SHAP, and official evaluation
├── src/                          # Modular Python source code (pipeline utilities)
│   └── .gitkeep
├── tests/                        # Test suites and validation checks
│   └── .gitkeep
├── .gitignore                    # Git exclusion rules
├── README.md                     # Comprehensive project documentation
└── requirements.txt              # Production dependency declarations
```

---

## 14. Saved Model Artifacts and How They Are Used

The `models/` directory houses the essential components needed for zero-training inference reproduction:

1. **`scaler_fd001.joblib` (1.48 KB)**:
   - Serialized `sklearn.preprocessing.StandardScaler` fitted on the 19 selected features across training engines.
   - Required to normalize incoming test telemetry to zero mean and unit variance.
2. **`lstm_fd001_config.json` (675 bytes)**:
   - JSON configuration declaring model class (`ImprovedRULLSTM`), sequence length (30), the 19 exact input feature names, and benchmarked metrics.
3. **`lstm_fd001.pth` (838 KB)**:
   - PyTorch `state_dict` of the trained `ImprovedRULLSTM` model. Reconstructed via:
     ```python
     import torch
     from models import ImprovedRULLSTM # or defined in notebook

     model = ImprovedRULLSTM(input_size=19, hidden_size=128)
     model.load_state_dict(torch.load("models/lstm_fd001.pth", map_location="cpu", weights_only=True))
     model.eval()
     ```
4. **`transformer_fd001.pth` (302 KB)**:
   - PyTorch `state_dict` for the self-attention `RULTransformer` model.

---

## 15. Limitations, Future Improvements, and References

### Limitations
- **Single Operating Regime**: Evaluated exclusively on FD001 (sea-level condition and single HPC fault mode). Multi-condition subsets (FD002, FD004) require operating condition clustering or normalization.
- **Fixed Sequence Length**: Requires at least 30 recorded cycles per engine before initial predictions can be produced without padding.
- **Piecewise Assumption**: Standard linear degradation models assume constant degradation rates toward end-of-life, whereas exponential wear may occur in final cycles.

### Future Improvements
- Extend pipelines to multi-condition datasets (FD002, FD003, FD004) using operating condition clustering.
- Implement Bidirectional LSTMs and Temporal Convolutional Networks (TCN) for comparison.
- Incorporate asymmetric loss functions (NASA scoring function $s = \sum e^{-d/a} - 1$) penalizing late predictions more strictly than early predictions.
- Package inference logic into a containerized FastAPI endpoint or ONNX runtime service.

### References
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* IEEE International Conference on Prognostics and Health Management.
- NASA Prognostics Data Repository: *Turbofan Engine Degradation Simulation Data Set*.