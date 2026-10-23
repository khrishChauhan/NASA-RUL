export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    if (
      msg.includes("Failed to fetch") ||
      msg.includes("NetworkError") ||
      msg.includes("fetch") ||
      msg.includes("Load failed")
    ) {
      throw new Error(
        "Unable to reach the predictive maintenance API. If the backend is waking up from sleep on Render (free tier), it may take 30–60 seconds for the initial cold start. Please wait a moment and try again."
      );
    }
    throw err;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = "An unexpected error occurred.";
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || errorJson.message || JSON.stringify(errorJson);
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/health`, { cache: "no-store" });
  return handleResponse<HealthResponse>(res);
}

export async function fetchOverview(): Promise<OverviewResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/overview`, { cache: "no-store" });
  return handleResponse<OverviewResponse>(res);
}

export async function fetchEngines(): Promise<EngineListItem[]> {
  const res = await apiFetch(`${API_BASE_URL}/api/engines`, { cache: "no-store" });
  return handleResponse<EngineListItem[]>(res);
}

export async function fetchEngineDetail(engineId: number): Promise<EngineDetailResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/engines/${engineId}`, { cache: "no-store" });
  return handleResponse<EngineDetailResponse>(res);
}

export async function predictEngine(engineId: number, modelName: string = "ImprovedRULLSTM"): Promise<PredictResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ engine_id: engineId, model_name: modelName }),
  });
  return handleResponse<PredictResponse>(res);
}

export async function predictUploadCsv(file: File, modelName: string = "ImprovedRULLSTM"): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_name", modelName);

  const res = await apiFetch(`${API_BASE_URL}/api/predict/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<PredictResponse>(res);
}

export async function fetchModelsInfo(): Promise<ModelsResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/models`, { cache: "no-store" });
  return handleResponse<ModelsResponse>(res);
}

export async function fetchSensorHistory(engineId: number): Promise<SensorHistoryResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/sensors/${engineId}`, { cache: "no-store" });
  return handleResponse<SensorHistoryResponse>(res);
}

export async function predictSample(): Promise<PredictResponse> {
  const res = await apiFetch(`${API_BASE_URL}/api/predict/sample`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  return handleResponse<PredictResponse>(res);
}

export async function fetchSampleTelemetry(): Promise<{
  label: string;
  rows: Record<string, number>[];
  features: string[];
  total_cycles: number;
}> {
  const res = await apiFetch(`${API_BASE_URL}/api/sample`, { cache: "no-store" });
  return handleResponse<{
    label: string;
    rows: Record<string, number>[];
    features: string[];
    total_cycles: number;
  }>(res);
}

export interface HealthResponse {
  status: string;
  models_loaded: Record<string, boolean>;
  scaler_loaded: boolean;
  dataset_loaded: boolean;
  device: string;
  version: string;
}

export interface EngineBenchmarkItem {
  engine_id: number;
  observed_cycles: number;
  predicted_rul: number;
  actual_rul: number;
  absolute_error: number;
  health_status: "Critical" | "Warning" | "Healthy";
}

export interface OverviewResponse {
  project_title: string;
  description: string;
  total_test_engines: number;
  total_train_engines: number;
  total_train_cycles: number;
  selected_model: string;
  best_test_mae: number;
  best_test_rmse: number;
  best_test_r2: number;
  benchmark_sample: EngineBenchmarkItem[];
  models_summary: {
    name: string;
    mae: number;
    rmse: number;
    r2: number;
    status: string;
  }[];
}

export interface EngineListItem {
  engine_id: number;
  observed_cycles: number;
  predicted_rul: number;
  actual_rul: number;
  absolute_error: number;
  health_status: "Critical" | "Warning" | "Healthy";
}

export interface EngineDetailResponse {
  engine_id: number;
  observed_cycles: number;
  predicted_rul: number;
  actual_rul: number;
  history: Record<string, number>[];
  final_sequence_features: string[];
}

export interface PredictResponse {
  model_name: string;
  predicted_rul: number;
  sequence_length_used: number;
  engine_identifier: string;
  health_status: "Critical" | "Warning" | "Healthy";
  status: string;
  note: string;
  actual_rul?: number | null;
  recent_telemetry?: Record<string, number>[] | null;
}

export interface ModelMetricDetail {
  model_name: string;
  mae: number;
  rmse: number;
  r2: number;
  status: string;
  description: string;
  is_active: boolean;
  weights_available: boolean;
  parameters?: number | null;
}

export interface ModelsResponse {
  models: ModelMetricDetail[];
  historical_benchmark_note: string;
  active_production_model: string;
}

export interface SensorStat {
  name: string;
  description: string;
  unit: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  correlation_with_rul: number;
}

export interface SensorHistoryResponse {
  engine_id: number;
  total_cycles: number;
  available_features: SensorStat[];
  readings: Record<string, number>[];
}

