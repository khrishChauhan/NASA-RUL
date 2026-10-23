"use client";

import { useEffect, useState } from "react";
import {
  BarChart3,
  CheckCircle2,
  TrendingDown,
  Layers,
  Info,
  Sparkles,
  Award,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from "recharts";
import { fetchModelsInfo, ModelsResponse } from "@/lib/api";
import { ErrorAlert } from "./ErrorAlert";

export function ModelComparison() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelsInfo()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load model comparison.");
        setLoading(false);
      });
  }, []);

  // Static fallback values in case backend is offline
  const models = data?.models || [
    {
      model_name: "ImprovedRULLSTM",
      mae: 17.6053,
      rmse: 24.0977,
      r2: 0.6637,
      status: "Production Active",
      description: "Two-layer stacked LSTM capturing temporal wear trends across 30 sliding cycles.",
      is_active: true,
      weights_available: true,
    },
    {
      model_name: "RULTransformer",
      mae: 18.898,
      rmse: 25.2003,
      r2: 0.6322,
      status: "Alternative Architecture",
      description: "Transformer Encoder with self-attention across 30 cycles.",
      is_active: false,
      weights_available: true,
    },
    {
      model_name: "XGBoost",
      mae: 24.3677,
      rmse: 34.1241,
      r2: 0.3257,
      status: "Tabular Baseline",
      description: "Gradient-boosted decision trees evaluated on isolated cycle snapshots.",
      is_active: false,
      weights_available: false,
    },
  ];

  const chartData = [
    { name: "LSTM", mae: 17.61, isBest: true },
    { name: "Transformer", mae: 18.9, isBest: false },
    { name: "XGBoost", mae: 24.37, isBest: false },
  ];

  return (
    <div className="space-y-6">
      {/* Benchmark Header Banner */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-2xs space-y-2.5">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-50 border border-blue-200 text-xs font-semibold text-blue-700">
          <Sparkles className="h-3.5 w-3.5 text-blue-600" />
          <span>Official FD001 Test Set Evaluation (100 Engines)</span>
        </div>
        <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">
          Model Benchmark Comparison
        </h2>
        <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
          Recorded experimental results benchmarked on the final 30 observed cycles of all 100 test engines in NASA C-MAPSS FD001.
          Evaluated against official ground truth Remaining Useful Life labels.
        </p>
      </div>

      {error && <ErrorAlert message={error} />}

      {/* 3 Model Cards (Mobile Stacked, Desktop 3-Col) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((m) => {
          const isLstm = m.model_name.includes("LSTM");
          return (
            <div
              key={m.model_name}
              className={`rounded-2xl border p-5 shadow-2xs flex flex-col justify-between transition-all ${
                isLstm
                  ? "bg-white border-blue-500 ring-2 ring-blue-500/10"
                  : "bg-white border-slate-200"
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <h3 className="font-bold text-slate-900 text-base flex items-center gap-1.5">
                    {isLstm && <Award className="h-4 w-4 text-blue-600" />}
                    <span>{isLstm ? "LSTM" : m.model_name.replace("RUL", "")}</span>
                  </h3>

                  {isLstm ? (
                    <span className="text-[11px] font-bold bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">
                      Best Overall
                    </span>
                  ) : (
                    <span className="text-[11px] font-medium bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
                      Comparison
                    </span>
                  )}
                </div>

                <p className="text-xs text-slate-500 mt-1 leading-snug">
                  {m.description}
                </p>

                {/* Metrics Breakdown */}
                <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                  <div className="bg-slate-50 p-2 rounded-xl border border-slate-200/80">
                    <span className="text-[10px] font-bold text-slate-400 block uppercase">
                      MAE
                    </span>
                    <span className="text-sm font-extrabold text-slate-900 mt-0.5 block">
                      {m.mae.toFixed(2)}
                    </span>
                    <span className="text-[9px] text-slate-400">cycles</span>
                  </div>

                  <div className="bg-slate-50 p-2 rounded-xl border border-slate-200/80">
                    <span className="text-[10px] font-bold text-slate-400 block uppercase">
                      RMSE
                    </span>
                    <span className="text-sm font-extrabold text-slate-900 mt-0.5 block">
                      {m.rmse.toFixed(2)}
                    </span>
                    <span className="text-[9px] text-slate-400">cycles</span>
                  </div>

                  <div className="bg-slate-50 p-2 rounded-xl border border-slate-200/80">
                    <span className="text-[10px] font-bold text-slate-400 block uppercase">
                      R²
                    </span>
                    <span className="text-sm font-extrabold text-slate-900 mt-0.5 block">
                      {m.r2.toFixed(4)}
                    </span>
                    <span className="text-[9px] text-slate-400">score</span>
                  </div>
                </div>
              </div>

              {isLstm && (
                <div className="mt-4 pt-3 border-t border-slate-100 flex items-center gap-1.5 text-xs text-emerald-700 font-semibold">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                  <span>Lowest error on official benchmark</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* MAE Visual Comparison Chart */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 shadow-2xs space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <div>
            <h3 className="text-sm sm:text-base font-bold text-slate-900 flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-blue-600" />
              <span>Mean Absolute Error (MAE) Comparison</span>
            </h3>
            <p className="text-xs text-slate-500">
              Average error in cycles (Lower bar represents superior precision)
            </p>
          </div>
          <span className="text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-200 self-start sm:self-auto">
            LSTM: 17.61 cycles (Best)
          </span>
        </div>

        <div className="h-48 sm:h-56 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" c" />
              <Tooltip
                formatter={(val: unknown) => [`${Number(val).toFixed(2)} cycles`, "MAE"]}
                contentStyle={{
                  backgroundColor: "#ffffff",
                  borderColor: "#cbd5e1",
                  borderRadius: "8px",
                  fontSize: "11px",
                }}
              />
              <Bar dataKey="mae" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isBest ? "#2563eb" : "#cbd5e1"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metric Definitions Guide */}
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 sm:p-5 space-y-3">
        <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
          <Info className="h-4 w-4 text-blue-600" />
          <span>Understanding the Evaluation Metrics</span>
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-slate-600">
          <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
            <span className="font-bold text-slate-900 block">MAE (Mean Absolute Error)</span>
            <p className="text-[11px] leading-relaxed text-slate-600">
              Average absolute difference between predicted and actual RUL in cycles. <strong>Lower is better</strong>.
            </p>
          </div>

          <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
            <span className="font-bold text-slate-900 block">RMSE (Root Mean Squared Error)</span>
            <p className="text-[11px] leading-relaxed text-slate-600">
              Penalizes larger outliers more heavily. Critical for early safety detection. <strong>Lower is better</strong>.
            </p>
          </div>

          <div className="p-3 bg-white rounded-xl border border-slate-200 space-y-1">
            <span className="font-bold text-slate-900 block">R² Score (Goodness of Fit)</span>
            <p className="text-[11px] leading-relaxed text-slate-600">
              Proportion of variance explained by model over a baseline mean. <strong>Higher is better</strong> (1.0 = perfect).
            </p>
          </div>
        </div>

        <div className="pt-1 text-[11px] text-slate-500 leading-snug">
          * Recorded experimental results from the official NASA C-MAPSS FD001 test benchmark. The platform does not claim state-of-the-art performance.
        </div>
      </div>
    </div>
  );
}
