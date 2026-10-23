"use client";

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { PredictResponse } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

interface PredictionResultProps {
  result: PredictResponse;
}

export function PredictionResult({ result }: PredictionResultProps) {
  const [showTable, setShowTable] = useState(false);

  // Interpretation helper
  const getInterpretation = (rul: number) => {
    if (rul < 30) {
      return {
        level: "Low Remaining Life",
        badge: "Critical",
        desc: "Sensor readings exhibit significant wear patterns near failure threshold.",
      };
    } else if (rul <= 80) {
      return {
        level: "Moderate Remaining Life",
        badge: "Warning",
        desc: "Active degradation trends detected across thermal and pressure channels.",
      };
    } else {
      return {
        level: "High Remaining Life",
        badge: "Healthy",
        desc: "Telemetry signals indicate standard operational envelope within steady state.",
      };
    }
  };

  const interpretation = getInterpretation(result.predicted_rul);

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-sm space-y-5 transition-all">
      {/* Result Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold text-slate-500 uppercase tracking-wider">
            <Sparkles className="h-3.5 w-3.5 text-blue-600" />
            <span>Predicted Remaining Useful Life</span>
          </div>
          <h3 className="text-base sm:text-lg font-bold text-slate-900 mt-0.5 truncate">
            {result.engine_identifier}
          </h3>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <StatusBadge status={interpretation.badge} />
          <span className="text-xs px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200">
            LSTM Model
          </span>
        </div>
      </div>

      {/* Main RUL Display Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
        {/* Metric 1: Predicted Cycles */}
        <div className="bg-blue-50/70 border border-blue-200/80 rounded-xl p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-blue-800">Forecasted Life</span>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-3xl sm:text-4xl font-extrabold text-blue-700 tracking-tight">
              {result.predicted_rul.toFixed(1)}
            </span>
            <span className="text-xs sm:text-sm font-bold text-blue-600">cycles</span>
          </div>
          <span className="text-[11px] text-blue-600 mt-1">
            Input: {result.sequence_length_used} cycles window
          </span>
        </div>

        {/* Metric 2: Condition Level */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-slate-600">Illustrative Condition</span>
          <div className="mt-1">
            <span className="text-lg sm:text-xl font-bold text-slate-900">
              {interpretation.level}
            </span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 leading-snug">
            {interpretation.desc}
          </span>
        </div>

        {/* Metric 3: Input Verification */}
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
          <span className="text-xs font-semibold text-slate-600">Architecture Evaluated</span>
          <div className="mt-1">
            <span className="text-sm sm:text-base font-bold text-slate-900">
              Stacked LSTM (2-Layer)
            </span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1">
            StandardScaler fitted on 19 features
          </span>
        </div>
      </div>

      {/* Illustrative Interpretation Callout */}
      <div className="p-3 sm:p-4 rounded-xl bg-amber-50/70 border border-amber-200 text-xs text-amber-900 flex items-start gap-2.5">
        <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
        <p className="leading-relaxed text-[11px] sm:text-xs">
          <strong>Illustrative Interpretation:</strong> {result.note} The condition band is an illustrative statistical interpretation of predicted cycles, not a certified airworthiness or flight safety classification.
        </p>
      </div>

      {/* Telemetry Trend Chart */}
      {result.recent_telemetry && result.recent_telemetry.length > 0 && (
        <div className="space-y-3 pt-2">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
            <h4 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-1.5">
              <Activity className="h-4 w-4 text-blue-600" />
              <span>Input Telemetry Wear Trends (30 Cycles)</span>
            </h4>
            <span className="text-[11px] text-slate-500">Key degradation channels</span>
          </div>

          <div className="h-56 sm:h-64 w-full bg-slate-50/50 rounded-xl border border-slate-200 p-2 sm:p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={result.recent_telemetry}
                margin={{ top: 10, right: 10, bottom: 0, left: -20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="cycle"
                  stroke="#94a3b8"
                  fontSize={10}
                  tickLine={false}
                />
                <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#ffffff",
                    borderColor: "#cbd5e1",
                    borderRadius: "8px",
                    fontSize: "11px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
                <Line
                  type="monotone"
                  dataKey="sensor_11"
                  name="HPC Pres (s11)"
                  stroke="#2563eb"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="sensor_4"
                  name="LPT Temp (s4)"
                  stroke="#dc2626"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="sensor_9"
                  name="Core Speed (s9)"
                  stroke="#16a34a"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="sensor_12"
                  name="Fuel Flow (s12)"
                  stroke="#9333ea"
                  dot={false}
                  strokeWidth={1.5}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Toggle Raw Data Table */}
          <div className="pt-1">
            <button
              type="button"
              onClick={() => setShowTable(!showTable)}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-semibold text-slate-700 transition-colors"
            >
              <span>{showTable ? "Hide 30-Cycle Raw Readings" : "View 30-Cycle Raw Readings"}</span>
              {showTable ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>

            {showTable && (
              <div className="mt-3 border border-slate-200 rounded-xl overflow-hidden max-h-60 overflow-y-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="bg-slate-100 text-slate-700 sticky top-0 border-b border-slate-200 font-semibold">
                    <tr>
                      <th className="py-2 px-3">Cycle</th>
                      <th className="py-2 px-3">s11 (HPC)</th>
                      <th className="py-2 px-3">s4 (LPT)</th>
                      <th className="py-2 px-3">s9 (Speed)</th>
                      <th className="py-2 px-3">s12 (Fuel)</th>
                      <th className="py-2 px-3">s2 (Temp)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {result.recent_telemetry.map((r, i) => (
                      <tr key={i} className="hover:bg-slate-50/70">
                        <td className="py-1.5 px-3 font-semibold text-slate-900">{r.cycle || i + 1}</td>
                        <td className="py-1.5 px-3 text-slate-600 font-mono">{r.sensor_11}</td>
                        <td className="py-1.5 px-3 text-slate-600 font-mono">{r.sensor_4}</td>
                        <td className="py-1.5 px-3 text-slate-600 font-mono">{r.sensor_9}</td>
                        <td className="py-1.5 px-3 text-slate-600 font-mono">{r.sensor_12}</td>
                        <td className="py-1.5 px-3 text-slate-600 font-mono">{r.sensor_2}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
