"use client";

import { useState } from "react";
import {
  Sparkles,
  Upload,
  FileText,
  AlertCircle,
  Download,
  Info,
  CheckCircle2,
  RefreshCw,
  Cpu,
} from "lucide-react";
import {
  predictSample,
  predictUploadCsv,
  fetchSampleTelemetry,
  PredictResponse,
} from "@/lib/api";
import { PredictionResult } from "./prediction-result";
import { ErrorAlert } from "./ErrorAlert";

export function AnalyzeEngine() {
  const [inputMode, setInputMode] = useState<"sample" | "upload">("sample");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [downloadingTemplate, setDownloadingTemplate] = useState(false);

  // Handle Generate Sample & Predict
  const handleGenerateSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictSample();
      setResult(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate sample prediction from backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // Handle CSV Upload & Predict
  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError("Please select a CSV or TXT file to upload.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictUploadCsv(selectedFile, "ImprovedRULLSTM");
      setResult(res);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Uploaded telemetry validation failed."
      );
    } finally {
      setLoading(false);
    }
  };

  // Download a real sample CSV template for users to test upload flow
  const handleDownloadSampleCsv = async () => {
    setDownloadingTemplate(true);
    try {
      const sample = await fetchSampleTelemetry();
      if (!sample || !sample.rows || sample.rows.length === 0) {
        throw new Error("Could not retrieve sample rows.");
      }
      const headers = ["cycle", ...sample.features];
      const rows = sample.rows.map((r, i) => {
        const cycle = r.cycle || i + 1;
        const vals = sample.features.map((f) => r[f] ?? 0.0);
        return [cycle, ...vals].join(",");
      });
      const csvContent = [headers.join(","), ...rows].join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "sample_engine_30_cycles.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to generate CSV template."
      );
    } finally {
      setDownloadingTemplate(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 shadow-2xs space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="space-y-1">
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 tracking-tight">
              Analyze Turbofan Engine Remaining Useful Life
            </h2>
            <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
              Run multivariate degradation inference using the trained 2-layer stacked LSTM model.
              Requires 30 consecutive operational cycles across the 19 selected sensor channels.
            </p>
          </div>

          <div className="shrink-0 self-start sm:self-auto">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-xs font-semibold text-blue-700">
              <Cpu className="h-3.5 w-3.5" />
              <span>LSTM Model Active</span>
            </span>
          </div>
        </div>

        {/* Input Mode Switcher (Option A vs Option B) */}
        <div className="pt-2 border-t border-slate-100 flex flex-col sm:flex-row gap-2">
          <button
            type="button"
            onClick={() => {
              setInputMode("sample");
              setError(null);
            }}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              inputMode === "sample"
                ? "bg-blue-600 text-white shadow-xs shadow-blue-200"
                : "bg-slate-100 hover:bg-slate-200/70 text-slate-700"
            }`}
          >
            <Sparkles className="h-4 w-4" />
            <span>Option A: Generate Sample Data</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setInputMode("upload");
              setError(null);
            }}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-xs sm:text-sm font-semibold transition-all ${
              inputMode === "upload"
                ? "bg-blue-600 text-white shadow-xs shadow-blue-200"
                : "bg-slate-100 hover:bg-slate-200/70 text-slate-700"
            }`}
          >
            <Upload className="h-4 w-4" />
            <span>Option B: Upload Sensor CSV</span>
          </button>
        </div>
      </div>

      {/* Mode A: Generate Sample Data */}
      {inputMode === "sample" && (
        <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 shadow-2xs space-y-4">
          <div className="space-y-1">
            <h3 className="text-sm sm:text-base font-bold text-slate-900 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-blue-600" />
              <span>Option A: Instant Sample Evaluation</span>
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Samples a real contiguous 30-cycle sequence from the NASA FD001 benchmark and evaluates it with the saved LSTM.
            </p>
          </div>

          <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-600 space-y-1.5">
            <div className="flex items-center gap-1.5 font-semibold text-slate-800">
              <Info className="h-4 w-4 text-blue-600 shrink-0" />
              <span>How Sample Generation Works</span>
            </div>
            <p className="text-[11px] sm:text-xs leading-relaxed text-slate-600">
              Picks a random engine from the FD001 dataset and extracts a contiguous 30-cycle operational window containing strictly the 19 sensor features (thermal, spool speed, pressure).
              Input telemetry is passed to the real backend and scaled with the trained <code className="bg-white px-1 py-0.5 rounded border border-slate-200 text-blue-700 font-mono">StandardScaler</code>.
            </p>
            <span className="text-[11px] text-amber-700 font-medium block">
              * Note: Sampled data is for benchmark evaluation, not measurements from a currently operating engine.
            </span>
          </div>

          {/* Prominent Action Button */}
          <button
            type="button"
            onClick={handleGenerateSample}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 px-5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-bold text-sm sm:text-base shadow-sm shadow-blue-200 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Running LSTM Model...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Generate Sample & Predict RUL</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Mode B: Upload Sensor CSV */}
      {inputMode === "upload" && (
        <form
          onSubmit={handleUploadSubmit}
          className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 shadow-2xs space-y-4"
        >
          <div className="space-y-1">
            <h3 className="text-sm sm:text-base font-bold text-slate-900 flex items-center gap-2">
              <Upload className="h-4 w-4 text-blue-600" />
              <span>Option B: Upload Custom Telemetry</span>
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed">
              Upload a CSV containing at least 30 consecutive flight cycles for an engine.
            </p>
          </div>

          {/* Drag & Drop File Zone */}
          <div className="border-2 border-dashed border-slate-300 hover:border-blue-400 rounded-2xl p-6 text-center transition-colors bg-slate-50/60">
            <Upload className="h-8 w-8 text-slate-400 mx-auto" />
            <div className="mt-2 text-xs sm:text-sm font-semibold text-slate-800">
              {selectedFile ? (
                <span className="text-blue-600 font-bold">{selectedFile.name}</span>
              ) : (
                <span>Tap or drag your sensor CSV file here</span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Supports .csv or space-delimited .txt (minimum 30 rows)
            </p>

            <input
              type="file"
              accept=".csv,.txt"
              id="csv-upload-input"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              className="hidden"
            />
            <label
              htmlFor="csv-upload-input"
              className="mt-3 inline-block px-4 py-2 rounded-xl border border-slate-300 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-700 cursor-pointer shadow-2xs transition-colors"
            >
              Select File
            </label>
          </div>

          {/* Schema & Download Template Row */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-3.5 bg-slate-50 rounded-xl border border-slate-200">
            <div className="text-xs text-slate-600">
              <span className="font-semibold text-slate-800 block">Need a valid file format?</span>
              <span className="text-[11px] text-slate-500">
                Download a verified 30-cycle sample CSV to test the upload flow.
              </span>
            </div>

            <button
              type="button"
              onClick={handleDownloadSampleCsv}
              disabled={downloadingTemplate}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-100 text-xs font-semibold text-slate-700 transition-colors shrink-0"
            >
              <Download className="h-3.5 w-3.5 text-blue-600" />
              <span>{downloadingTemplate ? "Generating..." : "Download Sample CSV"}</span>
            </button>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !selectedFile}
            className="w-full flex items-center justify-center gap-2 py-3 px-5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-bold text-sm sm:text-base shadow-sm shadow-blue-200 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Validating & Running Inference...</span>
              </>
            ) : (
              <>
                <FileText className="h-4 w-4" />
                <span>Upload & Predict RUL</span>
              </>
            )}
          </button>
        </form>
      )}

      {/* Error State Banner */}
      {error && (
        <ErrorAlert
          title="Analysis Failed"
          message={error}
          onRetry={inputMode === "sample" ? handleGenerateSample : undefined}
        />
      )}

      {/* Loading State Banner with Cold-Start Notice */}
      {loading && (
        <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-xl flex items-center gap-2.5 text-xs text-blue-800">
          <RefreshCw className="h-4 w-4 animate-spin text-blue-600 shrink-0" />
          <span>
            Communicating with model backend... (Note: Cloud hosting on free tiers such as Render may take 30–60s for cold starts if waking from sleep).
          </span>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !result && (
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-2xs animate-pulse space-y-4">
          <div className="h-4 w-36 bg-slate-200 rounded" />
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="h-20 bg-slate-200 rounded-xl" />
            <div className="h-20 bg-slate-200 rounded-xl" />
            <div className="h-20 bg-slate-200 rounded-xl" />
          </div>
          <div className="h-48 bg-slate-200 rounded-xl" />
        </div>
      )}

      {/* Prediction Result Component */}
      {result && <PredictionResult result={result} />}
    </div>
  );
}
