"use client";

import { useState, useEffect } from "react";
import {
  Cpu,
  BarChart3,
  Flame,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
} from "lucide-react";
import { fetchHealth, API_BASE_URL, HealthResponse } from "@/lib/api";
import { AnalyzeEngine } from "@/components/analyze-engine";
import { ModelComparison } from "@/components/model-comparison";

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<"analyze" | "models">("analyze");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    fetchHealth()
      .then((data) => {
        if (isMounted) {
          setHealth(data);
          setHealthLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setHealth(null);
          setHealthLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-between">
      {/* Top Application Bar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-2xs">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3">
          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="h-9 w-9 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-xs shadow-blue-200">
              <Flame className="h-5 w-5" />
            </div>
            <div>
              <div className="text-sm font-bold text-slate-900 leading-tight">
                NASA-RUL
              </div>
              <div className="text-[11px] font-medium text-slate-500 leading-tight">
                Turbofan Prognostics
              </div>
            </div>
          </div>

          {/* System Status Indicator */}
          <div className="flex items-center gap-2">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-xs">
              {healthLoading ? (
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
              ) : health?.status === "healthy" ? (
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
              ) : (
                <span className="h-2 w-2 rounded-full bg-amber-500" />
              )}
              <span className="text-[11px] font-medium text-slate-700 hidden sm:inline">
                {healthLoading
                  ? "Connecting..."
                  : health?.status === "healthy"
                  ? "API Online"
                  : "API Offline"}
              </span>
            </div>

            <a
              href={`${API_BASE_URL}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-slate-700 p-1.5 rounded-lg transition-colors"
              title="API Swagger Docs"
            >
              <ExternalLink className="h-4 w-4" />
            </a>
          </div>
        </div>

        {/* EXACTLY TWO MAIN TABS (Mobile-First) */}
        <div className="max-w-4xl mx-auto px-4 sm:px-6 pb-2.5">
          <div className="bg-slate-100 p-1 rounded-xl flex gap-1 border border-slate-200">
            <button
              type="button"
              onClick={() => setActiveTab("analyze")}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs sm:text-sm font-bold transition-all min-h-[44px] cursor-pointer ${
                activeTab === "analyze"
                  ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/50"
              }`}
            >
              <Cpu className={`h-4 w-4 ${activeTab === "analyze" ? "text-blue-600" : "text-slate-400"}`} />
              <span>1. Analyze Engine</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("models")}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs sm:text-sm font-bold transition-all min-h-[44px] cursor-pointer ${
                activeTab === "models"
                  ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-200/50"
              }`}
            >
              <BarChart3 className={`h-4 w-4 ${activeTab === "models" ? "text-blue-600" : "text-slate-400"}`} />
              <span>2. Model Comparison</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Tab Content Container */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-6">
        {activeTab === "analyze" ? <AnalyzeEngine /> : <ModelComparison />}
      </main>

      {/* Clean Technical Footer */}
      <footer className="border-t border-slate-200 bg-white py-5 px-4">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-center sm:text-left text-xs text-slate-500">
          <div>
            <strong>NASA-RUL</strong> — C-MAPSS FD001 Turbofan Prognostics
          </div>
          <div className="text-[11px] text-slate-400">
            PyTorch LSTM • Sequence Length: 30 • 19 Selected Sensor Channels
          </div>
        </div>
      </footer>
    </div>
  );
}
