import { useCallback, useEffect, useState } from "react";
import type { AlertDetail, AlertSummary, UserRole } from "../types";
import { fetchAlertDetail, fetchAlerts, login, setToken } from "../api";
import AlertQueue from "./AlertQueue";
import GraphViewer from "./GraphViewer";
import NarrativePanel from "./NarrativePanel";
import ReviewForm from "./ReviewForm";
import {
  Shield,
  Activity,
  WifiOff,
  User,
  ChevronDown,
  BarChart3,
} from "lucide-react";

const ROLES: UserRole[] = ["analyst", "supervisor", "admin"];

export default function Dashboard() {
  const [role, setRole] = useState<UserRole>("analyst");
  const [offline, setOffline] = useState(false);
  const [alerts, setAlerts] = useState<AlertSummary[]>([]);
  const [loadingAlerts, setLoadingAlerts] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState<AlertDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [highlightedNode, setHighlightedNode] = useState<string | null>(null);

  // Login whenever role changes
  useEffect(() => {
    (async () => {
      const token = await login(role);
      if (token) setToken(token.access_token);
    })();
  }, [role]);

  // Fetch alert list
  const refreshAlerts = useCallback(async () => {
    setLoadingAlerts(true);
    const { alerts: data, offline: off } = await fetchAlerts();
    setAlerts(data);
    setOffline(off);
    setLoadingAlerts(false);
  }, []);

  useEffect(() => { refreshAlerts(); }, [refreshAlerts]);

  const handleSelectAlert = async (alert: AlertSummary) => {
    setLoadingDetail(true);
    setHighlightedNode(null);
    const { alert: detail, offline: off } = await fetchAlertDetail(alert.alert_id);
    setSelectedAlert(detail);
    setOffline(off);
    setLoadingDetail(false);
  };

  const handleReviewSubmitted = (alertId: string, decision: "confirm" | "dismiss") => {
    // Optimistically update local state
    setAlerts((prev) =>
      prev.map((a) =>
        a.alert_id === alertId
          ? { ...a, status: decision === "confirm" ? "confirmed" : "dismissed" }
          : a
      )
    );
    if (selectedAlert?.alert_id === alertId) {
      setSelectedAlert((prev) =>
        prev ? { ...prev, status: decision === "confirm" ? "confirmed" : "dismissed" } : prev
      );
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* ─── Top Nav Bar ─── */}
      <header className="flex items-center justify-between px-6 py-3 bg-slate-900 border-b border-slate-800 z-10 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-emerald-400">
            <Shield size={22} strokeWidth={2} />
            <span className="font-bold text-lg tracking-tight text-white">LEDGER</span>
          </div>
          <span className="text-slate-600 text-sm">AML Compliance Platform</span>
        </div>

        <div className="flex items-center gap-4">
          {offline && (
            <div className="offline-banner">
              <WifiOff size={12} />
              <span>Offline — using mock data</span>
            </div>
          )}

          {/* Live activity indicator */}
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <Activity size={12} className="text-emerald-400 animate-pulse-slow" />
            <span>Live</span>
          </div>

          {/* Role selector */}
          <div className="relative">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700 cursor-pointer">
              <User size={14} className="text-slate-400" />
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
                className="bg-transparent text-sm text-slate-200 outline-none cursor-pointer appearance-none pr-4"
              >
                {ROLES.map((r) => (
                  <option key={r} value={r} className="bg-slate-800">
                    {r.charAt(0).toUpperCase() + r.slice(1)}
                  </option>
                ))}
              </select>
              <ChevronDown size={12} className="text-slate-500 pointer-events-none absolute right-2" />
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700 text-xs text-slate-400">
            <BarChart3 size={13} className="text-violet-400" />
            <span>
              <span className="text-rose-400 font-semibold">
                {alerts.filter((a) => a.status === "open").length}
              </span>{" "}
              open alerts
            </span>
          </div>
        </div>
      </header>

      {/* ─── Main 3-Column Layout ─── */}
      <div className="flex flex-1 min-h-0 gap-0 overflow-hidden">
        {/* LEFT: Alert Queue */}
        <aside className="w-72 flex-shrink-0 flex flex-col border-r border-slate-800 overflow-hidden">
          <AlertQueue
            alerts={alerts}
            selectedId={selectedAlert?.alert_id ?? null}
            onSelect={handleSelectAlert}
            loading={loadingAlerts}
          />
        </aside>

        {/* CENTER: Graph + Timeline */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden p-3 gap-3">
          <div className="flex-1 min-h-0">
            {loadingDetail ? (
              <div className="card h-full flex items-center justify-center">
                <div className="flex flex-col items-center gap-3 text-slate-600">
                  <div className="w-8 h-8 border-2 border-slate-700 border-t-emerald-500 rounded-full animate-spin" />
                  <span className="text-sm">Loading evidence…</span>
                </div>
              </div>
            ) : (
              <GraphViewer
                evidence={selectedAlert?.evidence ?? null}
                highlightedNodeId={highlightedNode}
                onNodeClick={(nodeId) => setHighlightedNode(nodeId)}
              />
            )}
          </div>

          {/* Alert metadata strip */}
          {selectedAlert && (
            <div className="flex-shrink-0 card px-4 py-2.5 flex items-center gap-6 text-xs animate-fade-in">
              <div>
                <span className="text-slate-500 mr-2">Alert ID</span>
                <span className="font-mono text-slate-300">{selectedAlert.alert_id}</span>
              </div>
              <div>
                <span className="text-slate-500 mr-2">Model</span>
                <span className="font-mono text-violet-400">{selectedAlert.model_version}</span>
              </div>
              <div>
                <span className="text-slate-500 mr-2">Run</span>
                <span className="font-mono text-slate-400">{selectedAlert.run_id}</span>
              </div>
              <div className="ml-auto">
                <span className="text-slate-500 mr-2">Created</span>
                <span className="text-slate-400">
                  {new Date(selectedAlert.created_at).toLocaleString()}
                </span>
              </div>
            </div>
          )}
        </main>

        {/* RIGHT: Narrative + Review */}
        <aside className="w-96 flex-shrink-0 flex flex-col border-l border-slate-800 overflow-y-auto gap-0">
          <div className="flex-1 min-h-0 overflow-y-auto p-3 flex flex-col gap-3">
            <NarrativePanel
              alert={selectedAlert}
              onHighlightNode={(id) => setHighlightedNode(id)}
            />
            <ReviewForm
              alert={selectedAlert}
              role={role}
              onReviewSubmitted={handleReviewSubmitted}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}
