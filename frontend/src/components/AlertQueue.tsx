import { useState } from "react";
import type { AlertStatus, AlertSummary } from "../types";
import { ShieldAlert, ChevronDown, ChevronUp } from "lucide-react";

interface AlertQueueProps {
  alerts: AlertSummary[];
  selectedId: string | null;
  onSelect: (alert: AlertSummary) => void;
  loading: boolean;
}

type SortKey = "confidence_score" | "created_at";

function ScoreBar({ score }: { score: number }) {
  const color =
    score > 0.8 ? "bg-rose-500" : score > 0.6 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="score-bar w-16">
      <div className={`score-fill ${color}`} style={{ width: `${score * 100}%` }} />
    </div>
  );
}

function StatusBadge({ status }: { status: AlertStatus }) {
  const cls =
    status === "open"
      ? "badge-open"
      : status === "confirmed"
      ? "badge-confirmed"
      : "badge-dismissed";
  return <span className={cls}>{status}</span>;
}

export default function AlertQueue({
  alerts,
  selectedId,
  onSelect,
  loading,
}: AlertQueueProps) {
  const [sortKey, setSortKey] = useState<SortKey>("confidence_score");
  const [sortAsc, setSortAsc] = useState(false);
  const [statusFilter, setStatusFilter] = useState<AlertStatus | "all">("all");

  const filtered = alerts
    .filter((a) => statusFilter === "all" || a.status === statusFilter)
    .sort((a, b) => {
      const va = sortKey === "confidence_score" ? a.confidence_score : new Date(a.created_at).getTime();
      const vb = sortKey === "confidence_score" ? b.confidence_score : new Date(b.created_at).getTime();
      return sortAsc ? va - vb : vb - va;
    });

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc((v) => !v);
    else { setSortKey(key); setSortAsc(false); }
  };

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? (sortAsc ? <ChevronUp size={11} /> : <ChevronDown size={11} />) : null;

  return (
    <div className="card flex flex-col h-full overflow-hidden">
      <div className="card-header justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert size={16} className="text-rose-400" />
          <span className="text-sm font-semibold text-slate-200">Alert Queue</span>
          <span className="badge bg-rose-500/15 text-rose-400 border border-rose-500/20">{alerts.length}</span>
        </div>
      </div>

      {/* Filters */}
      <div className="px-3 py-2 border-b border-slate-800 flex gap-2">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as AlertStatus | "all")}
          className="select flex-1 text-xs py-1.5"
        >
          <option value="all">All statuses</option>
          <option value="open">Open</option>
          <option value="confirmed">Confirmed</option>
          <option value="dismissed">Dismissed</option>
        </select>
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-[1fr_auto_auto] gap-2 px-3 py-1.5 border-b border-slate-800 text-xs text-slate-500">
        <span>Account</span>
        <button
          onClick={() => toggleSort("confidence_score")}
          className="flex items-center gap-1 hover:text-slate-300 transition-colors"
        >
          Score <SortIcon k="confidence_score" />
        </button>
        <button
          onClick={() => toggleSort("created_at")}
          className="flex items-center gap-1 hover:text-slate-300 transition-colors"
        >
          Time <SortIcon k="created_at" />
        </button>
      </div>

      {/* Alert rows */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="px-3 py-3 animate-pulse flex gap-3">
              <div className="flex-1 space-y-1.5">
                <div className="h-3 bg-slate-800 rounded w-2/3" />
                <div className="h-2.5 bg-slate-800 rounded w-1/2" />
              </div>
            </div>
          ))
        ) : filtered.length === 0 ? (
          <div className="px-4 py-8 text-center text-slate-600 text-sm">No alerts match filters.</div>
        ) : (
          filtered.map((a) => (
            <button
              key={a.alert_id}
              onClick={() => onSelect(a)}
              className={`w-full text-left px-3 py-3 transition-all duration-150 hover:bg-slate-800/60 ${
                selectedId === a.alert_id ? "bg-slate-800 border-l-2 border-emerald-500" : "border-l-2 border-transparent"
              }`}
            >
              <div className="grid grid-cols-[1fr_auto_auto] gap-2 items-center">
                <div className="min-w-0">
                  <div className="font-mono text-xs text-sky-400 truncate">{a.account_id}</div>
                  <div className="flex items-center gap-1.5 mt-1">
                    <StatusBadge status={a.status} />
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span
                    className="text-xs font-semibold tabular-nums"
                    style={{
                      color:
                        a.confidence_score > 0.8
                          ? "#f43f5e"
                          : a.confidence_score > 0.6
                          ? "#f59e0b"
                          : "#34d399",
                    }}
                  >
                    {(a.confidence_score * 100).toFixed(0)}%
                  </span>
                  <ScoreBar score={a.confidence_score} />
                </div>
                <div className="text-xs text-slate-600 text-right font-mono whitespace-nowrap">
                  {new Date(a.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
