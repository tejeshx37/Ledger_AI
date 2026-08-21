import type { AlertDetail } from "../types";
import { FileText, ExternalLink } from "lucide-react";


interface NarrativePanelProps {
  alert: AlertDetail | null;
  onHighlightNode: (nodeId: string) => void;
}

/**
 * Render narrative text where recognised account/tx IDs become clickable
 * hyperlinks that highlight the matching node in the graph viewer.
 */
function renderNarrative(
  text: string,
  nodeIds: Set<string>,
  onHighlight: (id: string) => void
) {
  // Build a regex from known node ids (longest-first to avoid partial matches)
  const sortedIds = [...nodeIds].sort((a, b) => b.length - a.length);
  if (!sortedIds.length) return <span className="text-slate-300">{text}</span>;

  const pattern = new RegExp(`(${sortedIds.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
  const parts = text.split(pattern);

  return (
    <>
      {parts.map((part, i) =>
        nodeIds.has(part) ? (
          <button
            key={i}
            onClick={() => onHighlight(part)}
            className="inline-flex items-center gap-0.5 font-mono text-sky-400 hover:text-sky-300
                       underline decoration-dotted underline-offset-2 transition-colors"
          >
            {part}
            <ExternalLink size={10} className="opacity-60" />
          </button>
        ) : (
          <span key={i} className="text-slate-300">{part}</span>
        )
      )}
    </>
  );
}

export default function NarrativePanel({ alert, onHighlightNode }: NarrativePanelProps) {
  const nodeIds = new Set(alert?.evidence?.nodes.map((n) => n.id) ?? []);

  return (
    <div className="card flex flex-col overflow-hidden">
      <div className="card-header">
        <FileText size={16} className="text-emerald-400" />
        <span className="text-sm font-semibold text-slate-200">Narrative Explanation</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!alert ? (
          <p className="text-slate-600 text-sm">Select an alert to view the narrative.</p>
        ) : !alert.narrative ? (
          <p className="text-slate-500 text-sm italic">No narrative available for this alert.</p>
        ) : (
          <div className="space-y-4 animate-fade-in">
            {/* Metadata pills */}
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="px-2 py-1 bg-slate-800 rounded-md text-slate-400">
                Alert: <span className="font-mono text-slate-200">{alert.alert_id}</span>
              </span>
              <span className="px-2 py-1 bg-slate-800 rounded-md text-slate-400">
                Account: <span className="font-mono text-slate-200">{alert.account_id}</span>
              </span>
              <span className="px-2 py-1 bg-slate-800 rounded-md text-slate-400">
                Score:{" "}
                <span
                  className="font-semibold"
                  style={{
                    color: alert.confidence_score > 0.8 ? "#f43f5e" : alert.confidence_score > 0.6 ? "#f59e0b" : "#34d399",
                  }}
                >
                  {(alert.confidence_score * 100).toFixed(1)}%
                </span>
              </span>
            </div>

            {/* Narrative text with clickable node IDs */}
            <p className="text-sm leading-relaxed">
              {renderNarrative(alert.narrative, nodeIds, onHighlightNode)}
            </p>

            {/* Evidence summary table */}
            {alert.evidence && (
              <div className="mt-4 border border-slate-800 rounded-lg overflow-hidden">
                <div className="px-3 py-2 bg-slate-800/60 text-xs font-semibold text-slate-400 uppercase tracking-wide">
                  Transaction Evidence
                </div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-800">
                      <th className="text-left px-3 py-2 text-slate-500 font-medium">From</th>
                      <th className="text-left px-3 py-2 text-slate-500 font-medium">To</th>
                      <th className="text-right px-3 py-2 text-slate-500 font-medium">Amount</th>
                      <th className="text-right px-3 py-2 text-slate-500 font-medium">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alert.evidence.edges.map((e) => (
                      <tr key={e.id} className="border-b border-slate-800/50 hover:bg-slate-800/40 transition-colors">
                        <td className="px-3 py-2 font-mono text-sky-400">
                          <button onClick={() => onHighlightNode(e.src)} className="hover:underline">{e.src}</button>
                        </td>
                        <td className="px-3 py-2 font-mono text-rose-400">
                          <button onClick={() => onHighlightNode(e.dst)} className="hover:underline">{e.dst}</button>
                        </td>
                        <td className="px-3 py-2 text-right text-emerald-400 font-mono">
                          ${e.amount.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-500 font-mono">
                          {new Date(e.timestamp).toLocaleTimeString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
