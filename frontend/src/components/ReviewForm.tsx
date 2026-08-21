import { useState } from "react";
import type { AlertDetail, UserRole } from "../types";
import { submitReview } from "../api";
import { CheckCircle, XCircle, AlertTriangle, Loader2 } from "lucide-react";

interface ReviewFormProps {
  alert: AlertDetail | null;
  role: UserRole;
  onReviewSubmitted: (alertId: string, decision: "confirm" | "dismiss") => void;
}

export default function ReviewForm({ alert, role, onReviewSubmitted }: ReviewFormProps) {
  const [decision, setDecision] = useState<"confirm" | "dismiss" | null>(null);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<"success" | "error" | null>(null);

  const canConfirm = role === "supervisor" || role === "admin";
  const alreadyReviewed = alert?.status !== "open";

  async function handleSubmit() {
    if (!alert || !decision || reason.trim().length < 5) return;

    setSubmitting(true);
    setResult(null);
    try {
      const { success } = await submitReview(alert.alert_id, {
        decision,
        reason: reason.trim(),
      });
      if (success) {
        setResult("success");
        onReviewSubmitted(alert.alert_id, decision);
        setDecision(null);
        setReason("");
      } else {
        setResult("error");
      }
    } catch {
      setResult("error");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card flex flex-col gap-4 p-4 animate-slide-in">
      <div className="flex items-center gap-2">
        <AlertTriangle size={16} className="text-amber-400" />
        <span className="text-sm font-semibold text-slate-200">Submit Review Decision</span>
      </div>

      {!alert ? (
        <p className="text-slate-600 text-sm">Select an alert to review.</p>
      ) : alreadyReviewed ? (
        <div className={`flex items-center gap-2 text-sm px-3 py-2 rounded-lg border ${
          alert.status === "confirmed"
            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
            : "bg-slate-500/10 border-slate-500/30 text-slate-400"
        }`}>
          {alert.status === "confirmed" ? <CheckCircle size={14} /> : <XCircle size={14} />}
          This alert has already been <strong className="ml-1 capitalize">{alert.status}</strong>.
        </div>
      ) : (
        <>
          {/* Decision buttons */}
          <div className="grid grid-cols-2 gap-2">
            <button
              disabled={!canConfirm}
              onClick={() => setDecision("confirm")}
              className={`flex items-center justify-center gap-2 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                decision === "confirm"
                  ? "bg-emerald-600 border-emerald-500 text-white shadow-lg shadow-emerald-900/30"
                  : canConfirm
                  ? "bg-slate-800 border-slate-700 text-slate-300 hover:border-emerald-500/50 hover:text-emerald-400"
                  : "bg-slate-900 border-slate-800 text-slate-600 cursor-not-allowed"
              }`}
              title={!canConfirm ? "Only supervisors can confirm alerts" : undefined}
            >
              <CheckCircle size={15} />
              Confirm SAR
            </button>
            <button
              onClick={() => setDecision("dismiss")}
              className={`flex items-center justify-center gap-2 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                decision === "dismiss"
                  ? "bg-rose-600 border-rose-500 text-white shadow-lg shadow-rose-900/30"
                  : "bg-slate-800 border-slate-700 text-slate-300 hover:border-rose-500/50 hover:text-rose-400"
              }`}
            >
              <XCircle size={15} />
              Dismiss
            </button>
          </div>

          {!canConfirm && (
            <p className="text-xs text-amber-500/80 flex items-center gap-1">
              <AlertTriangle size={11} /> Analysts cannot confirm alerts — supervisor role required.
            </p>
          )}

          {/* Reason textarea */}
          <div>
            <label className="block text-xs text-slate-400 mb-1.5">
              Justification <span className="text-slate-600">(min 5 characters)</span>
            </label>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={3}
              className="input resize-none"
              placeholder="Provide your rationale for this decision..."
              disabled={!decision}
            />
          </div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!decision || reason.trim().length < 5 || submitting}
            className="btn-primary w-full justify-center"
          >
            {submitting ? (
              <><Loader2 size={14} className="animate-spin" /> Submitting…</>
            ) : (
              `Submit ${decision === "confirm" ? "Confirmation" : decision === "dismiss" ? "Dismissal" : "Decision"}`
            )}
          </button>

          {result === "success" && (
            <div className="flex items-center gap-2 px-3 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-xs">
              <CheckCircle size={13} /> Review submitted successfully.
            </div>
          )}
          {result === "error" && (
            <div className="flex items-center gap-2 px-3 py-2 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-xs">
              <XCircle size={13} /> Failed to submit review. Please try again.
            </div>
          )}
        </>
      )}
    </div>
  );
}
