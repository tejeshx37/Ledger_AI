// Type definitions for LEDGER AML Platform

export type AlertStatus = "open" | "confirmed" | "dismissed";
export type UserRole = "analyst" | "supervisor" | "admin";

export interface AlertSummary {
  alert_id: string;
  account_id: string;
  confidence_score: number;
  status: AlertStatus;
  model_version: string;
  created_at: string;
}

export interface EvidenceNode {
  id: string;
  role: "source" | "mule" | "aggregator" | "destination";
  bank_id?: string;
  features?: Record<string, number>;
}

export interface EvidenceEdge {
  id: string;
  src: string;
  dst: string;
  amount: number;
  timestamp: string;
}

export interface Evidence {
  nodes: EvidenceNode[];
  edges: EvidenceEdge[];
}

export interface AlertDetail extends AlertSummary {
  run_id: string;
  evidence: Evidence | null;
  narrative: string | null;
}

export interface Review {
  decision: "confirm" | "dismiss";
  reason: string;
}

export interface Model {
  model_version: string;
  architecture: string;
  metrics: Record<string, number> | null;
  created_at: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
}
