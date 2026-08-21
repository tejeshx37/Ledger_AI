// API client with automatic mock fallback when backend is unreachable
import type { AlertDetail, AlertSummary, AuthToken, Model, Review } from "./types";
import {
  MOCK_ALERT_DETAIL,
  MOCK_ALERTS,
  MOCK_MODELS,
} from "./mockData";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

let authToken: string | null = null;

export function setToken(token: string) {
  authToken = token;
}

export function clearToken() {
  authToken = null;
}

export function getToken() {
  return authToken;
}

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<{ data: T; offline: boolean }> {
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
      ...(options.headers as Record<string, string>),
    };
    const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = (await res.json()) as T;
    return { data, offline: false };
  } catch {
    return { data: null as unknown as T, offline: true };
  }
}

export async function login(
  role: "analyst" | "supervisor" | "admin"
): Promise<AuthToken | null> {
  const body = new URLSearchParams({ username: role, password: "password" });
  try {
    const res = await fetch(`${BASE_URL}/v1/token`, {
      method: "POST",
      body,
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    if (!res.ok) return null;
    return (await res.json()) as AuthToken;
  } catch {
    // Mock token with role embedded
    return { access_token: `mock_token_${role}`, token_type: "bearer" };
  }
}

export async function fetchAlerts(
  minScore = 0,
  status?: string
): Promise<{ alerts: AlertSummary[]; offline: boolean }> {
  let path = `/v1/alerts?min_score=${minScore}&limit=50`;
  if (status) path += `&status_filter=${status}`;
  const { data, offline } = await apiFetch<AlertSummary[]>(path);
  return { alerts: offline ? MOCK_ALERTS : data, offline };
}

export async function fetchAlertDetail(
  alertId: string
): Promise<{ alert: AlertDetail; offline: boolean }> {
  const { data, offline } = await apiFetch<AlertDetail>(`/v1/alerts/${alertId}`);
  return {
    alert: offline
      ? { ...MOCK_ALERT_DETAIL, alert_id: alertId }
      : data,
    offline,
  };
}

export async function submitReview(
  alertId: string,
  review: Review
): Promise<{ success: boolean; offline: boolean }> {
  const { data, offline } = await apiFetch<{ status: string }>(
    `/v1/alerts/${alertId}/review`,
    { method: "POST", body: JSON.stringify(review) }
  );
  return { success: offline ? true : data?.status === "success", offline };
}

export async function fetchModels(): Promise<{ models: Model[]; offline: boolean }> {
  const { data, offline } = await apiFetch<Model[]>("/v1/models");
  return { models: offline ? MOCK_MODELS : data, offline };
}
