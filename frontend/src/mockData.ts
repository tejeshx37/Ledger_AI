// Mock data for offline/fallback mode
import type { AlertDetail, AlertSummary, Model } from "./types";

export const MOCK_ALERTS: AlertSummary[] = [
  {
    alert_id: "alert_1_A2_ring01",
    account_id: "1_A2",
    confidence_score: 0.93,
    status: "open",
    model_version: "graphsage-v1.0",
    created_at: "2022-09-01T03:12:00",
  },
  {
    alert_id: "alert_2_B1_ring02",
    account_id: "2_B1",
    confidence_score: 0.81,
    status: "open",
    model_version: "graphsage-v1.0",
    created_at: "2022-09-01T04:30:00",
  },
  {
    alert_id: "alert_3_C4_ring03",
    account_id: "3_C4",
    confidence_score: 0.77,
    status: "confirmed",
    model_version: "graphsage-v1.0",
    created_at: "2022-09-01T05:10:00",
  },
  {
    alert_id: "alert_4_D9_ring04",
    account_id: "4_D9",
    confidence_score: 0.64,
    status: "dismissed",
    model_version: "graphsage-v1.0",
    created_at: "2022-09-01T06:00:00",
  },
  {
    alert_id: "alert_5_E3_ring05",
    account_id: "5_E3",
    confidence_score: 0.58,
    status: "open",
    model_version: "graphsage-v1.0",
    created_at: "2022-09-01T06:45:00",
  },
];

export const MOCK_ALERT_DETAIL: AlertDetail = {
  alert_id: "alert_1_A2_ring01",
  account_id: "1_A2",
  confidence_score: 0.93,
  status: "open",
  model_version: "graphsage-v1.0",
  run_id: "test_explain_run",
  created_at: "2022-09-01T03:12:00",
  narrative:
    "Alert alert_1_A2_ring01 generated for account 1_A2 with confidence score 0.93. " +
    "The suspicious activity occurred within the window 2022-09-01T02:00:00 to 2022-09-01T03:12:00. " +
    "Contributing accounts involved: 1_A2, mule_tx_001, 1_B9. " +
    "The primary model feature attributions driving this alert are: temporal_tx_count (35.21%), " +
    "temporal_std_inter_arrival_seconds (20.15%), graph_pagerank (19.22%), " +
    "temporal_rolling_amount_mean (17.47%), temporal_burstiness (7.95%).",
  evidence: {
    nodes: [
      { id: "1_A2", role: "source", bank_id: "bank_us_01" },
      { id: "mule_tx_001", role: "mule", bank_id: "bank_offshore_03" },
      { id: "1_B9", role: "aggregator", bank_id: "bank_us_01" },
      { id: "shell_corp_Z", role: "destination", bank_id: "bank_cayman_99" },
    ],
    edges: [
      { id: "e1", src: "1_A2", dst: "mule_tx_001", amount: 49800, timestamp: "2022-09-01T02:01:00" },
      { id: "e2", src: "1_A2", dst: "mule_tx_001", amount: 49900, timestamp: "2022-09-01T02:15:00" },
      { id: "e3", src: "mule_tx_001", dst: "1_B9", amount: 99200, timestamp: "2022-09-01T02:45:00" },
      { id: "e4", src: "1_B9", dst: "shell_corp_Z", amount: 98500, timestamp: "2022-09-01T03:10:00" },
    ],
  },
};

export const MOCK_MODELS: Model[] = [
  {
    model_version: "graphsage-v1.0",
    architecture: "GraphSAGE",
    metrics: { roc_auc: 0.9231, precision: 0.8847, recall: 0.8012 },
    created_at: "2022-09-01T00:00:00",
  },
];
