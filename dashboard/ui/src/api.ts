// src/api.ts
// All fetch calls to the FastAPI backend. Polling every 3 seconds.

export interface FLRound {
  round: number;
  accuracy: number;
  f1: number;
  precision?: number;
  recall?: number;
  attack_precision?: number;
  attack_recall?: number;
  attack_f1?: number;
  clients: number;
  timestamp: string;
}

export interface FLMetrics {
  rounds: FLRound[];
}

export interface BaselineMetrics {
  final_metrics?: {
    accuracy: number;
    f1_weighted: number;
    precision_weighted: number;
    recall_weighted: number;
    attack_recall?: number;
    attack_precision?: number;
    attack_f1?: number;
    confusion_matrix: number[][];
    per_class_recall: number[];
    n_samples: number;
  };
  history?: {
    train_loss: number[];
    train_acc: number[];
    val_loss: number[];
    val_acc: number[];
  };
  class_names?: Record<string, string>;
}

export interface TwinEvent {
  substation?: string;
  substation_id?: string;
  true_class?: string;
  predicted_class?: string;
  traffic_type?: string;
  risk?: number;
  anomaly_score?: number;
  flagged?: boolean;
  was_injected?: boolean;
  tick?: number;
  severity?: string;
  src_bytes?: number;
  dst_bytes?: number;
  service?: string;
  timestamp?: string;
  [key: string]: unknown;
}

export interface NodeInfo {
  id: string;
  location: string;
  type: string;
  status: 'online' | 'offline' | 'unknown';
  accuracy: number | null;
  f1: number | null;
  last_seen: string | null;
}

export interface NodeStatus {
  nodes: NodeInfo[];
  fl_rounds_done: number;
}

const BASE = '/api';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  flMetrics:   () => get<FLMetrics>('/fl-metrics'),
  baseline:    () => get<BaselineMetrics>('/baseline'),
  twinEvents:  (limit = 2000) => get<TwinEvent[]>(`/twin-events?limit=${limit}`),
  nodeStatus:  () => get<NodeStatus>('/node-status'),
  health:      () => get<{ status: string; time: string }>('/health'),
};
