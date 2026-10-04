// Typed client for the AegisMind FastAPI backend.

export type Zone = 'user' | 'dmz' | 'server' | 'data'

export interface GNode {
  id: string
  kind: string
  zone: Zone
  criticality: number
  exposure: number
  is_crown: boolean
  is_entry: boolean
  is_target: boolean
  path_rank: number | null
  reach_probability: number
  alerts_in: number
  alerts_out: number
  services: string[]
  // filled in by the force layout
  x?: number
  y?: number
  z?: number
  fy?: number
}

export interface GLink {
  source: string | GNode
  target: string | GNode
  service: string
  relation: string
  exploitability: number
  technique_id: string
  technique: string
  tactic: string
  alerts: number
  events: number
  path_rank: number | null
}

export interface PathStep {
  step: number
  src: string
  dst: string
  service: string
  relation: string
  technique_id: string
  technique: string
  tactic: string
  exploitability: number
  alerts: number
  events: number
  reasons: string[]
}

export interface PredictedPath {
  rank: number
  path: string[]
  probability: number
  summary: string
  steps: PathStep[]
}

export interface Analysis {
  scenario: { id: string; entry: string; entry_vector: string; target: string }
  detector: { recall: number; fpr: number }
  kpis: {
    threat_level: 'Low' | 'Guarded' | 'Elevated' | 'Critical'
    risk_score: number
    alerts: number
    alerted_connections: number
    affected_assets: number
    assets: number
    crown_jewels: string[]
    crown_jewels_reachable: number
    blast_radius: number
  }
  nodes: GNode[]
  links: GLink[]
  paths: PredictedPath[]
  ground_truth: { path: string[] }
}

export interface ActionSpec {
  kind: string
  src: string | null
  dst: string | null
  asset: string | null
  service: string | null
  description: string
  rollback: string
}

export interface ActionEval {
  action: ActionSpec
  risk_before: number
  risk_after: number
  risk_reduction: number
  service_disruption: number
  action_cost: number
  uncertainty: number
  utility: number | null
  feasible: boolean
  rejected_reason: string | null
  attack_techniques_mitigated: string[]
}

export interface Recommendation {
  actions: ActionEval[]
  rejected: ActionEval[]
  plan: ActionEval[]
  plan_summary: { risk_before: number; risk_after: number; risk_reduction: number; service_disruption: number; actions: number }
  preference_penalty: Record<string, number>
  candidates_considered: number
  protected_assets: string[]
  config: { lam: number; mu: number; nu: number }
}

export interface WhatIf {
  evaluation: ActionEval
  changed_edges: { source: string; target: string; service: string; change: 'blocked' | 'hardened' }[]
  new_top_paths: { path: string[]; probability: number }[]
}

export interface ScenarioInfo {
  id: string
  entry: string
  entry_vector: string
  target: string
  hops: number
  assets: number
  edges: number
}

export interface Metrics {
  precision: number
  recall: number
  f1: number
  pr_auc: number | null
  false_positive_rate: number | null
}

export interface DetectorReport {
  file: string
  dataset: string
  split: string
  date_utc: string
  models: {
    model: string
    kind: string
    strategies: Record<string, { metrics: Metrics; families: Record<string, { rows: number; flagged_rate: number; seen_in_train: boolean }> }>
  }[]
}

export interface Experiments {
  detectors: DetectorReport[]
  pathbench: null | {
    runs: { test_recall: number; test_fpr: number; results: Record<string, Record<string, number>> }[]
    train_alerts: { recall: number; fpr: number }
    feature_importance?: Record<string, { feature: string; importance: number }[]>
  }
  benchmark: null | { results: Record<string, Record<string, number>> }
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!r.ok) {
    let detail = r.statusText
    try {
      detail = (await r.json()).detail ?? detail
    } catch {
      /* keep status text */
    }
    throw new Error(`${r.status}: ${detail}`)
  }
  return r.json() as Promise<T>
}

export const api = {
  scenarios: () => req<ScenarioInfo[]>('/api/scenarios'),
  analysis: (id: string, recall: number, fpr: number) =>
    req<Analysis>(`/api/scenarios/${id}/analysis?recall=${recall}&fpr=${fpr}`),
  recommend: (id: string, body: { recall: number; fpr: number; protect: string[] }) =>
    req<Recommendation>(`/api/scenarios/${id}/recommend`, { method: 'POST', body: JSON.stringify(body) }),
  whatif: (id: string, action: ActionSpec, recall: number, fpr: number) =>
    req<WhatIf>(`/api/scenarios/${id}/whatif`, {
      method: 'POST',
      body: JSON.stringify({
        recall,
        fpr,
        action: { kind: action.kind, src: action.src, dst: action.dst, asset: action.asset, service: action.service },
      }),
    }),
  feedback: (scenario: string, action: ActionSpec, decision: 'approve' | 'reject') =>
    req<{ preference_penalty: Record<string, number> }>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        scenario,
        decision,
        description: action.description,
        action: { kind: action.kind, src: action.src, dst: action.dst, asset: action.asset, service: action.service },
      }),
    }),
  experiments: () => req<Experiments>('/api/experiments'),
}
