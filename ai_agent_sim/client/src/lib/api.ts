/**
 * API client for FastAPI backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ─── Agent Simulation ──────────────────────────────────────────────
export interface AgentResponse {
  agentId: string;
  agentName: string;
  agentPersona: string;
  color: string;
  emoji: string;
  response: string;
}

export interface SimulationResult {
  predictedScore: number;
  agentsSpawned: number;
  responses: AgentResponse[];
}

export async function getAgentResponses(postText: string): Promise<SimulationResult> {
  const response = await fetch(`${API_BASE_URL}/simulation/get-responses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ postText }),
  });
  if (!response.ok) throw new Error(`Failed to get responses: ${response.statusText}`);
  return response.json();
}

export async function predictScore(postText: string) {
  const response = await fetch(`${API_BASE_URL}/simulation/predict-score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ postText }),
  });
  if (!response.ok) throw new Error(`Failed to predict score: ${response.statusText}`);
  return response.json();
}


// ─── Data Exploration ──────────────────────────────────────────────
export interface DataStats {
  total: number;
  ai_count: number;
  human_count: number;
  columns: string[];
  subreddits: string[];
}

export async function getDataStats(): Promise<DataStats> {
  const r = await fetch(`${API_BASE_URL}/data/stats`);
  if (!r.ok) throw new Error("Failed to get stats");
  return r.json();
}

export async function getFeatureList(): Promise<{ features: string[] }> {
  const r = await fetch(`${API_BASE_URL}/data/features`);
  if (!r.ok) throw new Error("Failed to get features");
  return r.json();
}

export interface ExploreParams {
  word_min?: number;
  word_max?: number;
  subreddits?: string;
  keyword?: string;
  limit?: number;
  offset?: number;
}

export async function getExploreData(params: ExploreParams) {
  const qs = new URLSearchParams();
  if (params.word_min !== undefined) qs.set("word_min", String(params.word_min));
  if (params.word_max !== undefined) qs.set("word_max", String(params.word_max));
  if (params.subreddits) qs.set("subreddits", params.subreddits);
  if (params.keyword) qs.set("keyword", params.keyword);
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));

  const r = await fetch(`${API_BASE_URL}/data/explore?${qs}`);
  if (!r.ok) throw new Error("Failed to get explore data");
  return r.json();
}

export interface ScatterPoint {
  [key: string]: number | string;
}

export async function getScatterData(x: string, y: string, sample?: number): Promise<{ points: ScatterPoint[]; x_col: string; y_col: string }> {
  const qs = new URLSearchParams({ x, y });
  if (sample) qs.set("sample", String(sample));
  const r = await fetch(`${API_BASE_URL}/data/scatter?${qs}`);
  if (!r.ok) throw new Error("Failed to get scatter data");
  return r.json();
}

export interface DistBin { center: number; human: number; ai: number; }
export interface BoxStats { min: number; q1: number; median: number; q3: number; max: number; mean: number; }

export async function getDistribution(feature: string, bins?: number): Promise<{
  feature: string;
  bins: DistBin[];
  box_human: BoxStats;
  box_ai: BoxStats;
}> {
  const qs = new URLSearchParams({ feature });
  if (bins) qs.set("bins", String(bins));
  const r = await fetch(`${API_BASE_URL}/data/distribution?${qs}`);
  if (!r.ok) throw new Error("Failed to get distribution");
  return r.json();
}

export async function getCorrelation(features: string[]): Promise<{ matrix: number[][]; labels: string[] }> {
  const r = await fetch(`${API_BASE_URL}/data/correlation?features=${features.join(",")}`);
  if (!r.ok) throw new Error("Failed to get correlation");
  return r.json();
}

export interface ClusterPoint {
  cluster: number;
  pc1: number;
  pc2: number;
  pc3?: number;
  author?: string;
  subreddit?: string;
}

export async function getClusters(sampleSize?: number, nClusters?: number, dimensions?: number): Promise<{
  points: ClusterPoint[];
  explained_variance: number;
  n_clusters: number;
  dimensions: number;
}> {
  const qs = new URLSearchParams();
  if (sampleSize) qs.set("sample_size", String(sampleSize));
  if (nClusters) qs.set("n_clusters", String(nClusters));
  if (dimensions) qs.set("dimensions", String(dimensions));
  const r = await fetch(`${API_BASE_URL}/data/clusters?${qs}`);
  if (!r.ok) throw new Error("Failed to get clusters");
  return r.json();
}

export async function getFeatureStats(): Promise<{
  human_means: Record<string, number>;
  ai_means: Record<string, number>;
  ranges: Record<string, { min: number; max: number }>;
  features: string[];
}> {
  const r = await fetch(`${API_BASE_URL}/data/feature-stats`);
  if (!r.ok) throw new Error("Failed to get feature stats");
  return r.json();
}

export interface FeatureImportance {
  Feature: string;
  Importance: number;
}

export async function getFeatureImportances(): Promise<{
  moltbook: FeatureImportance[];
  reddit: FeatureImportance[];
}> {
  const r = await fetch(`${API_BASE_URL}/data/feature-importances`);
  if (!r.ok) throw new Error("Failed to get feature importances");
  return r.json();
}


// ─── Inference ─────────────────────────────────────────────────────
export interface ClassificationResult {
  label: string;
  prediction: number;
  confidence: number;
  ai_prob: number;
  human_prob: number;
  word_count: number;
  char_count: number;
  features_used: number;
}

export async function classifyText(text: string): Promise<ClassificationResult> {
  const r = await fetch(`${API_BASE_URL}/inference/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("Classification failed");
  return r.json();
}

export interface RegressionResult {
  score: number;
  agents_to_spawn: number;
  word_count: number;
  char_count: number;
}

export async function predictEngagement(text: string): Promise<RegressionResult> {
  const r = await fetch(`${API_BASE_URL}/inference/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("Regression failed");
  return r.json();
}

export async function getModelInfo() {
  const r = await fetch(`${API_BASE_URL}/inference/model-info`);
  if (!r.ok) throw new Error("Failed to get model info");
  return r.json();
}

export async function extractFeatures(text: string): Promise<{ features: Record<string, number> }> {
  const r = await fetch(`${API_BASE_URL}/inference/extract-features`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("Feature extraction failed");
  return r.json();
}
