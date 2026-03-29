/**
 * API client for FastAPI backend
 * Replaces tRPC calls with direct HTTP requests
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

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

export interface ScorePrediction {
  score: number;
  agentsToSpawn: number;
  maxAgents: number;
}

/**
 * Predict engagement score for a post
 */
export async function predictScore(postText: string): Promise<ScorePrediction> {
  const response = await fetch(`${API_BASE_URL}/simulation/predict-score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ postText }),
  });

  if (!response.ok) {
    throw new Error(`Failed to predict score: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Start a new simulation
 */
export async function startSimulation(postText: string, userId: number = 1) {
  const response = await fetch(`${API_BASE_URL}/simulation/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ postText, userId }),
  });

  if (!response.ok) {
    throw new Error(`Failed to start simulation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get AI agent responses for a post
 */
export async function getAgentResponses(
  postText: string,
  userId: number = 1
): Promise<SimulationResult> {
  const response = await fetch(`${API_BASE_URL}/simulation/get-responses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ postText, userId }),
  });

  if (!response.ok) {
    throw new Error(`Failed to get responses: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get a specific simulation
 */
export async function getSimulation(simulationId: number) {
  const response = await fetch(
    `${API_BASE_URL}/simulation/simulations/${simulationId}`
  );

  if (!response.ok) {
    throw new Error(`Failed to get simulation: ${response.statusText}`);
  }

  return response.json();
}

/**
 * List all simulations for a user
 */
export async function listSimulations(userId: number = 1) {
  const response = await fetch(
    `${API_BASE_URL}/simulation/simulations?user_id=${userId}`
  );

  if (!response.ok) {
    throw new Error(`Failed to list simulations: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Login user
 */
export async function login(openId: string, name?: string, email?: string) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ openId, name, email, loginMethod: "oauth" }),
  });

  if (!response.ok) {
    throw new Error(`Failed to login: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Logout user
 */
export async function logout() {
  const response = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new Error(`Failed to logout: ${response.statusText}`);
  }

  return response.json();
}
