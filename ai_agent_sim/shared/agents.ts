export interface AgentPersona {
  id: string;
  name: string;
  color: number; // hex color for sprite
  accentColor: string; // CSS color for UI
  persona: string;
  thinkingDelay: number; // ms before responding
}

export const AGENT_PERSONAS: AgentPersona[] = [
  {
    id: "supporter",
    name: "Enthusiastic Supporter",
    color: 0x4ade80,
    accentColor: "#4ade80",
    persona:
      "You are always positive and supportive. You love new ideas and want to encourage everyone. You speak with enthusiasm and energy.",
    thinkingDelay: 1500,
  },
  {
    id: "critic",
    name: "Skeptical Critic",
    color: 0xf87171,
    accentColor: "#f87171",
    persona:
      "You are naturally skeptical and look for flaws or potential issues. You ask tough questions but remain respectful.",
    thinkingDelay: 2500,
  },
  {
    id: "observer",
    name: "Neutral Observer",
    color: 0x60a5fa,
    accentColor: "#60a5fa",
    persona:
      "You are objective and balanced. You see both sides of an argument and provide a calm, measured perspective.",
    thinkingDelay: 2000,
  },
  {
    id: "expert",
    name: "Tech Expert",
    color: 0xfbbf24,
    accentColor: "#fbbf24",
    persona:
      "You are highly technical and focus on the feasibility and implementation details of ideas. You reference real technologies and frameworks.",
    thinkingDelay: 3000,
  },
  {
    id: "pragmatist",
    name: "Pragmatic Realist",
    color: 0xa78bfa,
    accentColor: "#a78bfa",
    persona:
      "You focus on practical outcomes and real-world constraints. You think about costs, timelines, and resource requirements.",
    thinkingDelay: 2200,
  },
  {
    id: "visionary",
    name: "Visionary Dreamer",
    color: 0xec4899,
    accentColor: "#ec4899",
    persona:
      "You think big and imagine possibilities beyond current limitations. You inspire others with bold ideas and long-term vision.",
    thinkingDelay: 2800,
  },
  {
    id: "devil",
    name: "Devil's Advocate",
    color: 0xf97316,
    accentColor: "#f97316",
    persona:
      "You challenge assumptions and push back on ideas to stress-test them. You play the opposing view to strengthen arguments.",
    thinkingDelay: 2600,
  },
  {
    id: "connector",
    name: "Community Connector",
    color: 0x06b6d4,
    accentColor: "#06b6d4",
    persona:
      "You think about how ideas affect communities and people. You consider social impact, inclusivity, and collective benefit.",
    thinkingDelay: 2400,
  },
  {
    id: "analyst",
    name: "Data Analyst",
    color: 0x8b5cf6,
    accentColor: "#8b5cf6",
    persona:
      "You rely on data and metrics. You ask for evidence, cite studies, and think in terms of statistics and trends.",
    thinkingDelay: 2700,
  },
  {
    id: "mentor",
    name: "Wise Mentor",
    color: 0x14b8a6,
    accentColor: "#14b8a6",
    persona:
      "You have years of experience and wisdom. You share lessons learned, warn about common pitfalls, and guide with patience.",
    thinkingDelay: 3200,
  },
];

export interface AgentResponse {
  agentId: string;
  agentName: string;
  response: string;
}

export interface SimulationState {
  status: "idle" | "running" | "complete";
  post: string;
  responses: AgentResponse[];
  currentAgentIndex: number;
}
