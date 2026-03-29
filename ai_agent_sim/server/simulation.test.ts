import { describe, expect, it, vi } from "vitest";
import { appRouter } from "./routers";
import { AGENT_PERSONAS } from "../shared/agents";
import type { TrpcContext } from "./_core/context";

// Mock the LLM module
vi.mock("./_core/llm", () => ({
  invokeLLM: vi.fn().mockResolvedValue({
    id: "test-id",
    created: Date.now(),
    model: "test-model",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: "This is a test response from the AI agent.",
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 20,
      total_tokens: 30,
    },
  }),
}));

function createPublicContext(): TrpcContext {
  return {
    user: null,
    req: {
      protocol: "https",
      headers: {},
    } as TrpcContext["req"],
    res: {
      clearCookie: () => {},
    } as TrpcContext["res"],
  };
}

describe("simulation.getAgentResponse", () => {
  it("returns a response for a valid agent", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);

    const result = await caller.simulation.getAgentResponse({
      agentId: "supporter",
      post: "AI is the future of software development",
      conversationHistory: "",
    });

    expect(result).toBeDefined();
    expect(result.agentId).toBe("supporter");
    expect(result.agentName).toBe("Enthusiastic Supporter");
    expect(typeof result.response).toBe("string");
    expect(result.response.length).toBeGreaterThan(0);
  });

  it("returns a response for each defined agent persona", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);

    for (const persona of AGENT_PERSONAS) {
      const result = await caller.simulation.getAgentResponse({
        agentId: persona.id,
        post: "Testing all agents",
        conversationHistory: "",
      });

      expect(result.agentId).toBe(persona.id);
      expect(result.agentName).toBe(persona.name);
      expect(result.response).toBeTruthy();
    }
  });

  it("includes conversation history in the request", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);

    const history =
      "Enthusiastic Supporter: Great idea!\nSkeptical Critic: I have doubts.";

    const result = await caller.simulation.getAgentResponse({
      agentId: "observer",
      post: "AI will change everything",
      conversationHistory: history,
    });

    expect(result.agentId).toBe("observer");
    expect(result.agentName).toBe("Neutral Observer");
    expect(result.response).toBeTruthy();
  });

  it("throws an error for an invalid agent ID", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);

    await expect(
      caller.simulation.getAgentResponse({
        agentId: "nonexistent-agent",
        post: "Test post",
        conversationHistory: "",
      }),
    ).rejects.toThrow("Agent not found: nonexistent-agent");
  });

  it("rejects empty post content", async () => {
    const ctx = createPublicContext();
    const caller = appRouter.createCaller(ctx);

    await expect(
      caller.simulation.getAgentResponse({
        agentId: "supporter",
        post: "",
        conversationHistory: "",
      }),
    ).rejects.toThrow();
  });
});

describe("shared/agents", () => {
  it("has exactly 4 agent personas defined", () => {
    expect(AGENT_PERSONAS).toHaveLength(4);
  });

  it("each persona has required fields", () => {
    for (const persona of AGENT_PERSONAS) {
      expect(persona.id).toBeTruthy();
      expect(persona.name).toBeTruthy();
      expect(typeof persona.color).toBe("number");
      expect(persona.accentColor).toMatch(/^#[0-9a-fA-F]{6}$/);
      expect(persona.persona).toBeTruthy();
      expect(persona.thinkingDelay).toBeGreaterThan(0);
    }
  });

  it("all agent IDs are unique", () => {
    const ids = AGENT_PERSONAS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
