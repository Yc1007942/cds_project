import { useState, useCallback, useRef } from "react";
import * as api from "@/lib/api";
import type { AgentResponse } from "@/lib/api";
import { gameEvents } from "../game/SimulationScene";
import { AGENT_PERSONAS } from "../../../shared/agents";

export type SimStatus = "idle" | "running" | "complete";

export function useSimulation() {
  const [status, setStatus] = useState<SimStatus>("idle");
  const [responses, setResponses] = useState<AgentResponse[]>([]);
  const [currentAgentIndex, setCurrentAgentIndex] = useState(-1);
  const [post, setPost] = useState("");
  const [predictedScore, setPredictedScore] = useState<number | null>(null);
  const [agentsToSpawn, setAgentsToSpawn] = useState(0);
  const abortRef = useRef(false);

  const startSimulation = useCallback(async (userPost: string) => {
    if (!userPost.trim()) return;

    abortRef.current = false;
    setPost(userPost);
    setStatus("running");
    setResponses([]);
    setCurrentAgentIndex(-1);

    // Tell Phaser to show the post and start animation
    gameEvents.emit("show-post", { post: userPost });
    gameEvents.emit("start-simulation");

    // Wait for agents to arrive visually
    await new Promise((r) => setTimeout(r, 1500));

    try {
      // Get agent responses from FastAPI backend
      const result = await api.getAgentResponses(userPost);

      setPredictedScore(result.predictedScore);
      setAgentsToSpawn(result.agentsSpawned);

      // Process responses and emit events to Phaser
      for (let i = 0; i < result.responses.length; i++) {
        if (abortRef.current) break;

        const agentResponse = result.responses[i];
        setCurrentAgentIndex(i);

        // Tell Phaser this agent responded
        gameEvents.emit("agent-responded", {
          agentId: agentResponse.agentId,
          response: agentResponse.response,
        });

        setResponses((prev) => [...prev, agentResponse]);

        // Small delay between agents for visual effect
        await new Promise((r) => setTimeout(r, 500));
      }

      if (!abortRef.current) {
        setStatus("complete");
        setCurrentAgentIndex(-1);
      }
    } catch (error) {
      console.error("Error getting agent responses:", error);
      setStatus("idle");
    }
  }, []);

  const resetSimulation = useCallback(() => {
    abortRef.current = true;
    setStatus("idle");
    setResponses([]);
    setCurrentAgentIndex(-1);
    setPost("");
    setPredictedScore(null);
    setAgentsToSpawn(0);
    gameEvents.emit("reset-simulation");
  }, []);

  return {
    status,
    responses,
    currentAgentIndex,
    post,
    predictedScore,
    agentsToSpawn,
    agentCount: AGENT_PERSONAS.length,
    respondedCount: responses.length,
    startSimulation,
    resetSimulation,
  };
}
