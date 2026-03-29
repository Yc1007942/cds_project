import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Play,
  RotateCcw,
  Send,
  Users,
  MessageSquare,
  Zap,
  Loader2,
} from "lucide-react";
import PhaserGame from "@/game/PhaserGame";
import { useSimulation } from "@/hooks/useSimulation";
import { AGENT_PERSONAS } from "../../../shared/agents";

export default function Home() {
  const [inputText, setInputText] = useState("");
  const {
    status,
    responses,
    currentAgentIndex,
    agentCount,
    respondedCount,
    predictedScore,
    agentsToSpawn,
    startSimulation,
    resetSimulation,
  } = useSimulation();

  const handleStart = () => {
    if (!inputText.trim() || status === "running") return;
    startSimulation(inputText.trim());
  };

  const handleReset = () => {
    resetSimulation();
    setInputText("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleStart();
    }
  };

  const progressPercent =
    status === "idle" ? 0 : (respondedCount / agentCount) * 100;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm">
        <div className="container flex items-center justify-between h-14 px-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
              <Zap className="w-4 h-4 text-primary" />
            </div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">
              Agent Arena
            </h1>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="w-4 h-4" />
            <span>{agentCount} agents online</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container px-4 py-4 flex flex-col lg:flex-row gap-4 overflow-hidden">
        {/* Left Panel - Simulation Canvas */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {/* Canvas */}
          <Card className="flex-1 min-h-[350px] lg:min-h-0 overflow-hidden border-border/50 bg-card/30 p-0">
            <PhaserGame className="w-full h-full" />
          </Card>

          {/* Input Area */}
          <Card className="border-border/50 bg-card/50 p-4">
            <div className="flex gap-3">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your post here... What do you want the agents to discuss?"
                className="flex-1 bg-input/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground resize-none focus:outline-none focus:ring-2 focus:ring-ring/50 transition-all"
                rows={2}
                disabled={status === "running"}
              />
              <div className="flex flex-col gap-2">
                {status === "idle" || status === "complete" ? (
                  <Button
                    onClick={handleStart}
                    disabled={!inputText.trim()}
                    className="h-full px-6"
                    size="lg"
                  >
                    {status === "complete" ? (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Retry
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        Post
                      </>
                    )}
                  </Button>
                ) : (
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="h-full px-6 bg-transparent"
                    size="lg"
                  >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Stop
                  </Button>
                )}
              </div>
            </div>
          </Card>
        </div>

        {/* Right Panel - Conversation Log & Metrics */}
        <div className="w-full lg:w-[380px] flex flex-col gap-4">
          {/* Engagement Metrics */}
          <Card className="border-border/50 bg-card/50 p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-primary" />
                Engagement
              </h2>
              <span className="text-xs text-muted-foreground">
                {status === "idle"
                  ? "Waiting for post"
                  : status === "running"
                    ? "Simulation running..."
                    : "Simulation complete"}
              </span>
            </div>
            <Progress value={progressPercent} className="h-2 mb-3" />
            <div className="grid grid-cols-3 gap-3">
              <MetricCard
                label="Responded"
                value={`${respondedCount}/${agentCount}`}
                color="text-primary"
              />
              <MetricCard
                label="Status"
                value={
                  status === "idle"
                    ? "Idle"
                    : status === "running"
                      ? "Live"
                      : "Done"
                }
                color={
                  status === "running" ? "text-yellow-400" : "text-muted-foreground"
                }
              />
              <MetricCard
                label="Messages"
                value={`${responses.length}`}
                color="text-blue-400"
              />
            </div>
          </Card>

          {/* Conversation Log */}
          <Card className="flex-1 border-border/50 bg-card/50 flex flex-col overflow-hidden">
            <div className="p-4 pb-2">
              <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-primary" />
                Conversation Log
              </h2>
            </div>
            <Separator className="opacity-50" />
            <ScrollArea className="flex-1 p-4">
              {responses.length === 0 && status === "idle" && (
                <div className="text-center text-muted-foreground text-sm py-8">
                  <Users className="w-8 h-8 mx-auto mb-3 opacity-40" />
                  <p>Post something to start the discussion.</p>
                  <p className="text-xs mt-1 opacity-60">
                    AI agents will react to your post in real-time.
                  </p>
                </div>
              )}

              {responses.length === 0 && status === "running" && (
                <div className="text-center text-muted-foreground text-sm py-8">
                  <Loader2 className="w-8 h-8 mx-auto mb-3 animate-spin opacity-40" />
                  <p>Agents are gathering...</p>
                </div>
              )}

              <div className="space-y-4">
                {responses.map((res, idx) => {
                  const persona = AGENT_PERSONAS.find(
                    (p) => p.id === res.agentId,
                  );
                  return (
                    <div key={idx} className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
                      <div
                        className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold"
                        style={{
                          backgroundColor: persona?.accentColor + "30",
                          color: persona?.accentColor,
                          border: `1.5px solid ${persona?.accentColor}50`,
                        }}
                      >
                        {res.agentName.charAt(0)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span
                            className="text-xs font-semibold"
                            style={{ color: persona?.accentColor }}
                          >
                            {res.agentName}
                          </span>
                        </div>
                        <p className="text-sm text-foreground/85 leading-relaxed">
                          {res.response}
                        </p>
                      </div>
                    </div>
                  );
                })}

                {/* Currently thinking agent */}
                {status === "running" &&
                  currentAgentIndex >= 0 &&
                  currentAgentIndex < AGENT_PERSONAS.length && (
                    <div className="flex gap-3 opacity-60">
                      <div
                        className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold"
                        style={{
                          backgroundColor:
                            AGENT_PERSONAS[currentAgentIndex].accentColor + "30",
                          color:
                            AGENT_PERSONAS[currentAgentIndex].accentColor,
                          border: `1.5px solid ${AGENT_PERSONAS[currentAgentIndex].accentColor}50`,
                        }}
                      >
                        {AGENT_PERSONAS[currentAgentIndex].name.charAt(0)}
                      </div>
                      <div className="flex-1">
                        <span
                          className="text-xs font-semibold"
                          style={{
                            color:
                              AGENT_PERSONAS[currentAgentIndex].accentColor,
                          }}
                        >
                          {AGENT_PERSONAS[currentAgentIndex].name}
                        </span>
                        <div className="flex items-center gap-1 mt-1">
                          <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">
                            thinking...
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
              </div>
            </ScrollArea>
          </Card>

          {/* Reset Button */}
          {status === "complete" && (
            <Button
              onClick={handleReset}
              variant="outline"
              className="w-full bg-transparent"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset Simulation
            </Button>
          )}
        </div>
      </main>
    </div>
  );
}

function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="text-center p-2 rounded-lg bg-background/50">
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}
