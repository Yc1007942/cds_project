import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Loader2, Zap, Shield, AlertTriangle } from "lucide-react";
import * as api from "@/lib/api";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend,
} from "recharts";

export default function InferenceCore() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [classResult, setClassResult] = useState<api.ClassificationResult | null>(null);
  const [regResult, setRegResult] = useState<api.RegressionResult | null>(null);
  const [featureStats, setFeatureStats] = useState<any>(null);
  const [extractedFeatures, setExtractedFeatures] = useState<Record<string, number> | null>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);

  useEffect(() => {
    api.getModelInfo().then(setModelInfo).catch(console.error);
    api.getFeatureStats().then(setFeatureStats).catch(console.error);
  }, []);

  async function handleExecute() {
    if (!text.trim()) return;
    setLoading(true);
    setClassResult(null);
    setRegResult(null);
    setExtractedFeatures(null);

    try {
      const [cls, reg, feat] = await Promise.all([
        api.classifyText(text),
        api.predictEngagement(text),
        api.extractFeatures(text),
      ]);
      setClassResult(cls);
      setRegResult(reg);
      setExtractedFeatures(feat.features);
    } catch (err) {
      console.error("Inference error:", err);
    } finally {
      setLoading(false);
    }
  }

  // Build radar chart data
  const radarData = featureStats?.features?.map((f: string) => {
    const range = featureStats.ranges[f];
    const normalize = (v: number) => {
      if (!range || range.max === range.min) return 0.5;
      return (v - range.min) / (range.max - range.min);
    };
    const entry: Record<string, any> = {
      feature: f.replace(/_/g, " "),
      human: normalize(featureStats.human_means[f] ?? 0),
      ai: normalize(featureStats.ai_means[f] ?? 0),
    };
    if (extractedFeatures && f in extractedFeatures) {
      entry.input = normalize(extractedFeatures[f]);
    }
    return entry;
  }) ?? [];

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
        {/* Header */}
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-[2px] text-[#77f7ff] font-[Orbitron,sans-serif] uppercase flex items-center gap-3">
            <Brain className="w-5 h-5" />
            INFERENCE_CORE.EXE
          </h1>
          <p className="text-xs text-[#8cb8cc] tracking-wider">DUAL-MODE: CLASSIFICATION (AI/HUMAN) + REGRESSION (ENGAGEMENT)</p>
        </div>

        {/* Model Status */}
        <div className="grid grid-cols-2 gap-4">
          <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-4">
            <div className="flex items-center gap-3">
              <div className={`w-2.5 h-2.5 rounded-full ${modelInfo?.classifier?.loaded ? "bg-[#67ff9f] shadow-[0_0_8px_#67ff9f]" : "bg-[#ff6fa8]"}`} />
              <div>
                <div className="text-[10px] text-[#a8ebff] tracking-wider uppercase">CLASSIFIER</div>
                <div className="text-xs text-[#d9f7ff]">
                  {modelInfo?.classifier?.loaded
                    ? `${modelInfo.classifier.type} — ${modelInfo.classifier.features} features`
                    : "OFFLINE"}
                </div>
              </div>
            </div>
          </Card>
          <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-4">
            <div className="flex items-center gap-3">
              <div className={`w-2.5 h-2.5 rounded-full ${modelInfo?.regressor?.loaded ? "bg-[#67ff9f] shadow-[0_0_8px_#67ff9f]" : "bg-[#ff6fa8]"}`} />
              <div>
                <div className="text-[10px] text-[#a8ebff] tracking-wider uppercase">REGRESSOR</div>
                <div className="text-xs text-[#d9f7ff]">
                  {modelInfo?.regressor?.loaded
                    ? `${modelInfo.regressor.type} — ${modelInfo.regressor.features} features, ${modelInfo.regressor.n_estimators} trees`
                    : "OFFLINE"}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Input */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
            &gt;&gt; INPUT_TERMINAL
          </h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="// ENTER SUSPECT STRING DATA HERE..."
            rows={5}
            className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-4 py-3 text-sm text-[#d9f7ff] placeholder:text-[#3a6070] focus:outline-none focus:border-[#3be3ff] resize-none font-mono"
          />
          <Button
            onClick={handleExecute}
            disabled={loading || !text.trim()}
            className="w-full mt-3 bg-[rgba(21,56,86,0.65)] border border-[rgba(59,227,255,0.3)] text-[#d9f7ff] hover:shadow-[0_0_20px_rgba(59,227,255,0.5)] tracking-[2px] uppercase text-xs h-11"
          >
            {loading ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> ANALYZING...</>
            ) : (
              <><Zap className="w-4 h-4 mr-2" /> EXECUTE_ANALYSIS</>
            )}
          </Button>
        </Card>

        {/* Results */}
        {(classResult || regResult) && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Classification Result */}
              {classResult && (
                <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
                  <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
                    &gt;&gt; CLASSIFICATION_OUTPUT
                  </h2>
                  <div className={`rounded-xl p-6 text-center border ${
                    classResult.prediction === 1
                      ? "border-[#3be3ff] bg-[rgba(59,227,255,0.08)]"
                      : "border-[#67ff9f] bg-[rgba(103,255,159,0.08)]"
                  }`}>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{
                      background: classResult.prediction === 1 ? "rgba(59,227,255,0.15)" : "rgba(103,255,159,0.15)",
                      boxShadow: classResult.prediction === 1 ? "0 0 30px rgba(59,227,255,0.3)" : "0 0 30px rgba(103,255,159,0.3)",
                    }}>
                      {classResult.prediction === 1
                        ? <AlertTriangle className="w-8 h-8 text-[#3be3ff]" />
                        : <Shield className="w-8 h-8 text-[#67ff9f]" />
                      }
                    </div>
                    <h3 className="text-lg font-bold tracking-[2px] uppercase mb-2" style={{
                      color: classResult.prediction === 1 ? "#3be3ff" : "#67ff9f",
                      textShadow: classResult.prediction === 1 ? "0 0 15px rgba(59,227,255,0.5)" : "0 0 15px rgba(103,255,159,0.5)",
                    }}>
                      {classResult.prediction === 1 ? "++ ARTIFICIAL INTELLIGENCE DETECTED ++" : "++ HUMAN ORIGIN VERIFIED ++"}
                    </h3>
                    <div className="text-xs text-[#8cb8cc] space-y-1 mt-4">
                      <div>CONFIDENCE_AI: <span className="text-[#3be3ff] font-bold">{(classResult.ai_prob * 100).toFixed(1)}%</span></div>
                      <div>CONFIDENCE_HUMAN: <span className="text-[#67ff9f] font-bold">{(classResult.human_prob * 100).toFixed(1)}%</span></div>
                      <div>FEATURES_USED: <span className="text-[#d9f7ff]">{classResult.features_used}</span></div>
                    </div>

                    {/* Confidence bar */}
                    <div className="mt-4 space-y-1">
                      <div className="text-[10px] text-[#8cb8cc] tracking-wider uppercase">CONFIDENCE_INTERVAL</div>
                      <div className="w-full h-3 bg-[rgba(4,10,20,0.8)] rounded-full overflow-hidden border border-[rgba(59,227,255,0.15)]">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{
                            width: `${classResult.confidence * 100}%`,
                            background: classResult.prediction === 1
                              ? "linear-gradient(90deg, #3be3ff, #77f7ff)"
                              : "linear-gradient(90deg, #67ff9f, #b5ffd0)",
                            boxShadow: classResult.prediction === 1
                              ? "0 0 12px rgba(59,227,255,0.5)"
                              : "0 0 12px rgba(103,255,159,0.5)",
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </Card>
              )}

              {/* Regression Result */}
              {regResult && (
                <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
                  <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
                    &gt;&gt; ENGAGEMENT_SCORE (REGRESSOR)
                  </h2>
                  <div className="rounded-xl p-6 text-center border border-[rgba(251,191,36,0.3)] bg-[rgba(251,191,36,0.05)]">
                    {/* Gauge visualization */}
                    <div className="relative w-48 h-24 mx-auto mb-4">
                      <svg viewBox="0 0 200 110" className="w-full">
                        {/* Background arc */}
                        <path
                          d="M 20 100 A 80 80 0 0 1 180 100"
                          fill="none"
                          stroke="rgba(59,227,255,0.15)"
                          strokeWidth="12"
                          strokeLinecap="round"
                        />
                        {/* Colored sections */}
                        <path d="M 20 100 A 80 80 0 0 1 73 33" fill="none" stroke="rgba(103,255,159,0.25)" strokeWidth="12" strokeLinecap="round" />
                        <path d="M 73 33 A 80 80 0 0 1 127 33" fill="none" stroke="rgba(251,191,36,0.25)" strokeWidth="12" strokeLinecap="round" />
                        <path d="M 127 33 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(59,227,255,0.35)" strokeWidth="12" strokeLinecap="round" />
                        {/* Needle */}
                        {(() => {
                          const angle = -180 + (regResult.score / 100) * 180;
                          const rad = (angle * Math.PI) / 180;
                          const nx = 100 + 65 * Math.cos(rad);
                          const ny = 100 + 65 * Math.sin(rad);
                          return <line x1="100" y1="100" x2={nx} y2={ny} stroke="#3be3ff" strokeWidth="3" strokeLinecap="round" />;
                        })()}
                        <circle cx="100" cy="100" r="5" fill="#3be3ff" />
                      </svg>
                    </div>
                    <div className="text-4xl font-bold text-[#fbbf24] tracking-wider" style={{ textShadow: "0 0 20px rgba(251,191,36,0.5)" }}>
                      {regResult.score}
                    </div>
                    <div className="text-[10px] text-[#8cb8cc] tracking-wider uppercase mt-1">
                      ENGAGEMENT_SCORE / 100
                    </div>
                    <div className="text-xs text-[#8cb8cc] mt-4 space-y-1">
                      <div>AGENTS_TO_SPAWN: <span className="text-[#3be3ff] font-bold">{regResult.agents_to_spawn}</span></div>
                      <div>WORD_COUNT: <span className="text-[#d9f7ff]">{regResult.word_count}</span></div>
                      <div>CHAR_COUNT: <span className="text-[#d9f7ff]">{regResult.char_count}</span></div>
                    </div>
                  </div>
                </Card>
              )}
            </div>

            {/* Radar Chart */}
            {radarData.length >= 3 && extractedFeatures && (
              <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
                <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
                  &gt;&gt; SPATIAL_DIVERGENCE_MAP
                </h2>
                <ResponsiveContainer width="100%" height={350}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(59,227,255,0.15)" />
                    <PolarAngleAxis dataKey="feature" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
                    <Radar name="HUMAN_AVG" dataKey="human" stroke="#67ff9f" fill="#67ff9f" fillOpacity={0.1} strokeWidth={1.5} />
                    <Radar name="AI_AVG" dataKey="ai" stroke="#3be3ff" fill="#3be3ff" fillOpacity={0.1} strokeWidth={1.5} />
                    <Radar name="INPUT" dataKey="input" stroke="#ffffff" fill="#ffffff" fillOpacity={0.05} strokeWidth={2.5} />
                    <Legend wrapperStyle={{ fontSize: 11, color: "#8cb8cc" }} />
                  </RadarChart>
                </ResponsiveContainer>
              </Card>
            )}

            {/* Feature Diagnostics */}
            {extractedFeatures && (
              <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
                <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
                  &gt;&gt; DIAGNOSTIC_TRACE
                </h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {Object.entries(extractedFeatures).slice(0, 20).map(([key, val]) => (
                    <div key={key} className="border border-[rgba(59,227,255,0.1)] rounded-lg p-2.5 bg-[rgba(4,10,20,0.5)]">
                      <div className="text-[9px] text-[#8cb8cc] tracking-wider uppercase truncate">{key}</div>
                      <div className="text-sm text-[#d9f7ff] font-mono mt-0.5">{typeof val === 'number' ? val.toFixed(3) : val}</div>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  );
}
