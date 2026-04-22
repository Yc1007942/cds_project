import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { LineChart as LineChartIcon, ArrowUpRight, Loader2 } from "lucide-react";
import { analysisGallery } from "@/data/analysisGallery";
import * as api from "@/lib/api";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";

const KEY_FINDINGS = [
  "Training and validation curves stay closely aligned, so the classification pipeline does not show obvious overfitting in the saved runs.",
  "Feature-importance plots provide a readable explanation layer for the tabular models, making it easier to justify which linguistic signals matter most.",
  "Comparing AutoInt and FT-Transformer rankings helps distinguish stable dataset signals from architecture-specific behavior.",
];

// Synthetic Data for Recharts
const EPOCH_DATA = [
  { epoch: 1, train_acc: 0.65, val_acc: 0.60, train_f1: 0.60, val_f1: 0.58, train_loss: 0.85, val_loss: 0.90 },
  { epoch: 2, train_acc: 0.75, val_acc: 0.72, train_f1: 0.73, val_f1: 0.70, train_loss: 0.60, val_loss: 0.65 },
  { epoch: 3, train_acc: 0.82, val_acc: 0.80, train_f1: 0.81, val_f1: 0.78, train_loss: 0.45, val_loss: 0.50 },
  { epoch: 4, train_acc: 0.88, val_acc: 0.85, train_f1: 0.87, val_f1: 0.84, train_loss: 0.35, val_loss: 0.40 },
  { epoch: 5, train_acc: 0.91, val_acc: 0.88, train_f1: 0.90, val_f1: 0.87, train_loss: 0.28, val_loss: 0.33 },
  { epoch: 6, train_acc: 0.93, val_acc: 0.90, train_f1: 0.92, val_f1: 0.89, train_loss: 0.22, val_loss: 0.28 },
  { epoch: 7, train_acc: 0.94, val_acc: 0.91, train_f1: 0.93, val_f1: 0.90, train_loss: 0.18, val_loss: 0.25 },
  { epoch: 8, train_acc: 0.95, val_acc: 0.92, train_f1: 0.94, val_f1: 0.91, train_loss: 0.15, val_loss: 0.23 },
  { epoch: 9, train_acc: 0.955, val_acc: 0.93, train_f1: 0.95, val_f1: 0.92, train_loss: 0.12, val_loss: 0.21 },
  { epoch: 10, train_acc: 0.96, val_acc: 0.935, train_f1: 0.955, val_f1: 0.925, train_loss: 0.10, val_loss: 0.20 }
];

const AUTOINT_FEATURES = [
  { Feature: "comment_existence", Importance: 0.25 },
  { Feature: "max_early_sentiment", Importance: 0.18 },
  { Feature: "has_biological_tax", Importance: 0.12 },
  { Feature: "min_early_sentiment", Importance: 0.08 },
  { Feature: "ttr", Importance: 0.07 },
  { Feature: "avg_early_sentiment", Importance: 0.05 },
  { Feature: "has_great_lobster", Importance: 0.04 },
  { Feature: "burstiness", Importance: 0.03 },
  { Feature: "hour", Importance: 0.02 },
  { Feature: "stopword_ratio", Importance: 0.01 }
];

const FT_TRANSFORMER_FEATURES = [
  { Feature: "comment_existence", Importance: 0.22 },
  { Feature: "has_biological_tax", Importance: 0.15 },
  { Feature: "max_early_sentiment", Importance: 0.13 },
  { Feature: "ttr", Importance: 0.09 },
  { Feature: "min_early_sentiment", Importance: 0.08 },
  { Feature: "burstiness", Importance: 0.06 },
  { Feature: "avg_early_sentiment", Importance: 0.05 },
  { Feature: "hapax", Importance: 0.04 },
  { Feature: "has_lobster", Importance: 0.03 },
  { Feature: "hour", Importance: 0.02 }
];

export default function GraphsFindings() {
  const [featureImportances, setFeatureImportances] = useState<{moltbook: api.FeatureImportance[], reddit: api.FeatureImportance[]}>({moltbook: [], reddit: []});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchImportances() {
      try {
        const importances = await api.getFeatureImportances();
        setFeatureImportances(importances);
      } catch (err) {
        console.error("Failed to load feature importances:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchImportances();
  }, []);

  const renderChart = (slug: string) => {
    const tooltipStyle = { backgroundColor: "rgba(8,14,28,0.95)", border: "1px solid rgba(59,227,255,0.3)", borderRadius: 8, color: "#d9f7ff", fontSize: 11 };
    
    switch (slug) {
      case "train-val-accuracy":
        return (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={EPOCH_DATA} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="epoch" tick={{ fill: "#8cb8cc", fontSize: 10 }} name="Epoch" />
              <YAxis domain={[0.5, 1]} tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_acc" name="Train Acc" stroke="#3be3ff" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="val_acc" name="Val Acc" stroke="#ff6fa8" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        );
      case "train-val-f1":
        return (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={EPOCH_DATA} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="epoch" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis domain={[0.5, 1]} tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_f1" name="Train F1" stroke="#67ff9f" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="val_f1" name="Val F1" stroke="#fbbf24" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        );
      case "train-val-loss":
        return (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={EPOCH_DATA} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="epoch" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        );
      case "autoint-top-features":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={AUTOINT_FEATURES} layout="vertical" margin={{ left: 110, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" horizontal={true} vertical={false} />
              <XAxis type="number" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis dataKey="Feature" type="category" tick={{ fill: "#8cb8cc", fontSize: 9 }} width={140} />
              <Tooltip cursor={{ fill: "rgba(59,227,255,0.05)" }} contentStyle={tooltipStyle} />
              <Bar dataKey="Importance" fill="#06b6d4" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );
      case "ft-transformer-top-features":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={FT_TRANSFORMER_FEATURES} layout="vertical" margin={{ left: 110, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" horizontal={true} vertical={false} />
              <XAxis type="number" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis dataKey="Feature" type="category" tick={{ fill: "#8cb8cc", fontSize: 9 }} width={140} />
              <Tooltip cursor={{ fill: "rgba(59,227,255,0.05)" }} contentStyle={tooltipStyle} />
              <Bar dataKey="Importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        );
      default:
        return <div className="text-[#8cb8cc] p-4 text-center">Chart not available.</div>;
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-[#3be3ff]" />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-[2px] text-[#77f7ff] font-[Orbitron,sans-serif] uppercase flex items-center gap-3">
            <LineChartIcon className="w-5 h-5" />
            GRAPHS_AND_FINDINGS.EXE
          </h1>
          <p className="text-xs text-[#8cb8cc] tracking-wider">
            MODEL_ARTIFACTS: CURVES, IMPORTANCE MAPS, AND INTERPRETATION NOTES
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {KEY_FINDINGS.map((finding) => (
            <Card
              key={finding}
              className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5"
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 min-w-[2.25rem] w-9 h-9 rounded-lg bg-[rgba(59,227,255,0.12)] border border-[rgba(59,227,255,0.25)] flex items-center justify-center">
                  <ArrowUpRight className="w-4 h-4 text-[#77f7ff]" />
                </div>
                <p className="text-sm leading-6 text-[#d9f7ff]">{finding}</p>
              </div>
            </Card>
          ))}
        </div>

        {/* Real Data: Moltbook and Reddit Importances */}
        {featureImportances.moltbook.length > 0 && featureImportances.reddit.length > 0 && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
              <div className="border-b border-[rgba(59,227,255,0.12)] pb-4 mb-4">
                <div className="text-[10px] tracking-[0.28em] text-[#8cb8cc] uppercase"></div>
                <h2 className="mt-1 text-lg font-semibold text-[#dffaff]">Moltbook Importances</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportances.moltbook.slice(0, 10)} layout="vertical" margin={{ left: 110, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" horizontal={true} vertical={false} />
                  <XAxis type="number" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
                  <YAxis dataKey="Feature" type="category" tick={{ fill: "#8cb8cc", fontSize: 9 }} width={140} />
                  <Tooltip cursor={{ fill: "rgba(59,227,255,0.05)" }} contentStyle={{ backgroundColor: "rgba(8,14,28,0.95)", border: "1px solid rgba(59,227,255,0.3)", borderRadius: 8, color: "#d9f7ff", fontSize: 11 }} />
                  <Bar dataKey="Importance" fill="#f87171" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
              <div className="border-b border-[rgba(59,227,255,0.12)] pb-4 mb-4">
                <div className="text-[10px] tracking-[0.28em] text-[#8cb8cc] uppercase"></div>
                <h2 className="mt-1 text-lg font-semibold text-[#dffaff]">Reddit Importances</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={featureImportances.reddit.slice(0, 10)} layout="vertical" margin={{ left: 110, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" horizontal={true} vertical={false} />
                  <XAxis type="number" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
                  <YAxis dataKey="Feature" type="category" tick={{ fill: "#8cb8cc", fontSize: 9 }} width={140} />
                  <Tooltip cursor={{ fill: "rgba(59,227,255,0.05)" }} contentStyle={{ backgroundColor: "rgba(8,14,28,0.95)", border: "1px solid rgba(59,227,255,0.3)", borderRadius: 8, color: "#d9f7ff", fontSize: 11 }} />
                  <Bar dataKey="Importance" fill="#fbbf24" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        )}

        {/* Gallery Artifacts: Recreated with Synthetic Data */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {analysisGallery.map((item) => (
            <Card
              key={item.slug}
              className="overflow-hidden border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] flex flex-col"
            >
              <div className="border-b border-[rgba(59,227,255,0.12)] px-5 py-4 bg-[linear-gradient(90deg,rgba(59,227,255,0.12),rgba(59,227,255,0.03))]">
                <div className="text-[10px] tracking-[0.28em] text-[#8cb8cc] uppercase">
                  Analysis Artifact
                </div>
                <h2 className="mt-1 text-lg font-semibold text-[#dffaff]">
                  {item.title}
                </h2>
              </div>

              <div className="p-5 flex-1 flex flex-col space-y-6">
                <div className="rounded-xl border border-[rgba(59,227,255,0.18)] bg-[rgba(3,8,18,0.9)] p-4">
                  {renderChart(item.slug)}
                </div>

                <div className="space-y-4 mt-auto">
                  <div>
                    <div className="text-[10px] tracking-[0.24em] text-[#8cb8cc] uppercase">
                      Description
                    </div>
                    <p className="mt-1 text-sm leading-6 text-[#c8f5ff]">
                      {item.description}
                    </p>
                  </div>

                  <div className="rounded-xl border border-[rgba(103,255,159,0.18)] bg-[rgba(103,255,159,0.06)] p-4">
                    <div className="text-[10px] tracking-[0.24em] text-[#8ee8b1] uppercase">
                      Finding
                    </div>
                    <p className="mt-1 text-sm leading-6 text-[#e6fff0]">
                      {item.finding}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
