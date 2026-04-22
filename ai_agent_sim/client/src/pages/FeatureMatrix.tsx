import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Loader2, BarChart3 } from "lucide-react";
import * as api from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ScatterChart, Scatter,
} from "recharts";

const CLUSTER_COLORS = ["#3be3ff", "#67ff9f", "#ff6fa8", "#fbbf24", "#a78bfa", "#ec4899", "#f97316", "#06b6d4", "#8b5cf6", "#14b8a6", "#f87171", "#60a5fa"];

const CustomScatterTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[rgba(8,14,28,0.95)] border border-[rgba(59,227,255,0.3)] rounded-lg p-3 shadow-lg min-w-[120px]">
        {label && <p className="text-[#3be3ff] font-bold text-xs mb-2">{label}</p>}
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex gap-2 text-[11px] items-center mb-1 last:mb-0">
             <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: entry.color }} />
             <span className="font-bold text-white uppercase tracking-wider">{entry.name}:</span>
             <span className="text-white font-mono">
                {typeof entry.value === "number" ? entry.value.toFixed(3) : entry.value}
             </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function FeatureMatrix() {
  const [features, setFeatures] = useState<string[]>([]);
  const [selectedFeature, setSelectedFeature] = useState("score");
  const [distData, setDistData] = useState<api.DistBin[]>([]);
  const [boxHuman, setBoxHuman] = useState<api.BoxStats | null>(null);
  const [boxAi, setBoxAi] = useState<api.BoxStats | null>(null);
  const [corrMatrix, setCorrMatrix] = useState<number[][]>([]);
  const [corrLabels, setCorrLabels] = useState<string[]>([]);
  const [corrFeatures, setCorrFeatures] = useState<string[]>([]);
  const [clusterData, setClusterData] = useState<api.ClusterPoint[]>([]);
  const [clusterVariance, setClusterVariance] = useState(0);
  const [nClusters, setNClusters] = useState(5);
  const [scatterX, setScatterX] = useState("score");
  const [scatterY, setScatterY] = useState("burstiness");
  const [scatterData, setScatterData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadFeatures();
  }, []);

  async function loadFeatures() {
    try {
      const { features: f } = await api.getFeatureList();
      setFeatures(f);
      if (f.length > 0) {
        setSelectedFeature(f[0]);
        setCorrFeatures(f.slice(0, 5));
        await loadDist(f[0]);
        await loadCorrelation(f.slice(0, 5));
        await loadScatter(f[0], f.length > 1 ? f[1] : f[0]);
        await loadClusters(5);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadDist(feature: string) {
    try {
      const d = await api.getDistribution(feature, 50);
      setDistData(d.bins);
      setBoxHuman(d.box_human);
      setBoxAi(d.box_ai);
    } catch (err) { console.error(err); }
  }

  async function loadCorrelation(feats: string[]) {
    try {
      const c = await api.getCorrelation(feats);
      setCorrMatrix(c.matrix);
      setCorrLabels(c.labels);
    } catch (err) { console.error(err); }
  }

  async function loadScatter(x: string, y: string) {
    try {
      const s = await api.getScatterData(x, y, 1500);
      setScatterData(s.points);
    } catch (err) { console.error(err); }
  }

  async function loadClusters(k: number) {
    try {
      const c = await api.getClusters(1200, k, 2);
      setClusterData(c.points);
      setClusterVariance(c.explained_variance);
    } catch (err) { console.error(err); }
  }

  function handleFeatureChange(f: string) {
    setSelectedFeature(f);
    loadDist(f);
  }

  function handleScatterChange(axis: "x" | "y", val: string) {
    const x = axis === "x" ? val : scatterX;
    const y = axis === "y" ? val : scatterY;
    if (axis === "x") setScatterX(val); else setScatterY(val);
    loadScatter(x, y);
  }

  function handleClusterKChange(k: number) {
    setNClusters(k);
    loadClusters(k);
  }

  // Group cluster data by cluster id for coloring
  const clusterGroups: Record<number, api.ClusterPoint[]> = {};
  clusterData.forEach(p => {
    if (!clusterGroups[p.cluster]) clusterGroups[p.cluster] = [];
    clusterGroups[p.cluster].push(p);
  });

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
        {/* Header */}
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-[2px] text-[#77f7ff] font-[Orbitron,sans-serif] uppercase flex items-center gap-3">
            <BarChart3 className="w-5 h-5" />
            FEATURES.EXE
          </h1>
          <p className="text-xs text-[#8cb8cc] tracking-wider">SYSTEM STATUS: VISUALIZING LINGUISTIC VECTORS</p>
        </div>

        {/* Feature Distribution */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
              &gt;&gt; VECTOR_DISTRIBUTION
            </h2>
            <select
              value={selectedFeature}
              onChange={(e) => handleFeatureChange(e.target.value)}
              className="bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-1.5 text-xs text-[#d9f7ff] focus:outline-none uppercase tracking-wider"
            >
              {features.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={distData} barGap={0}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="center" tick={{ fill: "#8cb8cc", fontSize: 10 }} tickFormatter={(v) => v.toFixed(1)} />
              <YAxis tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip contentStyle={{ backgroundColor: "rgba(8,14,28,0.95)", border: "1px solid rgba(59,227,255,0.3)", borderRadius: 8, color: "#d9f7ff", fontSize: 11 }} itemStyle={{ color: "#ffffff", fontWeight: 500 }} labelStyle={{ color: "#3be3ff", fontWeight: "bold" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="human" name="HUMAN" fill="#67ff9f" fillOpacity={0.7} />
              <Bar dataKey="ai" name="AI" fill="#3be3ff" fillOpacity={0.7} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* 2D Scatter */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center gap-4 mb-4 flex-wrap">
            <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
              &gt;&gt; 2D_CORRELATION_MAP
            </h2>
            <div className="flex items-center gap-2 ml-auto">
              <label className="text-[10px] text-[#a8ebff] uppercase">X:</label>
              <select value={scatterX} onChange={(e) => handleScatterChange("x", e.target.value)}
                className="bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded px-2 py-1 text-xs text-[#d9f7ff]">
                {features.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
              <label className="text-[10px] text-[#a8ebff] uppercase ml-2">Y:</label>
              <select value={scatterY} onChange={(e) => handleScatterChange("y", e.target.value)}
                className="bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded px-2 py-1 text-xs text-[#d9f7ff]">
                {features.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey={scatterX} type="number" name={scatterX} stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis dataKey={scatterY} type="number" name={scatterY} stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip content={<CustomScatterTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Scatter name="HUMAN" data={scatterData.filter((p: any) => p.label === 1)} fill="#67ff9f" fillOpacity={0.5} />
              <Scatter name="AI" data={scatterData.filter((p: any) => p.label === 0)} fill="#3be3ff" fillOpacity={0.5} />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        {/* Correlation Heatmap */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
              &gt;&gt; CORRELATION_HEATMAP
            </h2>
            <div className="flex gap-1 flex-wrap">
              {features.map(f => (
                <button
                  key={f}
                  onClick={() => {
                    const next = corrFeatures.includes(f)
                      ? corrFeatures.filter(x => x !== f)
                      : [...corrFeatures, f];
                    if (next.length >= 2) {
                      setCorrFeatures(next);
                      loadCorrelation(next);
                    }
                  }}
                  className={`px-2 py-1 rounded text-[9px] tracking-wider uppercase transition-all border ${
                    corrFeatures.includes(f)
                      ? "bg-[rgba(59,227,255,0.15)] border-[rgba(59,227,255,0.4)] text-[#77f7ff]"
                      : "bg-transparent border-[rgba(59,227,255,0.1)] text-[#5a8899] hover:text-[#8cb8cc]"
                  }`}
                >
                  {f.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>
          {corrMatrix.length > 0 && (
            <div className="overflow-x-auto pt-20 pb-4 mt-4">
              <table className="mx-auto border-separate border-spacing-0">
                <thead>
                  <tr className="h-24">
                    <th />
                    {corrLabels.map(l => (
                      <th key={l} className="relative p-0 w-12 group">
                        <div className="absolute bottom-4 left-6 text-[9px] text-[#a8ebff] uppercase tracking-wider -rotate-45 origin-bottom-left whitespace-nowrap group-hover:text-[#3be3ff] transition-colors">
                          {l.replace(/_/g, " ")}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {corrMatrix.map((row, i) => (
                    <tr key={i}>
                      <td className="px-4 py-1 text-[10px] text-[#a8ebff] uppercase tracking-wider text-right whitespace-nowrap bg-[rgba(8,14,28,0.4)]">
                        {corrLabels[i].replace(/_/g, " ")}
                      </td>
                      {row.map((val, j) => {
                        const intensity = Math.abs(val);
                        const hue = val >= 0 ? 170 : 340;
                        return (
                          <td key={j} className="w-12 h-10 text-center text-[10px] font-mono border border-[rgba(59,227,255,0.1)] transition-colors hover:border-[#3be3ff]/50"
                            style={{ backgroundColor: `hsla(${hue}, 80%, 50%, ${intensity * 0.6})`, color: intensity > 0.6 ? "#fff" : "#d9f7ff" }}>
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* AI Agent Cluster Map */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div>
              <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
                &gt;&gt; AI_AGENT_CLUSTER_MAP
              </h2>
              <p className="text-[10px] text-[#8cb8cc] tracking-wider mt-1">
                PCA(2D) + KMEANS | Variance explained: {(clusterVariance * 100).toFixed(1)}%
              </p>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-[#a8ebff] uppercase tracking-wider">CLUSTERS_K:</label>
              <input
                type="range" min={2} max={12} value={nClusters}
                onChange={(e) => handleClusterKChange(Number(e.target.value))}
                className="w-24 accent-[#3be3ff]"
              />
              <span className="text-sm text-[#3be3ff] font-bold w-6">{nClusters}</span>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="pc1" type="number" name="PC1" stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis dataKey="pc2" type="number" name="PC2" stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip content={<CustomScatterTooltip />} cursor={{ strokeDasharray: '3 3' }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {Object.entries(clusterGroups).map(([clusterId, points]) => (
                <Scatter
                  key={clusterId}
                  name={`Cluster ${clusterId}`}
                  data={points}
                  fill={CLUSTER_COLORS[Number(clusterId) % CLUSTER_COLORS.length]}
                  fillOpacity={0.7}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
