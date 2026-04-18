import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Database, Search, Download, Loader2, Users, Bot, FileText } from "lucide-react";
import * as api from "@/lib/api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const FEATURE_LABELS: Record<string, string> = {
  score: "Score",
  burstiness: "Burstiness",
  ttr: "Type-Token Ratio",
  hedging_score: "Hedging",
  self_reference_rate: "Self Reference",
};

export default function DataExplorer() {
  const [stats, setStats] = useState<api.DataStats | null>(null);
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [totalFiltered, setTotalFiltered] = useState(0);
  const [loading, setLoading] = useState(true);
  const [tableLoading, setTableLoading] = useState(false);
  const [scatterLoading, setScatterLoading] = useState(true);
  const [visualLoading, setVisualLoading] = useState(true);
  const [scatterData, setScatterData] = useState<api.ScatterPoint[]>([]);
  const [scoreDistribution, setScoreDistribution] = useState<api.DistBin[]>([]);
  const [burstinessDistribution, setBurstinessDistribution] = useState<api.DistBin[]>([]);
  const [featureStats, setFeatureStats] = useState<Awaited<ReturnType<typeof api.getFeatureStats>> | null>(null);

  const [scoreRange, setScoreRange] = useState<[number, number]>([0, 100]);
  const [selectedLabel, setSelectedLabel] = useState<number | null>(null);
  const [maxRows, setMaxRows] = useState(300);

  useEffect(() => {
    void loadInitial();
  }, []);

  async function loadInitial() {
    try {
      const [statsData, scatterRes, scoreDist, burstDist, featureStatsRes] = await Promise.all([
        api.getDataStats(),
        api.getScatterData("score", "burstiness", 1200),
        api.getDistribution("score", 24),
        api.getDistribution("burstiness", 24),
        api.getFeatureStats(),
      ]);

      setStats(statsData);
      setScatterData(scatterRes.points);
      setScoreDistribution(scoreDist.bins);
      setBurstinessDistribution(burstDist.bins);
      setFeatureStats(featureStatsRes);
      await loadTable(0, 100, null, 300);
    } catch (err) {
      console.error("Data load error:", err);
    } finally {
      setLoading(false);
      setScatterLoading(false);
      setVisualLoading(false);
    }
  }

  async function loadTable(
    scoreMin: number,
    scoreMax: number,
    label: number | null,
    limit: number,
  ) {
    setTableLoading(true);
    try {
      const qs = new URLSearchParams();
      if (scoreMin > 0) qs.set("score_min", String(scoreMin));
      if (scoreMax < 999999) qs.set("score_max", String(scoreMax));
      if (label !== null) qs.set("label", String(label));
      qs.set("limit", String(limit));

      const response = await fetch(`http://localhost:8000/api/data/explore?${qs.toString()}`);
      if (!response.ok) throw new Error("Failed to load data");

      const result = await response.json();
      setRows(result.rows);
      setTotalFiltered(result.total_filtered);
    } catch (err) {
      console.error("Table load error:", err);
    } finally {
      setTableLoading(false);
    }
  }

  function handleFilter() {
    void loadTable(scoreRange[0], scoreRange[1], selectedLabel, maxRows);
  }

  function handleExportCSV() {
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const csv = [
      headers.join(","),
      ...rows.map((row) =>
        headers
          .map((header) => `"${String(row[header] ?? "").replace(/"/g, '""')}"`)
          .join(","),
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "filtered_data.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  const humanPoints = scatterData.filter((point) => point.label === 0);
  const aiPoints = scatterData.filter((point) => point.label === 1);

  const labelMixData = useMemo(() => {
    if (!stats) return [];
    return [
      { name: "Human", value: stats.human_count, color: "#67ff9f" },
      { name: "AI", value: stats.ai_count, color: "#3be3ff" },
    ];
  }, [stats]);

  const featureComparisonData = useMemo(() => {
    if (!featureStats) return [];
    return featureStats.features.map((feature) => ({
      feature: FEATURE_LABELS[feature] ?? feature.replace(/_/g, " "),
      human: Number((featureStats.human_means[feature] ?? 0).toFixed(3)),
      ai: Number((featureStats.ai_means[feature] ?? 0).toFixed(3)),
    }));
  }, [featureStats]);

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
            <Database className="w-5 h-5" />
            DATA_EXPLORER.EXE
          </h1>
          <p className="text-xs text-[#8cb8cc] tracking-wider">
            SYSTEM STATUS: ACCESSING RAW DATA_STREAMS
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            label="TOTAL_DOCS"
            value={stats?.total ?? 0}
            icon={<FileText className="w-4 h-4" />}
          />
          <MetricCard
            label="AI_AGENTS"
            value={stats?.ai_count ?? 0}
            icon={<Bot className="w-4 h-4" />}
            color="#3be3ff"
          />
          <MetricCard
            label="HUMAN_USERS"
            value={stats?.human_count ?? 0}
            icon={<Users className="w-4 h-4" />}
            color="#67ff9f"
          />
        </div>

        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
            &gt;&gt; QUERY_FILTERS
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">
                SCORE_MIN
              </label>
              <input
                type="number"
                value={scoreRange[0]}
                onChange={(e) => setScoreRange([Number(e.target.value), scoreRange[1]])}
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              />
            </div>

            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">
                SCORE_MAX
              </label>
              <input
                type="number"
                value={scoreRange[1]}
                onChange={(e) => setScoreRange([scoreRange[0], Number(e.target.value)])}
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              />
            </div>

            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">
                LABEL_FILTER
              </label>
              <select
                value={selectedLabel === null ? "all" : String(selectedLabel)}
                onChange={(e) =>
                  setSelectedLabel(e.target.value === "all" ? null : Number(e.target.value))
                }
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              >
                <option value="all">ALL</option>
                <option value="0">HUMAN (0)</option>
                <option value="1">AI (1)</option>
              </select>
            </div>

            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">
                MAX_ROWS
              </label>
              <select
                value={maxRows}
                onChange={(e) => setMaxRows(Number(e.target.value))}
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              >
                {[100, 300, 500, 1000].map((limit) => (
                  <option key={limit} value={limit}>
                    {limit}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-end">
              <Button
                onClick={handleFilter}
                className="w-full bg-[rgba(21,56,86,0.65)] border border-[rgba(59,227,255,0.3)] text-[#d9f7ff] hover:shadow-[0_0_16px_rgba(59,227,255,0.5)] tracking-wider uppercase text-xs"
              >
                <Search className="w-3.5 h-3.5 mr-2" />
                Execute Query
              </Button>
            </div>
          </div>
        </Card>

        {/* <div className="grid grid-cols-1 xl:grid-cols-[1.3fr_0.7fr] gap-6">
          <ChartCard
            title="TRAFFIC_SCAN // SCORE × BURSTINESS"
            loading={scatterLoading}
            subtitle="Sampled posts positioned by engagement score and burstiness."
          >
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
                <XAxis
                  dataKey="score"
                  type="number"
                  name="Score"
                  stroke="#8cb8cc"
                  tick={{ fill: "#8cb8cc", fontSize: 11 }}
                />
                <YAxis
                  dataKey="burstiness"
                  type="number"
                  name="Burstiness"
                  stroke="#8cb8cc"
                  tick={{ fill: "#8cb8cc", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(8,14,28,0.95)",
                    border: "1px solid rgba(59,227,255,0.3)",
                    borderRadius: 8,
                    color: "#d9f7ff",
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "#8cb8cc" }} />
                <Scatter name="HUMAN" data={humanPoints} fill="#67ff9f" fillOpacity={0.5} />
                <Scatter name="AI" data={aiPoints} fill="#3be3ff" fillOpacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="LABEL_MIX"
            loading={visualLoading}
            subtitle="Dataset composition by class label."
          >
            <ResponsiveContainer width="100%" height={360}>
              <PieChart>
                <Pie
                  data={labelMixData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={80}
                  outerRadius={120}
                  paddingAngle={4}
                >
                  {labelMixData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(8,14,28,0.95)",
                    border: "1px solid rgba(59,227,255,0.3)",
                    borderRadius: 8,
                    color: "#d9f7ff",
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: "#8cb8cc" }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div> */}

        {/* <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ChartCard
            title="SCORE_DISTRIBUTION"
            loading={visualLoading}
            subtitle="Histogram of engagement score counts for human and AI samples."
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={scoreDistribution} barGap={0}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
                <XAxis
                  dataKey="center"
                  tick={{ fill: "#8cb8cc", fontSize: 10 }}
                  tickFormatter={(value) => Number(value).toFixed(0)}
                />
                <YAxis tick={{ fill: "#8cb8cc", fontSize: 10 }} />
                <Tooltip
                  formatter={(value) => Number(value).toLocaleString()}
                  contentStyle={{
                    backgroundColor: "rgba(8,14,28,0.95)",
                    border: "1px solid rgba(59,227,255,0.3)",
                    borderRadius: 8,
                    color: "#d9f7ff",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="human" name="HUMAN" fill="#67ff9f" fillOpacity={0.75} />
                <Bar dataKey="ai" name="AI" fill="#3be3ff" fillOpacity={0.75} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="BURSTINESS_DISTRIBUTION"
            loading={visualLoading}
            subtitle="Histogram of temporal text burstiness across both classes."
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={burstinessDistribution} barGap={0}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
                <XAxis
                  dataKey="center"
                  tick={{ fill: "#8cb8cc", fontSize: 10 }}
                  tickFormatter={(value) => Number(value).toFixed(2)}
                />
                <YAxis tick={{ fill: "#8cb8cc", fontSize: 10 }} />
                <Tooltip
                  formatter={(value) => Number(value).toLocaleString()}
                  contentStyle={{
                    backgroundColor: "rgba(8,14,28,0.95)",
                    border: "1px solid rgba(59,227,255,0.3)",
                    borderRadius: 8,
                    color: "#d9f7ff",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="human" name="HUMAN" fill="#67ff9f" fillOpacity={0.75} />
                <Bar dataKey="ai" name="AI" fill="#3be3ff" fillOpacity={0.75} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div> */}
{/* 
        <ChartCard
          title="MEAN_SIGNAL_COMPARISON"
          loading={visualLoading}
          subtitle="Average values for the key features exposed by the inference layer."
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={featureComparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="feature" tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <YAxis tick={{ fill: "#8cb8cc", fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(8,14,28,0.95)",
                  border: "1px solid rgba(59,227,255,0.3)",
                  borderRadius: 8,
                  color: "#d9f7ff",
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="human" name="HUMAN_AVG" fill="#67ff9f" fillOpacity={0.8} />
              <Bar dataKey="ai" name="AI_AVG" fill="#3be3ff" fillOpacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard> */}

        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
            <div>
              <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
                &gt;&gt; QUERY_RESULTS
              </h2>
              <p className="text-[10px] text-[#8cb8cc] tracking-wider mt-1">
                FETCHED {totalFiltered.toLocaleString()} RECORDS MATCHING PARAMETERS
              </p>
            </div>

            <Button
              onClick={handleExportCSV}
              variant="outline"
              size="sm"
              className="border-[rgba(59,227,255,0.3)] text-[#8cb8cc] hover:text-[#d9f7ff] text-xs tracking-wider uppercase"
              disabled={!rows.length}
            >
              <Download className="w-3.5 h-3.5 mr-2" />
              Export CSV
            </Button>
          </div>

          <div className="overflow-x-auto rounded-lg border border-[rgba(59,227,255,0.15)]">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[rgba(59,227,255,0.2)] bg-[rgba(4,10,20,0.6)]">
                  {rows.length > 0 &&
                    Object.keys(rows[0]).map((header) => (
                      <th
                        key={header}
                        className="px-3 py-2.5 text-left text-[10px] text-[#a8ebff] tracking-wider uppercase font-medium whitespace-nowrap"
                      >
                        {header}
                      </th>
                    ))}
                </tr>
              </thead>

              <tbody>
                {tableLoading ? (
                  <tr>
                    <td colSpan={99} className="text-center py-8">
                      <Loader2 className="w-5 h-5 animate-spin text-[#3be3ff] mx-auto" />
                    </td>
                  </tr>
                ) : rows.length === 0 ? (
                  <tr>
                    <td colSpan={99} className="text-center py-8 text-[#8cb8cc]">
                      No results
                    </td>
                  </tr>
                ) : (
                  rows.slice(0, maxRows).map((row, index) => (
                    <tr
                      key={index}
                      className="border-b border-[rgba(59,227,255,0.08)] hover:bg-[rgba(59,227,255,0.04)] transition-colors"
                    >
                      {Object.values(row).map((value, cellIndex) => (
                        <td
                          key={cellIndex}
                          className="px-3 py-2 text-[#c8f5ff] max-w-[200px] truncate"
                        >
                          {typeof value === "number"
                            ? Number.isInteger(value)
                              ? value
                              : value.toFixed(3)
                            : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  loading,
  children,
}: {
  title: string;
  subtitle: string;
  loading: boolean;
  children: React.ReactNode;
}) {
  return (
    <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
      <div className="mb-4">
        <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] uppercase">
          &gt;&gt; {title}
        </h2>
        <p className="text-[10px] text-[#8cb8cc] tracking-wider mt-1 uppercase">{subtitle}</p>
      </div>

      {loading ? (
        <div className="h-[300px] flex items-center justify-center">
          <Loader2 className="w-6 h-6 animate-spin text-[#3be3ff]" />
        </div>
      ) : (
        children
      )}
    </Card>
  );
}

function MetricCard({
  label,
  value,
  icon,
  color = "#77f7ff",
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
  color?: string;
}) {
  return (
    <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-[rgba(59,227,255,0.1)] flex items-center justify-center text-[#3be3ff]">
          {icon}
        </div>
        <div>
          <div className="text-2xl font-bold" style={{ color, textShadow: `0 0 16px ${color}55` }}>
            {value.toLocaleString()}
          </div>
          <div className="text-[9px] text-[#8cb8cc] tracking-wider uppercase">{label}</div>
        </div>
      </div>
    </Card>
  );
}
