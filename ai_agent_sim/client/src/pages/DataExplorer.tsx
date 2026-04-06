import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Database, Search, Download, Loader2, Users, Bot, FileText } from "lucide-react";
import * as api from "@/lib/api";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";

export default function DataExplorer() {
  const [stats, setStats] = useState<api.DataStats | null>(null);
  const [rows, setRows] = useState<Record<string, any>[]>([]);
  const [totalFiltered, setTotalFiltered] = useState(0);
  const [loading, setLoading] = useState(true);
  const [tableLoading, setTableLoading] = useState(false);
  const [scatterData, setScatterData] = useState<any[]>([]);

  // Filters
  const [wordRange, setWordRange] = useState<[number, number]>([0, 2000]);
  const [selectedSubs, setSelectedSubs] = useState<string[]>([]);
  const [keyword, setKeyword] = useState("");
  const [maxRows, setMaxRows] = useState(300);

  useEffect(() => {
    loadInitial();
  }, []);

  async function loadInitial() {
    try {
      const [statsData, scatterRes] = await Promise.all([
        api.getDataStats(),
        api.getScatterData("word_count", "char_count", 1200),
      ]);
      setStats(statsData);
      setScatterData(scatterRes.points);
      // initial table load
      await loadTable(0, 2000, "", "", 300);
    } catch (err) {
      console.error("Data load error:", err);
    } finally {
      setLoading(false);
    }
  }

  async function loadTable(wMin: number, wMax: number, subs: string, kw: string, limit: number) {
    setTableLoading(true);
    try {
      const res = await api.getExploreData({
        word_min: wMin,
        word_max: wMax,
        subreddits: subs || undefined,
        keyword: kw || undefined,
        limit,
      });
      setRows(res.rows);
      setTotalFiltered(res.total_filtered);
    } catch (err) {
      console.error("Table load error:", err);
    } finally {
      setTableLoading(false);
    }
  }

  function handleFilter() {
    loadTable(wordRange[0], wordRange[1], selectedSubs.join(","), keyword, maxRows);
  }

  function handleExportCSV() {
    if (!rows.length) return;
    const headers = Object.keys(rows[0]);
    const csv = [headers.join(","), ...rows.map(r => headers.map(h => `"${String(r[h] ?? '').replace(/"/g, '""')}"`).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "filtered_data.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  // Split scatter for coloring
  const humanPoints = scatterData.filter((p: any) => p.label === 0);
  const aiPoints = scatterData.filter((p: any) => p.label === 1);

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
            <Database className="w-5 h-5" />
            DATA_EXPLORER.EXE
          </h1>
          <p className="text-xs text-[#8cb8cc] tracking-wider">SYSTEM STATUS: ACCESSING RAW DATA_STREAMS</p>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-4">
          <MetricCard label="TOTAL_DOCS" value={stats?.total ?? 0} icon={<FileText className="w-4 h-4" />} />
          <MetricCard label="AI_AGENTS" value={stats?.ai_count ?? 0} icon={<Bot className="w-4 h-4" />} color="#3be3ff" />
          <MetricCard label="HUMAN_USERS" value={stats?.human_count ?? 0} icon={<Users className="w-4 h-4" />} color="#67ff9f" />
        </div>

        {/* Filters */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
            &gt;&gt; QUERY_FILTERS
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">WORD_COUNT_MIN</label>
              <input
                type="number"
                value={wordRange[0]}
                onChange={(e) => setWordRange([Number(e.target.value), wordRange[1]])}
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              />
            </div>
            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">WORD_COUNT_MAX</label>
              <input
                type="number"
                value={wordRange[1]}
                onChange={(e) => setWordRange([wordRange[0], Number(e.target.value)])}
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] focus:outline-none focus:border-[#3be3ff]"
              />
            </div>
            <div>
              <label className="text-[10px] text-[#a8ebff] tracking-wider block mb-1.5 uppercase">SEARCH_PAYLOAD</label>
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="author/text contains..."
                className="w-full bg-[rgba(4,10,20,0.8)] border border-[rgba(59,227,255,0.2)] rounded-lg px-3 py-2 text-sm text-[#d9f7ff] placeholder:text-[#5a8899] focus:outline-none focus:border-[#3be3ff]"
              />
            </div>
            <div className="flex items-end gap-2">
              <Button
                onClick={handleFilter}
                className="flex-1 bg-[rgba(21,56,86,0.65)] border border-[rgba(59,227,255,0.3)] text-[#d9f7ff] hover:shadow-[0_0_16px_rgba(59,227,255,0.5)] tracking-wider uppercase text-xs"
              >
                <Search className="w-3.5 h-3.5 mr-2" />
                Execute Query
              </Button>
            </div>
          </div>
        </Card>

        {/* Scatter Chart */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <h2 className="text-sm font-bold tracking-wider text-[#77f7ff] mb-4 uppercase">
            &gt;&gt; TRAFFIC SCAN // WORD_COUNT × CHAR_COUNT
          </h2>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,227,255,0.1)" />
              <XAxis dataKey="word_count" type="number" name="Word Count" stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 11 }} />
              <YAxis dataKey="char_count" type="number" name="Char Count" stroke="#8cb8cc" tick={{ fill: "#8cb8cc", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "rgba(8,14,28,0.95)", border: "1px solid rgba(59,227,255,0.3)", borderRadius: 8, color: "#d9f7ff", fontSize: 12 }}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "#8cb8cc" }} />
              <Scatter name="HUMAN" data={humanPoints} fill="#67ff9f" fillOpacity={0.6} />
              <Scatter name="AI" data={aiPoints} fill="#3be3ff" fillOpacity={0.6} />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        {/* Data Table */}
        <Card className="border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)] p-5">
          <div className="flex items-center justify-between mb-4">
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
                  {rows.length > 0 && Object.keys(rows[0]).map((h) => (
                    <th key={h} className="px-3 py-2.5 text-left text-[10px] text-[#a8ebff] tracking-wider uppercase font-medium whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableLoading ? (
                  <tr><td colSpan={99} className="text-center py-8">
                    <Loader2 className="w-5 h-5 animate-spin text-[#3be3ff] mx-auto" />
                  </td></tr>
                ) : rows.length === 0 ? (
                  <tr><td colSpan={99} className="text-center py-8 text-[#8cb8cc]">No results</td></tr>
                ) : (
                  rows.slice(0, maxRows).map((row, i) => (
                    <tr key={i} className="border-b border-[rgba(59,227,255,0.08)] hover:bg-[rgba(59,227,255,0.04)] transition-colors">
                      {Object.values(row).map((v, j) => (
                        <td key={j} className="px-3 py-2 text-[#c8f5ff] max-w-[200px] truncate">
                          {typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : String(v)}
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

function MetricCard({ label, value, icon, color = "#77f7ff" }: { label: string; value: number; icon: React.ReactNode; color?: string }) {
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
