import { Link, useLocation } from "wouter";
import { Zap, Database, BarChart3, Brain, Menu, X, LineChart } from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = [
  { path: "/", label: "Agent Arena", icon: Zap, shortLabel: "01" },
  { path: "/data-explorer", label: "Data Explorer", icon: Database, shortLabel: "02" },
  { path: "/feature-matrix", label: "Feature Matrix", icon: BarChart3, shortLabel: "03" },
  { path: "/inference", label: "Inference Core", icon: Brain, shortLabel: "04" },
  { path: "/graphs-findings", label: "Graphs & Findings", icon: LineChart, shortLabel: "05" },
];

export default function Navigation() {
  const [location] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="border-b border-[var(--neon-line)] bg-[rgba(4,7,13,0.92)] backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 no-underline">
          <div className="w-8 h-8 rounded-lg bg-[rgba(59,227,255,0.15)] flex items-center justify-center border border-[rgba(59,227,255,0.3)]">
            <Zap className="w-4 h-4 text-[#3be3ff]" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-bold tracking-[2px] text-[#77f7ff] font-[Orbitron,sans-serif] uppercase">
              MOLTNET
            </span>
            <span className="text-[9px] tracking-[1px] text-[#3be3ff] opacity-60 -mt-0.5">
              NEURAL_OPS_DECK
            </span>
          </div>
        </Link>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map((item) => {
            const isActive = location === item.path;
            const Icon = item.icon;
            return (
              <Link key={item.path} href={item.path}>
                <div
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs tracking-wider transition-all cursor-pointer
                  ${isActive
                    ? "bg-[rgba(59,227,255,0.12)] text-[#77f7ff] border border-[rgba(59,227,255,0.35)] shadow-[0_0_12px_rgba(59,227,255,0.2)]"
                    : "text-[#8cb8cc] hover:text-[#badfe8] hover:bg-[rgba(59,227,255,0.06)] border border-transparent"
                  }`}
                >
                  <span className="text-[10px] opacity-50 font-mono">[{item.shortLabel}]</span>
                  <Icon className="w-3.5 h-3.5" />
                  <span className="font-medium uppercase">{item.label}</span>
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#67ff9f] shadow-[0_0_8px_#67ff9f] animate-pulse ml-1" />
                  )}
                </div>
              </Link>
            );
          })}
        </div>

        {/* Status pill */}
        <div className="hidden md:flex items-center gap-2 text-[10px] text-[#8cb8cc] tracking-wider">
          <span className="w-1.5 h-1.5 rounded-full bg-[#67ff9f] shadow-[0_0_6px_#67ff9f] animate-pulse" />
          LINK_STABLE
        </div>

        {/* Mobile toggle */}
        <button
          className="md:hidden p-2 text-[#77f7ff]"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-[rgba(59,227,255,0.15)] bg-[rgba(4,7,13,0.97)] px-4 py-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = location === item.path;
            const Icon = item.icon;
            return (
              <Link key={item.path} href={item.path}>
                <div
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm transition-all cursor-pointer
                  ${isActive
                    ? "bg-[rgba(59,227,255,0.12)] text-[#77f7ff] border border-[rgba(59,227,255,0.3)]"
                    : "text-[#8cb8cc] hover:bg-[rgba(59,227,255,0.06)]"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="uppercase tracking-wider">{item.label}</span>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
