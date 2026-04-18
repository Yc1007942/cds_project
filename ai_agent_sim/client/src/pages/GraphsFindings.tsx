import { Card } from "@/components/ui/card";
import { LineChart, ArrowUpRight } from "lucide-react";
import { analysisGallery } from "@/data/analysisGallery";

const KEY_FINDINGS = [
  "Training and validation curves stay closely aligned, so the classification pipeline does not show obvious overfitting in the saved runs.",
  "Feature-importance plots provide a readable explanation layer for the tabular models, making it easier to justify which linguistic signals matter most.",
  "Comparing AutoInt and FT-Transformer rankings helps distinguish stable dataset signals from architecture-specific behavior.",
];

export default function GraphsFindings() {
  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
        <div className="space-y-1">
          <h1 className="text-xl font-bold tracking-[2px] text-[#77f7ff] font-[Orbitron,sans-serif] uppercase flex items-center gap-3">
            <LineChart className="w-5 h-5" />
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
                <div className="mt-0.5 w-9 h-9 rounded-lg bg-[rgba(59,227,255,0.12)] border border-[rgba(59,227,255,0.25)] flex items-center justify-center">
                  <ArrowUpRight className="w-4 h-4 text-[#77f7ff]" />
                </div>
                <p className="text-sm leading-6 text-[#d9f7ff]">{finding}</p>
              </div>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {analysisGallery.map((item) => (
            <Card
              key={item.slug}
              className="overflow-hidden border-[rgba(59,227,255,0.2)] bg-[rgba(8,14,28,0.7)]"
            >
              <div className="border-b border-[rgba(59,227,255,0.12)] px-5 py-4 bg-[linear-gradient(90deg,rgba(59,227,255,0.12),rgba(59,227,255,0.03))]">
                <div className="text-[10px] tracking-[0.28em] text-[#8cb8cc] uppercase">
                  Analysis Artifact
                </div>
                <h2 className="mt-1 text-lg font-semibold text-[#dffaff]">
                  {item.title}
                </h2>
              </div>

              <div className="p-5 space-y-4">
                <div className="overflow-hidden rounded-xl border border-[rgba(59,227,255,0.18)] bg-[rgba(3,8,18,0.9)]">
                  <img
                    src={item.image}
                    alt={item.title}
                    className="w-full h-auto object-contain"
                  />
                </div>

                <div className="space-y-3">
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
