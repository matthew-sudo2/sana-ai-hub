import { BadgeCheck, Copy, Download } from "lucide-react";

const ReportPanel = () => {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">Executive Summary</h3>
        <div className="flex items-center gap-1.5">
          <BadgeCheck className="h-4 w-4 text-success" />
          <span className="font-display text-[11px] font-semibold text-success">Verified</span>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-5">
        {/* Paper container */}
        <div className="rounded-md border bg-card p-6 shadow-sm">
          <div className="mb-4 flex items-center gap-2">
            <div className="h-1 w-8 rounded-full bg-success" />
            <span className="font-display text-[10px] font-semibold uppercase tracking-widest text-success">
              Validation Complete
            </span>
          </div>

          <h4 className="font-display text-base font-bold text-foreground leading-snug">
            Multi-Source NLP Dataset Validation Report
          </h4>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">
            Generated 2026-03-18 • Pipeline ID: PL-20260318-A
          </p>

          <hr className="my-4 border-border" />

          <div className="space-y-4 font-body text-sm leading-relaxed text-foreground">
            <div>
              <h5 className="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                Overview
              </h5>
              <p className="text-[13px]">
                This report summarizes the validation of <strong>5,992 records</strong> collected from 7 academic sources across 6 research domains. The pipeline achieved an average quality score of <strong>89.6%</strong>, exceeding the 85% threshold.
              </p>
            </div>

            <div>
              <h5 className="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                Key Findings
              </h5>
              <ul className="list-inside space-y-1 text-[13px]">
                <li className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
                  <span>IEEE and ArXiv sources demonstrated the highest data quality (97% and 94% respectively).</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
                  <span>Cross-source correlation validated at r=0.92, indicating strong inter-source consistency.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-foreground/30" />
                  <span>Ethics domain data (SSRN) flagged for manual review due to sub-threshold quality score (78%).</span>
                </li>
              </ul>
            </div>

            <div>
              <h5 className="font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                Recommendation
              </h5>
              <p className="text-[13px]">
                The dataset is cleared for downstream model training with the exception of the SSRN Ethics subset, which requires additional manual labeling review before inclusion.
              </p>
            </div>
          </div>

          <hr className="my-4 border-border" />

          <div className="flex items-center gap-2">
            <button className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-[11px] font-medium text-muted-foreground transition-colors duration-150 hover:bg-muted">
              <Copy className="h-3.5 w-3.5" />
              Copy
            </button>
            <button className="flex h-8 items-center gap-1.5 rounded-md border px-3 font-display text-[11px] font-medium text-muted-foreground transition-colors duration-150 hover:bg-muted">
              <Download className="h-3.5 w-3.5" />
              Export PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportPanel;
