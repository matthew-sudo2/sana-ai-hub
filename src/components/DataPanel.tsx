import { ChevronLeft, ChevronRight } from "lucide-react";

const sampleData = [
  { id: "SC-001", source: "arxiv.org", field: "NLP", records: 1248, quality: 94 },
  { id: "SC-002", source: "pubmed.gov", field: "BioML", records: 876, quality: 89 },
  { id: "SC-003", source: "ieee.org", field: "CV", records: 2103, quality: 97 },
  { id: "SC-004", source: "nature.com", field: "GenAI", records: 534, quality: 91 },
  { id: "SC-005", source: "acm.org", field: "HCI", records: 312, quality: 86 },
  { id: "SC-006", source: "springer.com", field: "RL", records: 698, quality: 92 },
  { id: "SC-007", source: "ssrn.com", field: "Ethics", records: 221, quality: 78 },
];

const DataPanel = () => {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">Cleaned Data</h3>
        <span className="font-mono text-xs text-muted-foreground">7 sources</span>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-2 text-left font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">ID</th>
              <th className="px-4 py-2 text-left font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Source</th>
              <th className="px-4 py-2 text-left font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Field</th>
              <th className="px-4 py-2 text-right font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Rows</th>
              <th className="px-4 py-2 text-right font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">QS</th>
            </tr>
          </thead>
          <tbody>
            {sampleData.map((row) => (
              <tr key={row.id} className="border-b transition-colors duration-150 hover:bg-muted/30">
                <td className="px-4 py-2.5 font-mono text-xs text-foreground">{row.id}</td>
                <td className="px-4 py-2.5 font-mono text-xs text-foreground">{row.source}</td>
                <td className="px-4 py-2.5">
                  <span className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">
                    {row.field}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-xs text-foreground">
                  {row.records.toLocaleString()}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <span
                    className={`font-mono text-xs font-medium ${
                      row.quality >= 90 ? "text-success" : "text-muted-foreground"
                    }`}
                  >
                    {row.quality}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between border-t px-4 py-2.5">
        <span className="font-body text-[11px] text-muted-foreground">Page 1 of 3</span>
        <div className="flex gap-1">
          <button className="flex h-7 w-7 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted">
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <button className="flex h-7 w-7 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted">
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default DataPanel;
