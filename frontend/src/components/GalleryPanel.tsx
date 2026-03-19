import { Download, BarChart3, PieChart, TrendingUp, ScatterChart } from "lucide-react";

const charts = [
  {
    title: "Label Distribution by Category",
    type: "Bar Chart",
    icon: BarChart3,
    description: "Distribution of labeled data across 12 research categories",
    gradient: "from-primary/10 to-primary/5",
  },
  {
    title: "Quality Score Distribution",
    type: "Pie Chart",
    icon: PieChart,
    description: "Breakdown of data quality scores across all sources",
    gradient: "from-success/10 to-success/5",
  },
  {
    title: "Crawling Performance Over Time",
    type: "Line Chart",
    icon: TrendingUp,
    description: "Records collected per hour during the scouting phase",
    gradient: "from-secondary/10 to-secondary/5",
  },
  {
    title: "Source Correlation Matrix",
    type: "Scatter Plot",
    icon: ScatterChart,
    description: "Cross-source data correlation for validation",
    gradient: "from-primary/10 to-success/5",
  },
];

const GalleryPanel = () => {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">Visual Gallery</h3>
        <span className="font-mono text-xs text-muted-foreground">{charts.length} charts</span>
      </div>

      <div className="flex-1 space-y-3 overflow-auto p-4">
        {charts.map((chart) => (
          <div
            key={chart.title}
            className="group rounded-md border bg-card transition-colors duration-150 hover:border-muted-foreground/30"
          >
            {/* Chart Placeholder */}
            <div className={`flex h-32 items-center justify-center rounded-t-md bg-gradient-to-br ${chart.gradient}`}>
              <chart.icon className="h-10 w-10 text-muted-foreground/40" />
            </div>

            <div className="p-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="font-display text-xs font-semibold text-foreground">{chart.title}</p>
                  <p className="mt-0.5 font-body text-[11px] text-muted-foreground">{chart.description}</p>
                </div>
                <button className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border text-muted-foreground opacity-0 transition-all duration-150 hover:bg-muted group-hover:opacity-100">
                  <Download className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="mt-2">
                <span className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {chart.type}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GalleryPanel;
