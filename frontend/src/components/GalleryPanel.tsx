import { Download, Loader, SendHorizontal, Eye, X, Info, Sparkles, TrendingUp, BarChart2, ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useState, useCallback } from "react";
import { usePipeline } from "@/context/PipelineContext";
import {
  fetchImagesList,
  fetchChartExplanations,
  generateCustomChart,
  getImageURL,
  type ChartExplanation,
  type ChartExplanationsResponse,
} from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CHART_TYPE_ICON: Record<string, string> = {
  histogram: "📊",
  scatter: "✦",
  bar: "▊",
  pie: "◕",
  line: "↗",
  box: "⊡",
  heatmap: "⊞",
  bespoke: "✦",
  default: "◈",
};

function chartIcon(type: string): string {
  return CHART_TYPE_ICON[type] ?? CHART_TYPE_ICON.default;
}

// ---------------------------------------------------------------------------
// Sub-component: Explanation card shown inside each gallery item
// ---------------------------------------------------------------------------

interface ExplanationBadgeProps {
  explanation: ChartExplanation;
}

function ExplanationBadge({ explanation }: ExplanationBadgeProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-t bg-card/80">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="flex w-full items-center justify-between px-3 py-2 text-left transition-colors hover:bg-muted/50"
      >
        <span className="flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          <Info className="h-3 w-3 flex-shrink-0" />
          Insight
        </span>
        {open ? (
          <ChevronUp className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className="border-t px-3 pb-3 pt-2">
          <p className="font-body text-[11px] leading-relaxed text-foreground/80">
            {explanation.explanation}
          </p>

          {/* Inline stats row */}
          {explanation.stats && Object.keys(explanation.stats).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {explanation.stats.mean != null && (
                <StatPill label="mean" value={explanation.stats.mean} />
              )}
              {explanation.stats.median != null && (
                <StatPill label="median" value={explanation.stats.median} />
              )}
              {explanation.stats.std != null && (
                <StatPill label="std" value={explanation.stats.std} />
              )}
              {explanation.stats.outlier_count != null &&
                explanation.stats.outlier_count > 0 && (
                  <StatPill
                    label="outliers"
                    value={explanation.stats.outlier_count}
                    highlight
                  />
                )}
              {explanation.stats.trend && explanation.stats.trend !== "flat" && (
                <span className="inline-flex items-center gap-1 rounded-full border border-success/30 bg-success/10 px-2 py-0.5 font-mono text-[9px] text-success">
                  <TrendingUp className="h-2.5 w-2.5" />
                  {explanation.stats.trend}
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface StatPillProps {
  label: string;
  value: number;
  highlight?: boolean;
}

function StatPill({ label, value, highlight }: StatPillProps) {
  const formatted =
    Number.isInteger(value) ? String(value) : value.toFixed(2);
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[9px] ${
        highlight
          ? "border-destructive/30 bg-destructive/10 text-destructive"
          : "border-border bg-muted text-muted-foreground"
      }`}
    >
      <span className="opacity-70">{label}</span>
      <span className="font-semibold">{formatted}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Fullscreen viewer with explanation footer
// ---------------------------------------------------------------------------

interface ViewerProps {
  filename: string;
  runId: string;
  explanation: ChartExplanation | null;
  onClose: () => void;
  onDownload: (filename: string) => void;
}

function FullscreenViewer({
  filename,
  runId,
  explanation,
  onClose,
  onDownload,
}: ViewerProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}
    >
      <div
        className="relative flex h-5/6 max-w-5xl w-full flex-col overflow-hidden rounded-lg bg-background shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4 bg-muted/20">
          <div className="flex items-center gap-3 min-w-0">
            {explanation && (
              <span className="flex-shrink-0 text-lg">
                {chartIcon(explanation.chart_type)}
              </span>
            )}
            <div className="min-w-0">
              <h3 className="font-display text-sm font-semibold text-foreground truncate">
                {explanation?.title ?? filename}
              </h3>
              {explanation && (
                <p className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {explanation.chart_type}
                  {explanation.columns_used.length > 0 &&
                    ` · ${explanation.columns_used.join(", ")}`}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-md border text-muted-foreground hover:bg-muted ml-4 flex-shrink-0"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Image */}
        <div className="flex flex-1 items-center justify-center overflow-auto bg-muted/50 p-4">
          <img
            src={getImageURL(runId, filename)}
            alt={explanation?.title ?? filename}
            className="max-h-full max-w-full object-contain"
          />
        </div>

        {/* Explanation footer */}
        {explanation && (
          <div className="border-t bg-card px-6 py-4">
            <div className="flex items-start gap-2">
              <Sparkles className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-primary" />
              <p className="font-body text-sm leading-relaxed text-foreground/80">
                {explanation.explanation}
              </p>
            </div>

            {/* Stats row */}
            {explanation.stats && Object.keys(explanation.stats).length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {explanation.stats.mean != null && (
                  <StatPill label="mean" value={explanation.stats.mean} />
                )}
                {explanation.stats.median != null && (
                  <StatPill label="median" value={explanation.stats.median} />
                )}
                {explanation.stats.std != null && (
                  <StatPill label="std" value={explanation.stats.std} />
                )}
                {explanation.stats.min != null && (
                  <StatPill label="min" value={explanation.stats.min} />
                )}
                {explanation.stats.max != null && (
                  <StatPill label="max" value={explanation.stats.max} />
                )}
                {(explanation.stats.outlier_count ?? 0) > 0 && (
                  <StatPill
                    label="outliers"
                    value={explanation.stats.outlier_count!}
                    highlight
                  />
                )}
                {explanation.stats.trend &&
                  explanation.stats.trend !== "flat" && (
                    <span className="inline-flex items-center gap-1 rounded-full border border-success/30 bg-success/10 px-2 py-0.5 font-mono text-[9px] text-success">
                      <TrendingUp className="h-2.5 w-2.5" />
                      {explanation.stats.trend}
                    </span>
                  )}
              </div>
            )}
          </div>
        )}

        {/* Action bar */}
        <div className="flex items-center justify-end gap-2 border-t bg-muted/20 px-6 py-3">
          <button
            onClick={() => onDownload(filename)}
            className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground hover:bg-muted transition-colors"
          >
            <Download className="h-4 w-4" />
            Download
          </button>
          <button
            onClick={onClose}
            className="flex h-9 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground hover:bg-muted transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Key Insights banner
// ---------------------------------------------------------------------------

interface KeyInsightsBannerProps {
  insights: string[];
}

function KeyInsightsBanner({ insights }: KeyInsightsBannerProps) {
  const [expanded, setExpanded] = useState(false);
  if (!insights.length) return null;

  const visible = expanded ? insights : insights.slice(0, 2);

  return (
    <div className="mb-6 rounded-lg border border-primary/20 bg-primary/5 p-4">
      <div className="mb-2 flex items-center gap-2">
        <BarChart2 className="h-4 w-4 text-primary" />
        <span className="font-display text-xs font-semibold uppercase tracking-wide text-foreground">
          Dataset Insights
        </span>
      </div>
      <ul className="space-y-1">
        {visible.map((insight, i) => (
          <li key={i} className="flex items-start gap-2">
            <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
            <p className="font-body text-xs leading-relaxed text-foreground/80">
              {insight}
            </p>
          </li>
        ))}
      </ul>
      {insights.length > 2 && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 flex items-center gap-1 font-mono text-[10px] text-primary hover:underline"
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3 w-3" /> Show fewer
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3" /> {insights.length - 2} more
            </>
          )}
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main GalleryPanel component
// ---------------------------------------------------------------------------

const GalleryPanel = () => {
  const { images, isLoading, runId, phase } = usePipeline();
  const { toast } = useToast();
  const [instruction, setInstruction] = useState("");
  const [customLoading, setCustomLoading] = useState(false);
  const [displayImages, setDisplayImages] = useState<string[]>(images);
  const [viewingImage, setViewingImage] = useState<string | null>(null);

  // Explanation data keyed by filename
  const [explanationsMap, setExplanationsMap] = useState<
    Record<string, ChartExplanation>
  >({});
  const [keyInsights, setKeyInsights] = useState<string[]>([]);
  const [explanationsLoading, setExplanationsLoading] = useState(false);

  // Sync images from pipeline context
  useEffect(() => {
    setDisplayImages(images);
  }, [images]);

  // Fetch explanations once the artist/complete phase is reached
  const loadExplanations = useCallback(async () => {
    if (!runId) return;
    setExplanationsLoading(true);
    try {
      const data = await fetchChartExplanations(runId);
      if (data) {
        const map: Record<string, ChartExplanation> = {};
        for (const exp of data.explanations) {
          map[exp.filename] = exp;
        }
        setExplanationsMap(map);
        setKeyInsights(data.key_insights ?? []);
      }
    } catch {
      // Silently ignore — explanations are supplementary
    } finally {
      setExplanationsLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    // Load as soon as artist or later phase is active
    if (
      phase === "artist" ||
      phase === "validator" ||
      phase === "complete"
    ) {
      loadExplanations();
    }
  }, [runId, phase, loadExplanations]);

  const handleDownload = (filename: string) => {
    if (!runId) return;
    const url = getImageURL(runId, filename);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
  };

  const refreshImages = async () => {
    if (!runId) return;
    const list = await fetchImagesList(runId);
    setDisplayImages(list.images);
  };

  const handleGenerateCustomChart = async () => {
    if (!runId || !instruction.trim()) return;
    setCustomLoading(true);
    try {
      const response = await generateCustomChart(runId, instruction.trim());
      await refreshImages();
      // Reload explanations to pick up the new custom chart entry
      await loadExplanations();
      setInstruction("");
      toast({
        title: "Chart generated",
        description: `${response.filename} created from your instruction`,
      });
    } catch (error) {
      toast({
        title: "Chart generation failed",
        description:
          error instanceof Error
            ? error.message
            : "Unable to generate custom chart",
        variant: "destructive",
      });
    } finally {
      setCustomLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex-1 overflow-auto">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <div className="mx-auto max-w-7xl">
            {/* Dataset key insights */}
            {keyInsights.length > 0 && (
              <KeyInsightsBanner insights={keyInsights} />
            )}

            {/* Custom Visualization Creator */}
            <div className="mb-6 rounded-lg border bg-card p-5 shadow-sm">
              <p className="mb-3 font-display text-xs font-semibold text-foreground uppercase tracking-wide">
                CREATE CUSTOM VISUALIZATION
              </p>
              <div className="flex items-center gap-2 mb-2">
                <input
                  value={instruction}
                  onChange={(e) => setInstruction(e.target.value)}
                  onKeyDown={(e) =>
                    e.key === "Enter" && handleGenerateCustomChart()
                  }
                  placeholder="e.g., plot salary vs department as bar chart"
                  className="h-10 flex-1 rounded-lg border bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/50"
                />
                <button
                  onClick={handleGenerateCustomChart}
                  disabled={!runId || customLoading || !instruction.trim()}
                  className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {customLoading ? (
                    <Loader className="h-4 w-4 animate-spin" />
                  ) : (
                    <SendHorizontal className="h-4 w-4" />
                  )}
                </button>
              </div>
              <p className="font-body text-xs text-muted-foreground">
                💡 Tip: Use column names, chart type (bar, scatter, line), and
                conditions
              </p>
            </div>

            {/* Charts Gallery */}
            <div>
              <div className="mb-3 flex items-center justify-between">
                <h3 className="font-display text-sm font-semibold text-foreground">
                  Interactive Visualizations
                </h3>
                <div className="flex items-center gap-3">
                  {explanationsLoading && (
                    <span className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
                      <Loader className="h-3 w-3 animate-spin" />
                      loading insights…
                    </span>
                  )}
                  <span className="font-mono text-xs text-muted-foreground">
                    {isLoading ? (
                      <span className="flex items-center gap-1">
                        <Loader className="h-3 w-3 animate-spin" />
                        Generating...
                      </span>
                    ) : (
                      `${displayImages.length} chart${
                        displayImages.length !== 1 ? "s" : ""
                      }`
                    )}
                  </span>
                </div>
              </div>

              {displayImages.length === 0 ? (
                <div className="flex items-center justify-center rounded-lg border border-dashed py-16">
                  {isLoading ? (
                    <div className="text-center">
                      <Loader className="mx-auto mb-2 h-8 w-8 animate-spin text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        Generating visualizations...
                      </p>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No visualizations yet. Complete the Artist phase to see
                      charts.
                    </p>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {displayImages.map((filename) => {
                    const exp = explanationsMap[filename] ?? null;
                    return (
                      <div
                        key={filename}
                        className="group relative flex flex-col rounded-lg border bg-card overflow-hidden shadow-sm transition-all hover:border-primary/50 hover:shadow-md"
                      >
                        {/* Chart type badge */}
                        {exp && (
                          <div className="absolute left-2 top-2 z-10 flex items-center gap-1 rounded-full border border-border/60 bg-background/90 px-2 py-0.5 backdrop-blur-sm">
                            <span className="text-[10px]">
                              {chartIcon(exp.chart_type)}
                            </span>
                            <span className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                              {exp.chart_type}
                            </span>
                          </div>
                        )}

                        {/* Image Container */}
                        <div className="relative h-44 bg-muted/20 overflow-hidden">
                          {runId ? (
                            <>
                              <img
                                src={getImageURL(runId, filename)}
                                alt={exp?.title ?? filename}
                                className="h-full w-full object-cover transition-transform group-hover:scale-105"
                              />
                              {/* Overlay with Actions */}
                              <div className="absolute inset-0 flex items-center justify-center gap-3 bg-black/20 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                                <button
                                  onClick={() => setViewingImage(filename)}
                                  className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 text-black shadow-lg transition-all hover:bg-white"
                                  title="View full screen"
                                >
                                  <Eye className="h-5 w-5" />
                                </button>
                                <button
                                  onClick={() => handleDownload(filename)}
                                  className="flex h-10 w-10 items-center justify-center rounded-full bg-white/95 text-black shadow-lg transition-all hover:bg-white"
                                  title="Download"
                                >
                                  <Download className="h-5 w-5" />
                                </button>
                              </div>
                            </>
                          ) : (
                            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                              No run ID
                            </div>
                          )}
                        </div>

                        {/* Title Footer */}
                        <div className="border-b bg-card/50 px-3 py-2">
                          <p className="truncate font-display text-xs font-semibold text-foreground">
                            {exp?.title ??
                              filename.replace(/\.(png|jpg|jpeg)$/i, "")}
                          </p>
                          {exp?.columns_used && exp.columns_used.length > 0 && (
                            <p className="truncate font-mono text-[9px] text-muted-foreground">
                              {exp.columns_used.join(", ")}
                            </p>
                          )}
                        </div>

                        {/* Expandable explanation */}
                        {exp && <ExplanationBadge explanation={exp} />}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Fullscreen Viewer Modal */}
      {viewingImage && runId && (
        <FullscreenViewer
          filename={viewingImage}
          runId={runId}
          explanation={explanationsMap[viewingImage] ?? null}
          onClose={() => setViewingImage(null)}
          onDownload={handleDownload}
        />
      )}
    </div>
  );
};

export default GalleryPanel;