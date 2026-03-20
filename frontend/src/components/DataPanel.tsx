import { ChevronLeft, ChevronRight, Download, Loader, Sigma } from "lucide-react";
import { useEffect, useState } from "react";
import { usePipeline } from "@/context/PipelineContext";
import { downloadCSV, fetchSummaryStats, type ColumnStats } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const DataPanel = () => {
  const { csvData, isLoading, runId } = usePipeline();
  const { toast } = useToast();
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  const [stats, setStats] = useState<ColumnStats[] | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  const rows = csvData || [];
  const totalPages = rows.length > 0 ? Math.ceil(rows.length / rowsPerPage) : 1;
  const startIdx = (currentPage - 1) * rowsPerPage;
  const displayedRows = rows.slice(startIdx, startIdx + rowsPerPage);
  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  useEffect(() => {
    setCurrentPage((prev) => Math.min(prev, totalPages));
  }, [totalPages]);

  const handlePrevPage = () => {
    setCurrentPage((p) => Math.max(1, p - 1));
  };

  const handleNextPage = () => {
    setCurrentPage((p) => Math.min(totalPages, p + 1));
  };

  const handleDownloadCSV = async () => {
    if (!runId) return;
    try {
      const blob = await downloadCSV(runId);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `cleaned_data_${runId}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      toast({ title: "Downloaded", description: "Cleaned CSV downloaded successfully" });
    } catch (error) {
      toast({
        title: "Download failed",
        description: error instanceof Error ? error.message : "Could not download cleaned CSV",
        variant: "destructive",
      });
    }
  };

  const handleCalculateStats = async () => {
    if (!runId) return;
    setStatsLoading(true);
    setStatsError(null);
    try {
      const response = await fetchSummaryStats(runId);
      setStats(response.stats);
      if (response.stats.length === 0) {
        setStatsError("No numeric columns available for descriptive statistics.");
      }
    } catch (error) {
      setStatsError(error instanceof Error ? error.message : "Failed to calculate statistics");
      setStats(null);
    } finally {
      setStatsLoading(false);
    }
  };

  const formatMetric = (value: number | null): string => {
    if (value === null || Number.isNaN(value)) return "N/A";
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="font-display text-sm font-semibold text-foreground">Cleaned Data</h3>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">
            {isLoading ? (
              <span className="flex items-center gap-1">
                <Loader className="h-3 w-3 animate-spin" />
                Loading...
              </span>
            ) : (
              `${rows.length} rows`
            )}
          </span>
          <button
            onClick={handleDownloadCSV}
            disabled={!runId || rows.length === 0 || isLoading}
            className="flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5" />
            CSV
          </button>
          <button
            onClick={handleCalculateStats}
            disabled={!runId || rows.length === 0 || isLoading || statsLoading}
            className="flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
          >
            {statsLoading ? <Loader className="h-3.5 w-3.5 animate-spin" /> : <Sigma className="h-3.5 w-3.5" />}
            Stats
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {rows.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            {isLoading ? (
              <div className="space-y-2 w-full p-4">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-8 bg-muted animate-pulse rounded" />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                No data yet. Start a pipeline to see results.
              </p>
            )}
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b bg-muted/50 sticky top-0">
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-4 py-2 text-left font-display text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayedRows.map((row, idx) => (
                <tr key={idx} className="border-b transition-colors duration-150 hover:bg-muted/30">
                  {columns.map((col) => (
                    <td key={`${idx}-${col}`} className="px-4 py-2.5 font-mono text-xs text-foreground">
                      {String(row[col]).substring(0, 50)}
                      {String(row[col]).length > 50 ? "..." : ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {(statsLoading || statsError || (stats && stats.length > 0)) && (
          <div className="border-t bg-muted/20 px-4 py-3">
            <h4 className="mb-2 font-display text-xs font-semibold text-foreground">Descriptive Statistics</h4>
            {statsLoading ? (
              <p className="text-xs text-muted-foreground">Calculating statistics...</p>
            ) : statsError ? (
              <p className="text-xs text-destructive">{statsError}</p>
            ) : (
              <div className="space-y-2">
                {stats?.map((item) => (
                  <div key={item.column} className="rounded border bg-card p-2">
                    <p className="mb-1 font-display text-[11px] font-semibold text-foreground">{item.column}</p>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px] text-muted-foreground">
                      <span>Std Dev: {formatMetric(item.standard_deviation)}</span>
                      <span>Variance: {formatMetric(item.variance)}</span>
                      <span>Min: {formatMetric(item.min)}</span>
                      <span>Max: {formatMetric(item.max)}</span>
                      <span>Range: {formatMetric(item.range)}</span>
                      <span>Median: {formatMetric(item.median)}</span>
                      <span>Mode: {formatMetric(item.mode)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center justify-between border-t px-4 py-2.5">
        <span className="font-body text-[11px] text-muted-foreground">
          Page {rows.length > 0 ? currentPage : 0} of {totalPages}
        </span>
        <div className="flex gap-1">
          <button
            onClick={handlePrevPage}
            disabled={currentPage === 1 || rows.length === 0}
            className="flex h-7 w-7 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleNextPage}
            disabled={currentPage === totalPages || rows.length === 0}
            className="flex h-7 w-7 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default DataPanel;
