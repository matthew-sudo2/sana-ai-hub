import { ChevronLeft, ChevronRight, Download, Loader, Sigma, Home } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "@/context/PipelineContext";
import { downloadCSV, fetchSummaryStats, type ColumnStats } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const DataViewer = () => {
  const { csvData, isLoading, runId } = usePipeline();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 50;
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
    <div className="flex min-h-screen bg-background flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b px-6 py-3">
        <div>
          <h1 className="font-display text-base font-bold text-foreground">Data Viewer</h1>
          <p className="font-body text-[11px] text-muted-foreground">Full cleaned dataset explorer</p>
        </div>
        <button
          onClick={() => navigate("/")}
          className="flex h-9 items-center gap-2 rounded-md border px-3 text-sm text-muted-foreground transition-colors hover:bg-muted"
        >
          <Home className="h-4 w-4" />
          Back to Dashboard
        </button>
      </header>

      {/* Main content */}
      <div className="flex-1 overflow-auto">
        <div className="flex flex-col h-full">
          {/* Toolbar */}
          <div className="flex items-center justify-between border-b px-6 py-4 bg-muted/20">
            <div>
              <h3 className="font-display text-sm font-semibold text-foreground">Cleaned Data</h3>
              <p className="font-body text-xs text-muted-foreground mt-1">
                {isLoading ? (
                  <span className="flex items-center gap-1">
                    <Loader className="h-3 w-3 animate-spin" />
                    Loading...
                  </span>
                ) : (
                  `Total: ${rows.length} rows | Showing ${displayedRows.length} rows per page`
                )}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleDownloadCSV}
                disabled={!runId || rows.length === 0 || isLoading}
                className="flex h-9 items-center gap-1 rounded-md border px-3 text-sm text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                CSV
              </button>
              <button
                onClick={handleCalculateStats}
                disabled={!runId || rows.length === 0 || isLoading || statsLoading}
                className="flex h-9 items-center gap-1 rounded-md border px-3 text-sm text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
              >
                {statsLoading ? <Loader className="h-4 w-4 animate-spin" /> : <Sigma className="h-4 w-4" />}
                Stats
              </button>
            </div>
          </div>

          {/* Table */}
          <div className="flex-1 overflow-auto px-6 py-4">
            {rows.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                {isLoading ? (
                  <div className="space-y-2 w-full">
                    {[...Array(10)].map((_, i) => (
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
              <div className="border rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead>
                    <tr className="border-b bg-muted/50 sticky top-0">
                      {columns.map((col) => (
                        <th
                          key={col}
                          className="px-4 py-3 text-left font-display text-xs font-semibold uppercase tracking-wider text-muted-foreground whitespace-nowrap"
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
                          <td key={`${idx}-${col}`} className="px-4 py-3 font-mono text-xs text-foreground">
                            {String(row[col]).substring(0, 100)}
                            {String(row[col]).length > 100 ? "..." : ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between border-t px-6 py-4 bg-muted/20">
            <span className="font-body text-sm text-muted-foreground">
              Page {rows.length > 0 ? currentPage : 0} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                onClick={handlePrevPage}
                disabled={currentPage === 1 || rows.length === 0}
                className="flex h-9 w-9 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                onClick={handleNextPage}
                disabled={currentPage === totalPages || rows.length === 0}
                className="flex h-9 w-9 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics Panel */}
      {(statsLoading || statsError || (stats && stats.length > 0)) && (
        <div className="border-t bg-muted/20 px-6 py-4 max-h-96 overflow-auto">
          <h4 className="mb-4 font-display text-sm font-semibold text-foreground">Descriptive Statistics</h4>
          {statsLoading ? (
            <p className="text-sm text-muted-foreground">Calculating statistics...</p>
          ) : statsError ? (
            <p className="text-sm text-destructive">{statsError}</p>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {stats?.map((item) => (
                <div key={item.column} className="rounded border bg-card p-3">
                  <p className="mb-2 font-display text-xs font-semibold text-foreground">{item.column}</p>
                  <div className="grid grid-cols-2 gap-x-2 gap-y-1 font-mono text-[10px] text-muted-foreground">
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
  );
};

export default DataViewer;
