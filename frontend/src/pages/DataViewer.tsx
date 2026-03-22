import { ChevronLeft, ChevronRight, Download, Loader, Home, Search, ArrowUp, ArrowDown } from "lucide-react";
import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { usePipeline } from "@/context/PipelineContext";
import { downloadCSV } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const DataViewer = () => {
  const { csvData, isLoading, runId } = usePipeline();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 25;
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("none");
  const [sortOrder, setSortOrder] = useState("ascending");
  const [visibleColumns, setVisibleColumns] = useState<string[]>([]);

  const rows = csvData || [];
  const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

  // Update visible columns when data loads
  useEffect(() => {
    if (columns.length > 0 && visibleColumns.length === 0) {
      setVisibleColumns(columns);
    }
  }, [columns, visibleColumns]);

  // Calculate stats
  const stats = useMemo(() => {
    if (rows.length === 0) return { totalRows: 0, totalCols: 0, completeCols: 0, avgQuality: 0 };

    let completeCount = 0;
    let totalQuality = 0;

    columns.forEach((col) => {
      const complete = rows.filter((row) => row[col] != null && String(row[col]).trim() !== "").length;
      const quality = (complete / rows.length) * 100;
      if (quality === 100) completeCount++;
      totalQuality += quality;
    });

    return {
      totalRows: rows.length,
      totalCols: columns.length,
      completeCols: completeCount,
      avgQuality: columns.length > 0 ? totalQuality / columns.length : 0,
    };
  }, [rows, columns]);

  // Column quality details
  const columnQuality = useMemo(() => {
    const quality: Record<string, { complete: number; total: number; percentage: number }> = {};
    columns.forEach((col) => {
      const total = rows.length;
      const complete = rows.filter((row) => row[col] != null && String(row[col]).trim() !== "").length;
      quality[col] = { complete, total, percentage: total > 0 ? (complete / total) * 100 : 0 };
    });
    return quality;
  }, [rows, columns]);

  // Filter rows
  const filteredRows = useMemo(() => {
    let filtered = rows;

    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter((row) =>
        visibleColumns.some((col) => String(row[col]).toLowerCase().includes(term))
      );
    }

    if (sortBy !== "none") {
      filtered = [...filtered].sort((a, b) => {
        const aVal = a[sortBy];
        const bVal = b[sortBy];
        const cmp = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        return sortOrder === "ascending" ? cmp : -cmp;
      });
    }

    return filtered;
  }, [rows, searchTerm, sortBy, sortOrder, visibleColumns]);

  const totalPages = filteredRows.length > 0 ? Math.ceil(filteredRows.length / rowsPerPage) : 1;
  const startIdx = (currentPage - 1) * rowsPerPage;
  const displayedRows = filteredRows.slice(startIdx, startIdx + rowsPerPage);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, sortBy]);

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
      toast({ title: "Downloaded", description: "CSV downloaded successfully" });
    } catch (error) {
      toast({
        title: "Download failed",
        description: error instanceof Error ? error.message : "Could not download CSV",
        variant: "destructive",
      });
    }
  };

  const getQualityColor = (percentage: number) => {
    if (percentage === 100) return "bg-green-500/10 border-green-500/30";
    if (percentage >= 90) return "bg-yellow-500/10 border-yellow-500/30";
    if (percentage >= 70) return "bg-orange-500/10 border-orange-500/30";
    return "bg-red-500/10 border-red-500/30";
  };

  const getQualityLabel = (percentage: number) => {
    if (percentage === 100) return "Complete";
    if (percentage >= 90) return "Good";
    if (percentage >= 70) return "Fair";
    return "Poor";
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="font-display text-2xl font-bold text-foreground">Data Viewer</h1>
          <p className="font-body text-sm text-muted-foreground">Cleaned & processed dataset with quality metrics</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadCSV}
            disabled={rows.length === 0 || isLoading}
            className="flex h-10 items-center gap-2 rounded-md bg-primary px-4 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
          >
            <Download className="h-4 w-4" />
            Export CSV
          </button>
          <button
            onClick={() => navigate("/")}
            className="flex h-10 items-center gap-2 rounded-md border px-4 text-sm text-muted-foreground transition-colors hover:bg-muted"
          >
            <Home className="h-4 w-4" />
            Dashboard
          </button>
        </div>
      </header>

      {/* Stats Cards */}
      {rows.length > 0 && (
        <div className="grid grid-cols-4 gap-4 border-b px-6 py-4">
          <div className="rounded-lg border bg-card p-4">
            <p className="font-body text-sm text-muted-foreground">Total Rows</p>
            <p className="font-display text-3xl font-bold text-foreground">{stats.totalRows.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="font-body text-sm text-muted-foreground">Total Columns</p>
            <p className="font-display text-3xl font-bold text-foreground">{stats.totalCols}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="font-body text-sm text-muted-foreground">Complete Columns</p>
            <p className="font-display text-3xl font-bold text-green-500">{stats.completeCols}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="font-body text-sm text-muted-foreground">Data Quality</p>
            <p className="font-display text-3xl font-bold text-blue-500">{Math.round(stats.avgQuality)}%</p>
          </div>
        </div>
      )}

      {/* Column Quality Indicators */}
      {rows.length > 0 && (
        <div className="border-b px-6 py-4">
          <h3 className="mb-3 font-display text-sm font-semibold text-foreground">Column Quality Indicators</h3>
          <div className="grid auto-rows-auto gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))" }}>
            {columns.map((col) => {
              const quality = columnQuality[col];
              const percentage = quality.percentage;
              return (
                <div key={col} className={`rounded-lg border p-3 ${getQualityColor(percentage)}`}>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-display text-xs font-semibold text-foreground">{col}</span>
                    <span className="font-mono text-xs font-bold">{percentage.toFixed(0)}%</span>
                  </div>
                  <div className="mb-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className={`h-full transition-all ${
                        percentage === 100
                          ? "bg-green-500"
                          : percentage >= 90
                            ? "bg-yellow-500"
                            : percentage >= 70
                              ? "bg-orange-500"
                              : "bg-red-500"
                      }`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <p className="font-body text-[11px] text-muted-foreground">
                    {getQualityLabel(percentage)} • {quality.complete}/{quality.total}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Data Controls */}
      {rows.length > 0 && (
        <div className="border-b px-6 py-4">
          <h3 className="mb-3 font-display text-sm font-semibold text-foreground">Data Controls</h3>
          <div className="space-y-3">
            <div>
              <p className="mb-2 font-body text-xs text-muted-foreground">Search</p>
              <div className="flex items-center gap-2 rounded-lg border bg-background px-3 py-2">
                <Search className="h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search all columns..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="flex-1 bg-transparent text-sm outline-none"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="mb-2 font-body text-xs text-muted-foreground">Sort By</p>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none"
                >
                  <option value="none">None</option>
                  {visibleColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <p className="mb-2 font-body text-xs text-muted-foreground">Order</p>
                <select
                  value={sortOrder}
                  onChange={(e) => setSortOrder(e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none"
                >
                  <option value="ascending">Ascending</option>
                  <option value="descending">Descending</option>
                </select>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Columns Visible: {visibleColumns.length} of {columns.length}</span>
              <div className="flex gap-2">
                <button className="rounded px-2 py-1 hover:bg-muted">Show All</button>
                <button className="rounded px-2 py-1 hover:bg-muted">All visible</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Data Records Table */}
      <div className="flex-1 overflow-auto">
        <div className="px-6 py-4">
          {rows.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              {isLoading ? (
                <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
              ) : (
                <p className="text-sm text-muted-foreground">No data yet. Start a pipeline to see results.</p>
              )}
            </div>
          ) : (
            <div>
              <h3 className="mb-3 font-display text-sm font-semibold text-foreground">
                Data Records
              </h3>
              <p className="mb-3 font-body text-xs text-muted-foreground">
                Showing {startIdx + 1} to {Math.min(startIdx + rowsPerPage, filteredRows.length)} of {filteredRows.length} records
              </p>
              <div className="overflow-x-auto rounded-lg border">
                <table className="w-full">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      {visibleColumns.map((col) => (
                        <th key={col} className="px-4 py-3 text-left font-display text-xs font-semibold uppercase">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {displayedRows.map((row, idx) => (
                      <tr key={idx} className="border-b transition-colors hover:bg-muted/30">
                        {visibleColumns.map((col) => {
                          const value = row[col];
                          const isEmpty = value == null || String(value).trim() === "";
                          return (
                            <td
                              key={`${idx}-${col}`}
                              className={`px-4 py-3 font-mono text-xs ${
                                isEmpty ? "bg-red-500/10 text-red-500" : "text-foreground"
                              }`}
                            >
                              {isEmpty ? "∅" : String(value).substring(0, 50)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    Page {currentPage} of {totalPages}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="flex h-8 w-8 items-center justify-center rounded-md border text-muted-foreground transition-colors hover:bg-muted disabled:opacity-50"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataViewer;
