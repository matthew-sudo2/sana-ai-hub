import { X, Activity } from "lucide-react";
import { ColumnStatistics, formatStatValue } from "@/lib/statistics";

interface StatisticsPanelProps {
  stats: ColumnStatistics[];
  onClose: () => void;
}

const StatisticsPanel = ({ stats, onClose }: StatisticsPanelProps) => {
  const numericStats = stats.filter((s) => s.isNumeric);

  if (numericStats.length === 0) {
    return (
      <div className="rounded-lg border bg-card shadow-sm p-4">
        <div className="text-center py-8">
          <p className="text-sm text-muted-foreground">
            No numeric columns selected. Please select columns with numeric data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-card shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <h3 className="font-display text-sm font-semibold text-foreground">Statistical Analysis</h3>
          <p className="font-body text-xs text-muted-foreground">Descriptive statistics for numeric columns</p>
        </div>
        <button
          onClick={onClose}
          className="flex h-7 w-7 items-center justify-center rounded hover:bg-muted transition-colors"
        >
          <X className="h-4 w-4 text-muted-foreground" />
        </button>
      </div>

      {/* Statistics Grid */}
      <div className="p-4 space-y-6">
        {numericStats.map((colStats) => (
          <div key={colStats.columnName}>
            {/* Column Name Header */}
            <h4 className="font-display text-xs font-bold text-primary uppercase tracking-wide mb-3">
              {colStats.columnName}
            </h4>

            {/* Statistics Grid (2x3) */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {/* Mean */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>📊</span> MEAN
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.mean)}
                </p>
              </div>

              {/* Median */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>📌</span> MEDIAN
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.median)}
                </p>
              </div>

              {/* Mode */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>🎯</span> MODE
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.mode)}
                </p>
              </div>

              {/* Range */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>↔️</span> RANGE
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.range)}
                </p>
              </div>

              {/* Variance */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>📐</span> VARIANCE
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.variance)}
                </p>
              </div>

              {/* Standard Deviation */}
              <div className="rounded-lg border bg-background/50 p-3">
                <p className="font-body text-xs text-muted-foreground mb-1">
                  <span className="flex items-center gap-1">
                    <span>√</span> STD DEVIATION
                  </span>
                </p>
                <p className="font-display text-sm font-bold text-foreground">
                  {formatStatValue(colStats.stdDeviation)}
                </p>
              </div>
            </div>

            {/* Min/Max Footer */}
            <div className="mt-3 grid grid-cols-2 gap-3 pt-3 border-t">
              <div className="text-xs">
                <p className="text-muted-foreground">Minimum</p>
                <p className="font-display font-semibold text-foreground">
                  {formatStatValue(colStats.min)}
                </p>
              </div>
              <div className="text-xs">
                <p className="text-muted-foreground">Maximum</p>
                <p className="font-display font-semibold text-foreground">
                  {formatStatValue(colStats.max)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatisticsPanel;
