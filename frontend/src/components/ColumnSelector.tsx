import { X, CheckCircle2, Circle } from "lucide-react";

interface ColumnSelectorProps {
  columns: string[];
  selectedColumns: string[];
  onSelectionChange: (selected: string[]) => void;
  onClose: () => void;
  onAnalyze: () => void;
}

const ColumnSelector = ({
  columns,
  selectedColumns,
  onSelectionChange,
  onClose,
  onAnalyze,
}: ColumnSelectorProps) => {
  const handleColumnToggle = (column: string) => {
    if (selectedColumns.includes(column)) {
      onSelectionChange(selectedColumns.filter((c) => c !== column));
    } else {
      onSelectionChange([...selectedColumns, column]);
    }
  };

  const handleSelectAll = () => {
    if (selectedColumns.length === columns.length) {
      onSelectionChange([]);
    } else {
      onSelectionChange([...columns]);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="relative w-full max-w-md bg-background rounded-lg border shadow-lg">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <div>
            <h3 className="font-display text-sm font-semibold text-foreground">Select Columns for Analysis</h3>
            <p className="font-body text-xs text-muted-foreground mt-1">
              Choose which columns to include in statistical analysis
            </p>
          </div>
          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded hover:bg-muted transition-colors flex-shrink-0"
          >
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        {/* Select All */}
        <div className="border-b px-4 py-3">
          <button
            onClick={handleSelectAll}
            className="flex items-center gap-2 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
          >
            {selectedColumns.length === columns.length ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <Circle className="h-4 w-4" />
            )}
            {selectedColumns.length === columns.length ? "Deselect All" : "Select All"}
          </button>
        </div>

        {/* Column List */}
        <div className="max-h-64 overflow-y-auto p-4 space-y-2">
          {columns.map((column) => (
            <button
              key={column}
              onClick={() => handleColumnToggle(column)}
              className="w-full flex items-center gap-2 p-2 rounded-lg border hover:bg-muted/50 transition-colors text-left"
            >
              {selectedColumns.includes(column) ? (
                <CheckCircle2 className="h-4 w-4 text-success flex-shrink-0" />
              ) : (
                <Circle className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              )}
              <span className="font-body text-sm text-foreground flex-1">{column}</span>
            </button>
          ))}
        </div>

        {/* Footer */}
        <div className="border-t p-4 flex items-center gap-2 justify-end">
          <button
            onClick={onClose}
            className="px-3 py-2 text-xs font-medium text-muted-foreground border rounded-lg hover:bg-muted transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onAnalyze}
            disabled={selectedColumns.length === 0}
            className="px-3 py-2 text-xs font-medium text-primary-foreground bg-primary rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Analyze ({selectedColumns.length})
          </button>
        </div>
      </div>
    </div>
  );
};

export default ColumnSelector;
