import { Search, ArrowRight, Loader, RotateCw, AlertCircle, ChevronDown, ChevronUp, Plus, X } from "lucide-react";
import { useState, useRef } from "react";
import { usePipeline } from "@/context/PipelineContext";
import { useToast } from "@/hooks/use-toast";

const InputZone = () => {
  const { url, isRunning, isLoading, run, startPollingWithRunId, errorMessage, retry, status, debugLogs } = usePipeline();
  const [inputValue, setInputValue] = useState(url || "");
  const [showDebugLogs, setShowDebugLogs] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleSubmit = async () => {
    const source = selectedFile || inputValue.trim();
    
    if (!source) {
      toast({
        title: "Error",
        description: "Please enter a source (URL, file path, or upload a file)",
        variant: "destructive",
      });
      return;
    }

    try {
      // If file is selected, send as FormData
      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}/run`, {
          method: "POST",
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || "Failed to start pipeline with file");
        }
        
        const data = await response.json();
        const runId = data.run_id;
        
        if (!runId) {
          throw new Error("No run_id received from server");
        }
        
        // Start polling immediately with the run_id
        startPollingWithRunId(runId, selectedFile.name);
        
        toast({
          title: "Pipeline started",
          description: `Processing ${selectedFile.name}...`,
        });
      } else {
        // For URL/path input, use the normal run() function
        await run(inputValue.trim());
        
        toast({
          title: "Pipeline started",
          description: `Acquiring data from ${inputValue.trim()}`,
        });
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to start pipeline";
      toast({
        title: "Error",
        description: message,
        variant: "destructive",
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const validExtensions = [".csv", ".xlsx", ".xls", ".json"];
      const fileExt = "." + file.name.split(".").pop()?.toLowerCase();
      
      if (!validExtensions.includes(fileExt)) {
        toast({
          title: "Invalid file type",
          description: "Please upload a CSV, XLSX, or JSON file",
          variant: "destructive",
        });
        return;
      }
      
      setSelectedFile(file);
      setInputValue(""); // Clear text input when file is selected
      toast({
        title: "File selected",
        description: `Ready to process ${file.name}`,
      });
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isRunning && !isLoading) {
      handleSubmit();
    }
  };

  return (
    <div className="px-6 pb-4 space-y-3">
      {/* File Upload Section */}
      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls,.json"
          onChange={handleFileSelect}
          className="hidden"
          disabled={isRunning || isLoading}
        />
        
        {selectedFile ? (
          <div className="flex items-center gap-2 p-3 rounded-md bg-blue-50 border border-blue-200">
            <div className="flex-1 flex items-center gap-2 min-w-0">
              <div className="w-8 h-8 flex items-center justify-center rounded bg-blue-600 text-white text-xs font-semibold flex-shrink-0">
                {selectedFile.name.split(".").pop()?.toUpperCase() || "FILE"}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-blue-900 truncate">{selectedFile.name}</p>
                <p className="text-xs text-blue-700">{(selectedFile.size / 1024).toFixed(2)} KB</p>
              </div>
            </div>
            <button
              onClick={handleClearFile}
              disabled={isRunning || isLoading}
              className="flex-shrink-0 p-1 hover:bg-blue-100 rounded transition-colors disabled:opacity-50"
              title="Remove file"
            >
              <X className="h-4 w-4 text-blue-600" />
            </button>
          </div>
        ) : (
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRunning || isLoading}
            className="w-full flex items-center justify-center gap-2 p-3 rounded-md border-2 border-dashed border-muted-foreground/30 bg-card hover:border-success hover:bg-success/5 transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm font-medium text-muted-foreground">Upload dataset (CSV, XLSX, JSON)</span>
          </button>
        )}
      </div>

      {/* Text Input Section */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Or paste URL / file path:</p>
        <div className="relative flex items-center">
          <Search className="absolute left-4 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste a URL, local file path, or raw text to begin…"
            disabled={isRunning || isLoading || selectedFile !== null}
            className="h-11 w-full rounded-md border bg-card pl-11 pr-28 font-body text-sm text-foreground placeholder:text-muted-foreground focus:border-success focus:outline-none focus:ring-1 focus:ring-success transition-colors duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {status === "error" ? (
            <button
              onClick={async () => {
                await retry();
                toast({
                  title: "Retrying",
                  description: "Attempting to resume pipeline...",
                });
              }}
              disabled={isLoading}
              className="absolute right-1.5 flex h-8 items-center gap-1.5 rounded-md bg-orange-600 px-4 font-display text-xs font-medium text-white transition-colors duration-150 hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader className="h-3.5 w-3.5 animate-spin" />
                  Retrying
                </>
              ) : (
                <>
                  <RotateCw className="h-3.5 w-3.5" />
                  Retry
                </>
              )}
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={isRunning || isLoading || (!inputValue.trim() && !selectedFile)}
              className="absolute right-1.5 flex h-8 items-center gap-1.5 rounded-md bg-primary px-4 font-display text-xs font-medium text-primary-foreground transition-colors duration-150 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader className="h-3.5 w-3.5 animate-spin" />
                  Starting
                </>
              ) : (
                <>
                  Run
                  <ArrowRight className="h-3.5 w-3.5" />
                </>
              )}
            </button>
          )}
        </div>
      </div>
      {errorMessage && (
        <div className="mt-2 space-y-2">
          <div className="flex items-start gap-2 rounded-md bg-red-50 p-2.5">
            <AlertCircle className="h-4 w-4 shrink-0 text-red-600 mt-0.5" />
            <p className="text-xs text-red-600 flex-1">{errorMessage}</p>
          </div>
          {debugLogs.length > 0 && (
            <button
              onClick={() => setShowDebugLogs(!showDebugLogs)}
              className="flex items-center gap-1.5 text-xs text-red-600 hover:text-red-700 font-medium px-2.5 py-1 rounded hover:bg-red-50 transition-colors"
            >
              {showDebugLogs ? (
                <>
                  <ChevronUp className="h-3.5 w-3.5" />
                  Hide debug logs
                </>
              ) : (
                <>
                  <ChevronDown className="h-3.5 w-3.5" />
                  Show debug logs ({debugLogs.length})
                </>
              )}
            </button>
          )}
          {showDebugLogs && debugLogs.length > 0 && (
            <div className="rounded-md bg-red-950 p-3 max-h-48 overflow-y-auto">
              {debugLogs.map((log, idx) => (
                <div key={idx} className="text-xs text-red-100 font-mono py-0.5 break-all">
                  {log}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InputZone;
