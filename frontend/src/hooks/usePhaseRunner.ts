/**
 * usePhaseRunner: Custom hook for managing pipeline execution and polling.
 * Handles starting pipelines, polling for progress, and fetching results.
 */

import { useEffect, useRef, useState } from "react";
import {
  checkStatus,
  downloadCSV,
  fetchImagesList,
  fetchReport,
  parseCSV,
  startPipeline,
  APIError,
} from "@/lib/api";

export interface PhaseRunnerState {
  // Run metadata
  runId: string | null;
  url: string | null;
  startedAt: Date | null;

  // Current phase progress
  phase: string; // "scout", "labeler", "analyst", "artist", "validator", "complete"
  status: string; // "pending", "running", "success", "error"
  errorMessage: string | null;
  debugLogs: string[]; // Detailed error logs for debugging

  // Loading states
  isRunning: boolean;
  isLoading: boolean;

  // Results
  csvData: Record<string, string>[] | null;
  images: string[];
  reportContent: string | null;
}

const initialState: PhaseRunnerState = {
  runId: null,
  url: null,
  startedAt: null,
  phase: "pending",
  status: "pending",
  errorMessage: null,
  debugLogs: [],
  isRunning: false,
  isLoading: false,
  csvData: null,
  images: [],
  reportContent: null,
};

export function usePhaseRunner() {
  const [state, setState] = useState<PhaseRunnerState>(initialState);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Start a new pipeline run.
   */
  const run = async (url: string): Promise<void> => {
    setState((prev) => ({
      ...prev,
      isRunning: true,
      isLoading: true,
      errorMessage: null,
      debugLogs: [],
      phase: "pending",
      status: "pending",
      url,
      startedAt: new Date(),
      csvData: null,
      images: [],
      reportContent: null,
    }));

    try {
      setState((prev) => ({
        ...prev,
        debugLogs: [...prev.debugLogs, `[INIT] Starting pipeline for URL: ${url}`],
      }));
      
      let runId: string;
      try {
        runId = await startPipeline(url);
      } catch (fetchErr) {
        let detailedError = "Failed to fetch";
        
        if (fetchErr instanceof Error) {
          const errMsg = fetchErr.message.toLowerCase();
          if (errMsg.includes("failed to fetch")) {
            detailedError = "Cannot connect to backend API (http://localhost:8000). Is the server running? Try running: python start.py";
          } else if (errMsg.includes("cors")) {
            detailedError = "CORS error - backend may need to be restarted";
          } else if (errMsg.includes("network")) {
            detailedError = "Network error - cannot reach backend. Ensure backend is running on localhost:8000";
          } else {
            detailedError = `API Error: ${fetchErr.message}`;
          }
        }
        
        throw new Error(detailedError);
      }

      setState((prev) => ({
        ...prev,
        runId,
        isLoading: false,
        debugLogs: [...prev.debugLogs, `[INIT] Run created: ${runId}`],
      }));

      // Start polling
      startPolling(runId);
    } catch (error) {
      const message =
        error instanceof APIError
          ? error.message
          : error instanceof Error
            ? error.message
            : "Failed to start pipeline";

      setState((prev) => ({
        ...prev,
        isRunning: false,
        isLoading: false,
        errorMessage: message,
        status: "error",
        debugLogs: [...prev.debugLogs, `[ERROR] Failed to start pipeline: ${message}`],
      }));
    }
  };

  /**
   * Start polling for pipeline progress.
   */
  const startPolling = (runId: string): void => {
    // Clear existing interval
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    let pollRetries = 0;
    const maxRetries = 3;

    // Poll immediately, then every 2 seconds
    const poll = async () => {
      try {
        const status = await checkStatus(runId);
        pollRetries = 0; // Reset on success
        
        // Add debug log if error occurred
        const newDebugLog = status.error 
          ? `[${status.phase.toUpperCase()}] ${status.error}`
          : `[${status.phase.toUpperCase()}] Phase running...`;
        
        setState((prev) => ({
          ...prev,
          phase: status.phase,
          status: status.status,
          errorMessage: status.error,
          debugLogs: status.error && !prev.debugLogs.includes(newDebugLog)
            ? [...prev.debugLogs, newDebugLog]
            : prev.debugLogs,
        }));

        // If complete or error, stop polling and fetch results
        if (status.phase === "complete") {
          stopPolling();
          await fetchResults(runId);
        } else if (status.status === "error") {
          stopPolling();
        }
      } catch (error) {
        pollRetries++;
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error(`Polling error (attempt ${pollRetries}/${maxRetries}):`, error);

        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[POLLING] Attempt ${pollRetries}/${maxRetries}: ${errorMsg}`],
        }));

        // Stop polling after max retries
        if (pollRetries >= maxRetries) {
          stopPolling();
          setState((prev) => ({
            ...prev,
            isRunning: false,
            isLoading: false,
            errorMessage:
              "Connection lost. The pipeline may still be running. Refresh to check status.",
            status: "error",
            debugLogs: [...prev.debugLogs, `[CRITICAL] Max polling retries reached (${maxRetries})`],
          }));
        }
      }
    };

    // First poll immediately
    poll();

    // Then poll every 2 seconds
    pollIntervalRef.current = setInterval(poll, 2000);
  };

  /**
   * Stop polling.
   */
  const stopPolling = (): void => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  /**
   * Fetch all results after pipeline completes.
   */
  const fetchResults = async (runId: string): Promise<void> => {
    try {
      setState((prev) => ({
        ...prev,
        debugLogs: [...prev.debugLogs, `[RESULTS] Fetching CSV, images, and report...`],
      }));

      // Fetch each result independently with error tracking
      let csvBlob: Blob | null = null;
      let imagesList: { images: string[]; count: number } | null = null;
      let report: { content: string; filename: string } | null = null;

      // Fetch CSV
      try {
        csvBlob = await downloadCSV(runId);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] CSV downloaded successfully`],
        }));
      } catch (csvErr) {
        const csvErrorMsg = csvErr instanceof Error ? csvErr.message : String(csvErr);
        console.warn("CSV fetch failed:", csvErr);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] CSV fetch failed: ${csvErrorMsg}`],
        }));
      }

      // Fetch Images
      try {
        imagesList = await fetchImagesList(runId);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] Images list fetched (${imagesList?.count || 0} images)`],
        }));
      } catch (imgErr) {
        const imgErrorMsg = imgErr instanceof Error ? imgErr.message : String(imgErr);
        console.warn("Images list fetch failed:", imgErr);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] Images fetch failed: ${imgErrorMsg}`],
        }));
      }

      // Fetch Report
      try {
        report = await fetchReport(runId);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] Report fetched successfully`],
        }));
      } catch (reportErr) {
        const reportErrorMsg = reportErr instanceof Error ? reportErr.message : String(reportErr);
        console.warn("Report fetch failed:", reportErr);
        setState((prev) => ({
          ...prev,
          debugLogs: [...prev.debugLogs, `[RESULTS] Report fetch failed: ${reportErrorMsg}`],
        }));
      }

      // Parse results with defensive checks
      let csvData: Record<string, string>[] | null = null;
      if (csvBlob) {
        try {
          const csvText = await csvBlob.text();
          csvData = parseCSV(csvText);
        } catch (parseErr) {
          const parseErrorMsg = parseErr instanceof Error ? parseErr.message : String(parseErr);
          console.warn("CSV parsing failed:", parseErr);
          setState((prev) => ({
            ...prev,
            debugLogs: [...prev.debugLogs, `[RESULTS] CSV parsing failed: ${parseErrorMsg}`],
          }));
        }
      }

      const images = imagesList?.images ?? [];
      const reportContent = report?.content ?? null;

      setState((prev) => ({
        ...prev,
        isRunning: false,
        isLoading: false,
        csvData,
        images,
        reportContent,
        debugLogs: [...prev.debugLogs, `[RESULTS] Complete - CSV: ${csvData ? 'yes' : 'no'}, Images: ${images.length}, Report: ${reportContent ? 'yes' : 'no'}`],
      }));
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error("Failed to fetch results:", error);
      setState((prev) => ({
        ...prev,
        isRunning: false,
        isLoading: false,
        errorMessage: `Failed to fetch results: ${errorMsg}`,
        debugLogs: [...prev.debugLogs, `[RESULTS] CRITICAL ERROR: ${errorMsg}`],
      }));
    }
  };

  /**
   * Cancel the current run.
   */
  const cancel = (): void => {
    stopPolling();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setState(initialState);
  };

  /**
   * Reset state (for starting a new run).
   */
  const reset = (): void => {
    stopPolling();
    setState(initialState);
  };

  /**
   * Retry current run on error (checks status and resumes if still running).
   */
  const retry = async (): Promise<void> => {
    if (!state.runId) return;

    setState((prev) => ({
      ...prev,
      isLoading: true,
      errorMessage: null,
    }));

    try {
      const status = await checkStatus(state.runId);
      setState((prev) => ({
        ...prev,
        phase: status.phase,
        status: status.status,
        errorMessage: status.error,
        isLoading: false,
      }));

      // If still running, resume polling
      if (status.status === "pending" || status.status === "running") {
        startPolling(state.runId);
      }
    } catch (error) {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        errorMessage:
          error instanceof Error ? error.message : "Failed to retry",
      }));
    }
  };

  /**
   * Start polling with an existing run_id (for file uploads).
   */
  const startPollingWithRunId = (runId: string, sourceLabel: string = "uploaded file"): void => {
    setState((prev) => ({
      ...prev,
      runId,
      isRunning: true,
      isLoading: false,
      errorMessage: null,
      debugLogs: [
        ...prev.debugLogs,
        `[INIT] Starting polling for run: ${runId}`,
        `[INIT] Source: ${sourceLabel}`,
      ],
      phase: "pending",
      status: "pending",
      url: sourceLabel,
      startedAt: new Date(),
      csvData: null,
      images: [],
      reportContent: null,
    }));

    // Start polling
    startPolling(runId);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []);

  return {
    // State
    ...state,

    // Actions
    run,
    cancel,
    reset,
    retry,
    startPollingWithRunId,
  };
}

export type UsePhaseRunnerReturn = ReturnType<typeof usePhaseRunner>;
