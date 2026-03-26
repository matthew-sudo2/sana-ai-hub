/**
 * FeedbackWidget - User feedback component for quality score accuracy verification
 * Shows buttons to collect feedback on whether model's quality score was accurate
 */

import { useState, useEffect } from "react";
import { useFeedback } from "../hooks/useFeedback";
import { usePipeline } from "../context/PipelineContext";
import type { FeedbackResponse, FeedbackWidgetProps } from "../types/feedback"; 

export const FeedbackWidget = ({
  datasetHash,
  predictedScore,
  features = [],
  onFeedbackSubmitted,
}: FeedbackWidgetProps) => {
  const { runId } = usePipeline();
  const { submitFeedback, isLoading, error } = useFeedback();
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [response, setResponse] = useState<FeedbackResponse | null>(null);

  // Create a persistent key that survives tab switches
  // Use runId as primary key (persists across component remounts)
  // Fall back to datasetHash if runId not available
  const storageKey = runId ? `feedback_run_${runId}` : (datasetHash ? `feedback_${datasetHash}` : null);

  // Restore feedback state from localStorage when storage key changes
  useEffect(() => {
    if (storageKey) {
      try {
        const stored = localStorage.getItem(storageKey);
        if (stored) {
          const data = JSON.parse(stored);
          setResponse(data.response);
          setHasSubmitted(true);
        } else {
          // Reset state if no stored feedback for this key
          setResponse(null);
          setHasSubmitted(false);
        }
      } catch (e) {
        // Silently ignore localStorage errors - reset state
        setResponse(null);
        setHasSubmitted(false);
      }
    } else {
      // Reset if no storage key available
      setResponse(null);
      setHasSubmitted(false);
    }
  }, [storageKey]);

  const handleSubmit = async (label: number) => {
    const feedbackRequest = {
      dataset_hash: datasetHash,
      predicted_score: predictedScore,
      actual_quality: label,
      features: features && features.length === 8 ? features : [],
    };

    const result = await submitFeedback(feedbackRequest);
    if (result && storageKey) {
      setResponse(result);
      setHasSubmitted(true);
      // Persist to localStorage using runId key (survives tab switches)
      localStorage.setItem(storageKey, JSON.stringify({
        timestamp: new Date().toISOString(),
        response: result,
      }));
      onFeedbackSubmitted?.(result);
    }
  };

  if (hasSubmitted && response) {
    return (
      <div className="mt-4 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30"> 
        <div className="flex items-start gap-3">
          <div className="text-emerald-600 font-bold mt-1">✓</div>
          <div className="flex-1">
            <h3 className="font-display font-semibold text-emerald-800 dark:text-emerald-400 text-sm">Thank you for your feedback!</h3>
            <p className="font-body text-xs text-emerald-700 dark:text-emerald-500 mt-1">Your response has been recorded and will help improve our assessments.</p>   
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4 p-5 rounded-lg border bg-card shadow-sm">
      {error && (
        <div className="mb-4 p-3 rounded-md bg-destructive/10 border border-destructive/20">     
          <p className="text-xs text-destructive">⚠️ Error: {error}</p>
        </div>
      )}
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="font-display text-sm font-semibold text-foreground flex items-center gap-2">
            <span className="text-amber-500 text-base">⭐</span>
            How accurate was this quality score?
          </h3>
          <p className="font-body text-xs text-muted-foreground mt-1">
            Your feedback helps us improve the model's accuracy and future predictions
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <button
          onClick={() => handleSubmit(3)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 font-medium text-xs transition-colors hover:bg-emerald-500/20 disabled:opacity-50"
        >
          <span>✓</span> Perfect
        </button>
        <button
          onClick={() => handleSubmit(2)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-400 font-medium text-xs transition-colors hover:bg-blue-500/20 disabled:opacity-50"
        >
          <span>✓</span> Close enough
        </button>
        <button
          onClick={() => handleSubmit(1)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400 font-medium text-xs transition-colors hover:bg-amber-500/20 disabled:opacity-50"
        >
          <span>✗</span> Too high
        </button>
        <button
          onClick={() => handleSubmit(0)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 py-2.5 px-3 rounded-md border border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400 font-medium text-xs transition-colors hover:bg-red-500/20 disabled:opacity-50"
        >
          <span>✗</span> Very wrong
        </button>
      </div>

      {isLoading && (
        <div className="mt-4 text-center text-xs text-muted-foreground animate-pulse font-medium">    
          ⏳ Submitting feedback...
        </div>
      )}
    </div>
  );
};

